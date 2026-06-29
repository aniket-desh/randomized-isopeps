#!/usr/bin/env python3
"""Stage 1, experiment 3: the full implementation-free COST tally -- global vs local.

Stage 1 settled that accuracy ~ties and that the global structured QR is feasible. The
project is now a cost story, and the first read showed the variational disentangler is
~86% of the local move's SVD work. But that ignored the global side's own cost: the
sketch is ``ell`` matrix-MPS products at bond ``D * chi_sk``, which could eat the saving.
This experiment tallies the *complete* implementation-free FLOPs of both column
factorizations, at a matched vertical bond, on real isoTNS columns, swept over Lx -- so
the "global eliminates 86%" claim is turned into a net ratio (which may favour either side).

Per ``cost_model`` (implementation-free is PRIMARY; wall-clock is not used here):

* **local** (the real Moses column factorization) = svd1 + disentangler + svd2 SVD FLOPs,
  read from the actual matrices via ``MosesStats``.
* **global** = the rMPS sketch ``Y = C Omega`` (``ell`` matrix-MPS products, matrix-free
  MPO-MPS contractions at bond ``D * chi_sk``, ``x (1 + 2 n_power)`` for power iterations)
  + the TT-QR sweep SVDs + the residual ``R = Q^* C`` contraction.

The R-column zip-up absorption into the neighbour is the SAME on both sides (both peel an
``R``-column of similar bond), so it is excluded from the column-factorization comparison.
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
from rand_isopeps.column.structured_qr import structured_column_qr
from rand_isopeps.experiment_utils.cost_model import gemm_flops, svd_flops
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid

SUITE = "column_sketch"

SERIES = {
    "local_total": ("local Moses (svd1+disent+svd2)", "#0072b2", "o", "-"),
    "local_disent": ("  of which disentangler", "#56b4e9", "v", ":"),
    "global_total": ("global (sketch+TTQR+R)", "#d55e00", "s", "-"),
    "global_sketch": ("  of which sketch matvecs", "#e69f00", "^", ":"),
}
SERIES_ORDER = list(SERIES)


def _matvec_flops(op, chi_sk):
    """FLOPs of ONE matrix-free MPO-MPS product C*omega (apply_mpo_to_mps), probe bond chi_sk.

    Per site the einsum contracts the shared physical/input leg: cost
    ``2 * ml*mr * chi_sk^2 * o * s`` for an MPO core ``(ml,o,s,mr)`` and a probe core of bond
    ``chi_sk``. Boundary probe bonds are 1, bounded above by chi_sk."""
    tot = 0
    for core in op.cores:
        ml, o, s, mr = core.shape
        tot += 2 * ml * mr * (chi_sk ** 2) * o * s
    return tot


def _ttqr_flops(out_dims, ell, eta_q):
    """FLOPs of the TT-SVD left-canonical sweep of Y (n_out x ell) over the output legs."""
    tot = 0
    k = 1
    for i, o in enumerate(out_dims):
        m = k * o
        n = int(np.prod(out_dims[i + 1:])) * ell
        tot += svd_flops(m, max(n, 1))
        k = min(eta_q, m, max(n, 1))
    return tot


def _prepare_state(args, kind, lx, seed):
    from rand_isopeps.real_isotns.moses_move import random_isotns
    psi = random_isotns(lx, args.ly, bond=args.bond, phys=args.phys, chi=args.chi,
                        eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis, seed=seed)
    if kind.startswith("tfim@"):
        from rand_isopeps.real_isotns.tebd2 import imaginary_time, tfi_ham
        g = float(kind.split("@")[1])
        psi, _ = imaginary_time(psi, tfi_ham(lx, args.ly, g=g), taus=args.taus, steps=None,
                                chi=args.chi, eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis)
    return psi


def _run_trial(task):
    args, kind, lx, seed = task
    with with_blas_threads(args.blas_threads):
        from collections import defaultdict
        from rand_isopeps.real_isotns.instrument import MosesStats
        from rand_isopeps.real_isotns.moses_move import moses_move
        psi = _prepare_state(args, kind, lx, seed)
        jc, split = find_center_column(psi)

        # LOCAL: the real Moses column factorization (on a copy -- the move mutates)
        st = MosesStats()
        mm_errs = moses_move(psi.copy(), jc, args.chi, args.eta, args.cutoff, args.ndis,
                             orientation="col", sweep="up", split=split, renorm=True, stats=st)
        loc = defaultdict(float)
        for r in st.records:
            loc[r.stage] += svd_flops(r.m, r.n)
        local_total = sum(loc.values())
        local_acc = float(np.max(np.abs(mm_errs))) if mm_errs is not None else float("nan")

        # GLOBAL: structured QR at the matched vertical bond eta_q = eta
        op = from_quimb_column(psi, jc, split=split)
        c = op.materialize()
        ell = min(args.eta + args.oversample, c.shape[1])
        res = structured_column_qr(op, ell=ell, eta_q=args.eta, chi_sk=args.chi_sk,
                                   sketch_kind="rmps", n_power=args.n_power,
                                   rng=np.random.default_rng(args.seed + 7 * seed + 13 * lx),
                                   reference=c)
        sketch = ell * _matvec_flops(op, args.chi_sk) * (1 + 2 * args.n_power)
        ttqr = _ttqr_flops(op.output_dims, ell, args.eta)
        r_flops = gemm_flops(res.final_bond, c.shape[0], c.shape[1])  # R = Q^* C (dense upper bound)
        global_total = sketch + ttqr + r_flops

        return [{
            "state": kind, "lx": lx, "seed": seed, "center_j": jc, "ell": ell,
            "mpo_bond": op.mpo_bond, "n_out": c.shape[0], "n_in": c.shape[1],
            "local_total": local_total, "local_svd1": loc["svd1"],
            "local_disent": loc["disentangler_inner"], "local_svd2": loc["svd2"],
            "global_total": global_total, "global_sketch": sketch, "global_ttqr": ttqr,
            "global_r": r_flops, "local_acc": local_acc, "global_eps_proj": res.eps_proj,
            "flop_ratio": local_total / max(global_total, 1.0),
        }]


def run(args):
    workers = auto_worker_count(args.workers)
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp7-cost-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _series(rows, keys):
    present = [k for k in keys if any(float(r.get(k, 0)) > 0 for r in rows)]
    out = []
    for key in present:
        xs = sorted({int(r["lx"]) for r in rows})
        med = [float(np.median([float(r[key]) for r in rows if int(r["lx"]) == x])) for x in xs]
        label, color, marker, ls = SERIES[key]
        out.append(Series(label=label, x=[float(x) for x in xs], y=med, color=color,
                          marker=marker, linestyle=ls))
    return out


def make_plot(rows, fig_path, args):
    """Rows = state; cols = FLOPs vs Lx (with disentangler/sketch breakdown), and FLOP-ratio vs Lx."""
    grid = []
    for state in args.states:
        sub = [r for r in rows if r["state"] == state]
        label = "random" if state == "random" else state.replace("tfim@", "TFIM g=")
        xs = sorted({int(r["lx"]) for r in sub})
        ratio = [float(np.median([float(r["flop_ratio"]) for r in sub if int(r["lx"]) == x])) for x in xs]
        grid.append([
            Panel(f"{label}: column-factorization FLOPs", "Lx", "FLOPs", "log",
                  _series(sub, SERIES_ORDER)),
            Panel(f"{label}: local/global FLOP ratio", "Lx", "local FLOPs / global FLOPs", "log",
                  [Series(label="local/global", x=[float(x) for x in xs], y=ratio,
                          color="#222222", marker="o")]),
        ])
    write_panel_grid(fig_path, grid,
                     row_titles=["random" if s == "random" else s.replace("tfim@", "TFIM g=")
                                 for s in args.states],
                     col_titles=["column-factorization FLOPs (breakdown)", "local / global FLOP ratio"],
                     cell_width=440, cell_height=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--chi-sk", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    raw = args.taus if args.taus else [0.3, 0.1, 0.03]
    args.taus = [(t, args.tau_steps) for t in raw]
    if args.quick:
        args.lxs = [3, 4]
        args.states = ["random", "tfim@3.5"]
        args.seeds = 2
        args.taus = [(0.3, 3), (0.1, 3)]
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
