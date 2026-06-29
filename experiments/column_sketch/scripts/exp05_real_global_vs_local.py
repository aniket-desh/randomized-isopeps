#!/usr/bin/env python3
"""Stage 1, experiment 1: global-vs-local ACCURACY on REAL isoTNS columns.

The decisive test of the reframed thesis (per the theory review): on a *physical*
(area-law) column the global sketch and the local Moses move should TIE on accuracy
-- both hitting the Eckart-Young optimum, because the column is low flat-rank AND low
local-TN-rank -- while the global sketch should still beat the greedy local sweep on a
*random* column (high flat rank, greedy-suboptimal). If they tie on TFIM, the accuracy
story is gone for the physical regime and the real bet is matched-accuracy COST (fewer
passes/decompositions, possibly no disentangler).

For the same extracted real column ``C_j`` (``from_quimb_column``), at matched rank ``k``,
this compares the relative column error ``||C - C_hat||_F / ||C||_F`` of:

* ``eckart-young`` -- the optimal rank-``k`` tail (the floor everyone is racing to);
* ``global-gauss`` / ``global-rMPS`` / ``global-kron`` -- ``Q=orth(C Omega)``, ``C_hat=QQ*C``
  with a dense Gaussian / rMPS(chi_sk=8) / Kronecker(chi_sk=1) probe;
* ``local-det`` / ``local-rand`` -- the sequential greedy local Moses column QR
  (deterministic / randomized SVDs), truncating the carried column rank to ``k``.

States: ``random`` (``random_isotns``, the flat/hard case) vs ``tfim@g`` (2D
transverse-field Ising ground state via TEBD2, area-law). NB: this measures the
range-finder QB error ``eps_proj`` -- whether the captured range is a *valid low-bond
isometric isoPEPS column* is the separate Stage-1 feasibility gate (structured QR).
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
from rand_isopeps.column.global_range import global_column_range, reference_svd
from rand_isopeps.column.local_moses import local_column_qr
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import (
    METHOD_ORDER,
    Panel,
    Series,
    method_style,
    state_style,
    write_panel_grid,
)

SUITE = "column_sketch"


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
        psi = _prepare_state(args, kind, lx, seed)
        jc, split = find_center_column(psi)
        op = from_quimb_column(psi, jc, split=split)
        c = op.materialize()
        fd = op.input_dims
        n_out = c.shape[0]
        ref = reference_svd(c)
        s = ref[1] / max(float(np.linalg.norm(ref[1])), 1e-300)
        probe_rng = np.random.default_rng(args.seed + 7 * seed + 13 * lx)
        ks = sorted({k for k in args.ks if 1 <= k < min(n_out, c.shape[1])})
        rows = []
        for k in ks:
            ey = float(np.sqrt(np.sum(s[k:] ** 2)))
            # FAIR global: give the randomized sketch its standard oversampling (ell = k + p
            # matrix-MPS products), then truncate to the matched output rank k -- so every method
            # produces a rank-k approximation. ell=k (no oversampling) would handicap the sketch.
            ell = min(k + args.oversample, c.shape[1])

            def _glob(kind, chi):
                return global_column_range(c, fd, ell=ell, chi_sk=chi, sketch_kind=kind,
                                           target_rank=k, n_power=args.n_power, rng=probe_rng,
                                           ref_svd=ref).rank_r_approx_error
            res = {
                "eckart_young": ey,
                "global_gauss": _glob("gaussian", 0),
                "global_rmps": _glob("rmps", 8),
                "global_kron": _glob("kron", 1),
                "local_det": local_column_qr(op, k, randomized=False, reference=c).rel_error,
                "local_rand": local_column_qr(op, k, randomized=True, oversample=args.oversample,
                                              n_power=args.n_power, rng=probe_rng, reference=c).rel_error,
            }
            for m, val in res.items():
                rows.append({"state": kind, "lx": lx, "seed": seed, "k": k,
                             "n_out": n_out, "center_j": jc, "method": m,
                             "rel_error": float(val), "excess_error": float(val) - ey})
        return rows


def run(args):
    workers = auto_worker_count(args.workers)
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp5-real-gvl-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


# PLOT.md section 1: Kron does not belong in the physical-accuracy plot (it lives
# in the rMPS probe-quality figure), so it is dropped from both panels here. The
# excess panel also omits the Eckart-Young floor, which is zero by definition.
_ERROR_METHODS = ["eckart_young", "global_gauss", "global_rmps", "local_det", "local_rand"]
_EXCESS_METHODS = ["global_gauss", "global_rmps", "local_det", "local_rand"]


def _series(rows, value_key, methods, floor=1e-16):
    """Median + IQR bands over seeds for ``methods`` (canonical METHOD_ORDER),
    styled through ``method_style`` so the grammar lives in the plotting module.
    Values are floored at ``floor`` so nothing hits zero on the log axis."""
    order = [m for m in METHOD_ORDER if m in methods]
    bands = median_band(rows, group_key="method", x_key="k", value_key=value_key, group_order=order)
    out = []
    for key, (xs, med, lo, hi) in bands.items():
        label, color, marker, ls = method_style(key)
        out.append(Series(label=label, x=[float(v) for v in xs], y=[max(m, floor) for m in med],
                          ylow=[max(v, floor) for v in lo], yhigh=[max(v, floor) for v in hi],
                          color=color, marker=marker, linestyle=ls))
    return out


def make_plot(rows, fig_path, args):
    """One faceted grid: rows = states, cols = (column error, excess over EY) vs k.

    Col A is the relative column error with the Eckart-Young floor (dotted gray, no
    marker); col B is the excess over that optimum. Both log-y; bands are the 25-75
    IQR over seeds. Kron is dropped (probe-figure material); the EY floor is omitted
    from the excess panel since it is zero by definition.
    """
    grid = []
    for state in args.states:
        sub = [r for r in rows if r["state"] == state]
        grid.append([
            Panel("Column error", r"rank $k$", r"$\|C-\hat C\|_F/\|C\|_F$", "log",
                  _series(sub, "rel_error", _ERROR_METHODS)),
            Panel("Excess over Eckart-Young", r"rank $k$", "excess (method - optimal)", "log",
                  _series(sub, "excess_error", _EXCESS_METHODS)),
        ])
    write_panel_grid(fig_path, grid,
                     row_titles=[state_style(s)[0] for s in args.states],
                     col_titles=["Column error", "Excess over Eckart-Young"],
                     cell_width=440, cell_height=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    raw = args.taus if args.taus else [0.3, 0.1, 0.03]
    args.taus = [(t, args.tau_steps) for t in raw]
    if args.quick:
        args.lxs = [3, 4]
        args.states = ["random", "tfim@3.5"]
        args.ks = [1, 2, 4]
        args.seeds = 2
        args.taus = [(0.3, 3), (0.1, 3)]
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
