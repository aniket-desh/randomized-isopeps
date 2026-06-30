#!/usr/bin/env python3
"""Stage 1, experiment 4: MATCHED-ACCURACY matrix-free cost -- the honest cost test.

exp07 charged the global sketch at a FIXED operating point (``ell = eta + 6``,
``n_power = 1``, ``chi_sk = 8``) and found it 3-110x more expensive. But that point is
massive overkill for an effective-rank ~2 physical column: a fixed oversampling rule is
the right way to be *accuracy*-fair, not the right way to make a *cost* claim. The honest
question (theory review Q7) is **cost to reach matched accuracy**: the cheapest global
configuration that ties the local Moses move's error, with EVERY stage modeled
matrix-free.

Two corrections to exp07, both applied here:

1. **Search the operating point, do not fix it.** For each real column we sweep
   ``ell in {eta-2, eta, eta+2, eta+4, eta+8}``, ``n_power in {0, 1}``, ``chi_sk in {4, 8}``
   (vertical bond ``eta_q = eta``, matched to the local truncation budget) and keep the
   *cheapest* config whose structured-QR ``eps_proj`` is <= the local move's error. For
   contrast we also report the exp07 fixed point (``ell=eta+6, q=1, chi_sk=8``).

2. **Model the TT-QR and the residual matrix-free too.** exp07's sketch was already the
   matrix-free MPO-MPS count ``ell * sum_cores 2 ml mr chi_sk^2 o s`` (NOT a dense
   ``C Omega``), and that term dominated. But its TT-QR and ``R = Q* C`` used the dense
   remaining output dimension. Here the TT-SVD sweep's right-environment column dim is
   BOUNDED by the product-MPS bond ``ell * D * chi_sk`` (``apply_mpo_to_mps`` makes a
   bond-``D*chi_sk`` MPS), and ``R`` is the bond-``eta`` column contracted against the
   bond-``D`` MPO. At the reachable sizes ``d^Lx`` is tiny so this barely moves the
   verdict; it matters for the large-Lx projection where neither side can be materialized.

LOCAL is the SAME real block Moses as exp07: ``moses_move`` + ``MosesStats``, summing
``svd_flops`` over ``svd1 + disentangler_inner + svd2`` (the disentangler included). The
R-column absorption is excluded on BOTH sides (a shared zip-up of similar bond).

Headline: ``local_flops / global_matched_flops`` vs ``Lx``. Above 1, the global sketch is
cheaper at matched accuracy; this is the number exp07 should have reported.
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
from rand_isopeps.column.operator import DEFAULT_DENSE_MAX_GB, dense_column_nbytes
from rand_isopeps.column.structured_qr import make_column_qr
from rand_isopeps.experiment_utils.cost_model import svd_flops
from rand_isopeps.io_utils import IncrementalCsvWriter, output_paths, timestamp_slug
from rand_isopeps.parallel import (
    auto_worker_count,
    run_parallel_stream,
    with_blas_threads,
)
from rand_isopeps.plotting import Panel, Series, state_style, write_line_panels

SUITE = "column_sketch"


# --- matrix-free FLOP model for the global structured column QR -----------------------
def _matvec_flops(op, chi_sk):
    """ONE matrix-free MPO-MPS product C*omega (apply_mpo_to_mps), probe bond chi_sk.

    Per site the einsum ``axyb,lyr->laxrb`` contracts the shared input leg y: cost
    ``2 ml mr chi_sk^2 o s`` for an MPO core ``(ml,o,s,mr)`` and a probe of bond chi_sk.
    Summed over the column this is ``O(Lx D^2 chi_sk^2 d_in d_out)`` -- the access-model
    cost, no ``prod(o_i)`` dense vector ever formed."""
    return sum(2 * ml * mr * (chi_sk ** 2) * o * s
               for (ml, o, s, mr) in (c.shape for c in op.cores))


def _ttqr_mf_flops(out_dims, ell, eta_q, mpo_bond, chi_sk):
    """Matrix-free TT-SVD left-canonical sweep of the sampled range Y.

    Per output site, an SVD of ``(eta_q*o) x (right environment)``. The right-env column
    dim is the MIN of the dense remaining output product (``prod(o_{i+1:}) * ell``) and the
    product-MPS bond ``ell * D * chi_sk`` -- ``apply_mpo_to_mps`` yields a bond-``D*chi_sk``
    MPS per probe, so the matrix-free representation never exceeds that. Small Lx -> the
    dense product wins the min (and this == exp07's count); large Lx -> the MPS bond caps it."""
    tot = 0.0
    k = 1
    mps_cap = ell * mpo_bond * chi_sk
    for i, o in enumerate(out_dims):
        m = k * o
        dense_rem = int(np.prod(out_dims[i + 1:])) * ell
        n = min(dense_rem, mps_cap)
        tot += svd_flops(m, max(n, 1))
        k = min(eta_q, m, max(n, 1))
    return tot


def _r_mf_flops(op, eta_q):
    """Matrix-free residual ``R = Q_iso^* C``: contract the bond-``eta_q`` isometric column
    against the bond-``D`` column MPO over the output legs, site by site (an MPS over the
    absorbed legs). Per site ``~ 2 eta_q^2 ml mr o s``."""
    return sum(2 * (eta_q ** 2) * ml * mr * o * s
               for (ml, o, s, mr) in (c.shape for c in op.cores))


def _global_flops(op, ell, q, chi_sk, eta_q):
    """Total matrix-free global cost (sketch + TT-QR + residual) and its breakdown."""
    sketch = (1 + 2 * q) * ell * _matvec_flops(op, chi_sk)
    ttqr = _ttqr_mf_flops(op.output_dims, ell, eta_q, op.mpo_bond, chi_sk)
    rr = _r_mf_flops(op, eta_q)
    return sketch + ttqr + rr, sketch, ttqr, rr


def _local_cost_and_acc(psi, jc, split, args):
    """The real block Moses column factorization: total SVD FLOPs (svd1+disent+svd2) and
    the move's reported max truncation error -- the accuracy the global side must match."""
    from collections import defaultdict

    from rand_isopeps.real_isotns.instrument import MosesStats
    from rand_isopeps.real_isotns.moses_move import moses_move

    st = MosesStats()
    errs = moses_move(psi.copy(), jc, args.chi, args.eta, args.cutoff, args.ndis,
                      orientation="col", sweep="up", split=split, renorm=True, stats=st)
    loc = defaultdict(float)
    for r in st.records:
        loc[r.stage] += svd_flops(r.m, r.n)
    total = sum(loc.values())
    acc = float(np.max(np.abs(errs))) if errs is not None else float("nan")
    return total, acc, dict(loc)


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


def _candidate_ells(eta, n_in):
    return sorted({c for c in (eta - 2, eta, eta + 2, eta + 4, eta + 8) if 1 <= c <= n_in})


def _run_trial(task):
    args, kind, lx, seed = task
    with with_blas_threads(args.blas_threads):
        psi = _prepare_state(args, kind, lx, seed)
        jc, split = find_center_column(psi)
        local_flops, local_eps, _ = _local_cost_and_acc(psi, jc, split, args)

        op = from_quimb_column(psi, jc, split=split)
        # Budget-aware QR: dense reference (exact eps_proj) at small Lx, matrix-free
        # (estimated eps_proj, no (d*chi)^Lx allocation) once the column blows the
        # memory budget -- so large-Lx runs no longer materialize and OOM.
        qr_fn, _ = make_column_qr(op)
        n_in, n_out = op.n_in, op.n_out
        # local is only a meaningful accuracy target when the move itself succeeded; on an
        # incompressible random column local_eps ~ O(1) and "matching" it is vacuous.
        local_useful = local_eps < args.useful_eps
        # match: global must be at least as accurate as local (eps_proj <= local_eps*(1+tol)).
        # Give global every chance -- it may spend MORE vertical bond eta_q (it has no
        # disentangler to shrink it) as well as more probes/power to reach local's accuracy.
        thresh = local_eps * (1.0 + args.match_tol)
        eta_qs = sorted({e for e in (args.eta, args.eta + 2, args.eta + 4) if 1 <= e <= n_out})

        best = None          # cheapest config that matches local accuracy
        best_overall = None  # cheapest-by-eps fallback (for the infeasible case)
        for eta_q in eta_qs:
            for chi_sk in args.chi_sks:
                for ell in _candidate_ells(eta_q, n_in):
                    for q in args.n_powers:
                        rng = np.random.default_rng(args.seed + 7 * seed + 13 * lx + 101 * ell
                                                    + 1009 * q + 10007 * chi_sk + 100003 * eta_q)
                        res = qr_fn(ell, eta_q, chi_sk, q, rng)
                        gflops, sk, tq, rr = _global_flops(op, ell, q, chi_sk, eta_q)
                        cand = {"ell": ell, "q": q, "chi_sk": chi_sk, "eta_q": eta_q,
                                "eps": res.eps_proj, "flops": gflops, "sketch": sk, "ttqr": tq, "r": rr}
                        if best_overall is None or res.eps_proj < best_overall["eps"]:
                            best_overall = cand
                        if res.eps_proj <= thresh and (best is None or gflops < best["flops"]):
                            best = cand

        matched = best is not None
        chosen = best if matched else best_overall
        # exp07's fixed operating point, for the overcharge contrast
        fixed_ell = min(args.eta + 6, n_in)
        fixed_flops, *_ = _global_flops(op, fixed_ell, 1, 8, args.eta)

        return [{
            "state": kind, "lx": lx, "seed": seed, "center_j": jc, "n_in": n_in, "n_out": n_out,
            "mpo_bond": op.mpo_bond, "local_flops": local_flops, "local_eps": local_eps,
            "local_useful": int(local_useful), "matched": int(matched),
            "global_flops": chosen["flops"], "global_eps": chosen["eps"],
            "global_sketch": chosen["sketch"], "global_ttqr": chosen["ttqr"], "global_r": chosen["r"],
            "ell_star": chosen["ell"], "q_star": chosen["q"], "chi_sk_star": chosen["chi_sk"],
            "eta_q_star": chosen["eta_q"], "global_fixed_flops": fixed_flops,
            "ratio_matched": local_flops / max(chosen["flops"], 1.0),
            "ratio_fixed": local_flops / max(fixed_flops, 1.0),
        }]


def _est_task_bytes(args):
    """Per-trial peak RSS estimate -> caps the worker count so N processes cannot OOM.

    Max of two terms: the dense column peak (capped at the budget; above it the QR runs
    matrix-free) and the TEBD state prep, which DOMINATES and scales steeply with the
    vertical bond ``eta`` (exact-contraction RSS: ~3 GB at eta=4, ~21-30 GB at eta=6 on
    this suite -> ~(eta/4)^5.2). The prep term is what drove the 108 GB OOM, not the
    column. See reports/incidents/2026-06-29."""
    maxlx = max(args.lxs)
    dense_peak = min(3.0 * dense_column_nbytes(args.phys ** maxlx, args.chi ** maxlx),
                     3.0 * DEFAULT_DENSE_MAX_GB * (1024 ** 3))
    prep_peak = 3.0e9 * (args.eta / 4.0) ** 5.2 * (args.chi / 8.0) ** 5.2
    return int(max(dense_peak, prep_peak))


def run(args):
    workers = auto_worker_count(args.workers, est_bytes_per_task=_est_task_bytes(args))
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp8-matched-cost-{stamp}")

    # incremental write: a crash mid-sweep keeps every completed trial on disk
    rows = []
    with IncrementalCsvWriter(csv_path) as writer:
        for batch in run_parallel_stream(_run_trial, tasks, workers):
            writer.write(batch)
            rows.extend(batch)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _median_vs_lx(rows, key, lxs, predicate=None):
    out = []
    for lx in lxs:
        vals = [float(r[key]) for r in rows if int(r["lx"]) == lx
                and (predicate is None or predicate(r))]
        out.append(float(np.median(vals)) if vals else float("nan"))
    return out


def make_plot(rows, fig_path, args):
    """Two panels (physical columns only -- random is excluded because the local move
    itself fails there, so 'matched accuracy' is vacuous): matched-accuracy FLOPs (local
    vs cheapest accuracy-matched global, plus the exp07 fixed point) and the cost ratio,
    vs Lx. The break-even line at ratio=1 separates 'global cheaper' (above) from 'local
    cheaper' (below). The fraction of columns where global could even MATCH local's
    accuracy is annotated -- when it cannot, the global bar is its best-accuracy attempt."""
    useful = [r for r in rows if int(r["local_useful"]) == 1]
    pool = useful or rows  # fall back to all rows if nothing converged (e.g. --quick)
    lxs = sorted({int(r["lx"]) for r in pool})
    n_match = sum(int(r["matched"]) for r in useful)
    frac = f"{n_match}/{len(useful)} matched" if useful else "n/a"

    local = Series(label="local Moses (real, +disent.)",
                   x=[float(x) for x in lxs], y=_median_vs_lx(pool, "local_flops", lxs),
                   color="#0072b2", marker="o", linestyle="-")
    gmatch = Series(label="global accuracy-matched (matrix-free)",
                    x=[float(x) for x in lxs], y=_median_vs_lx(pool, "global_flops", lxs),
                    color="#009e73", marker="D", linestyle="-")
    gfixed = Series(label="global exp07 fixed point",
                    x=[float(x) for x in lxs], y=_median_vs_lx(pool, "global_fixed_flops", lxs),
                    color="#d55e00", marker="^", linestyle=":")
    panel_a = Panel("Matched-accuracy column-factorization FLOPs", r"column height $L_x$",
                    "FLOPs", "log", [local, gmatch, gfixed])

    ratio_m = Series(label="local / global (accuracy-matched)",
                     x=[float(x) for x in lxs], y=_median_vs_lx(pool, "ratio_matched", lxs),
                     color="#222222", marker="o", linestyle="-")
    ratio_f = Series(label="local / global (exp07 fixed)",
                     x=[float(x) for x in lxs], y=_median_vs_lx(pool, "ratio_fixed", lxs),
                     color="#d55e00", marker="^", linestyle=":")
    panel_b = Panel(f"Cost ratio (above 1: global cheaper) -- {frac}", r"column height $L_x$",
                    "local FLOPs / global FLOPs", "log", [ratio_m, ratio_f], hlines=[1.0])

    write_line_panels(fig_path, [panel_a, panel_b], width=980, height=380)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--chi-sks", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--n-powers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--match-tol", type=float, default=0.05,
                        help="accept global if eps_proj <= local_eps*(1+tol)")
    parser.add_argument("--useful-eps", type=float, default=0.5,
                        help="local is a meaningful accuracy target only if local_eps < this")
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8000)
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
