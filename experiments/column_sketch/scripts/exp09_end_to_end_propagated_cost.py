#!/usr/bin/env python3
"""Stage 1, experiment 5: END-TO-END propagated cost -- does the fatter global bond kill the tie?

exp08 showed that, **per column**, a matrix-free global structured QR can match the local Moses
move's accuracy at roughly equal FLOPs. But it matched only by spending a larger vertical bond
(eta_q = 6-8 vs local eta = 4), because it has no disentangler to shrink it. And that per-column
comparison was on the **boundary** centre column (output leg = physical d only, no horizontal
"away" bond). The columns that dominate a real Ly-wide sweep are INTERIOR: their output leg is
``d * (carry bond handed by the previous move)``. So the decisive question:

    Does global still tie/win after we charge the downstream cost of its fatter carry bond?

Measured ground truth (this file's premise, verified on converged TFIM lx=4):
  * After a LOCAL move, the horizontal bond between the new isometric column and the new centre is
    [4,2,2,1] -- i.e. local hands a carry bond <= eta = 4.
  * Global's residual R = Q* C has final_bond = r_bond = eta_q (6-8 to match accuracy) -- global
    hands a carry bond = eta_q.
  * An interior column's output leg is d * carry, so global's interior columns are (eta_q/eta)x
    fatter, inflating every downstream factorization.

================================  COST MODEL ASSUMPTIONS  ================================
A column sweep is Ly-1 moves; the orthogonality centre starts at a boundary and each move hands a
horizontal carry bond to the next centre. We charge implementation-free FLOPs (multiply-adds), no
BLAS/wall-clock. Per column: Lx rows, physical d, toward-bond chi (horizontal toward the
UNPROCESSED side), carry-bond b (handed from the previous move; boundary b -> output leg = d),
vertical bond eta_v. Charged dimensions per row: ``din = chi``, ``dout = d*b`` (interior) or ``d``
(boundary), vertical = eta_v.

* LOCAL is MEASURED, not modelled: we run the real ``moses_move`` for every column of a real partial
  sweep and sum ``svd_flops`` over its actual svd1 + disentangler + svd2 calls (``MosesStats``), plus
  an R-absorption proxy. This carries local's real bonds (carry = eta) with no analytic model.
* GLOBAL is MODELLED (there is no end-to-end matrix-free global move yet -- ``structured_qr`` is the
  dense per-column validation). Per column we count the matrix-free FLOPs exactly:
  ``sketch = (1+2q)*ell * sum_rows 2*eta_q^2*chi_sk^2*dout*din``;
  ``ttqr  = sum_rows svd_flops(eta_q*dout, min(dense_remaining, ell*eta_q*chi_sk))`` (the TT-SVD
  right-environment bounded by the product-MPS bond, not the dense output dim);
  ``R = sum_rows 2*eta_q^2*din*dout``. This is validated against exp08 at the boundary column.
* R-ABSORPTION (both sides, a proxy for quimb's zipup ``tensor_network_1d_compress``):
  ``Lx * svd_flops(d*carry, chi*carry)`` per move -- charged for local (carry=eta) and global
  (carry=eta_q), so global pays for absorbing its fatter residual.
* PROPAGATION: local carry = eta (measured [4,2,..]); global carry = eta_q (measured final_bond).
  ``global_fact`` evaluates global at LOCAL's carry (no penalty, ~ exp08); ``global_prop`` at global's
  own carry = eta_q. The multiplier ``global_prop/global_fact`` is the downstream penalty.

IGNORED / PROXIES (stated honestly): O(1) contraction-order constants; the exact accumulated-rank
profile of boundary columns (we use uniform steady-state chi for interior columns and MEASURE the
boundary); validation/scoring costs (eps_proj, materialize, the dense SVD reference) are NOT charged
to either side; the zip-up absorption is a single-SVD-per-row proxy, not quimb's exact compression;
global's interior columns are a steady-state parametric estimate (din=chi, dout=d*carry) -- their
*scaling* with the carry bond is the point, calibrated in spirit by validating the per-column model
against exp08's boundary measurement.
=========================================================================================

Pareto-fair: global tunes (ell, q, chi_sk, eta_q); local tunes the accuracy target eta (we sweep
eta in {3,4,6} = looser/default/stricter, and also report a cheap-disentangler local variant Ndis/2).
For each matched point we record ell*, q*, chi_sk*, eta_q*, final_bond, r_bond, eps_proj, local error.
Adversarial: flag near-full-rank matches (eta_q >= 0.9*n_out), flag degenerate lx=3, separate TFIM
g=3.5 / g=3.04, include random as a hardness control (local fails there, so it is excluded from the
physical-state conclusion). Honest reporting rule: do not claim global wins unless the PROPAGATED cost
wins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


# --------------------------- cost primitives (see header) --------------------------- #
def _absorb_flops(carry, chi, d, Lx):
    """R-column zip-up absorption proxy: Lx per-row SVDs of ~ (d*carry) x (chi*carry)."""
    return Lx * svd_flops(int(d * carry), int(chi * carry))


def _global_col_flops(din_dims, dout_dims, eta_q, ell, q, chi_sk):
    """Exact matrix-free FLOPs to factorize ONE column (per-row din/dout, vertical eta_q)."""
    matvec = sum(2 * eta_q * eta_q * chi_sk * chi_sk * o * s
                 for o, s in zip(dout_dims, din_dims))
    sketch = (1 + 2 * q) * ell * matvec
    tot, k, cap = 0.0, 1, ell * eta_q * chi_sk
    for i, o in enumerate(dout_dims):
        m = k * o
        n = min(int(np.prod(dout_dims[i + 1:])) * ell, cap)
        tot += svd_flops(m, max(n, 1))
        k = min(eta_q, m, max(n, 1))
    rr = sum(2 * eta_q * eta_q * s * o for o, s in zip(dout_dims, din_dims))
    return sketch + tot + rr


def _interior_dims(chi, carry, d, Lx):
    """Steady-state interior column: din=chi, dout=d*carry per row (the propagation channel)."""
    return (int(chi),) * Lx, (int(d * carry),) * Lx


def _prepare_state(args, kind, lx, seed, eta):
    """Prepare a state canonical at vertical bond ``eta`` -- the SAME bond the local sweep will
    truncate to, so the sweep is clean (the state's natural bond == the truncation budget). Prepping
    at a higher bond and truncating below it would make the sweep degrade and corrupt the cost."""
    from rand_isopeps.real_isotns.moses_move import random_isotns
    psi = random_isotns(lx, args.ly, bond=args.bond, phys=args.phys, chi=args.chi,
                        eta=eta, cutoff=args.cutoff, Ndis=args.ndis, seed=seed)
    if kind.startswith("tfim@"):
        from rand_isopeps.real_isotns.tebd2 import imaginary_time, tfi_ham
        g = float(kind.split("@")[1])
        psi, _ = imaginary_time(psi, tfi_ham(lx, args.ly, g=g), taus=args.taus, steps=None,
                                chi=args.chi, eta=eta, cutoff=args.cutoff, Ndis=args.ndis)
    return psi


def _real_local_sweep(psi, jc, split, eta, args):
    """Run the real Moses sweep from the centre boundary inward; MEASURE per-move SVD FLOPs.

    Returns ``(total_factorization_flops, boundary_eps, sweep_max_eps, n_moves)`` (on a copy).
    The accuracy TARGET is ``boundary_eps`` -- the FIRST (boundary centre) move's error, the clean
    single-column quantity comparable to exp08 and to global's matched boundary column; ``sweep_max_eps``
    is the worst error across the whole sweep (a diagnostic; a multi-move sweep on an under-converged
    state can degrade a late interior column, which should not corrupt the matched target).
    """
    from collections import defaultdict

    from rand_isopeps.real_isotns.instrument import MosesStats
    from rand_isopeps.real_isotns.moses_move import moses_move

    psi = psi.copy()
    js = range(jc, 0, -1) if split == "left" else range(jc, psi.Ly - 1)
    total, boundary_eps, worst, n = 0.0, float("nan"), 0.0, 0
    for j in js:
        st = MosesStats()
        errs = moses_move(psi, j, args.chi, eta, args.cutoff, args.ndis, orientation="col",
                          sweep="up", split=split, renorm=True, stats=st)
        loc = defaultdict(float)
        for r in st.records:
            loc[r.stage] += svd_flops(r.m, r.n)
        total += sum(loc.values()) + _absorb_flops(eta, args.chi, args.phys, psi.Lx)
        move_eps = float(np.max(np.abs(errs))) if errs is not None else float("nan")
        if n == 0:
            boundary_eps = move_eps
        worst = max(worst, move_eps if move_eps == move_eps else 0.0)  # nan-safe
        n += 1
    return total, boundary_eps, worst, n


def _candidate_ells(eta_q, n_in):
    return sorted({c for c in (eta_q - 2, eta_q, eta_q + 2, eta_q + 4) if 1 <= c <= n_in})


def _match_global(op, qr_fn, target_eps, args):
    """Cheapest global config (ell,q,chi_sk,eta_q) with eps_proj <= target. Returns dict or best-eps.

    Global searches a BROAD vertical-bond set (independent of the prep bond) so it can match a loose
    target cheaply (small eta_q) or a strict one (large eta_q) -- it is not handicapped to a fixed bond.
    ``qr_fn`` is the budget-aware QR closure (dense reference at small Lx, matrix-free above).
    """
    n_in, n_out = op.n_in, op.n_out
    eta_qs = sorted({e for e in (2, 3, 4, 6, 8) if 1 <= e <= n_out})
    best, best_overall = None, None
    for eta_q in eta_qs:
        for chi_sk in args.chi_sks:
            for ell in _candidate_ells(eta_q, n_in):
                for q in args.n_powers:
                    rng = np.random.default_rng(args.seed + 101 * ell + 1009 * q
                                                + 10007 * chi_sk + 100003 * eta_q)
                    res = qr_fn(ell, eta_q, chi_sk, q, rng)
                    cand = {"ell": ell, "q": q, "chi_sk": chi_sk, "eta_q": eta_q,
                            "eps": res.eps_proj, "final_bond": res.final_bond, "r_bond": res.r_bond}
                    if best_overall is None or res.eps_proj < best_overall["eps"]:
                        best_overall = cand
                    if res.eps_proj <= target_eps and (best is None or _cfg_cost(cand) < _cfg_cost(best)):
                        best = cand
    return (best, True) if best is not None else (best_overall, False)


def _cfg_cost(cfg):
    """Cheap proxy to rank configs during the search (favours small eta_q, ell, q, chi_sk)."""
    return (1 + 2 * cfg["q"]) * cfg["ell"] * cfg["eta_q"] ** 2 * cfg["chi_sk"] ** 2


def _run_trial(task):
    args, kind, lx, seed = task
    with with_blas_threads(args.blas_threads):
        d, chi, Ly = args.phys, args.chi, args.ly
        rows = []
        for eta in args.eta_targets:                       # accuracy targets: looser/default/stricter
            # prep AT this eta so prep bond == sweep bond (clean sweep, no truncation degradation)
            psi = _prepare_state(args, kind, lx, seed, eta)
            jc, split = find_center_column(psi)
            op = from_quimb_column(psi, jc, split=split)
            # budget-aware QR (dense reference small-Lx, matrix-free above) -- no
            # (d*chi)^Lx materialize, so interior/large-Lx columns no longer OOM.
            qr_fn, _ = make_column_qr(op)
            n_out = op.n_out
            din_boundary, dout_boundary = op.input_dims, op.output_dims

            local_flops, local_eps, sweep_max_eps, n_moves = _real_local_sweep(psi, jc, split, eta, args)
            if n_moves == 0 or not (local_eps == local_eps):   # skip nan boundary error
                continue
            cfg, matched = _match_global(op, qr_fn, local_eps * (1.0 + args.match_tol), args)
            eta_q, ell, q, chi_sk = cfg["eta_q"], cfg["ell"], cfg["q"], cfg["chi_sk"]

            # boundary column (1) + interior columns (n_moves-1), parametric steady state
            din_i, dout_i_loc = _interior_dims(chi, eta, d, lx)      # local carry = eta
            _, dout_i_glob = _interior_dims(chi, eta_q, d, lx)        # global carry = eta_q
            g_bnd = _global_col_flops(din_boundary, dout_boundary, eta_q, ell, q, chi_sk)
            g_int_fact = _global_col_flops(din_i, dout_i_loc, eta_q, ell, q, chi_sk)
            g_int_prop = _global_col_flops(din_i, dout_i_glob, eta_q, ell, q, chi_sk)
            n_int = max(n_moves - 1, 0)

            global_fact = g_bnd + n_int * g_int_fact
            global_prop = (g_bnd + n_int * g_int_prop
                           + n_moves * _absorb_flops(eta_q, chi, d, lx))

            rows.append({
                "state": kind, "lx": lx, "seed": seed, "eta_target": eta, "n_moves": n_moves,
                "local_eps": local_eps, "sweep_max_eps": sweep_max_eps,
                "global_eps": cfg["eps"], "matched": int(matched),
                "local_useful": int(local_eps < args.useful_eps),
                "local_flops": local_flops, "global_fact_flops": global_fact,
                "global_prop_flops": global_prop,
                "ratio_fact": local_flops / max(global_fact, 1.0),
                "ratio_prop": local_flops / max(global_prop, 1.0),
                "prop_multiplier": global_prop / max(global_fact, 1.0),
                "eta_q": eta_q, "ell": ell, "q": q, "chi_sk": chi_sk,
                "final_bond": cfg["final_bond"], "r_bond": cfg["r_bond"], "n_out": n_out,
                "near_full_rank": int(cfg["final_bond"] >= 0.9 * n_out),
                "degenerate_lx3": int(lx == 3),
            })
        return rows


def _est_task_bytes(args):
    """Conservative dense-column peak at the largest Lx, capped at the dense budget
    (above-budget trials run matrix-free, so the real per-worker peak is bounded)."""
    maxlx = max(args.lxs)
    dense_peak = 3.0 * dense_column_nbytes(args.phys ** maxlx, args.chi ** maxlx)
    return int(min(dense_peak, 3.0 * DEFAULT_DENSE_MAX_GB * (1024 ** 3)))


def run(args):
    workers = auto_worker_count(args.workers, est_bytes_per_task=_est_task_bytes(args))
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp9-propagated-cost-{stamp}")

    # incremental write: a crash mid-sweep keeps every completed trial on disk
    rows = []
    with IncrementalCsvWriter(csv_path) as writer:
        for batch in run_parallel_stream(_run_trial, tasks, workers):
            writer.write(batch)
            rows.extend(batch)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _phys(rows, eta_target):
    """Physical (local-useful) rows at one accuracy target, excluding random."""
    return [r for r in rows if int(r["local_useful"]) == 1 and r["state"] != "random"
            and int(float(r["eta_target"])) == eta_target]


def _median_vs_lx(rows, key, lxs):
    return [float(np.median([float(r[key]) for r in rows if int(r["lx"]) == x])) or float("nan")
            if any(int(r["lx"]) == x for r in rows) else float("nan") for x in lxs]


def make_plot(rows, fig_path, args):
    """Main figure: end-to-end cost ratios + propagation multiplier vs Lx (default target).

    Secondary (-diagnostics): chosen eta_q/final_bond and matched error vs Lx.
    Pareto (-pareto): matched error vs total PROPAGATED FLOPs. All on physical (TFIM) columns at the
    default accuracy target; random is excluded (local fails there)."""
    eta0 = args.eta_targets[len(args.eta_targets) // 2]    # default (middle) target
    phys = _phys(rows, eta0)
    lxs = sorted({int(r["lx"]) for r in phys}) or sorted({int(r["lx"]) for r in rows})

    def med(rs, key):
        return _median_vs_lx(rs, key, lxs)
    n_full = sum(int(r["near_full_rank"]) for r in phys)
    note = f"{n_full}/{len(phys)} near-full-rank" if phys else "n/a"

    # ---- MAIN ----
    rf = Series(label="local / global (factorization only)", x=[float(x) for x in lxs],
                y=med(phys, "ratio_fact"), color="#56b4e9", marker="s", linestyle=":")
    rp = Series(label="local / global (propagated)", x=[float(x) for x in lxs],
                y=med(phys, "ratio_prop"), color="#222222", marker="o", linestyle="-")
    pa = Panel("End-to-end cost ratio (above 1: global cheaper)", r"column height $L_x$",
               "local FLOPs / global FLOPs", "log", [rf, rp], hlines=[1.0])
    pm = Series(label="propagation multiplier", x=[float(x) for x in lxs],
                y=med(phys, "prop_multiplier"), color="#d55e00", marker="^", linestyle="-")
    pb = Panel("Downstream propagation penalty", r"column height $L_x$",
               r"global propagated / factorization-only", "log", [pm], hlines=[1.0])
    write_line_panels(fig_path, [pa, pb], width=980, height=380)

    # ---- DIAGNOSTICS ----
    sec = str(Path(fig_path).with_name(Path(fig_path).stem + "-diagnostics.pdf"))
    bond = Series(label=r"global $\eta_q$ (matched)", x=[float(x) for x in lxs],
                  y=med(phys, "eta_q"), color="#009e73", marker="D", linestyle="-")
    leta = Series(label=r"local $\eta$ (target)", x=[float(x) for x in lxs],
                  y=[float(eta0)] * len(lxs), color="#0072b2", marker="o", linestyle="--")
    d1 = Panel("Vertical bond to match accuracy", r"column height $L_x$", "vertical bond",
               "linear", [leta, bond])
    le = Series(label="local error", x=[float(x) for x in lxs], y=med(phys, "local_eps"),
                color="#0072b2", marker="o", linestyle="-")
    ge = Series(label="global error (matched)", x=[float(x) for x in lxs], y=med(phys, "global_eps"),
                color="#009e73", marker="D", linestyle="-")
    d2 = Panel("Matched error", r"column height $L_x$", r"$\|C-\hat C\|_F/\|C\|_F$", "log", [le, ge])
    write_line_panels(sec, [d1, d2], width=980, height=380)

    # ---- PARETO ----
    par = str(Path(fig_path).with_name(Path(fig_path).stem + "-pareto.pdf"))
    series = []
    for who, key, col, mk in (("local full", "local_flops", "#0072b2", "o"),
                              ("global propagated", "global_prop_flops", "#222222", "D")):
        xs = [float(r[key]) for r in phys]
        ys = [float(r["local_eps" if who == "local full" else "global_eps"]) for r in phys]
        order = np.argsort(xs)
        series.append(Series(label=who, x=[xs[i] for i in order], y=[max(ys[i], 1e-16) for i in order],
                             color=col, marker=mk, linestyle="-" if who == "local full" else ":"))
    pp = Panel(f"Pareto: error vs propagated cost ({note})", "total sweep FLOPs",
               r"$\|C-\hat C\|_F/\|C\|_F$", "log", series)
    write_line_panels(par, [pp], width=560, height=380)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--prep-eta", type=int, default=6, help="bond the state is prepared at (>= max target)")
    parser.add_argument("--eta-targets", type=int, nargs="+", default=[3, 4, 6],
                        help="local accuracy targets (move truncation bond): looser/default/stricter")
    parser.add_argument("--chi-sks", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--n-powers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--match-tol", type=float, default=0.05)
    parser.add_argument("--useful-eps", type=float, default=0.5)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    raw = args.taus if args.taus else [0.3, 0.1, 0.03]
    args.taus = [(t, args.tau_steps) for t in raw]
    if args.quick:
        args.lxs = [3, 4]
        args.states = ["random", "tfim@3.5"]
        args.eta_targets = [4]
        args.seeds = 1
        args.taus = [(0.3, 3), (0.1, 3)]
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
