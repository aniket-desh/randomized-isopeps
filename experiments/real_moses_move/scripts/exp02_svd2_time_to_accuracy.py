#!/usr/bin/env python3
"""E2-kernel -- matched-accuracy time-to-accuracy for the real-move SECOND SVD.

E1 (exp01) established that randomizing the local second SVD / disentangler search is
*accuracy*-neutral end-to-end on the real quimb Moses move, but found NO speedup: at
3x3 with a small source bond the move is overhead-bound (the SVD is ~11% of it) and at
fixed oversampling the tiny dense SVD beats the rsvd's fixed cost. This experiment asks
the *speed* question the way GPT/Roel framed it -- not raw runtime at fixed oversampling,
but **time-to-same-accuracy**: dial the randomized method down to just match the
deterministic accuracy, then compare wall-clock,

    speedup = T_det / min_{s,q : eps_rand <= (1+tol) eps_det} T_rand(s, q).

To make that a meaningful KERNEL comparison we isolate the SECOND-SVD matrix that
actually arises in the real move -- captured via ``MosesStats(keep_arrays=("svd2",))``
from a genuine deterministic Moses move, so the spectrum is real, not synthetic -- and
sweep into the rho2 << 1 regime where the dense SVD dominates Python/quimb overhead.

The rho2 lever is the SOURCE BOND, not eta/chi. The svd2 cut is (eta*n2) x (chi*n3)
with n2 the upward-carrier bond; empirically n2 ~ bond/4, so
``rho2 = eta/min(eta*n2, chi*n3) = eta/(eta*n2) ~ 4/bond`` -- a large source bond drives
rho2 down AND makes min(m,n)=eta*n2 large (SVD-dominated). Because this is a kernel
benchmark we never contract the PEPS, so the large bond does NOT hit the exact-overlap
memory wall that capped exp01.

Honesty notes:
- accuracy axis = the LOCAL second-cut reconstruction error ||A - A_k||/||A||, the
  faithful per-stage proxy for the state-error contribution of this stage (E1 mapped the
  svd2 excess onto the state-error excess). The matched-accuracy constraint is on it.
- rho2 uses the ACTUAL range-basis width (SVDResult.ell): for sparsestack that is
  zeta*ceil((k+s)/zeta) >= k+s, not the nominal k+s (the recording fix this depends on).
- gaussian + sparsestack only. khatri_rao is materialized (no matvec speedup yet), so a
  timing comparison would mislead; it is an accuracy experiment (exp14), not here.

Statistical protocol (project standard): paired over PEPS-instance seeds; each method
times the SAME captured matrix; median + IQR band over instances. Needs quimb.
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.experiment_utils.cost_model import compare as cost_compare
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.linalg.randomized_svd import relative_frobenius_error, rsvd_truncate, svd_truncate
from rand_isopeps.parallel import run_parallel, with_blas_threads
# matplotlib stays out of the import path (spawned workers re-import this module);
# rand_isopeps.plotting is imported lazily in make_plot only.

SUITE = "real_moses_move"

# sketch label, color, marker (gaussian + sparsestack: the two robust, matvec-honest kinds)
SKETCHES = [
    ("det",         "#000000", "o"),
    ("gaussian",    "#0072b2", "^"),
    ("sparsestack", "#009e73", "D"),
]


def _workers(requested):
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 6))


def _capture_svd2(chi, eta, bond, ndis, seed):
    """Run a deterministic real Moses move; return its largest SVD2 input matrix
    (the widest cut -> smallest rho2, the regime of interest)."""
    import quimb.tensor as qtn
    from rand_isopeps.real_isotns.instrument import MosesStats
    from rand_isopeps.real_isotns.moses_move import moses_move
    psi = qtn.PEPS.rand(3, 3, bond_dim=bond, phys_dim=2, seed=seed)
    # Per-tensor normalize, NOT psi.normalize(): the global norm is a full 9-site
    # boundary contraction that overflows to inf/nan at large bond. The kernel
    # benchmark only needs a realistic svd2 matrix from the move (a local operation),
    # not a globally normalized state, and SVD error/timing are scale-invariant.
    for t in psi:
        t.modify(data=t.data / np.linalg.norm(t.data))
    st = MosesStats(keep_arrays=("svd2",))
    moses_move(psi, j=2, chi=chi, eta=eta, cutoff=1e-12, Ndis=ndis, sweep="up",
               split="left", renorm=False, rand=None, stats=st)
    mats = [a for (s, a) in st.arrays if s == "svd2"]
    return max(mats, key=lambda a: min(a.shape))


def _median_time(fn, repeats):
    fn()  # warmup (BLAS/threadpool spin-up, first-call allocation)
    ts = []
    for _ in range(repeats):
        t = perf_counter()
        fn()
        ts.append(perf_counter() - t)
    return float(np.median(ts))


def _run_one(task):
    args, bond, inst = task
    chi, eta = args.chi, args.eta
    with with_blas_threads(args.blas_threads):
        A = _capture_svd2(chi, eta, bond, args.ndis, args.seed + 100003 * inst)
        m, n = A.shape
        mn = min(m, n)
        k = min(eta, mn)

        # deterministic reference: error + timed dense truncated SVD
        det_err = relative_frobenius_error(A, svd_truncate(A, k).reconstruct())
        t_det = _median_time(lambda: svd_truncate(A, k), args.repeats)

        base = {"chi": chi, "eta": eta, "bond": bond, "instance": inst,
                "m": m, "n": n, "min_mn": mn, "k": k}
        # det reference: cost ratios are vs itself = 1 (the baseline line)
        rows = [{**base, "sketch": "det", "rel_error": det_err, "excess": 0.0,
                 "time_s": t_det, "matched_speedup": 1.0, "flop_speedup": 1.0, "mem_ratio": 1.0,
                 "det_passes": 1, "rand_passes": 1, "rho2_used": k / mn, "cfg": "det"}]

        bound = max(args.tol * det_err, det_err + 1e-15)
        for sketch, _c, _mk in SKETCHES[1:]:
            # Pareto cloud over the tuning knobs (oversample s, power iters q)
            cloud = []  # (e, t, rho2, ell, s, q)
            for s in args.oversamples:
                for q in args.n_powers:
                    rng = np.random.default_rng(args.seed + 17 * inst + 131 * s + 7 * q)
                    res = rsvd_truncate(A, k, oversample=s, n_power=q, rng=rng, sketch=sketch)
                    e = relative_frobenius_error(A, res.reconstruct())
                    t = _median_time(
                        lambda s=s, q=q: rsvd_truncate(A, k, oversample=s, n_power=q,
                                                       rng=np.random.default_rng(0), sketch=sketch),
                        args.repeats)
                    cloud.append((e, t, res.ell / mn, res.ell, s, q))  # rho2 uses ACTUAL width
            valid = [c for c in cloud if c[0] <= bound]
            # matched-accuracy: cheapest valid config; else the most accurate (flagged *)
            chosen = min(valid, key=lambda c: c[1]) if valid else min(cloud, key=lambda c: c[0])
            e_b, t_b, rho2_b, ell_b, s_b, q_b = chosen
            # implementation-FREE algorithm cost at the matched config (FLOPs/passes/peak mem)
            cc = cost_compare(m, n, k, int(ell_b), n_power=q_b, sketch=sketch, zeta=4)
            rows.append({**base, "sketch": sketch, "rel_error": e_b, "excess": e_b / det_err - 1.0,
                         "time_s": t_b, "matched_speedup": (t_det / t_b) if valid else float("nan"),
                         "flop_speedup": cc.flop_speedup, "mem_ratio": cc.mem_ratio,
                         "det_passes": cc.det_passes, "rand_passes": cc.rand_passes,
                         "rho2_used": rho2_b, "cfg": f"s={s_b},q={q_b}" + ("" if valid else "*")})
        for r in rows:  # matrix rank fraction (method-independent regime axis)
            r["matrix_rho2"] = k / mn
        return rows


def run(args):
    tasks = [(args, bond, inst) for bond in args.bonds for inst in range(args.instances)]
    workers = _workers(args.workers)
    print(f"E2-kernel: {len(tasks)} tasks over {workers} workers "
          f"(bonds={args.bonds}, chi={args.chi}, eta={args.eta}, instances={args.instances}, "
          f"oversamples={args.oversamples}, n_powers={args.n_powers}, tol={args.tol})")
    nested = run_parallel(_run_one, tasks, workers)
    rows = [r for group in nested for r in group]
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp02-svd2-tta-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _series(rows, key, order):
    from rand_isopeps.plotting import Series
    bands = median_band(rows, "sketch", "bond", key, order)
    style = {m[0]: (m[1], m[2]) for m in SKETCHES}
    return [Series(label=g, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                   color=style[g][0], marker=style[g][1])
            for g, (xs, med, lo, hi) in bands.items()]


def make_plot(rows, fig_path):
    from rand_isopeps.plotting import Panel, Series, write_line_panels
    order = [m[0] for m in SKETCHES]
    rand_order = order[1:]
    rand_rows = [r for r in rows if r["sketch"] != "det"]

    # rho2 vs bond: geometry, method-independent -> one curve from the det rows
    rho_band = median_band([r for r in rows if r["sketch"] == "det"],
                           "sketch", "bond", "matrix_rho2", ["det"])
    xs, med, lo, hi = rho_band["det"]
    rho_series = [Series(label="rho2 = eta/min(m,n)", x=[float(v) for v in xs], y=med,
                         ylow=lo, yhigh=hi, color="#d55e00", marker="o")]

    panels = [
        # PRIMARY: implementation-free algorithm work (the fair cross-method/cross-sketch axis)
        Panel("FLOP-speedup (implementation-free)", "source bond", "det_flops / rand_flops", "log",
              _series(rand_rows, "flop_speedup", rand_order)),
        # SECONDARY: wall-clock on THIS machine -- gaussian >> sparsestack here is a dense-vs-
        # sparse-BLAS implementation gap, NOT an algorithm difference (see panel 1, where they match)
        Panel("wall-clock speedup (this machine)", "source bond", "T_det / T_rand", "log",
              _series(rand_rows, "matched_speedup", rand_order)),
        Panel("peak-memory ratio (det / rand)", "source bond", "mem ratio", "log",
              _series(rand_rows, "mem_ratio", rand_order)),
        Panel("passes over A (rand; det = 1)", "source bond", "passes", "linear",
              _series(rand_rows, "rand_passes", rand_order)),
        Panel("svd2 rank fraction", "source bond", "rho2 = eta / min(m,n)", "log", rho_series),
        Panel("excess at matched config", "source bond", "eps_rand/eps_det - 1", "linear",
              _series(rand_rows, "excess", rand_order)),
    ]
    write_line_panels(fig_path, panels, width=2280, height=400)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chi", type=int, default=24, help="horizontal bond (fixed; wide svd2 right index)")
    p.add_argument("--eta", type=int, default=8, help="second-SVD target rank (fixed)")
    p.add_argument("--bonds", type=int, nargs="+", default=[12, 16, 24, 32, 48, 64],
                   help="source PEPS bond sweep -- the rho2 lever (rho2 ~ 4/bond)")
    p.add_argument("--ndis", type=int, default=12)
    p.add_argument("--oversamples", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    p.add_argument("--n-powers", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tol", type=float, default=1.05, help="matched-accuracy tolerance: eps_rand <= tol*eps_det")
    p.add_argument("--instances", type=int, default=8, help="PEPS problem-instance seeds")
    p.add_argument("--repeats", type=int, default=7, help="timing repeats (median) per SVD config")
    p.add_argument("--seed", type=int, default=21000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--big", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.big:
        args.bonds = [12, 16, 24, 32, 48, 64, 80]
        args.instances = 12
        args.repeats = 9
    if args.quick:
        args.bonds = [16, 32]
        args.instances = 2
        args.oversamples = [0, 4]
        args.n_powers = [0, 1]
        args.repeats = 3
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
