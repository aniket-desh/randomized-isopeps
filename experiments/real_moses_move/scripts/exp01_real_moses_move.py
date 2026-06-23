#!/usr/bin/env python3
"""E1 -- real one-Moses-move STAGE ABLATION (quimb isoTNS).

Roadmap experiment E1. Runs one genuine block Moses Move on a quimb PEPS and
isolates *which* local tensor-ring SVD is randomized, using the Phase-0 per-stage
control (``MosesRandConfig``) + instrumentation (``MosesStats``). 3x3 contracts
exactly, so the headline metric is the represented-state error

    eps_state = 1 - |<psi0|psi_hat>|   (psi0 the exact original PEPS)

after one move on a boundary column (source bond > target eta -> the eta cut is
lossy).

Methods (the ablation axis -- which stage(s) use the randomized range finder;
sketch fixed by --sketch):
    det             all deterministic (baseline)
    RSVD1           first SVD only
    RSVD2           second SVD only
    sketch-Q        disentangler-search SVD only
    sketch-Q+RSVD2  disentangler search + second SVD
    all-rand        all three (control)

Hypothesis (and observed): RSVD2 and sketch-Q match det accuracy/gauge (excess at
machine precision); RSVD1 buys nothing because the first SVD has rho1 ~ 1 at these
eta-isolation settings -- generous chi makes chi*eta exceed the small-lattice n1, so
svd1 is effectively full rank and randomizing it is pure overhead that only adds a
degradation tail; all-rand is a control. The rank-fraction panel makes the rho1
(svd1, ~1) vs rho2 (svd2, < 1) story explicit -- the reason SVD2 is the reliable
target. The rho1 ~ 1/d low-rank regime for svd1 appears on larger lattices (E2 / E6),
where the actual n1 = chi*eta*d exceeds the kept rank chi*eta.

Statistical design (the standard protocol): paired over problem-instance seeds
(the PEPS) x sketch seeds; every method runs on the SAME psi0; excess is computed
against det on that same instance; plots show median + IQR band over instances x seeds.

Parallel: tasks fan out over processes (each rebuilds its psi0 from seed, so pairing
holds and nothing quimb crosses a process boundary). --workers 0 leaves headroom on
the machine; --blas-threads 1 avoids oversubscription. Needs quimb (`pip install quimb`).
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.experiment_utils.cost_model import rsvd_flops, svd_truncate_flops
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import run_parallel, with_blas_threads
# NOTE: rand_isopeps.plotting (-> matplotlib) is imported lazily in the plotting
# functions only. Spawned workers re-import this module; keeping matplotlib out of
# the import path avoids a BrokenProcessPool from matplotlib init in subprocesses.

SUITE = "real_moses_move"

# label, color, marker, stages randomized (MosesRandConfig field names)
METHODS = [
    ("det",            "#000000", "o", ()),
    ("RSVD1",          "#e69f00", "s", ("svd1",)),
    ("RSVD2",          "#0072b2", "^", ("svd2",)),
    ("sketch-Q",       "#009e73", "D", ("disentangler_inner",)),
    ("sketch-Q+RSVD2", "#cc79a7", "v", ("disentangler_inner", "svd2")),
    ("all-rand",       "#999999", "x", ("svd1", "disentangler_inner", "svd2")),
]


def _workers(requested):
    """Safe worker count: honor a positive request, else leave generous headroom
    (use the performance cores, keep the machine responsive -- never oversubscribe)."""
    if requested and requested > 0:
        return requested
    cpu = os.cpu_count() or 2
    return max(1, min(12, cpu - 6))


def _config(stages, sketch, oversample, rng):
    """A MosesRandConfig randomizing exactly ``stages`` (empty -> deterministic)."""
    from rand_isopeps.isotns import MosesRandConfig, RandSVD
    if not stages:
        return None
    r = RandSVD(sketch=sketch, oversample=oversample, rng=rng)
    return MosesRandConfig(**{stage: r for stage in stages})


def _stage_metrics(stats):
    """Per-stage timing / rank-fraction / size / tail from a MosesStats collector."""
    summ = stats.summary()
    def g(stage, key):
        return summ.get(stage, {}).get(key, float("nan"))
    def tail(stage):
        recs = [r for r in stats.records if r.stage == stage]
        return float(np.sqrt(sum(r.tail ** 2 for r in recs))) if recs else float("nan")
    return {
        "t_svd1": g("svd1", "t_svd"), "t_dis": g("disentangler_inner", "t_svd"),
        "t_svd2": g("svd2", "t_svd"),
        "rho1": g("svd1", "max_rho"), "rho2": g("svd2", "max_rho"),
        "dim1": g("svd1", "max_dim"), "dim2": g("svd2", "max_dim"),
        "tau1": tail("svd1"), "tau2": tail("svd2"),
    }


def _run_one(args, eta, inst, sk, mi, name, stages):
    import quimb.tensor as qtn
    from rand_isopeps.isotns import MosesStats, moses_move
    psi0 = qtn.PEPS.rand(args.lx, args.ly, bond_dim=args.bond, phys_dim=args.phys,
                         seed=args.seed + 100003 * inst)
    psi0.normalize()
    rng = (np.random.default_rng(args.seed + 100003 * inst + 1009 * eta + 131 * sk + 17 * mi)
           if stages else None)
    cfg = _config(stages, args.sketch, args.oversample, rng)
    stats = MosesStats(debug_tail=args.debug_tail)
    psi = psi0.copy()
    t = perf_counter()
    errs = moses_move(psi, j=args.ly - 1, chi=args.chi, eta=eta, cutoff=1e-12, Ndis=args.ndis,
                      sweep="up", split="left", renorm=False, rand=cfg, stats=stats)
    runtime = perf_counter() - t
    overlap = abs((psi0.H | psi).contract()) / (psi0.norm() * psi.norm())
    # implementation-free total FLOPs of the move's local SVDs (a stage is randomized when
    # its recorded ell > k); paired against det -> flop_speedup, the fair analogue of runtime.
    total_flops = sum(
        rsvd_flops(rec.m, rec.n, rec.k, rec.ell, n_power=1, sketch=args.sketch)
        if rec.ell > rec.k else svd_truncate_flops(rec.m, rec.n)
        for rec in stats.records)
    return {
        "method": name, "eta": eta, "instance": inst,
        "sketch": args.sketch if stages else "none", "sketch_seed": sk if stages else -1,
        "state_error": float(max(1.0 - overlap, 0.0)),
        "norm_ratio": float(psi.norm() / psi0.norm() - 1.0),
        "max_trunc_err": float(np.max(errs)), "runtime_s": runtime,
        "total_flops": float(total_flops),
        **_stage_metrics(stats),
    }


def _task(task):
    args, eta, inst, sk, mi, name, stages = task
    with with_blas_threads(args.blas_threads):
        return _run_one(args, eta, inst, sk, mi, name, stages)


def run(args):
    tasks = []
    for eta in args.etas:
        for inst in range(args.instances):
            for mi, (name, _c, _m, stages) in enumerate(METHODS):
                seeds = [0] if not stages else range(args.sketch_seeds)
                for sk in seeds:
                    tasks.append((args, eta, inst, sk, mi, name, stages))
    workers = _workers(args.workers)
    print(f"E1: {len(tasks)} tasks over {workers} workers "
          f"(etas={args.etas}, instances={args.instances}, sketch_seeds={args.sketch_seeds}, "
          f"bond={args.bond}, chi={args.chi}, sketch={args.sketch})")
    rows = run_parallel(_task, tasks, workers)
    _attach_excess(rows)
    _attach_flops(rows)
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp01-stage-ablation-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _attach_excess(rows):
    """Paired excess over det on the same (eta, instance)."""
    base = {(r["eta"], r["instance"]): r["state_error"] for r in rows if r["method"] == "det"}
    for r in rows:
        r["excess_state_error"] = r["state_error"] - base.get((r["eta"], r["instance"]), float("nan"))


def _attach_flops(rows):
    """Paired implementation-free FLOP-speedup over det on the same (eta, instance) -- the fair,
    machine-independent analogue of the wall-clock runtime panel."""
    base = {(r["eta"], r["instance"]): r["total_flops"] for r in rows if r["method"] == "det"}
    for r in rows:
        d = base.get((r["eta"], r["instance"]), float("nan"))
        r["flop_speedup"] = d / r["total_flops"] if r.get("total_flops") else float("nan")


def _series(rows, key, order, ci=None):
    from rand_isopeps.plotting import Series
    bands = median_band(rows, "method", "eta", key, order, ci=ci)
    style = {m[0]: (m[1], m[2]) for m in METHODS}
    return [Series(label=m, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                   color=style[m][0], marker=style[m][1])
            for m, (xs, med, lo, hi) in bands.items()]


def make_plot(rows, fig_path):
    from rand_isopeps.plotting import Panel, Series, write_line_panels
    order = [m[0] for m in METHODS]
    rho_rows = []
    for r in rows:
        if r["method"] == "det":
            rho_rows.append({"stage": "rho1 (svd1)", "eta": r["eta"], "val": r["rho1"]})
            rho_rows.append({"stage": "rho2 (svd2)", "eta": r["eta"], "val": r["rho2"]})
    rho_order = ["rho1 (svd1)", "rho2 (svd2)"]
    rho_bands = median_band(rho_rows, "stage", "eta", "val", rho_order)
    rho_series = [Series(label=g, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi, color=c, marker=mk)
                  for (g, (xs, med, lo, hi)), c, mk in
                  zip(rho_bands.items(), ["#d55e00", "#0072b2"], ["o", "s"])]

    panels = [
        Panel("state error after Moses move", "eta", "1 - |<psi0|psi>|", "log",
              _series(rows, "state_error", order)),
        Panel("excess over deterministic", "eta", "state err (rand - det)", "linear",
              _series(rows, "excess_state_error", order[1:])),
        Panel("FLOP-speedup (impl-free)", "eta", "det_flops / rand_flops", "log",
              _series([r for r in rows if r["method"] != "det"], "flop_speedup", order[1:])),
        Panel("Moses-move runtime (this machine)", "eta", "runtime (s)", "log",
              _series(rows, "runtime_s", order)),
        Panel("rank fraction by stage", "eta", "rho = (k+s)/min(m,n)", "linear", rho_series),
    ]
    write_line_panels(fig_path, panels, width=1850, height=420)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--bond", type=int, default=12, help="source PEPS bond (exceeds eta -> lossy eta cut)")
    p.add_argument("--phys", type=int, default=2, help="physical dim d (rho1 ~ 1/d only once chi*eta < n1; small lattices keep rho1 ~ 1)")
    p.add_argument("--chi", type=int, default=20, help="horizontal bond (generous: isolate the eta cut)")
    p.add_argument("--etas", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--ndis", type=int, default=20)
    p.add_argument("--oversample", type=int, default=8)
    p.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch", "sparsestack", "khatri_rao"],
                   default="gaussian", help="sketch family fixed across the stage ablation")
    p.add_argument("--instances", type=int, default=16, help="problem-instance (PEPS) seeds")
    p.add_argument("--sketch-seeds", type=int, default=4, help="sketch seeds per randomized method")
    p.add_argument("--seed", type=int, default=15000)
    p.add_argument("--workers", type=int, default=0, help="parallel workers; 0 -> safe default (leaves cores free)")
    p.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker (1 avoids oversubscription)")
    p.add_argument("--debug-tail", action="store_true", help="also record the exact deterministic tail (small cases)")
    p.add_argument("--big", action="store_true", help="comprehensive sweep (more instances/seeds)")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.big:
        args.instances = 32
        args.sketch_seeds = 6
    if args.quick:
        args.etas = [2, 4]
        args.instances = 2
        args.sketch_seeds = 2
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
