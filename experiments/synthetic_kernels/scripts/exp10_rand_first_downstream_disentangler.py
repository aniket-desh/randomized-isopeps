#!/usr/bin/env python3
"""does randomizing the first SVD pollute the downstream disentangler?

The first SVD does not just create reconstruction error -- it produces the
residual ``vtilde`` that the disentangler then optimizes over (it chooses an
orthogonal ``Q`` so that ``A(Q vtilde)`` has a small rank-``k2`` tail). So a
randomized first SVD perturbs the *input to the gauge problem*. This experiment
isolates that effect on the disentanglable ensemble by comparing four pipelines:

    exact1 + D + exact2     baseline (exact first SVD, alt-min gauge, exact SVD2)
    rand1  + D + exact2     randomized first SVD feeding the gauge search
    exact1 + sketchD + exact2   sketched-objective gauge search (control)
    exact1 + D + rand2      randomized only the final compression (control)

Recorded per pipeline: reconstruction error, the second-SVD tail energy
``tau2^2 = sum_{i>eta} sigma_i^2(A(Q vtilde))``, alt-min iterations, gauge /
isometry defects, runtime. The question: does ``rand1`` raise the tail or the
reconstruction error relative to the exact-first baseline?
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import scipy.linalg as la

from rand_isopeps.aggregate import median_band
from rand_isopeps.disentangler import cut_forward, cut_inverse, disentangle_altmin, tail_energy
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid
from rand_isopeps.randomized_svd import (
    isometry_defect_columns, isometry_defect_rows, relative_frobenius_error,
    rsvd_truncate, svd_truncate,
)
from rand_isopeps.synthetic_tensors import make_disentanglable_local
from rand_isopeps.tn_shapes import MosesDims

SUITE = "synthetic_kernels"

# label -> (rand_first, sketch_disentangler, rand_second) flags
PIPELINES = {
    "exact1+D+exact2": (False, False, False),
    "rand1+D+exact2": (True, False, False),
    "exact1+sketchD+exact2": (False, True, False),
    "exact1+D+rand2": (False, False, True),
}
_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7"]
_MARKERS = ["o", "s", "^", "D"]
STYLE = {p: (_COLORS[i], _MARKERS[i]) for i, p in enumerate(PIPELINES)}

COLUMNS = (
    ("rel_error", "relative error", "log", "reconstruction error"),
    ("tail2", "tail energy (>eta)", "log", "second-SVD tail"),
    ("iters", "alt-min iters", "linear", "gauge iterations"),
    ("runtime_s", "runtime (s)", "log", "pipeline cost"),
)


def _pipeline(b, dims, rand_first, sketch_dis, rand_second, args, rng):
    t0 = perf_counter()
    a1 = b.reshape(dims.n1, dims.n2 * dims.n3)
    if rand_first:
        first = rsvd_truncate(a1, dims.k1, oversample=args.oversample, n_power=args.n_power, rng=rng)
    else:
        first = svd_truncate(a1, dims.k1)
    u1, vtilde = first.u, first.s[:, None] * first.vh

    sketch = args.sketch if sketch_dis else None
    dis = disentangle_altmin(vtilde, dims, maxiter=args.dis_maxiter, sketch=sketch,
                             oversample=args.oversample, n_power=args.n_power, rng=rng)
    u1_eff = u1 @ dis.q.conj().T
    vtilde_g = dis.q @ vtilde
    m2 = cut_forward(vtilde_g, dims)
    tail2 = tail_energy(la.svdvals(m2), dims.k2)

    if rand_second:
        second = rsvd_truncate(m2, dims.k2, oversample=args.oversample, n_power=args.n_power, rng=rng)
    else:
        second = svd_truncate(m2, dims.k2)
    b_hat = (u1_eff @ cut_inverse((second.u * second.s) @ second.vh, dims)).reshape(dims.tensor_shape)
    runtime = perf_counter() - t0

    return {
        "rel_error": relative_frobenius_error(b, b_hat),
        "tail2": tail2,
        "iters": dis.iters,
        "first_iso_defect": isometry_defect_columns(u1_eff),
        "second_iso_defect": isometry_defect_rows(second.vh),
        "gauge_defect": isometry_defect_columns(dis.q),
        "runtime_s": runtime,
    }


def _run_trial(task: tuple[argparse.Namespace, int, int, int]) -> list[dict[str, object]]:
    args, d, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + 1000 * eta + trial)
        b = make_disentanglable_local(dims, rng, entangle=args.entangle, noise=args.noise)
        _pipeline(b, dims, False, False, False, args, np.random.default_rng(0))  # warmup
        rows = []
        for label, (rf, sd, rs) in PIPELINES.items():
            metrics = _pipeline(b, dims, rf, sd, rs, args, rng)
            rows.append({**dims.as_dict(), "pipeline": label, "trial": trial,
                         "entangle": args.entangle, "noise": args.noise, **metrics})
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, d, eta, trial) for d in args.ds for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp10-downstream-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds)
    return csv_path, fig_path


def _series(rows, key):
    bands = median_band(rows, group_key="pipeline", x_key="eta", value_key=key,
                        group_order=list(PIPELINES))
    return [
        Series(label=p, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
               color=STYLE[p][0], marker=STYLE[p][1])
        for p, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows, fig_path, ds) -> None:
    """Rows = d; columns = reconstruction error, second-SVD tail, gauge iters, runtime."""
    grid = [
        [
            Panel(title, "eta", ylabel, yscale, _series([r for r in rows if int(r["d"]) == d], key))
            for key, ylabel, yscale, title in COLUMNS
        ]
        for d in ds
    ]
    write_panel_grid(fig_path, grid, row_titles=[f"d={d}" for d in ds],
                     col_titles=[title for *_, title in COLUMNS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=3)
    parser.add_argument("--etas", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--entangle", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--dis-maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.etas = [3, 4]
        args.ds = [2, 3]
        args.trials = 2
        args.oversample = 4
        args.dis_maxiter = 40
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
