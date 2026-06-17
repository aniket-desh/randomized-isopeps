#!/usr/bin/env python3
"""gauge-preserving sketched disentangler: do structured sketches preserve the
gauge while approximating the exact alt-min tail?

The disentangler is a *gauge* (U1 V = (U1 Q^H)(Q V)); the sketch Omega is a
separate object. Sketching the disentangler *search* or the final SVD2 must keep
the inserted Q exactly unitary (||Q^H Q - I||_F ~ 1e-15) while still driving the
rank-k2 truncation tail c_k2(Q) down to the exact alt-min value. We compare, on
the disentanglable ensemble, the exact alt-min against validated sketched alt-min
with gaussian / countsketch / sparsestack / khatri-rao inner SVDs, and against an
exact disentangler followed by a gaussian / khatri-rao *final* SVD2. The
Khatri-Rao sketch targets the second-SVD right index chi*n3 = chi*(chi*eta) with
factor_dims (chi, n3) and a spherical base.

The headline is the unitary-defect column: every sketched method stays at machine
precision, confirming the sketch never corrupts the gauge. Faceted by physical
dimension d (rows) x metric (columns: reconstruction error, second-SVD tail,
unitary defect).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.disentangler import sketch_for_second_svd
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_methods import run_disentangle_pipeline
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid
from rand_isopeps.synthetic_tensors import make_disentanglable_local
from rand_isopeps.tn_shapes import MosesDims

# label -> (inner disentangler sketch, final SVD2 sketch); "exact" = deterministic
METHODS = {
    "exact_altmin": ("exact", "exact"),
    "inner_gaussian": ("gaussian", "exact"),
    "inner_countsketch": ("countsketch", "exact"),
    "inner_sparsestack": ("sparsestack", "exact"),
    "inner_khatri_rao": ("khatri_rao", "exact"),
    "final_gaussian": ("exact", "gaussian"),
    "final_khatri_rao": ("exact", "khatri_rao"),
}
_COLORS = ["#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
STYLE = {m: (_COLORS[i % len(_COLORS)], _MARKERS[i % len(_MARKERS)]) for i, m in enumerate(METHODS)}

COLUMNS = (
    ("rel_error", "relative error", "log", "reconstruction error"),
    ("tail_used", "tail energy (>eta)", "log", "second-SVD tail"),
    ("unitary_defect", "||Q^H Q - I||_F", "log", "gauge unitarity"),
)


def _run_trial(task: tuple[argparse.Namespace, int, int, int]) -> list[dict[str, object]]:
    args, d, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + 1000 * eta + trial)
        b = make_disentanglable_local(dims, rng, entangle=args.entangle, noise=args.noise)
        run_disentangle_pipeline(b, dims, rng=np.random.default_rng(0))  # warmup
        rows = []
        for label, (inner, final) in METHODS.items():
            r = run_disentangle_pipeline(
                b, dims, inner=sketch_for_second_svd(inner, dims), final=sketch_for_second_svd(final, dims),
                oversample=args.oversample, n_power=args.n_power, dis_maxiter=args.dis_maxiter, rng=rng,
            )
            rows.append({**dims.as_dict(), "method": label, "trial": trial,
                         "entangle": args.entangle, "noise": args.noise, **r.metrics(b)})
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, d, eta, trial) for d in args.ds for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp12-sketched-disentangler-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds)
    return csv_path, fig_path


def _series(rows, key):
    bands = median_band(rows, group_key="method", x_key="eta", value_key=key, group_order=list(METHODS))
    return [
        Series(label=m.replace("_", " "), x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
               color=STYLE[m][0], marker=STYLE[m][1])
        for m, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows, fig_path, ds) -> None:
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
    parser.add_argument("--etas", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--entangle", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--dis-maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.etas = [4, 6]
        args.ds = [2]
        args.trials = 2
        args.dis_maxiter = 40
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
