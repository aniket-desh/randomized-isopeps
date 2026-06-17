"""structured final SVD2 after an exact disentangler.

Isolates the *final compression* from the disentangler search: run the exact
alt-min disentangler, then truncate the gauged second-SVD matrix
M = A(Q vtilde) of shape (eta*n2, chi*n3) to rank k2 = eta with different
sketches -- deterministic, gaussian rsvd, sparsestack rsvd, and a Khatri-Rao
rsvd that exploits the product structure of M's right index chi*n3 =
chi*(chi*eta) (factor_dims (chi, n3), spherical base). After a good gauge the
second-cut spectrum decays fast, exactly the regime where randomized SVD is
safe, so the structured sketches should match the deterministic SVD2 at lower
notional cost. Reports the excess error over the exact SVD2, the rank fraction
rho2 = (k2+oversample)/min(m,n), the second-isometry defect, and runtime,
faceted by physical dimension d.
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

FINALS = ("exact", "gaussian", "sparsestack", "khatri_rao")
_COLORS = {"exact": "#000000", "gaussian": "#0072b2", "sparsestack": "#009e73", "khatri_rao": "#cc79a7"}
_MARKERS = {"exact": "o", "gaussian": "s", "sparsestack": "^", "khatri_rao": "D"}

COLUMNS = (
    ("excess_error", "rel error (rand - exact)", "linear", "excess over exact SVD2"),
    ("second_iso_defect", "||V2 V2^H - I||_F", "log", "second isometry"),
    ("runtime_s", "runtime (s)", "log", "pipeline cost"),
)


def _run_trial(task: tuple[argparse.Namespace, int, int, int]) -> list[dict[str, object]]:
    args, d, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + 1000 * eta + trial)
        b = make_disentanglable_local(dims, rng, entangle=args.entangle, noise=args.noise)
        run_disentangle_pipeline(b, dims, rng=np.random.default_rng(0))  # warmup
        rho2 = dims.rank_fraction("second", args.oversample)
        rows, exact_err = [], float("nan")
        for kind in FINALS:
            # same disentangler gauge each method: reseed so only the final SVD2 differs
            r = run_disentangle_pipeline(
                b, dims, inner="exact", final=sketch_for_second_svd(kind, dims),
                oversample=args.oversample, n_power=args.n_power,
                dis_maxiter=args.dis_maxiter, rng=np.random.default_rng(args.seed + trial),
            )
            m = r.metrics(b)
            if kind == "exact":
                exact_err = float(m["rel_error"])
            rows.append({**dims.as_dict(), "final": kind, "trial": trial, "rho2": rho2, **m})
        for row in rows:  # excess vs the exact final SVD2 on the same tensor/gauge
            row["excess_error"] = float(row["rel_error"]) - exact_err
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, d, eta, trial) for d in args.ds for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp14-structured-svd2-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds)
    return csv_path, fig_path


def _series(rows, key, group_order):
    bands = median_band(rows, group_key="final", x_key="eta", value_key=key, group_order=group_order)
    return [
        Series(label=g, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
               color=_COLORS[g], marker=_MARKERS[g])
        for g, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows, fig_path, ds) -> None:
    # the excess panel only shows the randomized finals (exact is the zero baseline)
    grid = []
    for d in ds:
        sub = [r for r in rows if int(r["d"]) == d]
        grid.append([
            Panel("excess over exact SVD2", "eta", "rel error (rand - exact)", "linear",
                  _series(sub, "excess_error", [f for f in FINALS if f != "exact"])),
            Panel("second isometry", "eta", "||V2 V2^H - I||_F", "log",
                  _series(sub, "second_iso_defect", list(FINALS))),
            Panel("pipeline cost", "eta", "runtime (s)", "log",
                  _series(sub, "runtime_s", list(FINALS))),
        ])
    write_panel_grid(fig_path, grid, row_titles=[f"d={d}" for d in ds],
                     col_titles=["excess over exact SVD2", "second isometry", "pipeline cost"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--etas", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--entangle", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--dis-maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=14000)
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
