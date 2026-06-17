#!/usr/bin/env python3
"""fixed-sketch overfitting stress test for the sketched disentangler.

The OSI/OSE guarantees are per-*fixed*-subspace: once the disentangler optimizes
Q over O(k1) against a *frozen* sketch, Q depends on the realized sketch and the
guarantee is void -- the optimizer can drive the *sketch surrogate* tail down
while the *true* tail c_k stalls or worsens (overfitting one Monte-Carlo draw).
Two mitigations: (i) draw a *fresh* independent sketch each iteration; (ii)
*validate* each candidate on the EXACT tail and reject non-improving steps.

This compares the four corners {fresh, fixed} x {validated, unvalidated} against
the exact alt-min baseline, with a deliberately *weak* inner sketch (small
oversample, no power iteration) so any overfitting is visible. Reported per mode:
the final EXACT tail c_k(Q), the optimism gap (exact tail - sketch surrogate),
the number of validation rejects, and runtime. The gauge stays exact in every
mode (the inserted Q is always a Procrustes/polar unitary).

What the data show (honest): a *frozen* sketch converges to a sketch-dependent
local minimum with a worse exact tail than exact alt-min, and validation cannot
rescue it (the bad basin is monotone-reachable, and the max_rejects bail can
stall validation at a worse minimum). The effective mitigation is a *fresh*
sketch per iteration, which escapes the trap and can even beat exact alt-min
(stochastic regularization of the non-convex greedy alt-min). Validation
guarantees per-step monotonicity of the exact tail but not a lower final tail.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_methods import run_disentangle_pipeline
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid
from rand_isopeps.synthetic_tensors import make_disentanglable_local
from rand_isopeps.tn_shapes import MosesDims

# label -> (inner, fresh_sketch, validate); "exact" inner ignores the flags
MODES = {
    "exact": ("exact", True, True),
    "fresh_val": ("gaussian", True, True),
    "fresh_noval": ("gaussian", True, False),
    "fixed_val": ("gaussian", False, True),
    "fixed_noval": ("gaussian", False, False),
}
_COLORS = ["#000000", "#0072b2", "#56b4e9", "#d55e00", "#e69f00"]
_MARKERS = ["o", "s", "^", "D", "v"]
STYLE = {m: (_COLORS[i], _MARKERS[i]) for i, m in enumerate(MODES)}

COLUMNS = (
    ("tail_used", "exact tail c_k", "log", "final exact tail"),
    ("exact_minus_surrogate", "exact - sketch tail", "linear", "sketch tail bias"),
    ("rejected", "rejected steps", "linear", "validation rejects"),
    ("runtime_s", "runtime (s)", "log", "cost"),
)


def _run_trial(task: tuple[argparse.Namespace, int, int, int]) -> list[dict[str, object]]:
    args, d, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + 1000 * eta + trial)
        b = make_disentanglable_local(dims, rng, entangle=args.entangle, noise=args.noise)
        run_disentangle_pipeline(b, dims, rng=np.random.default_rng(0))  # warmup
        rows = []
        for label, (inner, fresh, validate) in MODES.items():
            r = run_disentangle_pipeline(
                b, dims, inner=inner, final="exact", oversample=args.oversample,
                n_power=args.n_power, dis_maxiter=args.dis_maxiter,
                fresh_sketch=fresh, validate=validate, rng=rng,
            )
            extra = r.dis.extra or {}
            rows.append({
                **dims.as_dict(), "mode": label, "trial": trial, **r.metrics(b),
                "surrogate_final": extra.get("surrogate_final", float("nan")),
                "exact_minus_surrogate": extra.get("exact_minus_surrogate", float("nan")),
                "accepted": extra.get("accepted", float("nan")),
                "rejected": extra.get("rejected", float("nan")),
            })
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, d, eta, trial) for d in args.ds for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp13-overfitting-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds)
    return csv_path, fig_path


def _series(rows, key):
    bands = median_band(rows, group_key="mode", x_key="eta", value_key=key, group_order=list(MODES))
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
    # larger k1 = chi*eta (a wider O(k1) manifold) is where a frozen weak sketch
    # gets trapped at a sketch-dependent local minimum -- the overfitting regime.
    parser.add_argument("--chi", type=int, default=5)
    parser.add_argument("--etas", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--entangle", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=0, help="weak sketch: no oversampling provokes overfitting")
    parser.add_argument("--n-power", type=int, default=0)
    parser.add_argument("--dis-maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13000)
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
