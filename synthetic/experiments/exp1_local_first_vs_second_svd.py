#!/usr/bin/env python3
"""local two-SVD randomized insertion experiment.

Compares randomizing the first SVD, the second SVD, both, or neither, on the
local isometric tensor-ring decomposition. Reports the median over many trials
with an inter-quartile band, a warmup pass to remove BLAS cold-start spikes, and
an *excess error* panel (randomized minus deterministic reconstruction error on
the same tensor) that sharpens the otherwise-overlapping error curves.

Stress ensembles (``--ensemble gaussian|powerlaw|expdecay`` and larger
``--noise``) test whether the second-SVD advantage survives spectra that are not
exactly ring-compatible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_ring_decomp import LocalMode, local_ring_decomp
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import MARKERS, PALETTE, Panel, Series, write_line_panels
from rand_isopeps.synthetic_tensors import make_local_tensor
from rand_isopeps.tn_shapes import MosesDims


MODES: tuple[LocalMode, ...] = ("det", "rand_first", "rand_second", "rand_both")
RAND_MODES: tuple[LocalMode, ...] = ("rand_first", "rand_second", "rand_both")


def _run_eta_trial(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=args.d, p=args.p)
        base_rng = np.random.default_rng(args.seed + 1000 * eta + trial)
        tensor = make_local_tensor(
            dims,
            base_rng,
            ensemble=args.ensemble,
            noise=args.noise,
            decay=args.decay,
            spectrum_param=args.spectrum_param,
        )
        # warmup: a discarded decomposition warms BLAS/LAPACK for these sizes so
        # the first timed mode is not penalized by cold-start (fixes the
        # nonmonotone deterministic timing curve).
        local_ring_decomp(tensor, dims, mode="det")

        rows: list[dict[str, object]] = []
        det_rel_error = float("nan")
        for mode_index, mode in enumerate(MODES):
            rng = np.random.default_rng(args.seed + 17 * trial + 100 * eta + mode_index)
            result = local_ring_decomp(
                tensor,
                dims,
                mode=mode,
                oversample=args.oversample,
                n_power=args.n_power,
                sketch=args.sketch,
                rng=rng,
            )
            metrics = result.metrics(tensor)
            if mode == "det":
                det_rel_error = float(metrics["rel_error"])
            rows.append(
                {
                    **dims.as_dict(),
                    "trial": trial,
                    "ensemble": args.ensemble,
                    "noise": args.noise,
                    "spectrum_param": args.spectrum_param,
                    "oversample": args.oversample,
                    "n_power": args.n_power,
                    "sketch": args.sketch,
                    **metrics,
                }
            )
        # excess error vs deterministic on the *same* tensor (signed)
        for row in rows:
            row["excess_error"] = float(row["rel_error"]) - det_rel_error
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, eta, trial) for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_eta_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp1-local-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _series(rows, key, modes):
    bands = median_band(rows, group_key="mode", x_key="eta", value_key=key, group_order=modes)
    return [
        Series(
            label=mode.replace("_", " "),
            x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
            color=PALETTE[mode], marker=MARKERS[mode],
        )
        for mode, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows: list[dict[str, object]], fig_path: str) -> None:
    panels = [
        Panel("local two-svd timing", "eta", "runtime (s)", "linear", _series(rows, "total_time_s", MODES)),
        Panel("reconstruction error", "eta", "relative reconstruction error", "log", _series(rows, "rel_error", MODES)),
        Panel("excess error vs det", "eta", "rel error (rand - det)", "linear", _series(rows, "excess_error", RAND_MODES)),
    ]
    write_line_panels(fig_path, panels, width=1180, height=440)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--etas", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--ensemble",
        choices=["ring", "noisy_ring", "gaussian", "powerlaw", "expdecay"],
        default="noisy_ring",
    )
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.92)
    parser.add_argument("--spectrum-param", type=float, default=1.0, help="alpha (powerlaw) or xi (expdecay)")
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 selects a conservative local default")
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker; use 0 for library default")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.etas = [4, 6]
        args.trials = 3
        args.oversample = 4
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
