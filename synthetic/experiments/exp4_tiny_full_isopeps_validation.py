#!/usr/bin/env python3
"""tiny explicit isometry validation for the canonical-form intuition.

Validates the property the whole Moses Move rests on: an isometric center
preserves norm and has a vanishing isometry defect. Sweeps the center
(orthogonality) dimension and reports the median over many trials with an
inter-quartile band, so the figure shows both quantities staying at machine
precision across the sweep rather than reporting a single configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.randomized_svd import isometry_defect_columns
from rand_isopeps.plotting import Panel, Series, write_line_panels
from rand_isopeps.synthetic_tensors import random_complex, random_orthonormal_columns


def run(args: argparse.Namespace) -> tuple[str, str]:
    physical_dim = args.d ** (args.lx * args.ly)
    rows: list[dict[str, object]] = []
    for center_dim in args.center_dims:
        if center_dim > physical_dim:
            continue
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + 100 * center_dim + trial)
            iso = random_orthonormal_columns(physical_dim, center_dim, rng)
            center = random_complex((center_dim, args.p), rng)
            full_state_block = iso @ center
            norm_full = float(np.linalg.norm(full_state_block))
            norm_center = float(np.linalg.norm(center))
            rows.append(
                {
                    "trial": trial,
                    "lx": args.lx,
                    "ly": args.ly,
                    "d": args.d,
                    "p": args.p,
                    "physical_dim": physical_dim,
                    "center_dim": center_dim,
                    "norm_full": norm_full,
                    "norm_center": norm_center,
                    "relative_norm_error": abs(norm_full - norm_center) / norm_center,
                    "isometry_defect": isometry_defect_columns(iso),
                    "series": "isometry",
                }
            )
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp4-tiny-isometry-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _series(rows: list[dict[str, object]], key: str, color: str, marker: str) -> list[Series]:
    bands = median_band(rows, group_key="series", x_key="center_dim", value_key=key, group_order=("isometry",))
    return [
        Series(label="random isometry", x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi, color=color, marker=marker)
        for _, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows: list[dict[str, object]], fig_path: str) -> None:
    panels = [
        Panel("norm preservation", "center dim", "relative norm error", "log",
              _series(rows, "relative_norm_error", "#0072b2", "o")),
        Panel("isometry defect", "center dim", "||Q^H Q - I||_F", "log",
              _series(rows, "isometry_defect", "#009e73", "^")),
    ]
    write_line_panels(fig_path, panels, width=820, height=420)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=3)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--center-dims", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4567)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.center_dims = [4, 16]
        args.trials = 3
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
