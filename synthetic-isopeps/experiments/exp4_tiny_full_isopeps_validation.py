#!/usr/bin/env python3
"""tiny explicit isometry validation for the canonical-form intuition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rand_isopeps.io_utils import timestamp_slug, write_csv
from rand_isopeps.randomized_svd import isometry_defect_columns
from rand_isopeps.synthetic_tensors import random_complex, random_orthonormal_columns


def run(args: argparse.Namespace) -> str:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for trial in range(args.trials):
        physical_dim = args.d ** (args.lx * args.ly)
        center_dim = args.center_dim
        iso = random_orthonormal_columns(physical_dim, center_dim, rng)
        center = random_complex((center_dim, args.p), rng)
        full_state_block = iso @ center
        norm_full = np.linalg.norm(full_state_block)
        norm_center = np.linalg.norm(center)
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
            }
        )
    stamp = timestamp_slug()
    csv_path = f"outputs/data/exp4-tiny-isometry-{stamp}.csv"
    write_csv(csv_path, rows)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=3)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--center-dim", type=int, default=16)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4567)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.trials = 2
    return args


if __name__ == "__main__":
    data = run(parse_args())
    print(f"wrote {data}")
