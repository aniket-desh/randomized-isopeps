#!/usr/bin/env python3
"""synthetic R-column absorption / MPO-MPS compression experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.mpo_mps_absorb import run_absorption_case
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import MARKERS, PALETTE, Panel, Series, write_line_panels


METHODS = ("zipup_svd", "randomized")


def _run_absorption_task(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, eta, trial = task
    with with_blas_threads(args.blas_threads):
        # warm BLAS/LAPACK so the first timed compression is not cold-started
        np.linalg.svd(np.random.default_rng(0).standard_normal((eta * args.chi, eta * args.chi)))
        results = run_absorption_case(
            l_sites=args.l_sites,
            phys_dim=args.d * args.p,
            mps_bond=eta,
            mpo_bond=args.chi,
            target_bond=eta,
            oversample=args.oversample,
            n_power=args.n_power,
            sketch=args.sketch,
            seed=args.seed + 1000 * eta + trial,
        )
        rows = [
            {
                **result.as_dict(),
                "chi": args.chi,
                "eta": eta,
                "d": args.d,
                "p": args.p,
                "trial": trial,
                "oversample": args.oversample,
                "n_power": args.n_power,
                "sketch": args.sketch,
            }
            for result in results
        ]
        # excess error of randomized vs zip-up on the *same* product
        by_method = {str(r["method"]): float(r["rel_error_exact"]) for r in rows}
        zip_err = by_method.get("zipup_svd", float("nan"))
        for row in rows:
            row["excess_error"] = float(row["rel_error_exact"]) - zip_err
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, eta, trial) for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_absorption_task, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp3-absorption-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _series(rows: list[dict[str, object]], key: str, methods: tuple[str, ...]) -> list[Series]:
    bands = median_band(rows, group_key="method", x_key="eta", value_key=key, group_order=methods)
    return [
        Series(
            label=method.replace("_", " "),
            x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
            color=PALETTE[method], marker=MARKERS[method],
        )
        for method, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows: list[dict[str, object]], fig_path: str) -> None:
    panels = [
        Panel("R-column absorption", "eta", "compression runtime (s)", "linear", _series(rows, "runtime_s", METHODS)),
        Panel("compression error", "eta", "error vs exact product", "log", _series(rows, "rel_error_exact", METHODS)),
        Panel("excess error vs zip-up", "eta", "rel error (rand - zipup)", "linear", _series(rows, "excess_error", ("randomized",))),
    ]
    write_line_panels(fig_path, panels, width=1180, height=440)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-sites", type=int, default=8)
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--etas", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 selects a conservative local default")
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker; use 0 for library default")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.l_sites = 6
        args.etas = [4, 6]
        args.trials = 1
        args.oversample = 4
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
