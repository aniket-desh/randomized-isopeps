#!/usr/bin/env python3
"""column-level local error accumulation experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rand_isopeps.column_moses import run_column_moses_surrogate
from rand_isopeps.io_utils import timestamp_slug, write_csv
from rand_isopeps.local_ring_decomp import LocalMode
from rand_isopeps.parallel import auto_worker_count, run_parallel, with_blas_threads
from rand_isopeps.plotting import MARKERS, PALETTE, Panel, Series, write_line_panels
from rand_isopeps.tn_shapes import MosesDims


MODES: tuple[LocalMode, ...] = ("det", "rand_first", "rand_second", "rand_both")


def _run_column_task(task: tuple[argparse.Namespace, int, int, LocalMode]) -> dict[str, object]:
    args, lx, trial, mode = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=args.eta, d=args.d, p=args.p)
        result = run_column_moses_surrogate(
            dims,
            lx=lx,
            mode=mode,
            ensemble=args.ensemble,
            noise=args.noise,
            decay=args.decay,
            oversample=args.oversample,
            n_power=args.n_power,
            sketch=args.sketch,
            seed=args.seed + 10000 * lx + 100 * trial,
        )
        return {
            **result.as_dict(),
            "trial": trial,
            "ensemble": args.ensemble,
            "noise": args.noise,
            "oversample": args.oversample,
            "n_power": args.n_power,
            "sketch": args.sketch,
        }


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [
        (args, lx, trial, mode)
        for lx in args.lx_values
        for trial in range(args.trials)
        for mode in MODES
    ]
    rows = run_parallel(_run_column_task, tasks, workers)

    stamp = timestamp_slug()
    csv_path = f"outputs/data/exp2-column-{stamp}.csv"
    fig_path = f"outputs/figures/exp2-column-{stamp}.pdf"
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def grouped_mean(rows: list[dict[str, object]], key: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["lx"]))].append(float(row[key]))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for mode in MODES:
        xs = sorted({lx for m, lx in grouped if m == mode})
        ys = [float(np.mean(grouped[(mode, lx)])) for lx in xs]
        out[mode] = (xs, ys)
    return out


def make_plot(rows: list[dict[str, object]], fig_path: str) -> None:
    panels: list[Panel] = []
    for title, key, ylabel, yscale in [
        ("error accumulation", "rel_error", "column surrogate error", "log"),
        ("column cost", "runtime_s", "runtime (s)", "linear"),
    ]:
        series = grouped_mean(rows, key)
        panels.append(
            Panel(
                title=title,
                xlabel="Lx",
                ylabel=ylabel,
                yscale=yscale,
                series=[
                    Series(
                        label=mode.replace("_", " "),
                        x=[float(v) for v in xs],
                        y=ys,
                        color=PALETTE[mode],
                        marker=MARKERS[mode],
                    )
                    for mode, (xs, ys) in series.items()
                ],
            )
        )
    write_line_panels(fig_path, panels, width=820, height=420)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--eta", type=int, default=8)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--lx-values", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--ensemble", choices=["ring", "noisy_ring", "gaussian"], default="noisy_ring")
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.92)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 selects a conservative local default")
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker; use 0 for library default")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lx_values = [2, 4]
        args.trials = 1
        args.eta = min(args.eta, 6)
        args.oversample = 4
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
