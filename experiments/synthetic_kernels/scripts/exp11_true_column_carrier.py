#!/usr/bin/env python3
"""true recursive upward-carrier column Moses Move (replaces the exp2 surrogate).

exp2 measures error accumulation on a Kronecker product of *independent* local
tensors. This experiment instead runs a genuine sequential carrier: the residual
from each row is carried into the next row before its decomposition, so errors
compound through the actual carried tensor (``rand_isopeps.column_carrier``). The
column is a vertical MPS of height ``Lx`` with bond ``eta``; each Moses step is a
two-stage split mirroring the local two SVDs, so the same det / rand_first /
rand_second / rand_both modes are compared. The final column error is exact
(MPS transfer-matrix overlaps).

Faceted by physical dimension ``d`` (rows); per column the final column error,
the worst carrier conditioning, the worst second-SVD tail, and runtime are
plotted against ``Lx``. ``--ensemble flat`` is the stress case (lossy rank-eta
truncation); ``--ensemble decay`` is the easy case.
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column_carrier import make_column_mps, run_column_carrier
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import MARKERS, PALETTE, Panel, Series, write_panel_grid

SUITE = "synthetic_kernels"

MODES = ("det", "rand_first", "rand_second", "rand_both")
COLUMNS = (
    ("rel_error", "column relative error", "log", "error accumulation"),
    ("max_carrier_cond", "max carrier cond", "log", "carrier conditioning"),
    ("max_tail2", "max discarded fraction", "log", "carried tail fraction"),
    ("runtime_s", "runtime (s)", "log", "sweep cost"),
)


def _run_trial(task: tuple[argparse.Namespace, int, int, int]) -> list[dict[str, object]]:
    args, d, lx, trial = task
    with with_blas_threads(args.blas_threads):
        bond = args.bond_mult * args.eta
        tensors = make_column_mps(lx, d, bond, np.random.default_rng(args.seed + 1000 * lx + trial),
                                  ensemble=args.ensemble, decay=args.decay)
        run_column_carrier(tensors, args.eta, mode="det", rng=np.random.default_rng(0))  # warmup
        rows = []
        for mode in MODES:
            r = run_column_carrier(tensors, args.eta, mode=mode, oversample=args.oversample,
                                   n_power=args.n_power, sketch=args.sketch,
                                   rng=np.random.default_rng(args.seed + 7 * trial))
            rows.append({"d": d, "eta": args.eta, "lx": lx, "bond": bond, "mode": mode,
                         "trial": trial, "ensemble": args.ensemble, **r.summary()})
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, d, lx, trial) for d in args.ds for lx in args.lxs for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp11-carrier-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds)
    return csv_path, fig_path


def _series(rows, key):
    bands = median_band(rows, group_key="mode", x_key="lx", value_key=key, group_order=MODES)
    return [
        Series(label=mode.replace("_", " "), x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
               color=PALETTE[mode], marker=MARKERS[mode])
        for mode, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows, fig_path, ds) -> None:
    """Rows = d; columns = column error, carrier conditioning, carried tail, runtime; x = Lx."""
    grid = [
        [
            Panel(title, "Lx", ylabel, yscale, _series([r for r in rows if int(r["d"]) == d], key))
            for key, ylabel, yscale, title in COLUMNS
        ]
        for d in ds
    ]
    write_panel_grid(fig_path, grid, row_titles=[f"d={d}" for d in ds],
                     col_titles=[title for *_, title in COLUMNS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--bond-mult", type=int, default=3, help="exact bond = bond_mult * eta")
    parser.add_argument("--lxs", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ensemble", choices=["flat", "decay"], default="flat")
    parser.add_argument("--decay", type=float, default=0.45)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lxs = [2, 4, 6]
        args.ds = [2, 3]
        args.trials = 2
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
