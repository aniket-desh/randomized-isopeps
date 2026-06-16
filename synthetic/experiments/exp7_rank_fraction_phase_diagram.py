#!/usr/bin/env python3
"""rank-fraction phase diagram for the local two SVDs.

The economics of a randomized SVD are governed not by the spectrum alone but by
the dimensionless *rank fraction*

    rho = ell / min(m, n),   ell = k + oversample,

the kept sketch size as a fraction of the smaller matrix dimension. For the
first local SVD rho1 ~ 1/d (spin systems keep ~half the rows when d=2); for the
second rho2 ~ 1/eta. A large (chi, eta, d, oversample, ensemble) sweep collapses
onto this single axis: this experiment plots stage *speedup* (deterministic /
randomized stage time) against rho, colored by excess reconstruction error.

Expectation: SVD1 points sit at high rho (little room) for physical d=2,3,4 and
only march into the useful upper-left region (low rho, speedup > 1) as d grows;
SVD2 points sit at low rho and form the useful frontier. This explains *why*
larger d would make a randomized first SVD worthwhile -- the kept subspace
finally becomes a small enough fraction of the matrix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_ring_decomp import local_ring_decomp
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import ScatterPanel, ScatterSeries, write_scatter_grid
from rand_isopeps.synthetic_tensors import make_local_tensor
from rand_isopeps.tn_shapes import MosesDims

# target -> (rand mode, stage-time key, marker). Both compared against det.
TARGETS = {
    "first": ("rand_first", "first_time_s", "o"),
    "second": ("rand_second", "second_time_s", "s"),
}
EXCESS_FLOOR = 1e-10  # log-color floor for ~0 excess error


def _run_trial(task: tuple[argparse.Namespace, dict, int]) -> list[dict[str, object]]:
    args, combo, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=combo["chi"], eta=combo["eta"], d=combo["d"], p=args.p)
        rng = np.random.default_rng(args.seed + trial)
        tensor = make_local_tensor(
            dims, rng, ensemble=combo["ensemble"], noise=args.noise,
            spectrum_param=args.spectrum_param,
        )
        local_ring_decomp(tensor, dims, mode="det")  # warmup
        det = local_ring_decomp(tensor, dims, mode="det").metrics(tensor)
        rows: list[dict[str, object]] = []
        for target, (mode, time_key, _) in TARGETS.items():
            rnd = local_ring_decomp(
                tensor, dims, mode=mode, oversample=combo["oversample"],
                n_power=args.n_power, sketch=args.sketch, rng=rng,
            ).metrics(tensor)
            rows.append({
                **dims.as_dict(), "ensemble": combo["ensemble"], "target": target,
                "oversample": combo["oversample"], "trial": trial,
                "rho": dims.rank_fraction(target, combo["oversample"]),
                "speedup": float(det[time_key]) / max(float(rnd[time_key]), 1e-12),
                "excess_error": float(rnd["rel_error"]) - float(det["rel_error"]),
            })
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    combos = [
        {"chi": chi, "eta": eta, "d": d, "oversample": s, "ensemble": ens}
        for ens in args.ensembles for chi in args.chis for eta in args.etas
        for d in args.ds for s in args.oversamples
    ]
    workers = auto_worker_count(args.workers)
    tasks = [(args, combo, trial) for combo in combos for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp7-phase-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ensembles)
    return csv_path, fig_path


def _points(rows: list[dict[str, object]]) -> tuple[list[float], list[float], list[float]]:
    """Median (rho, speedup, log10 excess) per parameter group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["chi"], r["eta"], r["d"], r["oversample"])].append(r)
    xs, ys, cs = [], [], []
    for pts in groups.values():
        xs.append(float(pts[0]["rho"]))
        ys.append(float(np.median([p["speedup"] for p in pts])))
        excess = float(np.median([p["excess_error"] for p in pts]))
        cs.append(math.log10(max(excess, EXCESS_FLOOR)))
    return xs, ys, cs


def make_plot(rows: list[dict[str, object]], fig_path: str, ensembles: list[str]) -> None:
    """Rows = ensemble, columns = SVD target; x = rho, y = speedup, color = excess."""
    all_c = []
    grid: list[list[ScatterPanel]] = []
    for ens in ensembles:
        row_panels = []
        for target, (_, _, marker) in TARGETS.items():
            sub = [r for r in rows if r["ensemble"] == ens and r["target"] == target]
            x, y, c = _points(sub)
            all_c.extend(c)
            row_panels.append(ScatterPanel(
                title=f"SVD{1 if target == 'first' else 2}",
                xlabel=r"$\rho=(k+s)/\min(m,n)$", ylabel="speedup (det / rand)",
                xscale="log", yscale="log",
                series=[ScatterSeries(f"SVD{1 if target == 'first' else 2}", x, y, c, marker)],
                hlines=[1.0],
            ))
        grid.append(row_panels)

    vmin = math.floor(min(all_c)) if all_c else -10.0
    vmax = max(math.ceil(max(all_c)), vmin + 1) if all_c else 0.0
    write_scatter_grid(
        fig_path, grid, clabel=r"$\log_{10}$ excess error (rand $-$ det)",
        row_titles=list(ensembles),
        col_titles=[r"first SVD:  $\rho_1\sim 1/d$", r"second SVD:  $\rho_2\sim 1/\eta$"],
        cmap="viridis", vmin=vmin, vmax=vmax,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chis", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--etas", type=int, nargs="+", default=[6, 10, 16])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16])
    parser.add_argument("--oversamples", type=int, nargs="+", default=[2, 6, 12])
    parser.add_argument("--ensembles", nargs="+", default=["noisy_ring", "powerlaw", "expdecay"])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--spectrum-param", type=float, default=1.0)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.chis = [4]
        args.etas = [6, 10]
        args.ds = [2, 4, 8]
        args.oversamples = [2, 6]
        args.ensembles = ["noisy_ring", "powerlaw"]
        args.trials = 2
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
