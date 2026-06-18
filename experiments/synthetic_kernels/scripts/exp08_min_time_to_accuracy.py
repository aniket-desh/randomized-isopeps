#!/usr/bin/env python3
"""minimum time-to-accuracy: the fairest test of whether randomizing helps.

For each tensor and each randomized mode, sweep the tuning knobs
``oversample in {0,2,4,8,16}`` and ``n_power in {0,1,2}`` and ask: what is the
*fastest* configuration whose reconstruction error is within ``tol`` of the
deterministic baseline (with isometry defects at machine precision)? The plotted
quantity is the best valid speedup

    T_det / min_{s,q : eps <= tol * eps_det} T_rand(s,q).

This is stronger than fixing one ``(oversample, n_power)``: if a randomized mode
needs large ``s`` or ``q`` to match deterministic accuracy the speedup
evaporates. For spin-like ``d`` the first-SVD mode has no useful time-to-accuracy
advantage; the second-SVD mode does once its spectrum decays.
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_ring_decomp import LocalMode, local_ring_decomp
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import MARKERS, PALETTE, Panel, Series, write_panel_grid
from rand_isopeps.synthetic_tensors import make_local_tensor
from rand_isopeps.tn_shapes import MosesDims

SUITE = "synthetic_kernels"

RAND_MODES: tuple[LocalMode, ...] = ("rand_first", "rand_second", "rand_both")
ISO_TOL = 1e-10


def _best_speedup(tensor, dims, mode, det_time, det_err, oversamples, n_powers, tol, rng):
    """Fastest valid (oversample, n_power) speedup, or nan if none qualify."""
    best_time = float("inf")
    best_cfg = None
    bound = max(tol * det_err, det_err + 1e-12)
    for s in oversamples:
        for q in n_powers:
            m = local_ring_decomp(
                tensor, dims, mode=mode, oversample=s, n_power=q, rng=rng,
            ).metrics(tensor)
            ok = (
                float(m["rel_error"]) <= bound
                and float(m["first_iso_defect"]) < ISO_TOL
                and float(m["second_iso_defect"]) < ISO_TOL
            )
            if ok and float(m["total_time_s"]) < best_time:
                best_time = float(m["total_time_s"])
                best_cfg = (s, q)
    if best_cfg is None:
        return float("nan"), None
    return det_time / best_time, best_cfg


def _run_trial(task: tuple[argparse.Namespace, int, int, str, int]) -> list[dict[str, object]]:
    args, d, eta, ensemble, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + 1000 * eta + trial)
        tensor = make_local_tensor(dims, rng, ensemble=ensemble, noise=args.noise,
                                   spectrum_param=args.spectrum_param)
        local_ring_decomp(tensor, dims, mode="det")  # warmup
        det = local_ring_decomp(tensor, dims, mode="det").metrics(tensor)
        det_time, det_err = float(det["total_time_s"]), float(det["rel_error"])
        rows: list[dict[str, object]] = []
        for mode in RAND_MODES:
            row: dict[str, object] = {
                **dims.as_dict(), "ensemble": ensemble, "mode": mode, "trial": trial,
                "det_rel_error": det_err,
            }
            for tol in args.tols:
                sp, cfg = _best_speedup(tensor, dims, mode, det_time, det_err,
                                        args.oversamples, args.n_powers, tol, rng)
                tag = f"{tol:g}".replace(".", "p")
                row[f"speedup_{tag}"] = sp
                row[f"cfg_{tag}"] = "" if cfg is None else f"s={cfg[0]},q={cfg[1]}"
            rows.append(row)
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [
        (args, d, eta, ensemble, trial)
        for d in args.ds for eta in args.etas
        for ensemble in args.ensembles for trial in range(args.trials)
    ]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp8-tta-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ds, args.ensembles, args.tols[-1])
    return csv_path, fig_path


def _series(rows, key):
    bands = median_band(rows, group_key="mode", x_key="eta", value_key=key, group_order=RAND_MODES)
    return [
        Series(label=mode.replace("_", " "), x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
               color=PALETTE[mode], marker=MARKERS[mode])
        for mode, (xs, med, lo, hi) in bands.items()
    ]


def make_plot(rows, fig_path, ds, ensembles, tol) -> None:
    """Rows = d, columns = ensemble; best valid speedup (at the loosest tol) vs eta."""
    key = f"speedup_{tol:g}".replace(".", "p")
    grid = [
        [
            Panel(ensemble, "eta", "best valid speedup", "log",
                  _series([r for r in rows if int(r["d"]) == d and r["ensemble"] == ensemble], key))
            for ensemble in ensembles
        ]
        for d in ds
    ]
    write_panel_grid(fig_path, grid, row_titles=[f"d={d}" for d in ds], col_titles=list(ensembles))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--etas", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--ensembles", nargs="+", default=["noisy_ring", "powerlaw"])
    parser.add_argument("--oversamples", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    parser.add_argument("--n-powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tols", type=float, nargs="+", default=[1.01, 1.05])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--spectrum-param", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.etas = [4, 6]
        args.ds = [2, 4]
        args.oversamples = [0, 4]
        args.n_powers = [0, 1]
        args.ensembles = ["noisy_ring"]
        args.trials = 2
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
