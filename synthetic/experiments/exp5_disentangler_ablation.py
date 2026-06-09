#!/usr/bin/env python3
"""Disentangler ablation for the local Moses-Move second SVD.

Compares six ways to build the local isometric tensor-ring decomposition of a
*disentanglable* tensor, isolating the role of the unitary disentangler placed
before the second truncated SVD (docs/disentangling.pdf, docs/computing-isotn.pdf):

    A  two deterministic SVDs, no disentangler        (baseline truncation error)
    B  alt-min disentangler + deterministic SVD2      (value of disentanglement)
    B_riem  Riemannian (pymanopt) disentangler + SVD2 (paper's optimizer)
    C  disentangler + randomized SVD2                 (speedup of final compression)
    D  sketched-objective disentangler + exact SVD2   (randomized disentangler search)
    E  ALS on the ring cores from random init         (discover the ring from scratch)
    F  ALS warm-started from the greedy SVD+D cores    (polish the greedy solution)

The conceptual split is C (randomize the SVD) vs D (randomize the *search* for
the disentangler). The headline diagnostic is the singular spectrum entering the
second SVD: with a good disentangler it decays much faster, so the rank-eta
truncation throws away little -- "disentanglement improves the spectrum,
randomization exploits the improved spectrum."

Outputs a methods-vs-eta figure (reconstruction error, second-SVD tail energy,
runtime) and a spectrum figure (sigma_i with vs without the disentangler).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.local_methods import METHODS, run_method
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_line_panels
from rand_isopeps.synthetic_tensors import make_disentanglable_local
from rand_isopeps.tn_shapes import MosesDims

# Okabe-Ito + extensions; one (color, marker) per method, assigned by order.
_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#999999"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
METHOD_STYLE = {m: (_COLORS[i % len(_COLORS)], _MARKERS[i % len(_MARKERS)]) for i, m in enumerate(METHODS)}


def _tensor(args: argparse.Namespace, dims: MosesDims, trial: int) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 1000 * dims.eta + trial)
    return make_disentanglable_local(dims, rng, entangle=args.entangle, noise=args.noise)


def _run_eta_trial(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, eta, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=eta, d=args.d, p=args.p)
        b = _tensor(args, dims, trial)
        # warm BLAS for these sizes; the median over trials also absorbs the
        # one-time pymanopt import cost paid on the first Riemannian call.
        run_method("A_two_svd", b, dims, rng=np.random.default_rng(0))
        rows: list[dict[str, object]] = []
        for idx, method in enumerate(METHODS):
            rng = np.random.default_rng(args.seed + 17 * trial + 100 * eta + idx)
            result = run_method(
                method, b, dims,
                oversample=args.oversample, n_power=args.n_power, sketch=args.sketch,
                dis_maxiter=args.dis_maxiter, als_maxiter=args.als_maxiter, rng=rng,
            )
            rows.append({
                **dims.as_dict(),
                "trial": trial,
                "entangle": args.entangle,
                "noise": args.noise,
                "oversample": args.oversample,
                "n_power": args.n_power,
                "sketch": args.sketch,
                **result.metrics(b),
            })
        return rows


def run(args: argparse.Namespace) -> tuple[str, str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, eta, trial) for eta in args.etas for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_eta_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp5-disentangler-{stamp}")
    spectrum_path = fig_path.replace("exp5-disentangler-", "exp5-spectrum-")
    write_csv(csv_path, rows)
    make_methods_plot(rows, fig_path)
    make_spectrum_plot(args, spectrum_path)
    return csv_path, fig_path, spectrum_path


def _series(rows: list[dict[str, object]], key: str) -> list[Series]:
    bands = median_band(rows, group_key="method", x_key="eta", value_key=key, group_order=METHODS)
    return [
        Series(
            label=method.replace("_", " "),
            x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
            color=METHOD_STYLE[method][0], marker=METHOD_STYLE[method][1],
        )
        for method, (xs, med, lo, hi) in bands.items()
    ]


def make_methods_plot(rows: list[dict[str, object]], fig_path: str) -> None:
    panels = [
        Panel("reconstruction error", "eta", "relative error", "log", _series(rows, "rel_error")),
        Panel("second-SVD tail", "eta", "discarded energy (>eta)", "log", _series(rows, "tail_used")),
        Panel("method cost", "eta", "runtime (s)", "log", _series(rows, "runtime_s")),
    ]
    write_line_panels(fig_path, panels, width=1180, height=440)


def make_spectrum_plot(args: argparse.Namespace, fig_path: str) -> None:
    """Singular spectrum entering the second SVD, with vs without disentangler."""
    eta = max(args.etas)
    dims = MosesDims(chi=args.chi, eta=eta, d=args.d, p=args.p)
    b = _tensor(args, dims, trial=0)
    rng = np.random.default_rng(args.seed)
    no_d = run_method("A_two_svd", b, dims, rng=rng)
    with_d = run_method("B_disentangle_riem", b, dims)

    def _series(label, spectrum, color, marker):
        s = np.asarray(spectrum, dtype=float)
        idx = np.arange(1, s.shape[0] + 1)
        return Series(label=label, x=[float(i) for i in idx], y=[float(v) for v in s], color=color, marker=marker)

    panel = Panel(
        title=f"second-SVD spectrum (chi={args.chi}, eta={eta})",
        xlabel="singular index i", ylabel="singular value", yscale="log",
        series=[
            _series("no disentangler", no_d.spectrum_no_d, "#0072b2", "o"),
            _series("with disentangler", with_d.spectrum_used, "#009e73", "^"),
        ],
    )
    write_line_panels(fig_path, [panel], width=560, height=440)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=3)
    parser.add_argument("--etas", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--entangle", type=float, default=1.0, help="hidden bond-rotation strength in [0,1]")
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--dis-maxiter", type=int, default=100, help="alternating-min disentangler iterations")
    parser.add_argument("--als-maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 selects a conservative local default")
    parser.add_argument("--blas-threads", type=int, default=1, help="BLAS threads per worker; use 0 for library default")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.etas = [3, 4]
        args.trials = 1
        args.oversample = 4
        args.dis_maxiter = 40
        args.als_maxiter = 50
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig, spectrum = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
    print(f"wrote {spectrum}")
