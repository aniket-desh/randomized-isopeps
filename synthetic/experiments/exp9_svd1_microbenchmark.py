#!/usr/bin/env python3
"""SVD1 microbenchmark: isolate the first-SVD matrix and time it stage by stage.

Strips away the tensor reshapes and the second SVD to benchmark only the first
local SVD matrix ``B_{(1),(23)}`` of shape ``(chi*eta*d) x (chi*eta^2*p)`` at
rank ``k1 = chi*eta``. It compares a deterministic truncated SVD against a
randomized SVD whose time is broken into stages -- sketch ``A@Omega``, range QR,
power iterations, the small projected SVD ``Q^* A``, and reconstruction -- so we
can tell whether a weak ``rand_first`` gain is *algorithmic* (rho1 ~ 1/d too
large to help) rather than tensor-code reshape/allocation overhead. Sweeping d
drives rho1 = k1/n1 = 1/d down; the randomized total should only dip below
deterministic once d is large enough.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.linalg as la

from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid
from rand_isopeps.randomized_svd import orthonormalize, relative_frobenius_error
from rand_isopeps.synthetic_tensors import make_local_tensor
from rand_isopeps.tn_shapes import MosesDims

# metric key -> (color, marker) for the breakdown/accuracy panels
_STYLE = {
    "det_total": ("#0072b2", "o"),
    "rand_total": ("#d55e00", "s"),
    "sketch": ("#009e73", "^"),
    "qr": ("#cc79a7", "D"),
    "power": ("#e69f00", "v"),
    "small_svd": ("#56b4e9", "P"),
    "reconstruct": ("#999999", "X"),
    "det_error": ("#0072b2", "o"),
    "rand_error": ("#d55e00", "s"),
}


def _timed_rsvd(a: np.ndarray, k: int, oversample: int, n_power: int, rng) -> dict[str, float]:
    """Randomized SVD with per-stage timings (gaussian sketch)."""
    m, n = a.shape
    kk = min(k, m, n)
    ell = min(max(kk + oversample, kk), m, n)

    t = perf_counter()
    omega = rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))
    omega = (omega / np.sqrt(2.0)).astype(a.dtype, copy=False)
    y = a @ omega
    t_sketch = perf_counter() - t

    t = perf_counter()
    q = orthonormalize(y)
    t_qr = perf_counter() - t

    t = perf_counter()
    for _ in range(n_power):
        q = orthonormalize(a @ orthonormalize(a.conj().T @ q))
    t_power = perf_counter() - t

    t = perf_counter()
    b = q.conj().T @ a
    u_hat, s, vh = la.svd(b, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    t_small = perf_counter() - t

    t = perf_counter()
    u = (q @ u_hat)[:, :kk]
    approx = (u * s[:kk]) @ vh[:kk, :]
    t_recon = perf_counter() - t

    err = relative_frobenius_error(a, approx)
    total = t_sketch + t_qr + t_power + t_small + t_recon
    return {"sketch": t_sketch, "qr": t_qr, "power": t_power, "small_svd": t_small,
            "reconstruct": t_recon, "rand_total": total, "rand_error": err}


def _run_trial(task: tuple[argparse.Namespace, int, str, int]) -> list[dict[str, object]]:
    args, d, ensemble, trial = task
    with with_blas_threads(args.blas_threads):
        dims = MosesDims(chi=args.chi, eta=args.eta, d=d, p=args.p)
        rng = np.random.default_rng(args.seed + 100000 * d + trial)
        tensor = make_local_tensor(dims, rng, ensemble=ensemble, noise=args.noise,
                                   spectrum_param=args.spectrum_param)
        a = tensor.reshape(dims.n1, dims.n2 * dims.n3)
        k1 = dims.k1
        # warmup both paths at this size
        la.svd(a, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        _timed_rsvd(a, k1, args.oversample, args.n_power, rng)

        t = perf_counter()
        u, s, vh = la.svd(a, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        det_approx = (u[:, :k1] * s[:k1]) @ vh[:k1, :]
        det_total = perf_counter() - t
        det_error = relative_frobenius_error(a, det_approx)

        stages = _timed_rsvd(a, k1, args.oversample, args.n_power, rng)
        return [{
            **dims.as_dict(), "ensemble": ensemble, "trial": trial,
            "oversample": args.oversample, "n_power": args.n_power,
            "det_total": det_total, "det_error": det_error, **stages,
        }]


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [
        (args, d, ensemble, trial)
        for d in args.ds for ensemble in args.ensembles for trial in range(args.trials)
    ]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp9-svd1bench-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args.ensembles)
    return csv_path, fig_path


def _series(rows, keys):
    """Median+IQR band vs d for each metric key (single group per key)."""
    xs = sorted({int(r["d"]) for r in rows})
    out = []
    for key in keys:
        med, lo, hi = [], [], []
        for x in xs:
            vals = np.asarray([float(r[key]) for r in rows if int(r["d"]) == x], dtype=float)
            med.append(float(np.median(vals)))
            lo.append(float(np.percentile(vals, 25)))
            hi.append(float(np.percentile(vals, 75)))
        color, marker = _STYLE[key]
        out.append(Series(label=key.replace("_", " "), x=[float(x) for x in xs],
                          y=med, ylow=lo, yhigh=hi, color=color, marker=marker))
    return out


def make_plot(rows, fig_path, ensembles) -> None:
    """Rows = ensemble; columns = total time (det vs rand), rand stage breakdown, accuracy."""
    grid = [
        [
            Panel("det vs rand total", "d", "first-SVD time (s)", "log",
                  _series(sub, ["det_total", "rand_total"])),
            Panel("rand stage breakdown", "d", "stage time (s)", "log",
                  _series(sub, ["sketch", "qr", "power", "small_svd", "reconstruct"])),
            Panel("accuracy", "d", "relative error", "log",
                  _series(sub, ["det_error", "rand_error"])),
        ]
        for sub in (([r for r in rows if r["ensemble"] == e]) for e in ensembles)
    ]
    write_panel_grid(fig_path, grid, row_titles=list(ensembles),
                     col_titles=["det vs rand total", "rand stage breakdown", "accuracy"],
                     cell_width=400, cell_height=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=6)
    parser.add_argument("--eta", type=int, default=12)
    parser.add_argument("--ds", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16])
    parser.add_argument("--ensembles", nargs="+", default=["noisy_ring", "powerlaw"])
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--spectrum-param", type=float, default=1.0)
    parser.add_argument("--oversample", type=int, default=6)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.ds = [2, 4, 8]
        args.ensembles = ["noisy_ring"]
        args.trials = 3
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
