#!/usr/bin/env python3
"""How fast does an rMPS probe approach a dense Gaussian as the sketch bond grows?

Sweeps the sketch bond ``chi_sk`` densely and measures the subspace-injection
diagnostic ``OSI = sigma_min(V_r* Omega)^2`` (the same quantity as exp01) at a
fixed embedding ``ell`` and target rank ``r``, faceted/overlaid by column height
``Lx``. Three views of the same data:

1. **injection vs column height** -- one line per ``chi_sk`` (gradient) plus the
   dense-Gaussian reference. The family fills in between Kronecker (chi=1, which
   collapses with Lx) and Gaussian (flat).
2. **injection vs sketch bond** -- one line per ``Lx``; the climb of OSI as the
   probe gains bond.
3. **rMPS injectivity / Gaussian** -- the ratio ``OSI_rMPS / OSI_Gaussian`` vs
   ``chi_sk``, one line per ``Lx``, with a reference at 1.0. This is the clean
   "how rMPS compares to Gaussian" curve: each Lx saturates toward 1, and the
   ``chi_sk`` at which it saturates reveals the empirical scaling (the paper's
   thesis is ``chi_sk >~ Lx``).

Tiny materialized columns (cheap: chi_sk only enlarges the probe cores). A
mathematical diagnostic, not a speedup claim.
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

from rand_isopeps.column.global_range import global_column_range, reference_svd
from rand_isopeps.column.operator import random_column_operator
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_line_panels

SUITE = "column_sketch"


def _osi(c, factor_dims, ell, chi, r, n_power, rng, ref):
    kind = "kron" if chi == 1 else "rmps"
    return global_column_range(c, factor_dims, ell=ell, chi_sk=chi, sketch_kind=kind,
                               target_rank=r, n_power=n_power, rng=rng, ref_svd=ref).osi_sigma_min


def _run_trial(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, lx, trial = task
    with with_blas_threads(args.blas_threads):
        op = random_column_operator(lx, args.in_dim, args.out_dim, args.mpo_bond,
                                    np.random.default_rng(args.seed + 1000 * lx + trial),
                                    ensemble=args.ensemble)
        c = op.materialize()
        fd = op.input_dims
        n_in = c.shape[1]
        ref = reference_svd(c)
        r = min(args.target_rank, n_in)
        ell = min(args.osi_ell, n_in)
        rng = np.random.default_rng(args.seed + 7 * trial + 13 * lx)
        rows = []
        for chi in args.chis:
            # median OSI over a few fresh probe draws to tame per-draw variance
            vals = [_osi(c, fd, ell, chi, r, args.n_power, rng, ref) for _ in range(args.probe_draws)]
            rows.append({"lx": lx, "trial": trial, "chi_sk": chi, "osi": float(np.median(vals))})
        gvals = [global_column_range(c, fd, ell=ell, sketch_kind="gaussian", target_rank=r,
                                     n_power=args.n_power, rng=rng, ref_svd=ref).osi_sigma_min
                 for _ in range(args.probe_draws)]
        rows.append({"lx": lx, "trial": trial, "chi_sk": 0, "osi": float(np.median(gvals))})  # 0 = Gaussian
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, lx, trial) for lx in args.lxs for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp3-chi-saturation-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _median(rows, lx, chi):
    vals = [float(r["osi"]) for r in rows if int(r["lx"]) == lx and int(r["chi_sk"]) == chi]
    return float(np.median(vals)) if vals else float("nan")


def _chi_color(chis):
    cmap = colormaps["viridis"]
    rmps = [c for c in chis if c > 1]
    out = {}
    for i, c in enumerate(rmps):
        frac = i / (len(rmps) - 1) if len(rmps) > 1 else 0.5
        out[c] = to_hex(cmap(0.12 + 0.78 * frac))
    return out


def _lx_color(lxs):
    cmap = colormaps["plasma"]
    return {lx: to_hex(cmap(0.1 + 0.75 * (i / max(len(lxs) - 1, 1)))) for i, lx in enumerate(lxs)}


def make_plot(rows, fig_path, args) -> None:
    lxs = sorted({int(r["lx"]) for r in rows})
    chis = sorted({int(r["chi_sk"]) for r in rows if int(r["chi_sk"]) > 0})
    chi_col = _chi_color(chis)
    lx_col = _lx_color(lxs)

    # Panel 1: OSI vs Lx, one line per chi_sk (gradient) + Gaussian reference
    p1 = []
    for chi in chis:
        ys = [_median(rows, lx, chi) for lx in lxs]
        color = "#d55e00" if chi == 1 else chi_col[chi]
        label = "Kron (χ=1)" if chi == 1 else f"χ={chi}"
        p1.append(Series(label=label, x=[float(v) for v in lxs], y=ys, color=color,
                         marker="o", markevery=(0, 1)))
    p1.append(Series(label="Gaussian", x=[float(v) for v in lxs],
                     y=[_median(rows, lx, 0) for lx in lxs], color="#222222", marker="o", linestyle="--"))

    # Panel 2: OSI vs chi_sk, one line per Lx (gradient)
    p2 = []
    for lx in lxs:
        ys = [_median(rows, lx, chi) for chi in chis]
        p2.append(Series(label=f"Lx={lx}", x=[float(c) for c in chis], y=ys,
                         color=lx_col[lx], marker="o"))

    # Panel 3: OSI_rMPS / OSI_Gaussian vs chi_sk, one line per Lx (saturation toward 1)
    p3 = []
    for lx in lxs:
        g = _median(rows, lx, 0)
        ys = [(_median(rows, lx, chi) / g if g else float("nan")) for chi in chis]
        p3.append(Series(label=f"Lx={lx}", x=[float(c) for c in chis], y=ys,
                         color=lx_col[lx], marker="o"))

    panels = [
        Panel(rf"injection vs column height ($\ell$={args.osi_ell})",
              r"column height $L_x$", r"$\sigma_{\min}(V_r^*\Omega)^2$", "log", p1),
        Panel(rf"injection vs sketch bond ($\ell$={args.osi_ell})",
              r"sketch bond $\chi_{sk}$", r"$\sigma_{\min}(V_r^*\Omega)^2$", "log", p2),
        Panel("rMPS injectivity ÷ Gaussian",
              r"sketch bond $\chi_{sk}$", "OSI(rMPS) / OSI(Gaussian)", "linear", p3),
    ]
    panels[2].series.append(Series(label="Gaussian (=1)", x=[float(min(chis)), float(max(chis))],
                                   y=[1.0, 1.0], color="#222222", marker="", linestyle="--"))
    write_line_panels(fig_path, panels, width=1200, height=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--chis", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--in-dim", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=3)
    parser.add_argument("--mpo-bond", type=int, default=8)
    parser.add_argument("--target-rank", type=int, default=4)
    parser.add_argument("--osi-ell", type=int, default=8)
    parser.add_argument("--ensemble", default="gaussian",
                        choices=["gaussian", "decay", "identity_plus_noise"])
    parser.add_argument("--n-power", type=int, default=0)
    parser.add_argument("--probe-draws", type=int, default=6, help="OSI draws medianed per trial")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lxs = [3, 5, 7]
        args.chis = [1, 2, 4, 8, 16]
        args.trials = 4
        args.probe_draws = 3
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
