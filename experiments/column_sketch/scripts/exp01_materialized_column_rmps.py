#!/usr/bin/env python3
"""Phase 1+2: materialized global-column sketch with rMPS probes.

Builds tiny active columns whose whole-column map ``C_j`` (absorbed legs ->
retained legs) is small enough to materialize as a dense matrix, then runs the
global randomized range finder (:func:`rand_isopeps.column.global_range`) with
several probe distributions and compares them against the exact SVD:

* ``Kronecker (chi=1)`` -- rMPS with sketch bond 1 = Gaussian-Kronecker /
  Khatri-Rao, the baseline the paper warns fails by overwhelming orthogonality;
* ``rMPS chi=2,4,8`` -- random matrix product state probes of growing sketch bond;
* ``Gaussian (dense)`` -- the unstructured gold standard.

Faceted by column height ``Lx`` (= the tensor order ``t`` over which the probe is
a product), it plots, per probe distribution:

1. relative column approximation error ``||C - Q Q^* C||_F / ||C||_F`` vs ``ell``;
2. excess of the rank-``r`` randomized SVD over the Eckart-Young optimum vs ``ell``;
3. the OSI subspace-injection diagnostic ``sigma_min(V_r^* Omega)^2`` vs ``chi_sk``
   (the paper's reliability measure; a dense-Gaussian reference line is drawn);
4. the isometry defect ``||Q^* Q - I||_F`` vs ``ell`` (a sanity panel: the new
   column is genuinely isometric for *every* probe -- only the approximation
   quality differs).

Prediction (briefing sec 8): the dense Gaussian sketch is best; ``chi=1`` (Kron)
degrades as ``Lx`` grows; ``chi=2,4,8`` interpolate toward Gaussian. This is a
**mathematical validation**, not a speedup claim -- runtime is recorded in the
CSV but the matrix-free cost story is deferred to the global-vs-local experiment.
"""

from __future__ import annotations

import argparse

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column.global_range import global_column_range, reference_svd
from rand_isopeps.column.operator import controlled_spectrum_column_matrix, random_column_operator
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_line_panels, write_panel_grid

SUITE = "column_sketch"

# series key -> (label, color, marker, linestyle). Gradient from the failure
# (Kronecker, vermillion) through rMPS to the dense-Gaussian gold standard (black).
STYLE = {
    "kron": ("Kronecker (χ=1)", "#d55e00", "s", "-"),
    "rmps2": ("rMPS χ=2", "#e69f00", "^", "-"),
    "rmps4": ("rMPS χ=4", "#009e73", "D", "-"),
    "rmps8": ("rMPS χ=8", "#56b4e9", "v", "-"),
    "gaussian": ("Gaussian (dense)", "#222222", "o", "--"),
}


def _series_key(sketch: str, chi: int) -> str:
    if sketch == "gaussian":
        return "gaussian"
    if sketch == "kron" or chi == 1:
        return "kron"
    return f"rmps{chi}"


def _runs(chis: list[int]) -> list[tuple[str, int]]:
    """(sketch_kind, chi_sk) list: Kronecker for chi=1, rMPS for chi>1, plus dense Gaussian."""
    runs: list[tuple[str, int]] = []
    for chi in chis:
        runs.append(("kron", 1) if chi == 1 else ("rmps", chi))
    runs.append(("gaussian", 0))
    return runs


def _build_column(args: argparse.Namespace, lx: int, rng: np.random.Generator):
    """Return ``(C_dense, factor_dims)`` for one column instance under the chosen ensemble."""
    in_dims = (args.in_dim,) * lx
    out_dims = (args.out_dim,) * lx
    if args.ensemble in ("gaussian", "decay", "identity_plus_noise"):
        op = random_column_operator(lx, args.in_dim, args.out_dim, args.mpo_bond, rng,
                                    ensemble=args.ensemble, decay=args.decay)
        return op.materialize(), op.input_dims
    if args.ensemble in ("controlled_exp", "controlled_power"):
        kind = "exp" if args.ensemble == "controlled_exp" else "power"
        c = controlled_spectrum_column_matrix(out_dims, in_dims, rng,
                                               decay_kind=kind, parameter=args.spectrum_param)
        return c, in_dims
    raise ValueError(f"unknown ensemble: {args.ensemble}")


def _run_trial(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, lx, trial = task
    with with_blas_threads(args.blas_threads):
        c, factor_dims = _build_column(args, lx, np.random.default_rng(args.seed + 1000 * lx + trial))
        n_in = c.shape[1]
        ref = reference_svd(c)
        r = min(args.target_rank, n_in)
        ells = sorted({min(e, n_in) for e in args.ells if e >= r})
        probe_rng = np.random.default_rng(args.seed + 7 * trial + 13 * lx)
        rows: list[dict[str, object]] = []
        for sketch, chi in _runs(args.chis):
            for ell in ells:
                res = global_column_range(c, factor_dims, ell=ell, chi_sk=chi, sketch_kind=sketch,
                                          target_rank=r, n_power=args.n_power, rng=probe_rng, ref_svd=ref)
                rows.append({
                    "lx": lx, "trial": trial, "ensemble": args.ensemble,
                    "in_dim": args.in_dim, "out_dim": args.out_dim, "n_in": n_in, "n_out": c.shape[0],
                    "series": _series_key(sketch, chi), "sketch": sketch, "chi_sk": chi,
                    "ell": ell, "target_rank": r, **res.as_dict(),
                })
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, lx, trial) for lx in args.lxs for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp1-injection-{stamp}")
    write_csv(csv_path, rows)
    make_headline_plot(rows, fig_path, args)            # PRIMARY: the chi >~ Lx injection thesis
    _, detail_path = output_paths(SUITE, f"exp1-materialized-detail-{stamp}")
    make_detail_plot(rows, detail_path, args)           # SECONDARY: full per-Lx facet
    return csv_path, fig_path


# series drawn on the error/isometry panels, in legend order
_ERROR_SERIES = ("kron", "rmps2", "rmps4", "rmps8", "gaussian")


def _band_series(rows, value_key):
    """median+IQR vs ell, one Series per probe distribution."""
    present = [s for s in _ERROR_SERIES if any(r["series"] == s for r in rows)]
    bands = median_band(rows, group_key="series", x_key="ell", value_key=value_key, group_order=present)
    out = []
    for key, (xs, med, lo, hi) in bands.items():
        label, color, marker, ls = STYLE[key]
        out.append(Series(label=label, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                          color=color, marker=marker, linestyle=ls))
    return out


def _osi_vs_chi_series(rows, osi_ell):
    """OSI vs chi_sk: the rMPS curve (chi=1 point is Kronecker) + a dense-Gaussian reference line."""
    sub = [r for r in rows if int(r["ell"]) == osi_ell]
    chis = sorted({int(r["chi_sk"]) for r in sub if r["series"] != "gaussian"})
    xs, med, lo, hi = [], [], [], []
    for chi in chis:
        vals = np.asarray([float(r["osi_sigma_min"]) for r in sub if int(r["chi_sk"]) == chi
                           and r["series"] != "gaussian"], dtype=float)
        if not vals.size:
            continue
        xs.append(float(chi)); med.append(float(np.median(vals)))
        lo.append(float(np.percentile(vals, 25))); hi.append(float(np.percentile(vals, 75)))
    series = [Series(label="rMPS (χ sweep)", x=xs, y=med, ylow=lo, yhigh=hi,
                     color="#cc79a7", marker="p")]
    gvals = np.asarray([float(r["osi_sigma_min"]) for r in sub if r["series"] == "gaussian"], dtype=float)
    if gvals.size and xs:
        g = float(np.median(gvals))
        series.append(Series(label="Gaussian (dense)", x=xs, y=[g] * len(xs),
                             color="#222222", marker="o", linestyle="--"))
    return series


def _osi_vs_lx_series(rows, osi_ell):
    """OSI vs Lx, one Series per probe distribution -- the chi >~ Lx thesis in one panel."""
    sub = [r for r in rows if int(r["ell"]) == osi_ell]
    present = [s for s in _ERROR_SERIES if any(r["series"] == s for r in sub)]
    bands = median_band(sub, group_key="series", x_key="lx", value_key="osi_sigma_min", group_order=present)
    out = []
    for key, (xs, med, lo, hi) in bands.items():
        label, color, marker, ls = STYLE[key]
        out.append(Series(label=label, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                          color=color, marker=marker, linestyle=ls))
    return out


def make_headline_plot(rows, fig_path, args) -> None:
    """One row, three views of the subspace-injection thesis (the headline result)."""
    osi_ell = min(args.osi_ell, max(int(r["ell"]) for r in rows)) if rows else args.osi_ell
    max_lx = max(args.lxs)
    at_max = [r for r in rows if int(r["lx"]) == max_lx]
    panels = [
        Panel(rf"injection vs column height ($\ell$={osi_ell})", r"column height $L_x$ (= tensor order)",
              r"$\sigma_{\min}(V_r^*\Omega)^2$", "log", _osi_vs_lx_series(rows, osi_ell)),
        Panel(rf"injection vs sketch bond ($L_x$={max_lx})", r"sketch bond $\chi$",
              r"$\sigma_{\min}(V_r^*\Omega)^2$", "log", _osi_vs_chi_series(at_max, osi_ell)),
        Panel(rf"column error vs $\ell$ ($L_x$={max_lx})", r"embedding $\ell$",
              r"$\|C-QQ^*C\|_F/\|C\|_F$", "log", _band_series(at_max, "rel_error")),
    ]
    write_line_panels(fig_path, panels, width=1180, height=380)


def make_detail_plot(rows, fig_path, args) -> None:
    """Rows = Lx; cols = rel error vs ell, excess vs ell, OSI vs chi_sk, isometry vs ell."""
    osi_ell = min(args.osi_ell, max(int(r["ell"]) for r in rows)) if rows else args.osi_ell
    grid = []
    for lx in args.lxs:
        sub = [r for r in rows if int(r["lx"]) == lx]
        grid.append([
            Panel("column error", r"embedding $\ell$", r"$\|C-QQ^*C\|_F/\|C\|_F$", "log",
                  _band_series(sub, "rel_error")),
            Panel("excess over rank-r SVD", r"embedding $\ell$", "excess (rand − Eckart-Young)", "log",
                  _band_series(sub, "excess_error")),
            Panel(rf"subspace injection ($\ell$={osi_ell})", r"sketch bond $\chi$",
                  r"$\sigma_{\min}(V_r^*\Omega)^2$", "linear", _osi_vs_chi_series(sub, osi_ell)),
            Panel("isometry defect (sanity)", r"embedding $\ell$", r"$\|Q^*Q-I\|_F$", "log",
                  _band_series(sub, "isometry_defect")),
        ])
    write_panel_grid(fig_path, grid, row_titles=[f"Lx={lx}" for lx in args.lxs],
                     col_titles=["column error", "excess over rank-r SVD",
                                 "subspace injection", "isometry defect (sanity)"],
                     cell_width=380, cell_height=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--in-dim", type=int, default=2, help="absorbed leg dim per row (sketch factor)")
    parser.add_argument("--out-dim", type=int, default=3, help="retained leg dim per row")
    parser.add_argument("--mpo-bond", type=int, default=8, help="vertical MPO bond (intra-column entanglement)")
    parser.add_argument("--target-rank", type=int, default=4, help="r referenced by excess + OSI")
    parser.add_argument("--ells", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    parser.add_argument("--chis", type=int, nargs="+", default=[1, 2, 4, 8], help="rMPS sketch bonds")
    parser.add_argument("--osi-ell", type=int, default=8, help="ell at which the OSI-vs-chi panel is read")
    parser.add_argument("--ensemble", default="gaussian",
                        choices=["gaussian", "decay", "identity_plus_noise", "controlled_exp", "controlled_power"])
    parser.add_argument("--decay", type=float, default=0.6)
    parser.add_argument("--spectrum-param", type=float, default=3.0)
    parser.add_argument("--n-power", type=int, default=0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lxs = [3, 4, 5]
        args.ells = [4, 6, 8]
        args.trials = 3
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
