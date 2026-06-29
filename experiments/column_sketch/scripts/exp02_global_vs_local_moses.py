#!/usr/bin/env python3
"""Phase 4: global rMPS-sketched column QR vs the sequential local Moses move.

On the *same* tiny column operator ``C_j`` (absorbed legs -> retained legs), two
ways to build the new isometric column ``Q`` and residual ``R`` with ``C ~ Q R``
at a matched absorbed-column rank ``k`` are compared:

* **local Moses** (``local-det`` / ``local-rand``) -- the sequential sweep
  (:func:`rand_isopeps.column.local_moses.local_column_qr`): split each row into an
  output-isometric core, carry the residual up, truncate the carried rank to ``k``.
  Greedy: its error is a sum of local truncation errors. ``rand`` randomizes every
  local SVD (the "randomized local SVD2 Moses").
* **global sketch** (``global-gauss`` / ``global-kron`` / ``global-rmps χ``) -- one
  randomized range finder (:func:`rand_isopeps.column.global_range.global_column_range`)
  with ``ell = k`` probes: ``Q = orth(C Omega)``, ``R = Q^* C``. Targets the *flat*
  (whole-matrix) column rank. ``kron`` is the ``chi=1`` Khatri--Rao baseline.

The fork (briefing sec 5): global wins when the column has a low flat rank that is
better captured in one shot than by greedy local splits; the dense Gaussian is the
gold standard and rMPS should match it, while ``chi=1`` (Kron) lags more as ``Lx``
grows. Faceted by column height ``Lx``: column error vs ``k`` (with the
Eckart--Young floor), excess over that floor, the isometry defect (sanity), and
wall-clock (a labeled SECONDARY metric -- this is a **mathematical validation**, not
an end-to-end speedup claim, which would require the full matrix-free column
algorithm including absorption).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column.global_range import global_column_range, reference_svd
from rand_isopeps.column.local_moses import local_column_qr
from rand_isopeps.column.operator import controlled_spectrum_column_matrix, random_column_operator
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, method_style, write_panel_grid

SUITE = "column_sketch"

# method key -> (label, kind). kind: ("local", randomized) or ("global", sketch_kind,
# chi_sk). This dict drives DATA GENERATION in ``_run_trial`` (it maps each method key
# to the local/global + sketch + chi recipe), so it stays. Plot STYLING is *not* taken
# from here -- ``make_plot`` routes every series through ``rand_isopeps.plotting``'s
# canonical ``method_style`` (with the ``global_rmps8`` data key remapped to the
# canonical ``global_rmps`` grammar) so the visual grammar lives in one module.
METHODS = {
    "local_det": ("local Moses (det)", ("local", False)),
    "local_rand": ("local Moses (rand)", ("local", True)),
    "global_kron": ("global Kron (χ=1)", ("global", "kron", 1)),
    "global_rmps4": ("global rMPS χ=4", ("global", "rmps", 4)),
    "global_rmps8": ("global rMPS χ=8", ("global", "rmps", 8)),
    "global_gauss": ("global Gaussian", ("global", "gaussian", 0)),
}
METHOD_ORDER = list(METHODS)


def _build_column(args, lx, rng):
    in_dims = (args.in_dim,) * lx
    out_dims = (args.out_dim,) * lx
    if args.ensemble in ("gaussian", "decay", "identity_plus_noise"):
        return random_column_operator(lx, args.in_dim, args.out_dim, args.mpo_bond, rng,
                                      ensemble=args.ensemble, decay=args.decay)
    if args.ensemble in ("controlled_exp", "controlled_power"):
        kind = "exp" if args.ensemble == "controlled_exp" else "power"
        c = controlled_spectrum_column_matrix(out_dims, in_dims, rng,
                                               decay_kind=kind, parameter=args.spectrum_param)
        return c  # dense ndarray (no matrix-free form)
    raise ValueError(f"unknown ensemble: {args.ensemble}")


def _run_trial(task: tuple[argparse.Namespace, int, int]) -> list[dict[str, object]]:
    args, lx, trial = task
    with with_blas_threads(args.blas_threads):
        col = _build_column(args, lx, np.random.default_rng(args.seed + 1000 * lx + trial))
        is_operator = hasattr(col, "materialize")
        c_dense = col.materialize() if is_operator else col
        factor_dims = col.input_dims if is_operator else (args.in_dim,) * lx
        n_in, n_out = c_dense.shape[1], c_dense.shape[0]
        ref = reference_svd(c_dense)
        s = ref[1] / max(float(np.linalg.norm(ref[1])), 1e-300)
        # keep k strictly below the smaller dimension so no facet hits trivial full-rank recovery
        ks = sorted({k for k in args.ks if 1 <= k < min(n_in, n_out)})
        probe_rng = np.random.default_rng(args.seed + 7 * trial + 13 * lx)
        rows: list[dict[str, object]] = []
        for k in ks:
            ey_floor = float(np.sqrt(np.sum(s[k:] ** 2)))
            for key in METHOD_ORDER:
                _, kind, *_ = METHODS[key]
                if kind[0] == "local":
                    if not is_operator:
                        continue  # local sweep needs the MPO form
                    res = local_column_qr(col, k, randomized=kind[1], oversample=args.oversample,
                                          n_power=args.n_power, sketch=args.sketch, rng=probe_rng,
                                          reference=c_dense)
                    d = res.as_dict()
                    rel, iso, runtime = d["rel_error"], d["isometry_defect"], d["runtime_s"]
                    n_prim = d["n_svd"]
                else:
                    _, sk, chi = kind
                    res = global_column_range(c_dense, factor_dims, ell=k, chi_sk=chi, sketch_kind=sk,
                                              target_rank=k, n_power=args.n_power, rng=probe_rng, ref_svd=ref)
                    rel, iso, runtime = res.rel_error, res.isometry_defect, res.runtime_s
                    n_prim = 1  # one global range finder (vs Lx sequential local SVDs)
                rows.append({
                    "lx": lx, "trial": trial, "ensemble": args.ensemble, "k": k,
                    "n_in": n_in, "n_out": n_out, "method": key,
                    "rel_error": rel, "excess_error": float(rel) - ey_floor, "ey_floor": ey_floor,
                    "isometry_defect": iso, "n_primitives": n_prim, "runtime_s": runtime,
                })
        return rows


def run(args: argparse.Namespace) -> tuple[str, str]:
    workers = auto_worker_count(args.workers)
    tasks = [(args, lx, trial) for lx in args.lxs for trial in range(args.trials)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp2-global-vs-local-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


# Trimmed main-figure method set (PLOT.md: <=4-5 methods per panel). Each entry maps a
# DATA key in the CSV's ``method`` column to the canonical STYLE key looked up via
# ``method_style``. ``global_rmps8`` (chi=8 rMPS, the deployed probe) carries the
# canonical ``global_rmps`` grammar (green diamond, label "rMPS"); ``global_rmps4`` and
# ``global_kron`` are dropped from these figures to keep the comparison legible.
_PLOT_METHODS: dict[str, str] = {
    "global_gauss": "global_gauss",
    "global_rmps8": "global_rmps",
    "local_det": "local_det",
    "local_rand": "local_rand",
}
# Legend order: gold-standard dense Gaussian, then the rMPS probe, then the two local
# Moses variants (mirrors plotting.METHOD_ORDER minus the dropped probes).
_PLOT_ORDER = ["global_gauss", "global_rmps8", "local_det", "local_rand"]


def _method_series(rows, value_key, *, floor: float | None = None):
    """Median+IQR series over trials for the trimmed method set, styled canonically.

    Each present DATA key is summarized with :func:`median_band` and assigned its
    ``(label, color, marker, linestyle)`` from :func:`method_style` (the ``global_rmps8``
    -> ``global_rmps`` remap lives in ``_PLOT_METHODS``). ``floor`` clamps the median and
    band to a positive value so error/excess curves never hit 0 on the log axis.
    """
    present = [m for m in _PLOT_ORDER if any(r["method"] == m for r in rows)]
    bands = median_band(rows, group_key="method", x_key="k", value_key=value_key, group_order=present)
    out = []
    for data_key, (xs, med, lo, hi) in bands.items():
        label, color, marker, ls = method_style(_PLOT_METHODS[data_key])
        if floor is not None:
            med = [max(float(v), floor) for v in med]
            lo = [max(float(v), floor) for v in lo]
            hi = [max(float(v), floor) for v in hi]
        out.append(Series(label=label, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                          color=color, marker=marker, linestyle=ls))
    return out


def _ey_series(rows):
    """Eckart-Young floor vs k -- the optimal rank-k error, a probe-independent reference.

    Styled as the canonical ``eckart_young`` floor: gray dotted line with NO markers (the
    grammar's empty-string marker), so it reads as a baseline rather than a method.
    """
    label, color, marker, ls = method_style("eckart_young")
    ks = sorted({int(r["k"]) for r in rows})
    ys = [max(float(np.median([float(r["ey_floor"]) for r in rows if int(r["k"]) == k])), 1e-16)
          for k in ks]
    return [Series(label=label, x=[float(k) for k in ks], y=ys,
                   color=color, marker=marker, linestyle=ls)]


def make_plot(rows, fig_path, args) -> None:
    """Render the global-vs-local accuracy figures, faceted by column height ``Lx``.

    MAIN (``fig_path``): a rows=Lx x 2-col grid -- column error vs k (with the
    Eckart-Young floor) and excess over that floor -- for the trimmed method set
    (Gaussian, rMPS, local det, local rand). A SECONDARY ``-diagnostics`` figure holds
    the isometry-defect sanity check (with a 1e-12 threshold line) and the wall-clock
    secondary metric, so the headline accuracy claim is not buried in a 4-wide grid.
    """
    main = []
    for lx in args.lxs:
        sub = [r for r in rows if int(r["lx"]) == lx]
        main.append([
            Panel("Column error", "absorbed rank $k$", r"$\|C-QR\|_F/\|C\|_F$", "log",
                  _method_series(sub, "rel_error", floor=1e-16) + _ey_series(sub)),
            Panel("Excess over Eckart-Young", "absorbed rank $k$", "error $-$ optimal", "log",
                  _method_series(sub, "excess_error", floor=1e-16)),
        ])
    write_panel_grid(fig_path, main, row_titles=[f"Lx={lx}" for lx in args.lxs],
                     col_titles=["Column error (+ EY floor)", "Excess over Eckart-Young"],
                     cell_width=420, cell_height=300)

    diag = []
    for lx in args.lxs:
        sub = [r for r in rows if int(r["lx"]) == lx]
        diag.append([
            Panel("Isometry defect (sanity)", "absorbed rank $k$", r"$\|Q^*Q-I\|_F$", "log",
                  _method_series(sub, "isometry_defect", floor=1e-17), hlines=[1e-12]),
            Panel("Wall-clock (secondary)", "absorbed rank $k$", "runtime (s)", "log",
                  _method_series(sub, "runtime_s")),
        ])
    sec = str(Path(fig_path).with_name(Path(fig_path).stem + "-diagnostics" + ".pdf"))
    write_panel_grid(sec, diag, row_titles=[f"Lx={lx}" for lx in args.lxs],
                     col_titles=["Isometry defect (sanity)", "Wall-clock (secondary)"],
                     cell_width=420, cell_height=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--in-dim", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=3)
    parser.add_argument("--mpo-bond", type=int, default=8)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8, 12, 20])
    parser.add_argument("--ensemble", default="gaussian",
                        choices=["gaussian", "decay", "identity_plus_noise", "controlled_exp", "controlled_power"])
    parser.add_argument("--decay", type=float, default=0.6)
    parser.add_argument("--spectrum-param", type=float, default=3.0)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--sketch", choices=["gaussian", "rademacher", "countsketch"], default="gaussian")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lxs = [3, 4]
        args.ks = [2, 4, 6]
        args.trials = 3
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
