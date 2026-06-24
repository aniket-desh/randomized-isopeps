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

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column.global_range import global_column_range, reference_svd
from rand_isopeps.column.local_moses import local_column_qr
from rand_isopeps.column.operator import controlled_spectrum_column_matrix, random_column_operator
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, write_panel_grid

SUITE = "column_sketch"

# method key -> (label, kind, color, marker, linestyle). kind: ("local", randomized) or
# ("global", sketch_kind, chi_sk).
METHODS = {
    "local_det": ("local Moses (det)", ("local", False), "#0072b2", "o", "-"),
    "local_rand": ("local Moses (rand)", ("local", True), "#56b4e9", "s", "-"),
    "global_kron": ("global Kron (χ=1)", ("global", "kron", 1), "#d55e00", "X", ":"),
    "global_rmps4": ("global rMPS χ=4", ("global", "rmps", 4), "#009e73", "D", "-"),
    "global_rmps8": ("global rMPS χ=8", ("global", "rmps", 8), "#e69f00", "v", "-"),
    "global_gauss": ("global Gaussian", ("global", "gaussian", 0), "#222222", "o", "--"),
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


def _method_series(rows, value_key):
    present = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    bands = median_band(rows, group_key="method", x_key="k", value_key=value_key, group_order=present)
    out = []
    for key, (xs, med, lo, hi) in bands.items():
        label, _, color, marker, ls = METHODS[key]
        out.append(Series(label=label, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                          color=color, marker=marker, linestyle=ls))
    return out


def _ey_series(rows):
    """The Eckart-Young floor vs k (a probe-independent reference line on the error panel)."""
    ks = sorted({int(r["k"]) for r in rows})
    ys = [float(np.median([float(r["ey_floor"]) for r in rows if int(r["k"]) == k])) for k in ks]
    return [Series(label="Eckart-Young floor", x=[float(k) for k in ks], y=ys,
                   color="#999999", marker="o", linestyle=":")]


def make_plot(rows, fig_path, args) -> None:
    """Rows = Lx; cols = column error vs k (+EY floor), excess over EY, isometry, runtime."""
    grid = []
    for lx in args.lxs:
        sub = [r for r in rows if int(r["lx"]) == lx]
        grid.append([
            Panel("column error", "absorbed rank k", r"$\|C-QR\|_F/\|C\|_F$", "log",
                  _method_series(sub, "rel_error") + _ey_series(sub)),
            Panel("excess over Eckart-Young", "absorbed rank k", "excess (method − optimal)", "log",
                  _method_series(sub, "excess_error")),
            Panel("isometry defect (sanity)", "absorbed rank k", r"$\|Q^*Q-I\|_F$", "log",
                  _method_series(sub, "isometry_defect")),
            Panel("wall-clock (secondary)", "absorbed rank k", "runtime (s)", "log",
                  _method_series(sub, "runtime_s")),
        ])
    write_panel_grid(fig_path, grid, row_titles=[f"Lx={lx}" for lx in args.lxs],
                     col_titles=["column error (+ EY floor)", "excess over Eckart-Young",
                                 "isometry defect (sanity)", "wall-clock (secondary)"],
                     cell_width=380, cell_height=300)


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
