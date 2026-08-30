"""column-method and isometry campaign figures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rand_isopeps.campaign.aggregate import (
    isometry_summary,
    median_bands,
    paired_median_bands,
)
from rand_isopeps.plotting import apply_paper_style, clean_axis, state_style

from .plot_common import (
    converged_measurements,
    draw_bands,
    experiment_rows,
    finish,
    need_bands,
    positive_metric_rows,
    require_group_grid,
    require_groups,
)

import matplotlib.pyplot as plt


method_order = (
    "local_det_ndis0",
    "local_riemannian_ndis30",
    "local_rsvd2_gaussian",
    "local_rsvd2_sparsestack",
    "global_gaussian",
    "global_rademacher",
    "global_sparsestack",
    "global_rmps_bounded",
    "global_kron",
)


def plot_size_scaling(rows: Sequence[Mapping], path: str | Path) -> Path:
    selected = [
        row for row in rows
        if row.get("study") in {None, "method_comparison"}
    ]
    data = experiment_rows(
        selected, "column_moves", "random-versus-physical size scaling"
    )
    data = positive_metric_rows(converged_measurements(data), "state_infidelity")
    methods = (("global_rmps", "rmps column qr"), ("sequential_moses", "sequential moses"))
    require_groups(
        data,
        "method",
        tuple(method for method, _ in methods),
        "size-scaling",
    )
    require_group_grid(
        data,
        ("state", "lx"),
        "method",
        tuple(method for method, _ in methods),
        "size-scaling",
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, (method, title) in zip(axes, methods):
        subset = [row for row in data if str(row.get("method")) == method]
        states = sorted({str(row.get("state")) for row in subset})
        bands = {}
        for state in states:
            bands.update(paired_median_bands(
                [row for row in subset if str(row.get("state")) == state],
                "state",
                "lx",
                "state_infidelity",
                groups=(state,),
            ))
        need_bands(bands, f"{method} state-infidelity scaling")
        draw_bands(ax, bands, styles=lambda group, _index: state_style(str(group)))
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"column size $l_x$")
        ax.set_ylabel("state infidelity")
        clean_axis(ax, ylog=True)
    axes[0].legend()
    return finish(fig, path)


def plot_candidate_sketches(rows: Sequence[Mapping], path: str | Path) -> Path:
    selected = [
        row for row in rows
        if row.get("study") in {None, "method_comparison"}
        and str(row.get("method_label")) in method_order
    ]
    data = experiment_rows(
        selected, "column_moves", "candidate sketch comparison"
    )
    data = positive_metric_rows(converged_measurements(data), "state_infidelity")
    panel_states = ("random_raw", "tfim@3.5")
    if not any(row.get("state") == "random_raw" for row in data):
        panel_states = ("random", "tfim@3.5")
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
    for ax, state in zip(axes, panel_states):
        subset = [row for row in data if row.get("state") == state]
        require_groups(subset, "method_label", method_order, f"{state} candidate")
        for lx in sorted({int(row["lx"]) for row in subset}):
            expected = (
                method_order
                if state == "random_raw" or lx <= 4
                else tuple(
                    method for method in method_order
                    if not method.startswith("global_")
                    or method in {"global_rmps_bounded", "global_kron"}
                )
            )
            require_groups(
                [row for row in subset if int(row["lx"]) == lx],
                "method_label",
                expected,
                f"{state} L={lx} candidate",
            )
        bands = need_bands(
            paired_median_bands(
                subset,
                "method_label",
                "lx",
                "state_infidelity",
                groups=method_order,
            ),
            f"paired candidate sketches on {state}",
        )
        draw_bands(ax, bands)
        ax.set_title(state_style(state)[0], loc="left")
        ax.set_xlabel(r"column size $l_x$")
        ax.set_ylabel("state infidelity")
        clean_axis(ax, ylog=True)
    axes[0].legend(ncol=2, fontsize=6)
    return finish(fig, path)


def plot_controlled_spectra(rows: Sequence[Mapping], path: str | Path) -> Path:
    selected = [row for row in rows if row.get("study") == "controlled_spectrum"]
    data = positive_metric_rows(
        experiment_rows(selected, "column_moves", "controlled-spectrum sketches"),
        "projection_excess",
    )
    controls = (("controlled_exp", 4.0), ("controlled_power", 1.0))
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), layout="constrained")
    for ax, (family, decay) in zip(axes, controls):
        subset = [
            row for row in data
            if row.get("family") == family and float(row.get("decay", -1)) == decay
        ]
        groups = tuple(name for name in method_order if name.startswith("global_"))
        require_groups(
            subset,
            "method_label",
            groups,
            f"{family} controlled-spectrum",
        )
        require_group_grid(
            subset,
            ("lx",),
            "method_label",
            groups,
            f"{family} controlled-spectrum",
        )
        bands = need_bands(
            paired_median_bands(
                subset,
                "method_label",
                "lx",
                "projection_excess",
                groups=groups,
            ),
            f"{family} controlled spectrum",
        )
        draw_bands(ax, bands)
        ax.set_title(f"{family.removeprefix('controlled_')} decay {decay:g}", loc="left")
        ax.set_xlabel(r"column size $l_x$")
        ax.set_ylabel("excess over spectral floor")
        clean_axis(ax, ylog=True)
    axes[0].legend(ncol=2, fontsize=6)
    return finish(fig, path)


def plot_column_oat(rows: Sequence[Mapping], path: str | Path) -> Path:
    """plot one-factor RMPS tuning without pooling lattice sizes or states."""
    selected = [row for row in rows if row.get("study") == "one_at_a_time"]
    data = experiment_rows(selected, "column_moves", "one-factor column tuning")
    data = positive_metric_rows(converged_measurements(data), "state_infidelity")
    axes_names = ("eta", "ell", "chi_sk", "kappa", "n_power")
    states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
    sizes = (4, 6, 7)
    apply_paper_style()
    fig, plot_axes = plt.subplots(
        len(sizes),
        len(axes_names),
        figsize=(15.0, 7.8),
        layout="constrained",
        squeeze=False,
    )
    for row_index, lx in enumerate(sizes):
        for column_index, axis_name in enumerate(axes_names):
            ax = plot_axes[row_index, column_index]
            points = []
            for row in data:
                if int(row.get("lx", -1)) != lx:
                    continue
                baseline = row.get("baseline")
                config = row.get("method_config")
                if not isinstance(baseline, Mapping) or not isinstance(config, Mapping):
                    continue
                if any(
                    config.get(other) != baseline.get(other)
                    for other in axes_names
                    if other != axis_name
                ):
                    continue
                x = config.get(axis_name)
                if x is not None:
                    points.append({**row, "x": x})
            require_groups(
                points,
                "state",
                states,
                f"L={lx} {axis_name} one-factor",
            )
            require_group_grid(
                points,
                ("x",),
                "state",
                states,
                f"L={lx} {axis_name} one-factor",
            )
            bands = need_bands(
                median_bands(
                    points,
                    "state",
                    "x",
                    "state_infidelity",
                    groups=states,
                ),
                f"L={lx} {axis_name} one-factor sweep",
            )
            draw_bands(
                ax,
                bands,
                styles=lambda group, _index: state_style(str(group)),
            )
            if row_index == 0:
                ax.set_title(axis_name.replace("_", " "), loc="left")
            if column_index == 0:
                ax.set_ylabel(f"L={lx}\nstate infidelity")
            ax.set_xlabel(axis_name.replace("_", " "))
            clean_axis(ax, ylog=True)
    plot_axes[0, 0].legend(fontsize=6)
    return finish(fig, path)


isometry_states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
isometry_routes = (
    "boundary_right",
    "interior_right",
    "interior_left",
    "round_trip",
)


def _isometry_inputs(rows: Sequence[Mapping]):
    data = converged_measurements(
        experiment_rows(rows, "isometry", "isometry preservation heatmap")
    )
    normalized = [{**row, "regime": row.get("regime", "all")} for row in data]
    summary = isometry_summary(
        normalized,
        group_keys=(
            "lx", "ly", "state", "regime", "method_label", "route",
        ),
    )
    if not summary:
        raise ValueError("missing data for isometry max and 95th percentile")
    facets = sorted({
        (int(row["lx"]), int(row["ly"]), str(row["regime"]))
        for row in summary
    })
    lookup = {
        (
            int(row["lx"]), int(row["ly"]), str(row["state"]),
            str(row["regime"]), str(row["method_label"]), str(row["route"]),
        ): row
        for row in summary
    }
    return normalized, facets, lookup


def _expected_isometry_methods(state: str, lx: int) -> tuple[str, ...]:
    if state == "random_raw" or lx <= 3:
        return method_order
    return tuple(
        method for method in method_order
        if not method.startswith("global_")
        or method in {"global_rmps_bounded", "global_kron"}
    )


def _isometry_panel(normalized, lookup, lx, ly, state, regime):
    subset = [
        row
        for row in normalized
        if int(row.get("lx", -1)) == lx
        and int(row.get("ly", -1)) == ly
        and row.get("state") == state
        and row.get("regime") == regime
    ]
    expected = _expected_isometry_methods(state, lx)
    label = f"{lx}x{ly} {state} {regime} isometry"
    require_groups(subset, "method_label", expected, label)
    require_groups(subset, "route", isometry_routes, label)
    require_group_grid(
        subset, ("route",), "method_label", expected, label,
    )
    return np.asarray([
        [
            math.log10(max(float(lookup[key]["p95"]), 1e-18))
            if (key := (lx, ly, state, regime, method, route)) in lookup
            else np.nan
            for route in isometry_routes
        ]
        for method in method_order
    ])


def _isometry_panels(normalized, lookup, facets):
    return {
        (lx, ly, state, regime): _isometry_panel(
            normalized, lookup, lx, ly, state, regime,
        )
        for lx, ly, regime in facets
        for state in isometry_states
    }


def _draw_isometry_panel(ax, matrix, lookup, panel, row_index, column_index, limits):
    lx, ly, state, regime = panel
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="magma",
        vmin=limits[0],
        vmax=limits[1],
    )
    if row_index == 0:
        ax.set_title(state.replace("_", " "), loc="left")
    if column_index == 0:
        ax.set_ylabel(f"{lx}x{ly}, {regime.replace('_', ' ')}")
        ax.set_yticks(
            range(len(method_order)),
            [method.replace("_", " ") for method in method_order],
        )
    else:
        ax.set_yticks([])
    ax.set_xticks(
        range(len(isometry_routes)),
        [route.replace("_", " ") for route in isometry_routes],
        rotation=25,
        ha="right",
    )
    for i, method in enumerate(method_order):
        for j, route in enumerate(isometry_routes):
            key = (lx, ly, state, regime, method, route)
            if key in lookup:
                ax.text(
                    j,
                    i,
                    f"{math.log10(max(float(lookup[key]['max']), 1e-18)):.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
    return image


def plot_isometry_heatmap(rows: Sequence[Mapping], path: str | Path) -> Path:
    normalized, facets, lookup = _isometry_inputs(rows)
    panels = _isometry_panels(normalized, lookup, facets)
    finite = np.concatenate([
        matrix[np.isfinite(matrix)] for matrix in panels.values()
    ])
    limits = float(np.min(finite)), float(np.max(finite))
    apply_paper_style()
    fig, axes = plt.subplots(
        len(facets), len(isometry_states),
        figsize=(15.5, 2.8 * len(facets)),
        layout="constrained",
        squeeze=False,
    )
    image = None
    for row_index, (lx, ly, regime) in enumerate(facets):
        for column_index, state in enumerate(isometry_states):
            panel = lx, ly, state, regime
            image = _draw_isometry_panel(
                axes[row_index, column_index], panels[panel], lookup, panel,
                row_index, column_index, limits,
            )
    fig.colorbar(
        image,
        ax=axes,
        label=r"$\log_{10}$ p95 defect; text is $\log_{10}$ maximum",
        shrink=0.8,
    )
    return finish(fig, path)
