"""Dektor reproduction and Hamiltonian-robustness figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rand_isopeps.campaign.aggregate import (
    MissingDataError,
    finite_float,
    median_bands,
    trajectory_groups,
)
from rand_isopeps.plotting import apply_paper_style, clean_axis

from .plot_common import (
    campaign_method_style,
    converged_measurements,
    draw_bands,
    final_converged_measurements,
    finish,
    need_bands,
    require_group_grid,
    require_groups,
    validate_plot_revisions,
)
from .plot_physics import _relative_errors

import matplotlib.pyplot as plt


def _physics(rows: Sequence[Mapping], study: str) -> list[dict]:
    return [
        dict(row) for row in rows
        if row.get("experiment") == "physics"
        and row.get("status") == "ok"
        and row.get("study") == study
    ]


def _contains_panel(row: Mapping, panel: str) -> bool:
    panels = row.get("dektor_panels", ())
    return isinstance(panels, Sequence) and not isinstance(panels, (str, bytes)) and panel in panels


def _state_curve_style(group, index):
    p, state_index = group
    colors = ("#0072b2", "#d55e00", "#009e73")
    lines = ("-", "--", ":")
    return (
        rf"$p={int(p)},\ \alpha={int(state_index)}$",
        colors[(int(p) - 1) % len(colors)],
        ("o", "s", "^")[int(state_index) % 3],
        lines[int(state_index) % len(lines)],
    )


def plot_dektor_convergence(rows: Sequence[Mapping], path: str | Path) -> Path:
    data = validate_plot_revisions(converged_measurements([
        row
        for row in _physics(rows, "dektor_reproduction")
        if _contains_panel(row, "figure_2")
    ]), "Dektor Figure 2")
    require_groups(
        data,
        "method_label",
        ("local_riemannian_ndis30", "global_rmps_bounded"),
        "Dektor Figure 2",
    )
    require_group_grid(
        data,
        ("states", "iteration"),
        "method_label",
        ("local_riemannian_ndis30", "global_rmps_bounded"),
        "Dektor Figure 2",
    )
    methods = sorted({str(row.get("method_label", row.get("method"))) for row in data})
    if not methods:
        raise MissingDataError("missing converged Dektor Figure 2 trajectories")
    apply_paper_style()
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.2 * len(methods), 3.1),
        layout="constrained",
        squeeze=False,
    )
    for ax, method in zip(axes.flat, methods):
        points = []
        for row in data:
            if str(row.get("method_label", row.get("method"))) != method:
                continue
            for state_index, value in enumerate(_relative_errors(row)):
                if value is not None:
                    points.append({
                        **row,
                        "series": (int(row.get("states", 1)), state_index),
                        "value": max(value, 1e-18),
                    })
        groups = sorted({row["series"] for row in points})
        bands = need_bands(
            median_bands(points, "series", "iteration", "value", groups=groups),
            f"Dektor Figure 2 {method}",
        )
        draw_bands(ax, bands, styles=_state_curve_style)
        ax.set_title(campaign_method_style(method)[0], loc="left")
        ax.set_xlabel("imaginary-time iteration")
        ax.set_ylabel("relative eigenvalue error")
        clean_axis(ax, ylog=True)
        ax.legend(fontsize=6, ncol=2)
    return finish(fig, path)


def _size_style(group, index):
    ham, method, state_index = group
    base_label, color, marker, _ = campaign_method_style(str(method), index)
    ham_label = str(ham).replace("tfim@", "g=").replace("heis", "Heisenberg")
    return (
        f"{ham_label}, {base_label}, $\\alpha={state_index}$",
        color,
        marker,
        ("-", "--")[int(state_index) % 2],
    )


def plot_dektor_size_scaling(rows: Sequence[Mapping], path: str | Path) -> Path:
    data = _physics(rows, "dektor_reproduction")
    panels = (("figure_3", "TFI"), ("figure_4", "Heisenberg"))
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), layout="constrained")
    for row_index, (panel, title) in enumerate(panels):
        panel_rows = validate_plot_revisions([
            row for row in data if _contains_panel(row, panel)
        ], f"Dektor {panel}")
        panel_rows = final_converged_measurements(panel_rows)
        require_groups(
            panel_rows,
            "method_label",
            ("local_riemannian_ndis30", "global_rmps_bounded"),
            f"Dektor {panel}",
        )
        require_group_grid(
            panel_rows,
            ("hamiltonian", "lx", "states"),
            "method_label",
            ("local_riemannian_ndis30", "global_rmps_bounded"),
            f"Dektor {panel}",
        )
        finals = panel_rows
        for column_index, p in enumerate((1, 2)):
            ax = axes[row_index, column_index]
            points = []
            for row in finals:
                if int(row.get("states", 1)) != p:
                    continue
                method = str(row.get("method_label", row.get("method", "unknown")))
                for state_index, value in enumerate(_relative_errors(row)):
                    if value is not None:
                        points.append({
                            **row,
                            "series": (str(row.get("hamiltonian")), method, state_index),
                            "value": max(value, 1e-18),
                        })
            groups = sorted({row["series"] for row in points}, key=str)
            bands = need_bands(
                median_bands(points, "series", "lx", "value", groups=groups),
                f"Dektor {panel} p={p}",
            )
            draw_bands(ax, bands, styles=_size_style)
            ax.set_title(f"{title}, p={p}", loc="left")
            ax.set_xlabel("lattice side length $L$")
            ax.set_ylabel("relative eigenvalue error")
            clean_axis(ax, ylog=True)
            ax.legend(fontsize=5.5)
    return finish(fig, path)


comparison_methods = ("local_det", "global_rmps_bounded")


def _final_study_rows(rows, study, label):
    data = validate_plot_revisions(_physics(rows, study), label)
    return final_converged_measurements(data)


def _bond_points(rows, ham):
    points = []
    for row in rows:
        if row.get("hamiltonian") != ham:
            continue
        errors = _relative_errors(row)
        if not errors or errors[0] is None:
            continue
        method = str(row.get("method_label", row.get("method", "unknown")))
        points.append({
            **row,
            "value": max(errors[0], 1e-18),
            "series": (int(row["lx"]), method),
        })
    return points


def _bond_style(group, index):
    label, color, _, _ = campaign_method_style(group[1], index)
    return (
        f"L={group[0]}, {label}",
        color,
        ("o", "s", "^")[int(group[0]) % 3],
        "-",
    )


def _draw_bond_panel(ax, rows, ham):
    subset = [row for row in rows if row.get("hamiltonian") == ham]
    label = f"{ham} bond sweep"
    require_groups(subset, "method_label", comparison_methods, label)
    require_group_grid(
        subset,
        ("lx", "chi", "eta"),
        "method_label",
        comparison_methods,
        label,
    )
    points = _bond_points(rows, ham)
    groups = sorted({row["series"] for row in points}, key=str)
    bands = need_bands(
        median_bands(points, "series", "chi", "value", groups=groups),
        label,
    )
    draw_bands(ax, bands, styles=_bond_style)
    ax.set_title(
        ham.replace("tfim@", "TFI g=").replace("heis", "Heisenberg"),
        loc="left",
    )
    ax.set_xlabel(r"paired bond budget $\chi$ ($\eta$ recorded)")
    ax.set_ylabel("relative ground-energy error")
    clean_axis(ax, ylog=True)
    ax.legend(fontsize=6)


def _robustness_points(rows):
    xxz_points = []
    compass_points = []
    for row in rows:
        errors = _relative_errors(row)
        if not errors or errors[0] is None:
            continue
        ham = str(row.get("hamiltonian"))
        method = str(row.get("method_label", row.get("method", "unknown")))
        point = {**row, "value": max(errors[0], 1e-18), "method_group": method}
        if ham == "compass":
            compass_points.append(point)
            continue
        x = finite_float(ham.partition("@")[2])
        if x is not None:
            xxz_points.append({
                **point,
                "x": x,
                "series": (int(row["lx"]), method),
            })
    return xxz_points, compass_points


def _validate_robustness(xxz_points, compass_points):
    require_groups(xxz_points, "method_group", comparison_methods, "XXZ robustness")
    require_groups(
        compass_points, "method_group", comparison_methods, "compass robustness",
    )
    require_group_grid(
        xxz_points,
        ("lx", "x"),
        "method_group",
        comparison_methods,
        "XXZ robustness",
    )


def _draw_xxz_panel(ax, points):
    groups = sorted({row["series"] for row in points}, key=str)
    bands = need_bands(
        median_bands(points, "series", "x", "value", groups=groups),
        "XXZ robustness",
    )
    draw_bands(ax, bands)
    ax.set_title("XXZ", loc="left")
    ax.set_xlabel(r"anisotropy $\Delta$")
    ax.set_ylabel("relative ground-energy error")
    clean_axis(ax, ylog=True)
    ax.legend(fontsize=6)


def _draw_compass_panel(ax, points):
    groups = [
        method
        for method in comparison_methods
        if any(row["method_group"] == method for row in points)
    ]
    labels = []
    for index, method in enumerate(groups):
        values = np.asarray([
            row["value"] for row in points if row["method_group"] == method
        ])
        label, color, marker, _ = campaign_method_style(method, index)
        labels.append(label)
        center = float(np.median(values))
        low, high = np.percentile(values, (10, 90))
        ax.errorbar(
            index,
            center,
            yerr=[[center - low], [high - center]],
            color=color,
            marker=marker,
            capsize=3,
            label=label,
        )
    ax.set_title("compass", loc="left")
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("relative ground-energy error")
    clean_axis(ax, ylog=True)
    ax.legend(fontsize=6)


def plot_bond_hamiltonian_sweeps(rows: Sequence[Mapping], path: str | Path) -> Path:
    bond = _final_study_rows(rows, "bond_sweep", "bond-sweep physics")
    robust = _final_study_rows(
        rows, "hamiltonian_robustness", "Hamiltonian-robustness physics",
    )
    if not bond or not robust:
        raise MissingDataError("missing converged bond or Hamiltonian robustness cells")
    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.0), layout="constrained")
    for ax, ham in zip(axes.flat[:3], ("tfim@3", "tfim@3.5", "heis")):
        _draw_bond_panel(ax, bond, ham)
    xxz_points, compass_points = _robustness_points(robust)
    _validate_robustness(xxz_points, compass_points)
    _draw_xxz_panel(axes.flat[3], xxz_points)
    _draw_compass_panel(axes.flat[4], compass_points)
    axes.flat[5].axis("off")
    return finish(fig, path)
