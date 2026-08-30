"""comparable-cell physics and literature-baseline figures."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rand_isopeps.campaign.aggregate import (
    MissingDataError,
    field,
    finite_float,
    median_bands,
    paired_median_bands,
    trajectory_groups,
)
from rand_isopeps.plotting import apply_paper_style, clean_axis

from .plot_common import (
    campaign_method_style,
    converged_measurements,
    draw_bands,
    experiment_rows,
    final_converged_measurements,
    finish,
    first_number,
    need_bands,
    require_group_grid,
    require_groups,
    validate_plot_revisions,
)
from .table_2_validation import (
    table_2_exact_relative_tolerance,
    table_2_path,
    validate_table_2_exact_references,
)

import matplotlib.pyplot as plt


def _cell_key(row: Mapping) -> str:
    payload = {
        key: row.get(key)
        for key in (
            "study",
            "hamiltonian",
            "lx",
            "ly",
            "chi",
            "eta",
            "states",
            "trotter_order",
            "schedule",
        )
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _select_cell(rows: Sequence[Mapping], role: str) -> list[dict]:
    physics = [row for row in rows if row.get("experiment") == "physics"]
    marked = [row for row in physics if row.get("plot_role") == role]
    selected = marked or [row for row in physics if row.get("plot_role") is None]
    candidates = converged_measurements(experiment_rows(
        selected, "physics", f"{role} physics trajectories"
    ))
    groups = defaultdict(list)
    for row in candidates:
        groups[_cell_key(row)].append(row)
    if not groups:
        raise MissingDataError(f"missing data for {role} physics cell")
    if marked and len(groups) != 1:
        raise ValueError(f"plot role {role!r} spans incomparable physics cells")
    selected_key = max(groups, key=lambda key: (len(groups[key]), key))
    selected = groups[selected_key]
    if marked:
        local_label = (
            "local_riemannian_ndis30" if role == "low_energy" else "local_det"
        )
        require_groups(
            selected,
            "method_label",
            (local_label, "global_rmps_bounded"),
            f"{role} physics",
        )
    trajectories = trajectory_groups(selected)
    if not trajectories:
        raise MissingDataError(f"missing ordered trajectories for {role}")
    return [row for trajectory in trajectories.values() for row in trajectory]


def plot_physics_trajectories(rows: Sequence[Mapping], path: str | Path) -> Path:
    data = _select_cell(rows, "physics_trajectory")
    derived = []
    for row in data:
        energy_error = first_number(row, ("ground_energy_errors", "energy_error"))
        residual = first_number(row, ("relative_residuals", "residual_norms"))
        base = {
            **row,
            "method_group": str(row.get("method_label", row.get("method", "unknown"))),
        }
        if energy_error is not None:
            derived.append({**base, "metric": "energy", "value": max(abs(energy_error), 1e-18)})
        if residual is not None:
            derived.append({**base, "metric": "residual", "value": max(abs(residual), 1e-18)})
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, metric, ylabel in zip(
        axes,
        ("energy", "residual"),
        ("absolute ground-energy error", "relative residual"),
    ):
        subset = [row for row in derived if row["metric"] == metric]
        groups = sorted({row["method_group"] for row in subset})
        bands = need_bands(
            paired_median_bands(
                subset, "method_group", "iteration", "value", groups=groups
            ),
            f"physics {metric} trajectories",
        )
        draw_bands(ax, bands)
        ax.set_xlabel("imaginary-time iteration")
        ax.set_ylabel(ylabel)
        clean_axis(ax, ylog=True)
    axes[0].legend(ncol=2)
    return finish(fig, path)


def _state_style(group, index):
    method, state_index = group
    label, color, marker, _ = campaign_method_style(str(method), index)
    linestyles = ("-", "--", ":", "-.")
    return f"{label}, $\\alpha={state_index}$", color, marker, linestyles[int(state_index) % 4]


def plot_energy_gap(rows: Sequence[Mapping], path: str | Path) -> Path:
    data = _select_cell(rows, "low_energy")
    errors, gaps = [], []
    for row in data:
        method = str(row.get("method_label", row.get("method", "unknown")))
        values = row.get("ground_energy_errors")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for state_index, value in enumerate(values):
                found = finite_float(value)
                if found is not None:
                    errors.append({
                        **row,
                        "series": (method, state_index),
                        "value": max(abs(found), 1e-18),
                    })
        energies = row.get("energies")
        references = row.get("reference_energies")
        gap_error = None
        if (
            isinstance(energies, Sequence)
            and isinstance(references, Sequence)
            and len(energies) >= 2
            and len(references) >= 2
        ):
            numbers = [finite_float(value) for value in (*energies[:2], *references[:2])]
            if all(value is not None for value in numbers):
                e0, e1, r0, r1 = numbers
                gap_error = abs((e1 - e0) - (r1 - r0))
        if gap_error is not None:
            gaps.append({**row, "method_group": method, "value": max(gap_error, 1e-18)})
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), layout="constrained")
    state_groups = sorted({row["series"] for row in errors}, key=str)
    state_bands = need_bands(
        paired_median_bands(errors, "series", "iteration", "value", groups=state_groups),
        "state-resolved low-energy errors",
    )
    draw_bands(axes[0], state_bands, styles=_state_style)
    axes[0].set_xlabel("imaginary-time iteration")
    axes[0].set_ylabel(r"absolute eigenvalue error $|E_\alpha-E_{\alpha,ref}|$")
    clean_axis(axes[0], ylog=True)
    axes[0].legend(ncol=2, fontsize=6)
    gap_groups = sorted({row["method_group"] for row in gaps})
    gap_bands = need_bands(
        paired_median_bands(gaps, "method_group", "iteration", "value", groups=gap_groups),
        "energy-gap trajectory",
    )
    draw_bands(axes[1], gap_bands)
    axes[1].set_xlabel("imaginary-time iteration")
    axes[1].set_ylabel(r"absolute gap error $|\Delta E-\Delta E_{ref}|$")
    clean_axis(axes[1], ylog=True)
    axes[1].legend()
    return finish(fig, path)


def _final_rows(rows: Sequence[Mapping], role: str) -> list[dict]:
    selected = [row for row in rows if row.get("plot_role") == role]
    data = experiment_rows(selected, "physics", f"{role} final physics cells")
    return final_converged_measurements(data)


def _relative_errors(row: Mapping) -> list[float | None]:
    errors = row.get("ground_energy_errors")
    references = row.get("reference_energies")
    if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)):
        return []
    values = []
    for index, error in enumerate(errors):
        found = finite_float(error)
        reference = (
            finite_float(references[index])
            if isinstance(references, Sequence) and len(references) > index
            else None
        )
        values.append(
            None if found is None else abs(found) / max(abs(reference or 1.0), 1e-18)
        )
    return values


def _table_2_executed_bands(executed: Sequence[Mapping], state_index: int) -> dict:
    subset = [
        {**row, "series": (row["method"], row["chi"], row["eta"])}
        for row in executed
        if int(row["state_index"]) == state_index
    ]
    groups = sorted({row["series"] for row in subset}, key=str)
    return median_bands(
        subset,
        "series",
        "g",
        "value",
        groups=groups,
    )


def plot_physics_sweeps(rows: Sequence[Mapping], path: str | Path) -> Path:
    final = _final_rows(rows, "physics_sweep")
    if not final:
        raise MissingDataError("missing converged physical sketch sweeps")
    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), layout="constrained")
    for ax, axis, xlabel in zip(
        axes.flat[:3],
        ("ell", "chi_sk", "kappa"),
        (r"sketch width $\ell$", r"sketch bond $\chi_{sk}$", r"residual cap $\kappa$"),
    ):
        points = []
        for row in final:
            if row.get("study") != "sketch_parameter_sweep":
                continue
            sweep_axis = row.get("sweep_axis")
            if sweep_axis not in {axis, "baseline"}:
                continue
            config = row.get("method_config", row.get("task_method", {}))
            x = finite_float(config.get(axis) if isinstance(config, Mapping) else None)
            errors = _relative_errors(row)
            if x is None or not errors or errors[0] is None:
                continue
            points.append({
                **row,
                "x": x,
                "value": max(errors[0], 1e-18),
                "group": f"p={int(row.get('states', 1))}",
            })
        groups = sorted({row["group"] for row in points})
        require_group_grid(
            points,
            ("x",),
            "group",
            ("p=1", "p=2"),
            f"physical {axis} sweep",
        )
        bands = need_bands(
            median_bands(points, "group", "x", "value", groups=groups),
            f"physical {axis} sweep",
        )
        draw_bands(ax, bands)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("relative ground-energy error")
        clean_axis(ax, ylog=True)
        ax.legend()

    for ax, axis, xlabel in zip(
        axes.flat[3:5],
        ("chi", "eta"),
        (r"vertical bond $\chi$", r"column bond $\eta$"),
    ):
        points = []
        for row in final:
            if row.get("study") != "bond_parameter_sweep":
                continue
            if row.get("sweep_axis") not in {axis, "baseline"}:
                continue
            x = finite_float(row.get(axis))
            errors = _relative_errors(row)
            if x is None or not errors or errors[0] is None:
                continue
            points.append({
                **row,
                "x": x,
                "value": max(errors[0], 1e-18),
                "group": str(row.get("method", "unknown")),
            })
        groups = sorted({row["group"] for row in points})
        require_group_grid(
            points,
            ("x",),
            "group",
            ("peps_local", "peps_sketch"),
            f"physical {axis} sweep",
        )
        bands = need_bands(
            paired_median_bands(points, "group", "x", "value", groups=groups),
            f"physical {axis} sweep",
        )
        draw_bands(ax, bands)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("relative ground-energy error")
        clean_axis(ax, ylog=True)
        ax.legend()

    points = []
    for row in final:
        if row.get("sweep_axis") != "states":
            continue
        errors = [value for value in _relative_errors(row) if value is not None]
        if not errors:
            continue
        points.append({
            **row,
            "x": float(row.get("states", 1)),
            "value": max(max(errors), 1e-18),
            "group": str(row.get("method_label", row.get("method", "unknown"))),
        })
    groups = sorted({row["group"] for row in points})
    require_group_grid(
        points,
        ("x",),
        "group",
        ("local_det", "global_rmps_bounded"),
        "physical block-size sweep",
    )
    bands = need_bands(
        median_bands(points, "group", "x", "value", groups=groups),
        "physical block-size sweep",
    )
    draw_bands(axes.flat[5], bands)
    axes.flat[5].set_xlabel(r"block size $p$")
    axes.flat[5].set_ylabel("worst low-lying relative error")
    clean_axis(axes.flat[5], ylog=True)
    axes.flat[5].legend()
    return finish(fig, path)


def plot_table_2_comparison(rows: Sequence[Mapping], path: str | Path) -> Path:
    table_rows = [
        row for row in rows
        if row.get("experiment") == "physics"
        and row.get("status") == "ok"
        and row.get("study") == "dektor_reproduction"
        and isinstance(row.get("dektor_panels"), Sequence)
        and "table_2" in row["dektor_panels"]
    ]
    tolerances = {
        float(row.get(
            "published_reference_relative_tolerance",
            table_2_exact_relative_tolerance,
        ))
        for row in table_rows
    }
    if len(tolerances) != 1:
        raise MissingDataError(
            "table 2 rows are missing one consistent exact-reference tolerance"
        )
    try:
        validate_table_2_exact_references(
            rows,
            relative_tolerance=next(iter(tolerances)),
        )
    except ValueError as exc:
        raise MissingDataError(str(exc)) from exc

    payload = json.loads(table_2_path.read_text(encoding="utf-8"))
    published = defaultdict(list)
    labels = {
        "block_isopeps_chi12_eta20": r"block isoPEPS $\chi=12$ [published]",
        "block_isopeps_chi4_eta8": r"block isoPEPS $\chi=4$ [published]",
        "peps_tangent_chi2": r"tangent PEPS $\chi=2$ [published]",
        "peps_tangent_chi4": r"tangent PEPS $\chi=4$ [published]",
    }
    for row in payload["records"]:
        for method, value in row["published_relative_errors"].items():
            if value is not None:
                published[(int(row["state_index"]), method)].append((float(row["g"]), float(value)))

    executed = []
    physics = [
        row for row in table_rows
        if row.get("lx") == 4
        and row.get("states") == 2
        and str(row.get("hamiltonian", "")).startswith("tfim@")
        and isinstance(row.get("dektor_panels"), Sequence)
        and "table_2" in row["dektor_panels"]
    ]
    physics = validate_plot_revisions(physics, "Dektor Table 2")
    physics = final_converged_measurements(physics)
    for chi, eta in ((4, 8), (12, 20)):
        subset = [
            row for row in physics
            if int(row.get("chi", -1)) == chi and int(row.get("eta", -1)) == eta
        ]
        require_groups(
            subset,
            "method_label",
            ("local_riemannian_ndis30", "global_rmps_bounded"),
            f"Table 2 chi={chi} eta={eta}",
        )
        require_group_grid(
            subset,
            ("hamiltonian",),
            "method_label",
            ("local_riemannian_ndis30", "global_rmps_bounded"),
            f"Table 2 chi={chi} eta={eta}",
        )
    for row in physics:
        try:
            g = float(str(row["hamiltonian"]).split("@", 1)[1])
        except (KeyError, ValueError):
            continue
        for state_index, value in enumerate(_relative_errors(row)):
            if value is not None:
                executed.append({
                    "state_index": state_index,
                    "g": g,
                    "value": max(value, 1e-18),
                    "method": str(row.get("method_label", row.get("method", "unknown"))),
                    "chi": int(row.get("chi", -1)),
                    "eta": int(row.get("eta", -1)),
                })

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), layout="constrained")
    colors = plt.cm.tab10.colors
    for state_index, ax in enumerate(axes):
        for index, ((alpha, method), values) in enumerate(sorted(published.items())):
            if alpha != state_index:
                continue
            values = sorted(values)
            ax.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                label=labels[method],
                color=colors[index % len(colors)],
                marker="o",
                linestyle="--",
            )
        bands = _table_2_executed_bands(executed, state_index)

        def executed_style(group, index):
            method, chi, eta = group
            _, color, marker, _ = campaign_method_style(str(method), index)
            return (
                f"{str(method).replace('_', ' ')}, "
                rf"$\chi={chi},\eta={eta}$ [executed]",
                color,
                marker,
                "-",
            )

        draw_bands(ax, bands, styles=executed_style)
        ax.set_title(rf"state $\alpha={state_index}$", loc="left")
        ax.set_xlabel("transverse field $g$")
        ax.set_ylabel("relative eigenvalue error")
        clean_axis(ax, ylog=True)
    axes[0].legend(fontsize=6)
    return finish(fig, path)
