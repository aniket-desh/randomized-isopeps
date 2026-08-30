"""dense-oracle correctness ladders for single-state and block evolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rand_isopeps.campaign.aggregate import (
    MissingDataError,
    field,
    finite_float,
    paired_median_bands,
)
from rand_isopeps.plotting import apply_paper_style, clean_axis

from .plot_common import (
    campaign_method_style,
    draw_bands,
    experiment_rows,
    final_converged_measurements,
    finish,
    need_bands,
    require_group_grid,
    require_groups,
)

import matplotlib.pyplot as plt


hamiltonians = ("tfim@3.5", "heis")
single_lattices = ((2, 2), (2, 3), (3, 3))
block_cells = ((2, 2, 2), (2, 3, 2), (2, 2, 3))
single_methods = (
    "dense_exact",
    "dense_strang",
    "peps_full",
    "peps_local",
    "peps_sketch",
)
block_methods = (
    "dense_exact",
    "dense_first_order",
    "local_full_rank",
    "rmps_full_rank",
    "local_truncated",
    "rmps_truncated",
)


def _method_name(row: Mapping) -> str:
    method = row.get("method")
    if isinstance(method, str):
        return method
    config = row.get("method_config", row.get("task_method", method))
    return str(config.get("name", "unknown")) if isinstance(config, Mapping) else "unknown"


def _final_rows(rows: Sequence[Mapping], study: str) -> list[dict]:
    selected = [row for row in rows if row.get("study") == study]
    return final_converged_measurements(
        experiment_rows(
            selected,
            "physics",
            f"{study} correctness trajectories",
        )
    )


def _energy_vector(row: Mapping) -> list[float] | None:
    values = row.get("energies")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    numbers = [finite_float(value) for value in values]
    return None if not numbers or any(value is None for value in numbers) else numbers


def _pair_key(row: Mapping, *, include_tau: bool) -> tuple:
    key = (
        str(row.get("hamiltonian")),
        int(row.get("lx", -1)),
        int(row.get("ly", -1)),
        int(row.get("states", 1)),
        field(row, "seeds.problem", None),
    )
    return (*key, finite_float(row.get("tau"))) if include_tau else key


def _block_method(row: Mapping) -> str:
    name = _method_name(row)
    if name in {"dense_exact", "dense_first_order"}:
        return name
    regime = str(row.get("regime"))
    prefix = "local" if name == "peps_local" else "rmps"
    suffix = "full_rank" if regime == "full_rank_oracle" else "truncated"
    return f"{prefix}_{suffix}"


def _relative_to_dense(rows: Sequence[Mapping], *, include_tau: bool) -> list[dict]:
    references = {
        _pair_key(row, include_tau=include_tau): _energy_vector(row)
        for row in rows
        if _method_name(row) == "dense_exact"
    }
    output = []
    for row in rows:
        values = _energy_vector(row)
        reference = references.get(_pair_key(row, include_tau=include_tau))
        if values is None or reference is None or len(values) != len(reference):
            raise MissingDataError("correctness row is missing its paired dense energy")
        error = max(
            abs(value - exact) / max(abs(exact), 1e-300)
            for value, exact in zip(values, reference)
        )
        output.append({
            **row,
            "series": _method_name(row) if include_tau else _block_method(row),
            "relative_dense_energy_error": float(error),
        })
    return output


def _single_style(group, index):
    labels = {
        "dense_exact": "exact step",
        "dense_strang": "dense Strang",
        "peps_full": "full PEPS",
        "peps_local": "local Moses",
        "peps_sketch": "rMPS column QR",
    }
    _, color, marker, linestyle = campaign_method_style(str(group), index)
    return labels[str(group)], color, marker, linestyle


def _at_plot_floor(rows: Sequence[Mapping]) -> list[dict]:
    positive = [
        float(row["relative_dense_energy_error"])
        for row in rows
        if float(row["relative_dense_energy_error"]) > 0.0
    ]
    floor = min(positive, default=1e-14) / 10.0
    return [
        {
            **row,
            "relative_dense_energy_error": max(
                float(row["relative_dense_energy_error"]), floor
            ),
        }
        for row in rows
    ]


def _single_panel(ax, rows: Sequence[Mapping], hamiltonian: str, lattice: tuple[int, int]):
    lx, ly = lattice
    selected = [
        row for row in rows
        if row.get("hamiltonian") == hamiltonian
        and int(row.get("lx", -1)) == lx
        and int(row.get("ly", -1)) == ly
    ]
    selected = _at_plot_floor(selected)
    require_groups(selected, "series", single_methods, f"{hamiltonian} {lx}x{ly}")
    require_group_grid(
        selected,
        ("tau", "seeds.problem"),
        "series",
        single_methods,
        f"{hamiltonian} {lx}x{ly} correctness",
    )
    taus = {finite_float(row.get("tau")) for row in selected}
    if taus != {0.1, 0.05, 0.025, 0.0125}:
        raise MissingDataError(f"incomplete {hamiltonian} {lx}x{ly} tau grid")
    bands = need_bands(
        paired_median_bands(
            selected,
            "series",
            "tau",
            "relative_dense_energy_error",
            groups=single_methods,
        ),
        f"{hamiltonian} {lx}x{ly} correctness",
    )
    draw_bands(ax, bands, styles=_single_style)
    ax.set_title(f"{hamiltonian}, {lx}x{ly}", loc="left")
    ax.set_xlabel(r"imaginary-time step $\tau$")
    clean_axis(ax, ylog=True)


def _block_panel(ax, rows: Sequence[Mapping], hamiltonian: str, cell: tuple[int, int, int]):
    lx, ly, states = cell
    selected = [
        row for row in rows
        if row.get("hamiltonian") == hamiltonian
        and int(row.get("lx", -1)) == lx
        and int(row.get("ly", -1)) == ly
        and int(row.get("states", -1)) == states
    ]
    selected = _at_plot_floor(selected)
    require_groups(selected, "series", block_methods, f"{hamiltonian} {lx}x{ly} p={states}")
    colors = ("#222222", "#666666", "#009e73", "#0072b2", "#e69f00", "#7b3294")
    markers = ("o", "s", "^", "D", "v", "P")
    for index, method in enumerate(block_methods):
        values = [
            float(row["relative_dense_energy_error"])
            for row in selected
            if row["series"] == method
        ]
        if len(values) != 1:
            raise MissingDataError(
                f"expected one {method} result for {hamiltonian} {lx}x{ly} p={states}"
            )
        ax.scatter(index, values[0], color=colors[index], marker=markers[index], zorder=3)
    ax.set_xticks(
        range(len(block_methods)),
        ("exact", "dense\n1st", "local\nfull", "rMPS\nfull", "local\ntrunc", "rMPS\ntrunc"),
        fontsize=6,
    )
    ax.set_title(f"{hamiltonian}, {lx}x{ly}, p={states}", loc="left")
    clean_axis(ax, ylog=True)


def plot_correctness(rows: Sequence[Mapping], path: str | Path) -> Path:
    """render every dense-oracle cell without pooling lattice or Hamiltonian data."""
    single = _relative_to_dense(
        _final_rows(rows, "correctness_ladder"), include_tau=True
    )
    block = _relative_to_dense(
        _final_rows(rows, "block_correctness"), include_tau=False
    )
    apply_paper_style()
    fig, axes = plt.subplots(4, 3, figsize=(11.0, 10.5), layout="constrained")
    for row_index, hamiltonian in enumerate(hamiltonians):
        for column_index, lattice in enumerate(single_lattices):
            _single_panel(axes[row_index, column_index], single, hamiltonian, lattice)
        for column_index, cell in enumerate(block_cells):
            _block_panel(axes[row_index + 2, column_index], block, hamiltonian, cell)
    axes[0, 0].legend(ncol=2, fontsize=6)
    axes[0, 0].set_ylabel("single-state relative energy error\n(zero at plot floor)")
    axes[1, 0].set_ylabel("single-state relative energy error\n(zero at plot floor)")
    axes[2, 0].set_ylabel("block max relative energy error\n(zero at plot floor)")
    axes[3, 0].set_ylabel("block max relative energy error\n(zero at plot floor)")
    return finish(fig, path)
