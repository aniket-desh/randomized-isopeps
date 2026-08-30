"""shared plotting helpers for campaign figures."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np


def _configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache = Path(tempfile.gettempdir()) / "rand_isopeps-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".matplotlib"
    for fontlist in source.glob("fontlist-v*.json") if source.exists() else ():
        target = cache / fontlist.name
        if not target.exists():
            shutil.copyfile(fontlist, target)
    os.environ["MPLCONFIGDIR"] = str(cache)


_configure_matplotlib_cache()

import matplotlib.pyplot as plt

from rand_isopeps.campaign.aggregate import (
    MissingDataError,
    field,
    finite_float,
    require_rows,
    trajectory_groups,
)
from rand_isopeps.plotting import method_style, savefig


extra_colors = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#7b3294",
    "#666666",
)


def campaign_method_style(name: str, index: int = 0) -> tuple[str, str, str, str]:
    candidates = {
        "local_riemannian_ndis30": (
            "riemannian Moses, 30 sweeps",
            "#d55e00",
            "D",
            "-",
        ),
        "local_rsvd2_gaussian": ("local gaussian", "#56b4e9", "s", "-"),
        "local_rsvd2_sparsestack": ("local sparsestack", "#cc79a7", "^", "-"),
        "global_rademacher": ("rademacher", "#e69f00", "v", "-"),
        "global_sparsestack": ("sparsestack", "#7b3294", "P", "-"),
        "dense_exact": ("dense exact", "#222222", "o", "-"),
        "dense_strang": ("dense strang", "#666666", "s", "--"),
        "dense_first_order": ("dense first order", "#666666", "s", "--"),
        "peps_full": ("full trotter", "#999999", "^", ":"),
    }
    if name in candidates:
        return candidates[name]
    canonical = {
        "local_det": "local_det",
        "local_det_ndis0": "local_det",
        "sequential_moses": "local_det",
        "local_gaussian": "local_rand",
        "local_sparsestack": "local_rand",
        "global_gaussian": "global_gauss",
        "global_rmps": "global_rmps",
        "global_rmps_bounded": "global_rmps",
        "global_kron": "global_kron",
        "peps_local": "local_det",
        "peps_sketch": "global_rmps",
    }.get(name)
    if canonical is not None:
        return method_style(canonical)
    return name.replace("_", " "), extra_colors[index % len(extra_colors)], "o", "-"


def finish(fig, path: str | Path) -> Path:
    output = savefig(fig, path)
    plt.close(fig)
    return output


def experiment_rows(
    rows: Iterable[Mapping], experiment: str, description: str
) -> list[dict]:
    selected = require_rows(
        rows,
        lambda row: row.get("experiment") == experiment and row.get("status") == "ok",
        description,
    )
    return validate_plot_revisions(selected, description)


def validate_plot_revisions(
    rows: Iterable[Mapping], description: str
) -> list[dict]:
    """reject mixed revisions within any campaign family."""
    selected = [dict(row) for row in rows]
    revisions = {}
    for row in selected:
        family = str(row.get("campaign_family", row.get("experiment", "legacy")))
        revision = (
            row.get("campaign_revision"),
            row.get("runtime_source_fingerprint"),
        )
        revisions.setdefault(family, set()).add(revision)
    mixed = [family for family, values in revisions.items() if len(values) > 1]
    if mixed:
        raise ValueError(
            f"{description} mixes revisions for: {', '.join(sorted(mixed))}"
        )
    return selected


def require_groups(
    rows: Iterable[Mapping], key: str, expected: Iterable[str], description: str
) -> None:
    """fail a final figure when a required comparison group is absent."""
    present = {str(field(row, key, "")) for row in rows}
    missing = [value for value in expected if value not in present]
    if missing:
        raise MissingDataError(
            f"missing {description} groups: {', '.join(missing)}"
        )


def require_group_grid(
    rows: Iterable[Mapping],
    facet_keys: Sequence[str],
    group_key: str,
    expected: Iterable[str],
    description: str,
) -> None:
    """require every observed facet to contain every comparison group."""
    selected = [dict(row) for row in rows]
    expected_groups = tuple(expected)
    facets = {
        tuple(field(row, key, None) for key in facet_keys)
        for row in selected
    }
    missing = []
    for facet in sorted(facets, key=str):
        present = {
            str(field(row, group_key, ""))
            for row in selected
            if tuple(field(row, key, None) for key in facet_keys) == facet
        }
        absent = [group for group in expected_groups if group not in present]
        if absent:
            missing.append(f"{facet}: {','.join(absent)}")
    if missing:
        raise MissingDataError(
            f"incomplete {description} grid; " + "; ".join(missing)
        )


def draw_bands(
    ax,
    bands: Mapping,
    *,
    styles: Callable[[object, int], tuple[str, str, str, str]] | None = None,
) -> None:
    for index, (group, band) in enumerate(bands.items()):
        label, color, marker, linestyle = (
            styles(group, index)
            if styles is not None
            else campaign_method_style(str(group), index)
        )
        x = np.asarray(band["x"], dtype=float)
        median = np.asarray(band["median"], dtype=float)
        low = np.asarray(band["low"], dtype=float)
        high = np.asarray(band["high"], dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            x,
            median,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markeredgecolor="white",
        )


def need_bands(bands: Mapping, description: str) -> Mapping:
    if not bands:
        raise MissingDataError(f"missing data for {description}")
    return bands


def positive_metric_rows(rows: Iterable[Mapping], key: str) -> list[dict]:
    output = []
    for raw in rows:
        value = finite_float(field(raw, key, None))
        if value is None:
            continue
        output.append({**raw, key: max(value, 1e-18)})
    return output


def first_number(row: Mapping, keys: Sequence[str], index: int = 0) -> float | None:
    for key in keys:
        value = field(row, key, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = value[index] if len(value) > index else None
        found = finite_float(value)
        if found is not None:
            return found
    return None


def converged_measurements(rows: Iterable[Mapping]) -> list[dict]:
    """reject accuracy figures containing unconverged contractions."""
    selected = [dict(row) for row in rows]
    unconverged = [
        str(row.get("task_id", "unknown"))
        for row in selected
        if row.get("measurement_converged") is False
        or field(row, "preparation.converged", None) is False
    ]
    if unconverged:
        raise MissingDataError(
            f"unconverged measurement cells: {len(set(unconverged))}"
        )
    return selected


def final_converged_measurements(rows: Iterable[Mapping]) -> list[dict]:
    """select each trajectory endpoint, then apply the contraction gate."""
    finals = [
        trajectory[-1]
        for trajectory in trajectory_groups(rows).values()
    ]
    return converged_measurements(finals)
