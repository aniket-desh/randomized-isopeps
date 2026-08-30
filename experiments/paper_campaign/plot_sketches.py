"""rmps and Gaussian-limit campaign figures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rand_isopeps.campaign.aggregate import field, finite_float, median_bands
from rand_isopeps.plotting import apply_paper_style, clean_axis

from .plot_common import (
    draw_bands,
    experiment_rows,
    finish,
    need_bands,
    positive_metric_rows,
)

import matplotlib.pyplot as plt


def _replicate_key(row: Mapping) -> tuple:
    """identify one paired draw after runner provenance normalization."""
    replicate_seed = field(row, "replicate_seeds.problem", None)
    if replicate_seed is not None:
        return ("replicate_seed", replicate_seed)
    return ("legacy", field(row, "seeds.problem", None), row.get("replicate"))


def _pooled_variance(rows: Sequence[Mapping]) -> float | None:
    count = 0
    mean = 0.0
    m2 = 0.0
    trace = None
    for row in rows:
        n = finite_float(row.get("samples"))
        trial_mean = finite_float(row.get("quadratic_sample_mean"))
        trial_m2 = finite_float(row.get("quadratic_sample_m2"))
        trial_trace = finite_float(row.get("trace_value"))
        if None in (n, trial_mean, trial_m2, trial_trace) or n < 2:
            continue
        n = int(n)
        if trace is not None and not np.isclose(trace, trial_trace):
            raise ValueError("pooled variance trials have inconsistent trace values")
        trace = trial_trace
        total = count + n
        delta = trial_mean - mean
        m2 += trial_m2 + delta**2 * count * n / total
        mean += delta * n / total
        count = total
    if count < 2 or trace is None:
        return None
    return float(m2 / (count - 1) / trace**2)


def _variance_bands(rows: Sequence[Mapping], method: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("method") == method:
            grouped[int(row.get("chi_sk", 0))].append(row)
    if not grouped:
        return {}
    xs, center, low, high, counts = [], [], [], [], []
    for chi_sk in sorted(grouped):
        trials = grouped[chi_sk]
        values = [
            value for row in trials
            if (value := finite_float(row.get("normalized_quadratic_variance"))) is not None
        ]
        if not values:
            continue
        xs.append(chi_sk)
        center.append(_pooled_variance(trials) or float(np.median(values)))
        low.append(float(np.percentile(values, 10)))
        high.append(float(np.percentile(values, 90)))
        counts.append(len(values))
    return {method: {"x": xs, "median": center, "low": low, "high": high, "n": counts}}


def plot_gaussian_moments_osi(rows: Sequence[Mapping], path: str | Path) -> Path:
    selected = [
        row for row in rows
        if row.get("benchmark")
        in {"rmps_figure2_variance", "rmps_figure2_nystrom"}
    ]
    data = experiment_rows(selected, "gaussian_limit", "rmps figure 2")
    variance = [row for row in data if row.get("benchmark") == "rmps_figure2_variance"]
    nystrom = [row for row in data if row.get("benchmark") == "rmps_figure2_nystrom"]
    variance_bands = need_bands(_variance_bands(variance, "rmps"), "rmps quadratic-form variance")

    nystrom_rows = []
    for row in positive_metric_rows(nystrom, "relative_nuclear_error"):
        label = (
            "gaussian nystrom"
            if row.get("method") == "gaussian_nystrom"
            else f"mps gram nystrom, $\\chi={int(row['chi_sk'])}$"
        )
        nystrom_rows.append({**row, "method_group": label})
    groups = sorted({row["method_group"] for row in nystrom_rows})
    nystrom_bands = need_bands(
        median_bands(
            nystrom_rows,
            "method_group",
            "embedding_dim",
            "relative_nuclear_error",
            groups=groups,
        ),
        "rmps figure 2 nystrom errors",
    )

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), layout="constrained")
    draw_bands(
        axes[0], variance_bands,
        styles=lambda _group, _index: ("rmps", "#009e73", "o", "-"),
    )
    gaussian_rows = [row for row in variance if row.get("method") == "global_gaussian"]
    gaussian_values = [
        value for row in gaussian_rows
        if (value := finite_float(row.get("normalized_quadratic_variance"))) is not None
    ]
    if not gaussian_values:
        raise ValueError("missing dense Gaussian variance trials")
    gaussian_center = _pooled_variance(gaussian_rows) or float(np.median(gaussian_values))
    axes[0].axhspan(
        float(np.percentile(gaussian_values, 10)),
        float(np.percentile(gaussian_values, 90)),
        color="#222222",
        alpha=0.10,
        linewidth=0,
    )
    axes[0].axhline(gaussian_center, color="#222222", linestyle="--", label="dense gaussian")
    for multiple in range(1, 7):
        axes[0].axvline(10 * multiple, color="#999999", linewidth=0.5, alpha=0.35)
    axes[0].set_xlabel(r"rmps bond $\chi$")
    axes[0].set_ylabel(r"$\mathrm{var}(\omega^T A\omega)/\mathrm{tr}(A)^2$")
    clean_axis(axes[0], ylog=True)
    axes[0].legend()

    def nystrom_style(group, index):
        if group == "gaussian nystrom":
            return group, "#222222", "o", "--"
        color = plt.cm.viridis(index / max(len(nystrom_bands) - 1, 1))
        return group, color, "o", "-"

    draw_bands(axes[1], nystrom_bands, styles=nystrom_style)
    axes[1].set_xlabel(r"embedding dimension $k$")
    axes[1].set_ylabel(r"$\|A-\widehat A\|_*/\|A\|_*$")
    clean_axis(axes[1], ylog=True)
    axes[1].legend(ncol=2, fontsize=6)
    return finish(fig, path)


def plot_rmps_gaussian_ratio(rows: Sequence[Mapping], path: str | Path) -> Path:
    selected = [row for row in rows if row.get("benchmark") == "column_embedding"]
    data = experiment_rows(
        selected, "gaussian_limit", "rmps-to-gaussian error ratios"
    )
    gaussian = defaultdict(list)
    for row in data:
        if row.get("method") != "global_gaussian":
            continue
        value = finite_float(row.get("projection_error"))
        if value is None:
            continue
        key = (
            *_replicate_key(row),
            row.get("lx"),
            row.get("subspace"),
            row.get("ell"),
        )
        gaussian[key].append(value)
    ratios = []
    for row in data:
        if row.get("method") != "rmps":
            continue
        value = finite_float(row.get("projection_error"))
        key = (
            *_replicate_key(row),
            row.get("lx"),
            row.get("subspace"),
            row.get("ell"),
        )
        if value is None or key not in gaussian:
            continue
        ratios.append({
            "status": "ok",
            "lx": row["lx"],
            "chi_sk": row["chi_sk"],
            "ratio": max(value / max(float(np.median(gaussian[key])), 1e-18), 1e-18),
        })
    bands = need_bands(
        median_bands(ratios, "lx", "chi_sk", "ratio"),
        "paired rmps-to-gaussian projection errors",
    )
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.1, 3.0), layout="constrained")
    draw_bands(
        ax,
        bands,
        styles=lambda group, index: (
            f"$l_x={int(group)}$",
            plt.cm.viridis(index / max(len(bands) - 1, 1)),
            "o",
            "-",
        ),
    )
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"sketch bond $\chi_{sk}$")
    ax.set_ylabel("projection error / gaussian error")
    clean_axis(ax, ylog=True, xlog=True)
    ax.legend(ncol=2)
    return finish(fig, path)
