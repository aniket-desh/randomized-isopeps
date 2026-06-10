"""Matplotlib plotting helpers for compact, paper-style research figures.

The figures aim for the restrained style of the block-isoPEPS paper: white
background, small multiples grouped by parameter, log axes for error
quantities, thin lines with clear markers, and a single shared legend.

Matplotlib is imported with the non-interactive ``Agg`` backend so it never
touches a GUI toolkit. An earlier note in this repo warned that the matplotlib
font cache hung in a managed sandbox; in this environment the cache builds
instantly, so we use matplotlib directly and emit vector PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter


# Okabe-Ito colorblind-safe palette, keyed by experiment series name.
PALETTE = {
    "det": "#0072b2",          # blue
    "rand_first": "#d55e00",   # vermillion
    "rand_second": "#009e73",  # bluish green
    "rand_both": "#cc79a7",    # reddish purple
    "zipup_svd": "#0072b2",
    "randomized": "#d55e00",
    "src": "#009e73",          # bluish green
}

# Series name -> matplotlib marker code.
MARKERS = {
    "det": "o",
    "rand_first": "s",
    "rand_second": "^",
    "rand_both": "D",
    "zipup_svd": "o",
    "randomized": "s",
    "src": "^",
}

# Accept both the legacy descriptive names and matplotlib codes.
_MARKER_ALIASES = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
}


def _apply_style() -> None:
    """Set rcParams for a clean, restrained research-plot look."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
            "lines.markeredgewidth": 0.8,
        }
    )


@dataclass(frozen=True)
class Series:
    label: str
    x: list[float]
    y: list[float]
    color: str
    marker: str = "o"
    ylow: list[float] | None = None  # optional lower band (e.g. 25th percentile)
    yhigh: list[float] | None = None  # optional upper band (e.g. 75th percentile)
    linestyle: str = "-"
    # marker subsampling for dense curves: int step or (offset, step) tuple so
    # overlapping series can stagger their markers. None marks every point.
    markevery: int | tuple[int, int] | None = None


@dataclass(frozen=True)
class Panel:
    title: str
    xlabel: str
    ylabel: str
    yscale: Literal["linear", "log"]
    series: list[Series]
    vlines: list[float] | None = None  # optional vertical reference lines (e.g. k1, k2)


def _log_span_decades(panel: Panel) -> float:
    """Number of base-10 decades spanned by a panel's positive y-values.

    Used to decide whether a "log" panel actually benefits from a log axis. A
    span below one decade renders more cleanly as a linear axis with a shared
    scientific multiplier, so this returns 0 when there is no usable data.
    """
    ys = [y for s in panel.series for y in s.y if y is not None and y > 0]
    if len(ys) < 2:
        return 0.0
    lo, hi = min(ys), max(ys)
    if lo <= 0.0:
        return 0.0
    import math

    return math.log10(hi / lo)


def write_line_panels(
    path: str | Path,
    panels: list[Panel],
    width: int = 960,
    height: int = 380,
) -> Path:
    """Render compact line-panel figures with matplotlib and save as PDF.

    ``width``/``height`` are interpreted as approximate pixel dimensions at the
    figure DPI and converted to inches, preserving the call sites that passed
    pixel sizes to the old SVG writer.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    _apply_style()

    dpi = plt.rcParams["figure.dpi"]
    n = max(len(panels), 1)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(width / dpi, height / dpi),
        squeeze=False,
        layout="constrained",
    )
    # Generous padding so rotated y-axis labels and left-aligned titles get
    # room instead of being clipped, with clear gaps between panels.
    fig.get_layout_engine().set(w_pad=0.16, h_pad=0.10, wspace=0.10, hspace=0.0)
    axes = axes[0]

    # Stable legend order across panels: first-seen label wins.
    legend: dict[str, Line2D] = {}

    log_axis = {id(panel): (panel.yscale == "log" and _log_span_decades(panel) >= 1.0) for panel in panels}

    for ax, panel in zip(axes, panels):
        is_log = log_axis[id(panel)]
        for s in panel.series:
            has_band = s.ylow is not None and s.yhigh is not None
            lows = s.ylow if has_band else [None] * len(s.x)
            highs = s.yhigh if has_band else [None] * len(s.x)
            rows = list(zip(s.x, s.y, lows, highs))
            if is_log:
                rows = [r for r in rows if r[1] is not None and r[1] > 0]
            if not rows:
                continue
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            marker = _MARKER_ALIASES.get(s.marker, s.marker)
            if has_band:
                los = [r[2] for r in rows]
                his = [r[3] for r in rows]
                if is_log:
                    # keep the band strictly positive so it renders on a log axis
                    los = [max(lo, y * 1e-3) for lo, y in zip(los, ys)]
                ax.fill_between(xs, los, his, color=s.color, alpha=0.15, linewidth=0, zorder=1)
            (line,) = ax.plot(
                xs,
                ys,
                color=s.color,
                marker=marker,
                markevery=s.markevery,
                linestyle=s.linestyle,
                markeredgecolor="white",
                label=s.label,
                clip_on=False,
                zorder=3,
            )
            legend.setdefault(s.label, line)

        for xv in panel.vlines or []:
            ax.axvline(xv, color="#888888", linestyle="--", linewidth=0.8, zorder=2)

        ax.set_title(panel.title, pad=8, loc="left")
        ax.set_xlabel(panel.xlabel, labelpad=4)
        ax.set_ylabel(panel.ylabel, labelpad=5)
        ax.margins(x=0.08, y=0.10)

        if is_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.grid(True, which="major", axis="both", color="#e6e6e6", linewidth=0.7)
            ax.grid(True, which="minor", axis="y", color="#f2f2f2", linewidth=0.5)
        else:
            # Sub-decade error data reads cleanly as a linear axis with a single
            # shared scientific multiplier (e.g. ticks 0.96 0.98 1.00 with a
            # "x10^-4" offset) rather than awkward mixed-exponent log labels.
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_size(8)
            ax.grid(True, which="major", axis="both", color="#e6e6e6", linewidth=0.7)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.tick_params(length=3)
        ax.set_axisbelow(True)

    if legend:
        handles = list(legend.values())
        labels = list(legend.keys())
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(len(labels), 4),
            handlelength=1.8,
            columnspacing=1.8,
            borderaxespad=0.0,
        )

    fig.savefig(out)
    plt.close(fig)
    return out


# Backward-compatible alias for the previous SVG entry point. The extension on
# the supplied path now determines the format (matplotlib infers it), so PDF
# paths produce PDF and any lingering .svg path still produces valid SVG.
def write_line_panels_svg(
    path: str | Path,
    panels: list[Panel],
    width: int = 960,
    height: int = 300,
) -> Path:
    return write_line_panels(path, panels, width=width, height=height)
