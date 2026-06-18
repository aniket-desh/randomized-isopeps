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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter


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


def _draw_panel(ax, panel: Panel, legend: dict[str, Line2D]) -> None:
    """Render one Panel onto ``ax``; accumulate first-seen legend handles.

    Shared by the single-row (``write_line_panels``) and faceted-grid
    (``write_panel_grid``) renderers so the per-panel styling stays in one place.
    The caller owns the title (grids title only the top row).
    """
    is_log = panel.yscale == "log" and _log_span_decades(panel) >= 1.0
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
            xs, ys, color=s.color, marker=marker, markevery=s.markevery,
            linestyle=s.linestyle, markeredgecolor="white", label=s.label,
            clip_on=False, zorder=3,
        )
        legend.setdefault(s.label, line)

    for xv in panel.vlines or []:
        ax.axvline(xv, color="#888888", linestyle="--", linewidth=0.8, zorder=2)

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


def _shared_legend(fig, legend: dict[str, Line2D], ncol: int | None = None) -> None:
    if not legend:
        return
    fig.legend(
        list(legend.values()), list(legend.keys()),
        loc="outside lower center", ncol=ncol or min(len(legend), 4),
        handlelength=1.8, columnspacing=1.8, borderaxespad=0.0,
    )


def _row_header(ax, text: str) -> None:
    """Vertical row label anchored just left of a leftmost axis' y-label."""
    ax.annotate(
        text, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 30, 0),
        xycoords=ax.yaxis.label, textcoords="offset points",
        rotation=90, va="center", ha="center", fontsize=11, fontweight="semibold",
    )


def write_line_panels(
    path: str | Path,
    panels: list[Panel],
    width: int = 960,
    height: int = 380,
) -> Path:
    """Render a single row of line panels with matplotlib and save as PDF.

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
        1, n, figsize=(width / dpi, height / dpi), squeeze=False, layout="constrained",
    )
    # Generous padding so rotated y-axis labels and left-aligned titles get
    # room instead of being clipped, with clear gaps between panels.
    fig.get_layout_engine().set(w_pad=0.16, h_pad=0.10, wspace=0.10, hspace=0.0)

    legend: dict[str, Line2D] = {}  # stable order: first-seen label wins
    for ax, panel in zip(axes[0], panels):
        _draw_panel(ax, panel, legend)
        ax.set_title(panel.title, pad=8, loc="left")

    _shared_legend(fig, legend)
    fig.savefig(out)
    plt.close(fig)
    return out


def write_panel_grid(
    path: str | Path,
    grid: Sequence[Sequence[Panel | None]],
    row_titles: Sequence[str] | None = None,
    col_titles: Sequence[str] | None = None,
    cell_width: int = 380,
    cell_height: int = 300,
) -> Path:
    """Render a 2D grid of Panels (rows x cols) with one shared legend.

    ``grid`` is a list of rows, each a list of Panels (``None`` blanks a cell).
    ``row_titles`` label each row on the left (e.g. ``"d=3"``); ``col_titles``
    title the top row only. Used for the larger faceted parameter sweeps where a
    single row of panels would be unreadable.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _apply_style()

    dpi = plt.rcParams["figure.dpi"]
    nrows = max(len(grid), 1)
    ncols = max((len(r) for r in grid), default=1)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_width * ncols / dpi, cell_height * nrows / dpi),
        squeeze=False, layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.16, h_pad=0.16, wspace=0.10, hspace=0.08)

    legend: dict[str, Line2D] = {}
    for r, row in enumerate(grid):
        for c in range(ncols):
            ax = axes[r][c]
            panel = row[c] if c < len(row) else None
            if panel is None:
                ax.set_visible(False)
                continue
            _draw_panel(ax, panel, legend)
            if r == 0:
                ax.set_title(col_titles[c] if col_titles else panel.title, pad=8, loc="left")
        if row_titles and row and row[0] is not None:
            _row_header(axes[r][0], row_titles[r])

    _shared_legend(fig, legend)
    fig.savefig(out)
    plt.close(fig)
    return out


@dataclass(frozen=True)
class ScatterSeries:
    label: str
    x: list[float]
    y: list[float]
    c: list[float]  # color values mapped through the figure's shared colormap
    marker: str = "o"


@dataclass(frozen=True)
class ScatterPanel:
    title: str
    xlabel: str
    ylabel: str
    xscale: Literal["linear", "log"]
    yscale: Literal["linear", "log"]
    series: list[ScatterSeries]
    hlines: list[float] | None = None  # e.g. speedup = 1 reference
    vlines: list[float] | None = None  # e.g. rho = 1/d reference


def _short_log_tick(x, _pos=None) -> str:
    """Readable decimal log-tick labels (0.05, 0.1, 0.2, 1, 2, 10) for small ranges."""
    if x <= 0:
        return ""
    return f"{x:.2g}"


def write_scatter_grid(
    path: str | Path,
    grid: Sequence[Sequence[ScatterPanel | None]],
    clabel: str,
    row_titles: Sequence[str] | None = None,
    col_titles: Sequence[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cell_width: int = 560,
    cell_height: int = 320,
) -> Path:
    """Faceted scatter grid with a single shared colorbar (rank-fraction phase diagram).

    Each point carries (x, y) and a color value ``c`` mapped through one shared
    ``cmap``/``Normalize``. Tuned for dense phase diagrams: only the bottom row
    gets x-labels and the left column y-labels, log ticks use short decimal
    labels, x-limits are shared per column, markers are small/translucent (the
    second SVD stacks many points at one rho), and there is no marker legend
    because the columns already encode the SVD target.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _apply_style()

    dpi = plt.rcParams["figure.dpi"]
    nrows = max(len(grid), 1)
    ncols = max((len(r) for r in grid), default=1)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(cell_width * ncols / dpi, cell_height * nrows / dpi),
        squeeze=False, layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.22, h_pad=0.16, wspace=0.06, hspace=0.05)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Share x-limits within each column so hiding intermediate ticks stays honest.
    col_xlims: list[tuple[float, float] | None] = []
    for c in range(ncols):
        xs = [x for r in grid if c < len(r) and r[c] is not None
              for s in r[c].series for x in s.x if x is not None and x > 0]
        if xs:
            lo, hi = min(xs), max(xs)
            pad = (hi / lo) ** 0.06 if hi > lo else 1.1
            col_xlims.append((lo / pad, hi * pad))
        else:
            col_xlims.append(None)

    for r, row in enumerate(grid):
        for c in range(ncols):
            ax = axes[r][c]
            panel = row[c] if c < len(row) else None
            if panel is None:
                ax.set_visible(False)
                continue
            for s in panel.series:
                marker = _MARKER_ALIASES.get(s.marker, s.marker)
                ax.scatter(s.x, s.y, c=s.c, cmap=cmap, norm=norm, marker=marker,
                           s=26, alpha=0.78, edgecolors="white", linewidths=0.35, zorder=3)
            for yv in panel.hlines or []:
                ax.axhline(yv, color="#888888", linestyle="--", linewidth=0.8, zorder=2)
            for xv in panel.vlines or []:
                ax.axvline(xv, color="#888888", linestyle=":", linewidth=0.8, zorder=2)

            ax.set_xscale(panel.xscale)
            ax.set_yscale(panel.yscale)
            if col_xlims[c] is not None:
                ax.set_xlim(*col_xlims[c])
            if panel.xscale == "log":
                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
                ax.xaxis.set_major_formatter(FuncFormatter(_short_log_tick))
                ax.xaxis.set_minor_formatter(NullFormatter())
            if panel.yscale == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
                ax.yaxis.set_major_formatter(FuncFormatter(_short_log_tick))
                ax.yaxis.set_minor_formatter(NullFormatter())

            # label only the outer axes
            if r == nrows - 1:
                ax.set_xlabel(panel.xlabel, labelpad=4)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(panel.ylabel, labelpad=5)

            ax.margins(x=0.05, y=0.12)
            ax.grid(True, which="major", axis="both", color="#e6e6e6", linewidth=0.7)
            ax.tick_params(length=3)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(col_titles[c] if col_titles else panel.title, pad=8, loc="left")
        if row_titles and row and row[0] is not None:
            _row_header(axes[r][0], row_titles[r])

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label=clabel,
                        shrink=0.82, pad=0.035, fraction=0.045)
    cbar.ax.tick_params(length=3)
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
