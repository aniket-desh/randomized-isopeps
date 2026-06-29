"""Guardrails for the canonical paper-plot style (docs/PLOT.md).

These lock in the two project-wide invariants the style depends on -- every figure
is written as BOTH .pdf and .png, and the method/state visual grammar lives in one
module -- plus back-compat for the legacy series keys older suites still import.
Matplotlib renders headless (Agg) so these run with no display and no optional deps.
"""

from pathlib import Path

from rand_isopeps.plotting import (
    COLORS,
    MARKERS,
    METHOD_ORDER,
    PALETTE,
    Panel,
    Series,
    method_style,
    savefig,
    state_style,
    write_line_panels,
    write_panel_grid,
)


def _panel(title="t"):
    lab, c, m, ls = method_style("global_rmps")
    s = Series(label=lab, x=[1.0, 2.0, 3.0], y=[1e-1, 1e-3, 1e-5], color=c, marker=m, linestyle=ls)
    return Panel(title, "rank $k$", "rel. error", "log", [s])


def test_savefig_writes_both_pdf_and_png(tmp_path):
    """savefig emits a .pdf and a .png regardless of the suffix it is handed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1e-2, 1e-4])
    out = savefig(fig, tmp_path / "fig.png")  # hand it a .png; still want both
    plt.close(fig)
    assert out.suffix == ".pdf"
    for ext in (".pdf", ".png"):
        p = tmp_path / f"fig{ext}"
        assert p.exists() and p.stat().st_size > 0


def test_renderers_emit_both_formats(tmp_path):
    """The Panel renderers (line row + facet grid) both dual-output via savefig."""
    line = tmp_path / "line.pdf"
    write_line_panels(line, [_panel("A"), _panel("B")], width=680, height=270)
    grid = tmp_path / "grid.pdf"
    write_panel_grid(grid, [[_panel("A"), _panel("B")]], col_titles=["A", "B"])
    for stem in ("line", "grid"):
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
        assert (tmp_path / f"{stem}.png").stat().st_size > 0


def test_method_style_is_the_canonical_grammar():
    """One color/marker/linestyle per method; the EY floor is a no-marker line."""
    # every ordered method resolves to its registered color
    for key in METHOD_ORDER:
        label, color, marker, ls = method_style(key)
        assert color == COLORS[key]
        assert isinstance(label, str) and label
    # the Eckart-Young floor draws as a line with NO marker (empty marker code)
    assert method_style("eckart_young")[2] == ""
    # methods are visually distinct: the primary algorithmic curves use distinct colors
    primaries = ["global_gauss", "global_rmps", "global_kron", "local_det", "local_rand"]
    assert len({COLORS[k] for k in primaries}) == len(primaries)


def test_state_style_grammar():
    """random is the gray dashed baseline; the critical TFIM point gets the accent."""
    assert state_style("random")[1:] == ("#888888", "s", "--")
    crit = state_style("tfim@3.04")  # g <= 3.1 -> critical accent, solid
    para = state_style("tfim@3.5")   # deeper paramagnet -> distinct color, solid
    assert crit[0] == "TFIM g=3.04" and crit[3] == "-"
    assert para[0] == "TFIM g=3.5" and para[3] == "-"
    assert crit[1] != para[1]        # distinguishable states


def test_legacy_series_keys_still_resolve():
    """Older suites import PALETTE/MARKERS by their synthetic series keys -- keep them."""
    for key in ("det", "rand_first", "rand_second", "rand_both"):
        assert key in PALETTE and key in MARKERS


def test_panel_hlines_render(tmp_path):
    """A horizontal reference line (sanity threshold) renders without error."""
    p = Panel("sanity", "x", "defect", "log", [_panel().series[0]], hlines=[1e-12])
    write_line_panels(tmp_path / "h.pdf", [p], width=360, height=270)
    assert (tmp_path / "h.png").stat().st_size > 0
