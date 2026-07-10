#!/usr/bin/env python3
"""Hamiltonian-survey report generator (re-runnable, append-as-data-lands).

Reads whatever exp04/exp09/exp10 CSVs exist under ``--data-dir`` and regenerates the
survey figures into ``reports/figures/column_sketch/`` in the repo plot style. Safe to
re-run after each NERSC wave -- it only draws the experiments/states/Lx actually present,
so the running report (``reports/hamiltonian_survey_report.md``) stays current.

    python experiments/column_sketch/scripts/survey_report.py [--data-dir DIR]

The scientific question: does the disentangled-global column sketch (fixed vertical bond
eta=4) beat the local Moses move, and does that track how low-effective-rank the physical
column is? Figures: (1) the ladder -- m2/local vs Lx per Hamiltonian; (2) the mechanism --
the disentangler's residual-truncation loss per Lx (low = reorganizable/low-rank).
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

from rand_isopeps.plotting import Panel, Series, state_style, write_line_panels

SUITE = "column_sketch"
VALID = ("random", "tfim@", "heis", "xxz@", "compass")
FIGDIR = Path(__file__).resolve().parents[3] / "reports" / "figures" / SUITE


def _valid_state(s: str) -> bool:
    return any(s == v or s.startswith(v) for v in VALID)


def _load(data_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(data_dir / f"{prefix}*.csv"))):
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    # drop rows corrupted by a same-second slug collision (fixed in io.timestamp_slug):
    # a merged 'state' field, or an impossible normalized error > 1.
    return [r for r in rows if _valid_state(r.get("state", ""))
            and _as_float(r.get("local_eps", "0")) is not None
            and _as_float(r.get("local_eps", "0")) <= 1.0]


def _as_float(x):
    try:
        v = float(x)
        return v if v == v else None   # drop NaN
    except (TypeError, ValueError):
        return None


def _med(rows, key):
    xs = [_as_float(r.get(key)) for r in rows]
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _by_state_lx(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["state"], int(r["lx"]))].append(r)
    return g


def _series_per_state(g, ynum, yden=None, hardness=None):
    """One Series per state: median(ynum)/median(yden) vs Lx, styled by state_style."""
    states = sorted({s for s, _ in g}, key=lambda s: (hardness or {}).get(s, 0.0))
    out = []
    for s in states:
        lxs = sorted(lx for st_, lx in g if st_ == s)
        xs, ys = [], []
        for lx in lxs:
            num = _med(g[s, lx], ynum)
            den = _med(g[s, lx], yden) if yden else 1.0
            if num is None or not den:
                continue
            xs.append(float(lx))
            ys.append(num / den)
        if xs:
            label, color, marker, ls = state_style(s)
            out.append(Series(label=label, x=xs, y=ys, color=color, marker=marker, linestyle=ls))
    return out


def build(data_dir: Path) -> list[str]:
    e10 = _load(data_dir, "exp10-disentangler")
    e09 = _load(data_dir, "exp9-propagated-cost")
    e04 = _load(data_dir, "exp4-")
    written: list[str] = []

    if e10:
        g = _by_state_lx(e10)
        # hardness = median m2/local at each state's largest Lx (data-driven legend order)
        hardness = {}
        for s in {st_ for st_, _ in g}:
            lx = max(lx for st_, lx in g if st_ == s)
            num, den = _med(g[s, lx], "dis_eps_bestcase"), _med(g[s, lx], "local_eps")
            hardness[s] = (num / den) if (num and den) else 0.0
        pa = Panel(r"Disentangler holds $\eta=4$: does it beat local?",
                   r"column height $L_x$", r"$\epsilon_{\mathrm{dis}} / \epsilon_{\mathrm{local}}$",
                   "log", _series_per_state(g, "dis_eps_bestcase", "local_eps", hardness),
                   hlines=[1.0])
        pb = Panel(r"Mechanism: residual-truncation loss (low $\Rightarrow$ reorganizable)",
                   r"column height $L_x$", r"$\tau_{\mathrm{dis}} = \sqrt{\sum \mathrm{dis\_tail}}/\|Y\|$",
                   "log", _series_per_state(g, "residual_loss", None, hardness))
        written.append(str(write_line_panels(FIGDIR / "hamsurvey-ladder.pdf", [pa, pb],
                                              width=1000, height=380)))

    if e09:
        g = _by_state_lx(e09)
        pa = Panel(r"Plain global: vertical bond $\eta_q$ to match local",
                   r"column height $L_x$", r"matched $\eta_q$", "linear",
                   _series_per_state(g, "eta_q"))
        pb = Panel(r"Propagated cost ratio (local/global; $>1$ = global cheaper)",
                   r"column height $L_x$", r"ratio$_{\mathrm{prop}}$", "log",
                   _series_per_state(g, "ratio_prop"), hlines=[1.0])
        written.append(str(write_line_panels(FIGDIR / "hamsurvey-cost.pdf", [pa, pb],
                                              width=1000, height=380)))

    if e04:
        g = _by_state_lx(e04)
        # exp04 spectrum diagnostic: effective rank rho2 (the hypothesis' direct measurement).
        key = next((k for k in ("rho2", "eff_rank", "n2") if any(k in r for r in e04)), None)
        if key:
            pa = Panel(r"Spectrum diagnostic: effective column rank",
                       r"column height $L_x$", key, "linear", _series_per_state(g, key))
            written.append(str(write_line_panels(FIGDIR / "hamsurvey-spectrum.pdf", [pa],
                                                  width=560, height=380)))

    _print_summary(e10, e09)
    return written


def _print_summary(e10, e09):
    if e10:
        print("\n=== exp10 3-way at fixed eta=4 (median/seeds; dis<=local => WIN) ===")
        g = _by_state_lx(e10)
        print(f"{'state':12s} {'Lx':>2s} {'local':>8s} {'plain':>8s} {'dis':>8s} {'win':>4s} {'resloss':>9s}")
        for s, lx in sorted(g):
            loc, m1, m2 = _med(g[s, lx], "local_eps"), _med(g[s, lx], "plain_eps"), _med(g[s, lx], "dis_eps_bestcase")
            if None in (loc, m1, m2):
                continue
            print(f"{s:12s} {lx:2d} {loc:8.4f} {m1:8.4f} {m2:8.4f} {'YES' if m2 <= loc else 'no':>4s} "
                  f"{_med(g[s, lx], 'residual_loss'):9.2e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "outputs" / SUITE / "data",
                   help="dir with the exp04/09/10 CSVs (e.g. a results-branch worktree's outputs/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    figs = build(args.data_dir)
    print("\nwrote:")
    for f in figs:
        print(" ", f)
