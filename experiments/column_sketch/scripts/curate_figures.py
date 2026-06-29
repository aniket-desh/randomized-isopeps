#!/usr/bin/env python3
"""Render curated headline PNGs for the column_sketch suite into reports/figures/.

Reads the most recent sweep CSVs from ``outputs/column_sketch/data/`` and re-renders
the headline panels (reusing each experiment's own plot functions) as committed PNGs
under ``reports/figures/column_sketch/``, the tracked figure location. Run after a
full sweep:

    python experiments/column_sketch/scripts/exp01_materialized_column_rmps.py --lxs 3 4 5 6 7 8 --trials 16
    python experiments/column_sketch/scripts/exp02_global_vs_local_moses.py   --lxs 4 5 6 7   --trials 16
    python experiments/column_sketch/scripts/curate_figures.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS, "..", "..", ".."))
DATA = os.path.join(ROOT, "outputs", "column_sketch", "data")
DEST = os.path.join(ROOT, "reports", "figures", "column_sketch")


def _load(module_file: str):
    path = os.path.join(SCRIPTS, module_file)
    spec = importlib.util.spec_from_file_location(module_file[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _latest(prefix: str) -> list[dict[str, object]]:
    matches = sorted(glob.glob(os.path.join(DATA, f"{prefix}-*.csv")))
    if not matches:
        raise SystemExit(f"no CSV for {prefix} in {DATA}; run the experiment first")
    print(f"  reading {os.path.basename(matches[-1])}")
    with open(matches[-1]) as f:
        return list(csv.DictReader(f))


def _lxs(rows) -> list[int]:
    return sorted({int(r["lx"]) for r in rows})


def main() -> None:
    os.makedirs(DEST, exist_ok=True)

    exp01 = _load("exp01_materialized_column_rmps.py")
    rows1 = _latest("exp1-injection")
    a1 = argparse.Namespace(lxs=_lxs(rows1), osi_ell=8)
    exp01.make_headline_plot(rows1, os.path.join(DEST, "exp01-injection.png"), a1)
    exp01.make_detail_plot(rows1, os.path.join(DEST, "exp01-materialized-detail.png"), a1)

    exp02 = _load("exp02_global_vs_local_moses.py")
    rows2 = _latest("exp2-global-vs-local")
    a2 = argparse.Namespace(lxs=_lxs(rows2))
    exp02.make_plot(rows2, os.path.join(DEST, "exp02-global-vs-local.png"), a2)

    names = ["exp01-injection.png", "exp01-materialized-detail.png", "exp02-global-vs-local.png"]
    def _state_order(rs):
        return sorted({r["state"] for r in rs}, key=lambda s: (s != "random", s))

    for mod, prefix, png, build_args in (
        ("exp04_real_column_spectrum.py", "exp4-real-spectrum", "exp04-real-column-spectrum.png",
         lambda rs: argparse.Namespace(states=_state_order(rs), eta=4)),
        ("exp05_real_global_vs_local.py", "exp5-real-gvl", "exp05-real-global-vs-local.png",
         lambda rs: argparse.Namespace(states=_state_order(rs))),
        ("exp06_structured_qr_feasibility.py", "exp6-structured-feasibility", "exp06-structured-qr-feasibility.png",
         lambda rs: argparse.Namespace(lxs=_lxs(rs), states=_state_order(rs))),
        ("exp07_cost_comparison.py", "exp7-cost", "exp07-cost-comparison.png",
         lambda rs: argparse.Namespace(states=_state_order(rs))),
    ):
        try:  # each needs quimb to have produced a CSV; skip cleanly if it never ran
            m = _load(mod)
            rs = [r for r in _latest(prefix) if any(r.values())]
            m.make_plot(rs, os.path.join(DEST, png), build_args(rs))
            names.append(png)
        except SystemExit:
            print(f"  (skipping {png} -- no {prefix} CSV found)")

    for name in names:
        print(f"  wrote reports/figures/column_sketch/{name}")


if __name__ == "__main__":
    main()
