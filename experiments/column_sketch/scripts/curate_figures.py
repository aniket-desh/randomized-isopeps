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
    try:  # exp04 needs quimb to have produced a CSV; skip cleanly if it never ran
        exp04 = _load("exp04_real_column_spectrum.py")
        rows4 = [r for r in _latest("exp4-real-spectrum") if r.get("col_spectrum")]
        states = sorted({r["state"] for r in rows4}, key=lambda s: (s != "random", s))
        a4 = argparse.Namespace(states=states, eta=4)
        exp04.make_plot(rows4, os.path.join(DEST, "exp04-real-column-spectrum.png"), a4)
        names.append("exp04-real-column-spectrum.png")
    except SystemExit:
        print("  (skipping exp04 figure -- no exp4 CSV found)")

    for name in names:
        print(f"  wrote reports/figures/column_sketch/{name}")


if __name__ == "__main__":
    main()
