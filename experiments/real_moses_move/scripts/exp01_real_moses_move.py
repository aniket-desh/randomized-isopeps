#!/usr/bin/env python3
"""randomized SVD / disentangler inside the REAL isoTNS Moses Move (quimb).

Unlike exp1-14 (synthetic numpy kernels), this runs the genuine block Moses Move
on a quimb PEPS via ``rand_isopeps.isotns`` -- the vertical carrier *and* the
sideways R-column zip-up absorption -- and swaps the **local tensor-ring SVDs**
(first SVD, disentangler search, second SVD) for our randomized range finder by
passing ``RandSVD(sketch=...)``. (The R-column zip-up absorption itself stays
quimb's deterministic compress -- not yet randomized here.) It measures the
represented-state error directly: 1 - |<psi0|psi>| after one Moses move on a
column, where psi0 is the exact original PEPS (3x3 contracts exactly).

Sweeps the vertical truncation bond eta (the lossy second cut; the source PEPS
bond exceeds it) and compares deterministic vs gaussian / sparsestack /
countsketch sketches. Headline: on the real Moses Move the randomized SVD tracks
the deterministic state error -- accuracy-neutral, as the synthetic exp1/exp12-14
predicted, now end-to-end in the real algorithm.

Needs quimb (`pip install quimb`).
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.plotting import Panel, Series, write_line_panels

SUITE = "real_moses_move"

METHODS = [
    ("det", None, "#000000", "o"),
    ("gaussian", "gaussian", "#0072b2", "s"),
    ("sparsestack", "sparsestack", "#009e73", "^"),
    ("countsketch", "countsketch", "#d55e00", "D"),
]


def _run_trial(args, eta, trial):
    import quimb.tensor as qtn  # local import so --help works without quimb
    from rand_isopeps.isotns import moses_move, RandSVD

    psi0 = qtn.PEPS.rand(args.lx, args.ly, bond_dim=args.bond, phys_dim=args.phys,
                         seed=args.seed + 100 * eta + trial)
    psi0.normalize()
    rows = []
    for name, sketch, *_ in METHODS:
        psi = psi0.copy()
        rand = None if sketch is None else RandSVD(sketch=sketch, oversample=args.oversample,
                                                   rng=np.random.default_rng(args.seed + trial))
        t = perf_counter()
        errs = moses_move(psi, j=args.ly - 1, chi=args.chi, eta=eta, cutoff=1e-12, Ndis=args.ndis,
                          sweep="up", split="left", renorm=False, rand=rand)
        runtime = perf_counter() - t
        overlap = abs((psi0.H | psi).contract()) / (psi0.norm() * psi.norm())
        rows.append({"method": name, "eta": eta, "trial": trial,
                     "state_error": float(max(1.0 - overlap, 0.0)),
                     "max_trunc_err": float(np.max(errs)), "runtime_s": runtime})
    det = next(r for r in rows if r["method"] == "det")
    for r in rows:
        r["excess_state_error"] = r["state_error"] - det["state_error"]
    return rows


def run(args):
    rows = []
    for eta in args.etas:
        for trial in range(args.trials):
            rows.extend(_run_trial(args, eta, trial))
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp15-real-moses-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path)
    return csv_path, fig_path


def _series(rows, key, order):
    bands = median_band(rows, group_key="method", x_key="eta", value_key=key, group_order=order)
    style = {m[0]: (m[2], m[3]) for m in METHODS}
    return [Series(label=m, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                   color=style[m][0], marker=style[m][1])
            for m, (xs, med, lo, hi) in bands.items()]


def make_plot(rows, fig_path):
    order = [m[0] for m in METHODS]
    panels = [
        Panel("state error after Moses move", "eta", "1 - |<psi0|psi>|", "log", _series(rows, "state_error", order)),
        Panel("excess over deterministic", "eta", "state err (rand - det)", "log",
              _series(rows, "excess_state_error", order[1:])),
        Panel("Moses-move cost", "eta", "runtime (s)", "log", _series(rows, "runtime_s", order)),
    ]
    write_line_panels(fig_path, panels, width=1180, height=420)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lx", type=int, default=3)
    p.add_argument("--ly", type=int, default=3)
    p.add_argument("--bond", type=int, default=4, help="source PEPS bond (exceeds eta -> lossy)")
    p.add_argument("--phys", type=int, default=2)
    p.add_argument("--chi", type=int, default=16, help="horizontal bond (generous: isolate the eta cut)")
    p.add_argument("--etas", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--ndis", type=int, default=20)
    p.add_argument("--oversample", type=int, default=8)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed", type=int, default=15000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.etas = [2, 4]
        args.trials = 2
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
