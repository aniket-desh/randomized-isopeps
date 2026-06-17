#!/usr/bin/env python3
"""Two advisor-facing figures for the randomized-disentangler update.

Fig 1 (disentangler): the second-SVD truncation tail c_k(Q) collapses when the
disentangler is applied, and a *sketched* disentangler search (gaussian /
khatri-rao / sparsestack inner SVD, validated) reaches the same tail as the exact
alternating minimisation -- while the gauge stays exactly unitary (||Q^H Q - I||
~ 1e-14).

Fig 2 (second SVD): after an exact disentangler, a structured randomized final
SVD2 (gaussian / sparsestack / khatri-rao over the product index chi*n3) matches
the deterministic SVD2 reconstruction error; the excess is at machine precision.

Run from the repo root:  python3 synthetic/make_advisor_figures.py
Writes synthetic/figures/update-disentangler.png and update-svd2.png.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.disentangler import sketch_for_second_svd
from rand_isopeps.local_methods import run_disentangle_pipeline
from rand_isopeps.plotting import Panel, Series, write_line_panels
from rand_isopeps.synthetic_tensors import make_disentanglable_local
from rand_isopeps.tn_shapes import MosesDims

FIG_DIR = Path(__file__).resolve().parents[0] / "figures"
CHI, D, P = 4, 2, 1
ETAS = [4, 6, 8, 10, 12]
TRIALS = 10
OVERSAMPLE = 8
SEED = 4242

# (label, color, marker) for the disentangler-search methods
DIS = [
    ("no disentangler", "#999999", "o"),
    ("exact alt-min", "#000000", "s"),
    ("gaussian sketch", "#0072b2", "^"),
    ("khatri-rao sketch", "#cc79a7", "D"),
    ("sparsestack sketch", "#009e73", "v"),
]
# (final SVD2, color, marker)
SVD2 = [
    ("exact", "#000000", "o"),
    ("gaussian", "#0072b2", "s"),
    ("sparsestack", "#009e73", "^"),
    ("khatri_rao", "#cc79a7", "D"),
]


def _band(rows, key, group_order):
    return median_band(rows, group_key="g", x_key="eta", value_key=key, group_order=group_order)


def collect_disentangler():
    rows = []
    for eta in ETAS:
        dims = MosesDims(chi=CHI, eta=eta, d=D, p=P)
        for trial in range(TRIALS):
            b = make_disentanglable_local(dims, np.random.default_rng(SEED + 100 * eta + trial),
                                          entangle=1.0, noise=1e-6)
            rng = np.random.default_rng(SEED + 7 * trial + eta)
            for label, inner in (("exact alt-min", "exact"), ("gaussian sketch", "gaussian"),
                                 ("khatri-rao sketch", "khatri_rao"), ("sparsestack sketch", "sparsestack")):
                r = run_disentangle_pipeline(b, dims, inner=sketch_for_second_svd(inner, dims),
                                             final="exact", oversample=OVERSAMPLE, rng=rng)
                m = r.metrics(b)
                rows.append({"g": label, "eta": eta, "tail": m["tail_used"], "udef": m["unitary_defect"]})
                if label == "exact alt-min":  # the Q=I baseline tail comes for free
                    rows.append({"g": "no disentangler", "eta": eta,
                                 "tail": r.dis.tail_initial, "udef": 0.0})
    return rows


def collect_svd2():
    rows = []
    for eta in ETAS:
        dims = MosesDims(chi=CHI, eta=eta, d=D, p=P)
        for trial in range(TRIALS):
            b = make_disentanglable_local(dims, np.random.default_rng(SEED + 100 * eta + trial),
                                          entangle=1.0, noise=1e-6)
            exact_err = float("nan")
            local = []
            for final, *_ in SVD2:
                r = run_disentangle_pipeline(b, dims, inner="exact", final=sketch_for_second_svd(final, dims),
                                             oversample=OVERSAMPLE, rng=np.random.default_rng(SEED + trial))
                err = float(r.metrics(b)["rel_error"])
                if final == "exact":
                    exact_err = err
                local.append({"g": final, "eta": eta, "rel_error": err})
            for row in local:
                row["excess"] = row["rel_error"] - exact_err
            rows.extend(local)
    return rows


def _series(bands, styles):
    out = []
    for label, color, marker in styles:
        if label not in bands:
            continue
        xs, med, lo, hi = bands[label]
        out.append(Series(label=label, x=[float(v) for v in xs], y=med, ylow=lo, yhigh=hi,
                          color=color, marker=marker))
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    dis = collect_disentangler()
    order = [s[0] for s in DIS]
    fig1 = [
        Panel(f"second-SVD tail c_k(Q)  (chi={CHI}, d={D})", "eta", "discarded energy (>eta)", "log",
              _series(_band(dis, "tail", order), DIS)),
        Panel("gauge unitarity", "eta", "||Q^H Q - I||_F", "log",
              _series(_band(dis, "udef", order[1:]), DIS[1:])),  # drop Q=I baseline (defect 0)
    ]
    write_line_panels(FIG_DIR / "update-disentangler.png", fig1, width=900, height=400)
    print("wrote", FIG_DIR / "update-disentangler.png")

    svd2 = collect_svd2()
    order2 = [s[0] for s in SVD2]
    fig2 = [
        Panel(f"reconstruction error  (chi={CHI}, d={D})", "eta", "relative error ||B-B_hat||/||B||", "log",
              _series(_band(svd2, "rel_error", order2), SVD2)),
        Panel("excess over exact SVD2", "eta", "rel error (rand - exact)", "log",
              _series(_band(svd2, "excess", order2[1:]), SVD2[1:])),  # exact is the zero baseline
    ]
    write_line_panels(FIG_DIR / "update-svd2.png", fig2, width=900, height=400)
    print("wrote", FIG_DIR / "update-svd2.png")


if __name__ == "__main__":
    main()
