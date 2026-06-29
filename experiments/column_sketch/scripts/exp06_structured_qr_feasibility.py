#!/usr/bin/env python3
"""Stage 1, experiment 2: the feasibility gate -- does range(C*Omega) round to a
LOW-BOND isometric isoPEPS column?

The reframed thesis rides on one clause: a global range finder ties the Moses move's
accuracy at lower cost *provided the sampled range rounds to a low-bond isometric
column*. Range capture alone only gives a dense matrix isometry; the arrow-compatible
structured column QR (:mod:`rand_isopeps.column.structured_qr`) turns the sampled range
``Y = C Omega`` into a genuinely isometric isoPEPS column by a TT-SVD sweep that
truncates each vertical bond to ``eta_q``. The correct theorem (Q4 of the theory review)
is **range-capture + small TT-rounding tail => valid low-bond isometric column with
controlled error** -- so we sweep ``eta_q`` and watch whether a *small* one keeps the
projection error ``eps_proj`` low.

For the real extracted centre column ``C_j`` (``from_quimb_column``), at a fixed sketch
width ``ell``, sweep the vertical-bond cap ``eta_q`` and record:

* ``eps_proj``    = ``||(I - Q_iso Q_iso^*) C||/||C||`` -- did the rounded isometric column capture C?
* ``tau_round``   -- the TT-rounding tail of ``Y`` (relative);
* ``delta_local`` / ``delta_global`` -- per-site / global isometry defect (the arrows -- should be ~eps);
* ``final_bond``  -- the carried residual (R-column) bond.

Prediction: for a physical (TFIM, area-law) column the sampled range rounds cheaply
(``eps_proj`` flattens at small ``eta_q``); a random column needs a large ``eta_q`` (or
never gets small, being high flat-rank). The isometry defect stays ~1e-15 throughout --
the structured sweep preserves the arrows by construction, the whole point of the reframe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rand_isopeps.aggregate import median_band
from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
from rand_isopeps.column.structured_qr import structured_column_qr
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, state_style, write_line_panels

SUITE = "column_sketch"


def _prepare_state(args, kind, lx, seed):
    from rand_isopeps.real_isotns.moses_move import random_isotns
    psi = random_isotns(lx, args.ly, bond=args.bond, phys=args.phys, chi=args.chi,
                        eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis, seed=seed)
    if kind.startswith("tfim@"):
        from rand_isopeps.real_isotns.tebd2 import imaginary_time, tfi_ham
        g = float(kind.split("@")[1])
        psi, _ = imaginary_time(psi, tfi_ham(lx, args.ly, g=g), taus=args.taus, steps=None,
                                chi=args.chi, eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis)
    return psi


def _run_trial(task):
    args, kind, lx, seed = task
    with with_blas_threads(args.blas_threads):
        psi = _prepare_state(args, kind, lx, seed)
        jc, split = find_center_column(psi)
        op = from_quimb_column(psi, jc, split=split)
        c = op.materialize()
        rng = np.random.default_rng(args.seed + 7 * seed + 13 * lx)
        rows = []
        for eta_q in args.eta_qs:
            res = structured_column_qr(op, ell=args.ell, eta_q=eta_q, chi_sk=args.chi_sk,
                                       sketch_kind=args.sketch, n_power=args.n_power,
                                       rng=rng, reference=c)
            rows.append({"state": kind, "lx": lx, "seed": seed, "eta_q": eta_q,
                         "n_out": c.shape[0], **res.as_dict()})
        return rows


def run(args):
    workers = auto_worker_count(args.workers)
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp6-structured-feasibility-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _series(rows, value_key, lx, floor=0.0):
    """Median+IQR bands over seeds at a fixed Lx, one Series per physical state.

    The comparison axis is the state (random worst case vs TFIM ground states), so
    each curve gets its canonical ``state_style`` color/marker. ``random`` is kept
    first for a stable legend order; ``floor`` clips for safe log-axis rendering.
    """
    sub = [r for r in rows if int(r["lx"]) == lx]
    present = sorted({r["state"] for r in sub}, key=lambda s: (s != "random", s))
    bands = median_band(sub, group_key="state", x_key="eta_q", value_key=value_key, group_order=present)
    out = []
    for key, (xs, med, lo, hi) in bands.items():
        label, color, marker, ls = state_style(key)
        out.append(Series(label=label, x=[float(v) for v in xs], y=[max(m, floor) for m in med],
                          ylow=[max(v, floor) for v in lo], yhigh=[max(v, floor) for v in hi],
                          color=color, marker=marker, linestyle=ls))
    return out


def make_plot(rows, fig_path, args):
    """Two clean 2-panel figures at a fixed Lx (the largest swept); series = states.

    MAIN (``fig_path``): the feasibility claim -- range capture ``eps_proj`` and the
    TT-rounding tail ``tau_round`` vs the vertical-bond cap ``eta_q``. SECONDARY
    (``-sanity``): the arrow-isometry defect (should hug ~1e-15, well under the
    1e-12 threshold) and the carried residual R-bond. The Lx facet is collapsed to
    ``max(args.lxs)`` so the headline reads as one question per panel.
    """
    lx = max(args.lxs)

    main = [
        Panel(f"Range capture after structured QR (Lx={lx})", r"vertical bond $\eta_Q$",
              r"$\|(I-QQ^*)C\|_F/\|C\|_F$", "log", _series(rows, "eps_proj", lx, floor=1e-16)),
        Panel("TT-rounding tail", r"vertical bond $\eta_Q$",
              r"$\tau_{\mathrm{round}}$", "log", _series(rows, "tau_round", lx, floor=1e-16)),
    ]
    write_line_panels(fig_path, main, width=980, height=320)

    sec_path = str(Path(fig_path).with_name(Path(fig_path).stem + "-sanity" + ".pdf"))
    sanity = [
        Panel("Arrow isometry defect", r"vertical bond $\eta_Q$",
              r"$\max_i\|q_i^*q_i-I\|$", "log", _series(rows, "delta_local", lx, floor=1e-17),
              hlines=[1e-12]),
        Panel("Residual bond", r"vertical bond $\eta_Q$",
              "final R-bond", "linear", _series(rows, "final_bond", lx)),
    ]
    write_line_panels(sec_path, sanity, width=980, height=320)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--eta-qs", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--ell", type=int, default=12, help="sketch width (>= column rank to capture)")
    parser.add_argument("--chi-sk", type=int, default=8)
    parser.add_argument("--sketch", choices=["gaussian", "rmps", "kron"], default="rmps")
    parser.add_argument("--n-power", type=int, default=1)
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    raw = args.taus if args.taus else [0.3, 0.1, 0.03]
    args.taus = [(t, args.tau_steps) for t in raw]
    if args.quick:
        args.lxs = [3, 4]
        args.states = ["random", "tfim@3.5"]
        args.eta_qs = [1, 2, 4]
        args.seeds = 2
        args.taus = [(0.3, 3), (0.1, 3)]
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
