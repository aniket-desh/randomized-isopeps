#!/usr/bin/env python3
"""Sweep the global sketch's vertical bond eta_q = 4..10 and watch the projection error.

The companion to exp10's fixed-bond comparison: instead of three methods at one bond, this is
ONE method (plain global rMPS sketch) across a range of vertical bonds, so you can see the
accuracy-vs-bond curve directly -- how fast eps_proj = ||(I-QQ*)C||/||C|| falls as you spend
more vertical bond, and where it crosses the local Moses target. The disentangler's promise is
to reach the local line WITHOUT climbing this curve (holding eta=4).

Memory: this is a small-Lx dense validation by construction. The dense column is 2^Lx x 8^Lx
(<= 16 MB at Lx=5), state prep peaks ~3 GB at eta=4, single-process -- it will NOT crash a Mac.
A hard guard refuses any (Lx) whose dense column would exceed --max-dense-gb.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
from rand_isopeps.column.operator import dense_column_nbytes
from rand_isopeps.column.structured_qr import structured_column_qr
from rand_isopeps.io_utils import output_paths, timestamp_slug
from rand_isopeps.plotting import Panel, Series, state_style, write_line_panels

SUITE = "column_sketch"


def _prepare(lx, ly, g, eta, seed, chi, bond, phys, cutoff, ndis, taus):
    from rand_isopeps.real_isotns.moses_move import random_isotns
    from rand_isopeps.real_isotns.tebd2 import imaginary_time, tfi_ham
    psi = random_isotns(lx, ly, bond=bond, phys=phys, chi=chi, eta=eta, cutoff=cutoff,
                        Ndis=ndis, seed=seed)
    psi, _ = imaginary_time(psi, tfi_ham(lx, ly, g=g), taus=taus, steps=None,
                            chi=chi, eta=eta, cutoff=cutoff, Ndis=ndis)
    return psi


def _local_eps(psi, jc, split, eta, chi, cutoff, ndis):
    from rand_isopeps.real_isotns.instrument import MosesStats
    from rand_isopeps.real_isotns.moses_move import moses_move
    st = MosesStats()
    errs = moses_move(psi.copy(), jc, chi, eta, cutoff, ndis, orientation="col",
                      sweep="up", split=split, renorm=True, stats=st)
    return float(np.max(np.abs(errs)))


def run(args):
    taus = [(t, args.tau_steps) for t in (args.taus or [0.3, 0.1, 0.03])]
    # per (state, lx): eps_proj at each eta_q, and the local target
    eps_by = defaultdict(lambda: defaultdict(list))     # (lx) -> eta_q -> [eps...]
    local_by = defaultdict(list)                         # (lx) -> [local_eps...]
    for lx in args.lxs:
        peak_gb = 3.0 * dense_column_nbytes(args.phys ** lx, args.chi ** lx) / (1024 ** 3)
        if peak_gb > args.max_dense_gb:
            print(f"SKIP Lx={lx}: dense column peak ~{peak_gb:.1f} GB > --max-dense-gb={args.max_dense_gb}")
            continue
        for g in args.gs:
            for seed in range(args.seeds):
                psi = _prepare(lx, args.ly, g, args.eta, seed, args.chi, args.bond,
                               args.phys, args.cutoff, args.ndis, taus)
                jc, split = find_center_column(psi)
                op = from_quimb_column(psi, jc, split=split)
                c = op.materialize()
                n_out = op.n_out
                local_by[lx].append(_local_eps(psi, jc, split, args.eta, args.chi,
                                                args.cutoff, args.ndis))
                for eq in args.eta_qs:
                    if eq > n_out:
                        continue
                    rng = np.random.default_rng(args.seed + 17 * seed + 101 * eq)
                    res = structured_column_qr(op, ell=max(args.ell, eq + 4), eta_q=eq,
                                               chi_sk=args.chi_sk, n_power=1, rng=rng, reference=c)
                    eps_by[lx][eq].append(res.eps_proj)
                print(f"  done Lx={lx} g={g} seed={seed} (n_out={n_out})")

    stamp = timestamp_slug()
    _, fig_path = output_paths(SUITE, f"exp10-etaq-sweep-{stamp}")
    make_plot(eps_by, local_by, args, fig_path)
    # also drop it straight into the tracked reports dir
    dest = str(Path(__file__).resolve().parents[3] / "reports" / "figures" / SUITE /
               "exp10-etaq-sweep.pdf")
    make_plot(eps_by, local_by, args, dest)
    return fig_path, dest


def make_plot(eps_by, local_by, args, fig_path):
    lxs = sorted(eps_by)
    series_eps, series_loc = [], []
    for lx in lxs:
        label, color, marker, _ = state_style(f"tfim@{3.04 if lx % 2 else 3.5}")  # per-Lx colour
        color = ["#0072b2", "#009e73", "#d55e00", "#7b3294"][lxs.index(lx) % 4]
        xs = [float(e) for e in args.eta_qs if eps_by[lx].get(e)]
        ys = [float(np.median(eps_by[lx][e])) for e in args.eta_qs if eps_by[lx].get(e)]
        series_eps.append(Series(label=rf"global sketch, $L_x={lx}$", x=xs, y=ys,
                                 color=color, marker="o", linestyle="-"))
        if local_by[lx]:
            loc = float(np.median(local_by[lx]))
            series_loc.append(Series(label=rf"local target, $L_x={lx}$", x=[float(args.eta_qs[0]),
                              float(args.eta_qs[-1])], y=[loc, loc], color=color, marker="",
                              linestyle="--"))
    pa = Panel(r"Global sketch: projection error vs vertical bond $\eta_q$",
               r"vertical bond $\eta_q$", r"$\|(I-QQ^*)C\|_F/\|C\|_F$", "log",
               series_eps + series_loc)
    # second panel: same curves normalized to the local target (>1 above local, <1 below)
    series_rel = []
    for lx in lxs:
        if not local_by[lx]:
            continue
        loc = float(np.median(local_by[lx]))
        color = ["#0072b2", "#009e73", "#d55e00", "#7b3294"][lxs.index(lx) % 4]
        xs = [float(e) for e in args.eta_qs if eps_by[lx].get(e)]
        ys = [float(np.median(eps_by[lx][e])) / loc for e in args.eta_qs if eps_by[lx].get(e)]
        series_rel.append(Series(label=rf"$L_x={lx}$", x=xs, y=ys, color=color, marker="o",
                                 linestyle="-"))
    pb = Panel(r"Relative to local (crosses 1 where the bond matches local)",
               r"vertical bond $\eta_q$", r"eps$_{\mathrm{global}}$ / eps$_{\mathrm{local}}$",
               "log", series_rel, hlines=[1.0])
    write_line_panels(fig_path, [pa, pb], width=980, height=380)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lxs", type=int, nargs="+", default=[4, 5])
    p.add_argument("--eta-qs", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--gs", type=float, nargs="+", default=[3.5, 3.04])
    p.add_argument("--ly", type=int, default=4)
    p.add_argument("--bond", type=int, default=3)
    p.add_argument("--phys", type=int, default=2)
    p.add_argument("--chi", type=int, default=8)
    p.add_argument("--eta", type=int, default=4, help="prep/local vertical bond (the target)")
    p.add_argument("--chi-sk", type=int, default=8)
    p.add_argument("--ell", type=int, default=16)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--cutoff", type=float, default=1e-10)
    p.add_argument("--ndis", type=int, default=10)
    p.add_argument("--taus", type=float, nargs="+", default=None)
    p.add_argument("--tau-steps", type=int, default=5)
    p.add_argument("--max-dense-gb", type=float, default=2.0, help="refuse any Lx above this dense peak")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    out, dest = run(a)
    print(f"wrote {out}")
    print(f"wrote {dest}")
