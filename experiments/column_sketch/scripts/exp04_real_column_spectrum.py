#!/usr/bin/env python3
"""Stage 0 (make-or-break): does a REAL isoTNS column have a decaying spectrum?

The global rMPS column sketch can only beat the local Moses move if the whole-column
map ``C_j`` has a low *flat* rank -- a sharply decaying singular spectrum. The
synthetic study showed random columns are flat (no win); the question is whether a
physical (area-law) column decays. This experiment measures it directly on real
quimb isoTNS states, contrasting:

* ``random``  -- ``random_isotns`` (a random PEPS Moses-swept to canonical form): the
  flat, hard, worst case (the synthetic analogue).
* ``tfim@g``  -- a 2D transverse-field-Ising state prepared by imaginary-time TEBD2
  (``tfi_ham``/``imaginary_time``), the rMPS paper's physical model, at field ``g``
  (``g >~ 3.04`` is the 2D-TFIM critical point; smaller ``g`` is more ordered, larger
  is a deeper paramagnet -- lower entanglement).

Two spectra are measured per state (after canonicalising the centre to column 0):

1. **Column-level** ``C_0`` (the object the global sketch QRs): extracted by
   :func:`rand_isopeps.column.from_quimb.from_quimb_column`, its dense singular
   spectrum.
2. **Move-anchored** ``svd2`` (what the local move actually truncates to ``eta``):
   captured gauge-free via ``MosesStats(keep_arrays=("svd2",))`` from a real move.

GATE: if the TFIM spectra decay appreciably faster than the random one -- and the
gap **survives criticality and grows with Lx** -- green-light the sketch direction.
If the physical column is no better than flat random, the thesis is dead. This is a
mathematical-validation diagnostic, not a speedup claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rand_isopeps.column.from_quimb import from_quimb_column
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.parallel import auto_worker_count, flatten, run_parallel, with_blas_threads
from rand_isopeps.plotting import Panel, Series, state_style, write_line_panels

SUITE = "column_sketch"


def _spectrum_metrics(s, eta):
    s = np.asarray(s, dtype=float)
    s = s[s > 0]
    s = s / s[0]
    s2 = s ** 2
    total = float(np.sum(s2))
    eff = int((np.cumsum(s2) / total < 0.99).sum() + 1)
    top_eta = float(np.sum(s2[: eta]) / total)
    decades = float(np.log10(s[0] / max(s[-1], 1e-30)))
    return s, eff, top_eta, decades


def _prepare_state(args, kind, lx, seed):
    """Return a canonical isoTNS (centre at column 0) for kind in {'random', 'tfim@g'}."""
    from rand_isopeps.real_isotns.moses_move import random_isotns
    psi = random_isotns(lx, args.ly, bond=args.bond, phys=args.phys, chi=args.chi,
                        eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis, seed=seed)
    if kind.startswith("tfim@"):
        from rand_isopeps.real_isotns.tebd2 import imaginary_time, tfi_ham
        g = float(kind.split("@")[1])
        psi, _ = imaginary_time(psi, tfi_ham(lx, args.ly, g=g), taus=args.taus, steps=None,
                                chi=args.chi, eta=args.eta, cutoff=args.cutoff, Ndis=args.ndis)
    # NB: do NOT canonize_column(0) -- it makes an off-centre column isometric (flat) rather
    # than relocating the 2D orthogonality centre. The centre is detected at extraction time.
    return psi


def _run_trial(task):
    args, kind, lx, seed = task
    with with_blas_threads(args.blas_threads):
        from rand_isopeps.column.from_quimb import find_center_column
        from rand_isopeps.real_isotns.instrument import MosesStats
        from rand_isopeps.real_isotns.moses_move import moses_move
        psi = _prepare_state(args, kind, lx, seed)

        # extract the ACTUAL orthogonality-centre column (boundary 0 or Ly-1) -- not column 0,
        # which may be an off-centre isometry (flat) depending on the last sweep direction.
        jc, split = find_center_column(psi)

        # column-level C_jc spectrum (the global-sketch object)
        op = from_quimb_column(psi, jc, split=split)
        col_s = np.linalg.svd(op.materialize(), compute_uv=False)
        col_spec, col_eff, col_top, col_dec = _spectrum_metrics(col_s, args.eta)

        # move-anchored svd2 spectrum (what the local move truncates) -- on a COPY (the move mutates)
        st = MosesStats(keep_arrays=("svd2",))
        moses_move(psi.copy(), jc, args.chi, args.eta, args.cutoff, args.ndis,
                   orientation="col", sweep="up", split=split, renorm=True, stats=st)
        svd2_mats = [arr for stg, arr in st.arrays if stg == "svd2"]
        if svd2_mats:
            big = svd2_mats[int(np.argmax([m.shape[0] * m.shape[1] for m in svd2_mats]))]
            sv2_spec, sv2_eff, sv2_top, sv2_dec = _spectrum_metrics(
                np.linalg.svd(big, compute_uv=False), args.eta)
        else:
            sv2_spec, sv2_eff, sv2_top, sv2_dec = np.array([1.0]), 1, 1.0, 0.0

        return [{
            "state": kind, "lx": lx, "ly": args.ly, "seed": seed,
            "center_j": jc, "split": split,
            "n_in": op.n_in, "n_out": op.n_out, "mpo_bond": op.mpo_bond,
            "col_eff_rank99": col_eff, "col_top_eta_frac": col_top, "col_decades": col_dec,
            "svd2_eff_rank99": sv2_eff, "svd2_top_eta_frac": sv2_top, "svd2_decades": sv2_dec,
            "col_spectrum": ";".join(f"{v:.6g}" for v in col_spec[:32]),
            "svd2_spectrum": ";".join(f"{v:.6g}" for v in sv2_spec[:32]),
        }]


def run(args):
    workers = auto_worker_count(args.workers)
    tasks = [(args, kind, lx, seed)
             for kind in args.states for lx in args.lxs for seed in range(args.seeds)]
    rows = flatten(run_parallel(_run_trial, tasks, workers))

    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(SUITE, f"exp4-real-spectrum-{stamp}")
    write_csv(csv_path, rows)
    make_plot(rows, fig_path, args)
    return csv_path, fig_path


def _median_vs_lx(rows, state, key, lxs):
    ys = []
    for lx in lxs:
        v = [float(r[key]) for r in rows if r["state"] == state and int(r["lx"]) == lx]
        ys.append(float(np.median(v)) if v else float("nan"))
    return ys


def _rep_spectrum(rows, state, lx, field):
    """A representative (median-eff-rank) spectrum string for state at lx."""
    sub = [r for r in rows if r["state"] == state and int(r["lx"]) == lx]
    if not sub:
        return None
    ekey = "col_eff_rank99" if field == "col_spectrum" else "svd2_eff_rank99"
    effs = [float(r[ekey]) for r in sub]
    pick = sub[int(np.argsort(effs)[len(effs) // 2])]
    return [float(v) for v in pick[field].split(";") if v]


def _vs_lx_series(rows, states, key, lxs):
    """One Series per state of the per-Lx median of ``key``, styled via state_style."""
    out = []
    for s in states:
        label, color, marker, linestyle = state_style(s)
        out.append(Series(label=label, x=[float(x) for x in lxs],
                          y=_median_vs_lx(rows, s, key, lxs),
                          color=color, marker=marker, linestyle=linestyle))
    return out


def make_plot(rows, fig_path, args):
    """Two clean 2-panel figures (docs/PLOT.md section 3).

    MAIN (fig_path): the headline decay story -- a representative column
    spectrum at the largest Lx (A) and how the column's effective rank scales
    with column height (B). SECONDARY (``-diagnostics``): the truncation /
    move-anchored sanity checks kept out of the headline.
    """
    states = list(args.states)
    lxs = sorted({int(r["lx"]) for r in rows})
    max_lx = max(lxs)

    # --- MAIN: column spectrum decay + effective-rank scaling ----------------
    spectrum_series = []
    for s in states:
        spec = _rep_spectrum(rows, s, max_lx, "col_spectrum")
        if spec:
            label, color, marker, linestyle = state_style(s)
            spectrum_series.append(Series(
                label=label, x=list(range(1, len(spec) + 1)), y=spec,
                color=color, marker=marker, linestyle=linestyle))
    main_panels = [
        Panel("Column spectrum", r"singular index $i$", r"$\sigma_i/\sigma_1$",
              "log", spectrum_series),
        Panel("Column effective rank", r"column height $L_x$", "eff. rank (99% energy)",
              "linear", _vs_lx_series(rows, states, "col_eff_rank99", lxs)),
    ]
    write_line_panels(fig_path, main_panels, width=680, height=300)

    # --- SECONDARY: truncation kept-mass + move-anchored svd2 rank -----------
    sec = str(Path(fig_path).with_name(Path(fig_path).stem + "-diagnostics" + ".pdf"))
    sec_panels = [
        Panel(rf"Kept mass in top $\eta$={args.eta}", r"column height $L_x$",
              r"top-$\eta$ mass frac", "linear",
              _vs_lx_series(rows, states, "col_top_eta_frac", lxs)),
        Panel("Move svd2 effective rank", r"column height $L_x$", "svd2 eff. rank (99%)",
              "linear", _vs_lx_series(rows, states, "svd2_eff_rank99", lxs)),
    ]
    write_line_panels(sec, sec_panels, width=680, height=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["random", "tfim@3.5", "tfim@3.04"])
    parser.add_argument("--bond", type=int, default=3)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--taus", type=float, nargs="+", default=None,
                        help="flat list of imaginary-time step sizes (each run for --tau-steps)")
    parser.add_argument("--tau-steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    # imaginary-time schedule: anneal the step down
    raw = args.taus if args.taus else [0.3, 0.1, 0.03]
    args.taus = [(t, args.tau_steps) for t in raw]
    if args.quick:
        args.lxs = [3, 4]
        args.states = ["random", "tfim@3.5"]
        args.seeds = 2
        args.taus = [(0.3, 3), (0.1, 3)]
        args.workers = 1
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
