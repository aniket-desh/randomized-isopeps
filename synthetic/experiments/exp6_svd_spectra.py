#!/usr/bin/env python3
"""SVD spectrum diagnostic for the local two-SVD decomposition.

Explains the exp1 result. For each synthetic ensemble it plots the singular
values of the first-SVD matrix ``B_{n1,(n2 n3)}`` and of the second-SVD matrix
``A(vtilde)``, normalized by the leading value, with dashed markers at the
truncation ranks ``k1 = chi*eta`` and ``k2 = eta``.

Expectation (advisor): the first SVD keeps ``k1 = chi*eta`` of ``n1 = chi*eta*d``
rows -- for spin systems ``d=2`` that is half the row space, so its spectrum is
*not* steeply truncatable and randomizing it buys little. The second SVD keeps
``k2 = eta`` of a much larger matrix, so a fast-decaying spectrum there explains
why randomizing the second SVD is safe. Stress ensembles (gaussian, power-law,
exponential) show how far the favorable second-SVD spectrum persists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.linalg as la

from rand_isopeps.disentangler import cut_forward
from rand_isopeps.io_utils import output_paths, timestamp_slug, write_csv
from rand_isopeps.plotting import Panel, Series, write_line_panels
from rand_isopeps.synthetic_tensors import make_local_tensor
from rand_isopeps.tn_shapes import MosesDims

# ensemble -> (color, marker)
_STYLE = {
    "ring": ("#0072b2", "o", "-"),
    "noisy_ring": ("#009e73", "^", "--"),
    "gaussian": ("#d55e00", "s", "-."),
    "powerlaw": ("#cc79a7", "D", ":"),
    "expdecay": ("#e69f00", "v", (0, (3, 1, 1, 1))),
}


def _spectra(dims: MosesDims, ensemble: str, args: argparse.Namespace, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized singular values of the first- and second-SVD matrices."""
    rng = np.random.default_rng(args.seed + 1000 * offset)
    b = make_local_tensor(
        dims, rng, ensemble=ensemble, noise=args.noise, decay=args.decay, spectrum_param=args.spectrum_param
    )
    first_matrix = b.reshape(dims.n1, dims.n2 * dims.n3)
    s1 = la.svdvals(first_matrix)
    # form the residual after the first truncation, then the second-SVD matrix
    u1, sv1, vh1 = la.svd(first_matrix, full_matrices=False, check_finite=False)
    vtilde = sv1[: dims.k1, None] * vh1[: dims.k1, :]
    s2 = la.svdvals(cut_forward(vtilde, dims))
    return s1 / s1[0], s2 / s2[0]


def _series(name: str, spectrum: np.ndarray, offset: int) -> Series:
    color, marker, linestyle = _STYLE.get(name, ("#999999", "o", "-"))
    idx = np.arange(1, spectrum.shape[0] + 1)
    # sparse, staggered markers (~10 per curve) so dense overlapping spectra stay
    # legible instead of merging into blobs; distinct line styles disambiguate the
    # flat noise-floor tails where every ensemble overlaps.
    step = max(1, spectrum.shape[0] // 10)
    return Series(
        label=name.replace("_", " "),
        x=[float(i) for i in idx], y=[float(v) for v in spectrum],
        color=color, marker=marker, linestyle=linestyle,
        markevery=(offset % step, step),
    )


def run(args: argparse.Namespace) -> tuple[str, str]:
    dims = MosesDims(chi=args.chi, eta=args.eta, d=args.d, p=args.p)
    rows: list[dict[str, object]] = []
    first_series, second_series = [], []
    for offset, ensemble in enumerate(args.ensembles):
        s1, s2 = _spectra(dims, ensemble, args, offset)
        first_series.append(_series(ensemble, s1, offset))
        second_series.append(_series(ensemble, s2, offset))
        for cut, spectrum in (("first", s1), ("second", s2)):
            for i, val in enumerate(spectrum, start=1):
                rows.append({**dims.as_dict(), "ensemble": ensemble, "cut": cut, "index": i, "sigma_norm": float(val)})

    panels = [
        Panel(
            f"first SVD spectrum (keep k1={dims.k1} of n1={dims.n1})",
            "singular index i", "sigma_i / sigma_1", "log", first_series, vlines=[float(dims.k1)],
        ),
        Panel(
            f"second SVD spectrum (keep k2={dims.k2} of {dims.eta*dims.n2}x{dims.chi*dims.n3})",
            "singular index i", "sigma_i / sigma_1", "log", second_series, vlines=[float(dims.k2)],
        ),
    ]
    stamp = timestamp_slug()
    csv_path, fig_path = output_paths(__file__, f"exp6-spectra-{stamp}")
    write_csv(csv_path, rows)
    write_line_panels(fig_path, panels, width=900, height=440)
    return csv_path, fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--eta", type=int, default=8)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument(
        "--ensembles", nargs="+",
        default=["noisy_ring", "gaussian", "powerlaw", "expdecay"],
    )
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.92)
    parser.add_argument("--spectrum-param", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6789)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.ensembles = ["noisy_ring", "gaussian"]
        args.eta = min(args.eta, 6)
    return args


if __name__ == "__main__":
    data, fig = run(parse_args())
    print(f"wrote {data}")
    print(f"wrote {fig}")
