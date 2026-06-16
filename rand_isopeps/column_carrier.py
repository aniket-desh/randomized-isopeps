"""true recursive upward-carrier column Moses Move (sequential carrier).

Upgrades the exp2 product-column *surrogate* (independent locals) to a genuine
sequential carrier: the residual from each row is carried into the next row
before its decomposition, so approximation errors compound through an actual
carried tensor rather than through an artificial Kronecker product.

The column is modelled as a vertical MPS with bond ``eta`` and physical leg
``d`` per site. Sweeping bottom -> top, each Moses step performs a *two-stage*
split that mirrors the local two SVDs of ``local_ring_decomp``:

  * SVD1 -- isometry extraction: canonicalize the carried row (split the
    down-bond + physical index from the up-bond), producing the new isometric
    column tensor. ``rand_first`` randomizes this split (rho1 ~ 1, no room).
  * SVD2 -- bond compression to rank ``eta``: the residual ``Sigma V^H`` is the
    carrier passed to the next row. ``rand_second`` randomizes this step (the
    one with real room, rho2 ~ eta / bond).

The exact column (untruncated MPS) and the Moses-compressed column are both
MPS, so the final relative error is computed *exactly* by transfer-matrix
overlaps with no exponential materialization. This is the briefing's "task 5":
a faithful sequential carrier for advisor-facing column-error accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np
import scipy.linalg as la

from .randomized_svd import (
    SketchKind, isometry_defect_columns, rsvd_truncate, svd_truncate,
)

ColumnMode = Literal["det", "rand_first", "rand_second", "rand_both"]


def _mode_flags(mode: ColumnMode) -> tuple[bool, bool]:
    return {"det": (False, False), "rand_first": (True, False),
            "rand_second": (False, True), "rand_both": (True, True)}[mode]


def _trunc(a, k, randomized, rng, oversample, n_power, sketch):
    if randomized:
        return rsvd_truncate(a, k, oversample=oversample, n_power=n_power, rng=rng, sketch=sketch)
    return svd_truncate(a, k)


def _random_complex(shape, rng):
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return z / np.sqrt(2.0)


def make_column_mps(lx: int, d: int, bond: int, rng, ensemble: str = "flat", decay: float = 0.45):
    """Random vertical column MPS (boundary bonds 1, interior bond ``bond``).

    ``flat``: Gaussian tensors (flat-ish Schmidt spectra -> rank-eta truncation
    is genuinely lossy, the stress case). ``decay``: the right-bond index is
    geometrically suppressed so the bond spectrum decays and truncation to eta
    is near-lossless (the easy case).
    """
    bonds = [1] + [bond] * (lx - 1) + [1]
    tensors = []
    for i in range(lx):
        l, r = bonds[i], bonds[i + 1]
        a = _random_complex((l, d, r), rng)
        if ensemble == "decay" and r > 1:
            a = a * (decay ** np.arange(r))[None, None, :]
        tensors.append(a)
    return tensors


def mps_overlap(a, b) -> complex:
    """<a|b> for two MPS over the same physical legs (transfer-matrix contraction)."""
    e = np.ones((1, 1), dtype=complex)
    for aa, ab in zip(a, b):
        e = np.einsum("xy,xpr,yps->rs", e, aa.conj(), ab, optimize=True)
    return complex(e[0, 0])


def mps_relative_error(exact, approx) -> float:
    na = mps_overlap(exact, exact).real
    nb = mps_overlap(approx, approx).real
    ov = mps_overlap(exact, approx)
    err2 = max(na + nb - 2.0 * ov.real, 0.0)
    return float(np.sqrt(err2 / na)) if na > 0 else float(np.sqrt(err2))


@dataclass
class CarrierResult:
    mode: ColumnMode
    lx: int
    rel_error: float
    runtime_s: float
    row_errors: list[float] = field(default_factory=list)
    carrier_norms: list[float] = field(default_factory=list)
    carrier_conds: list[float] = field(default_factory=list)
    tail2: list[float] = field(default_factory=list)
    iso_defects: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        def _max(xs):
            return float(max(xs)) if xs else 0.0
        return {
            "rel_error": self.rel_error,
            "runtime_s": self.runtime_s,
            "max_row_error": _max(self.row_errors),
            "max_carrier_cond": _max(self.carrier_conds),
            "max_tail2": _max(self.tail2),
            "max_iso_defect": _max(self.iso_defects),
        }


def run_column_carrier(
    tensors: list[np.ndarray],
    eta: int,
    mode: ColumnMode = "det",
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    rng: np.random.Generator | None = None,
) -> CarrierResult:
    """Sweep the Moses carrier bottom -> top, compressing the vertical bond to ``eta``."""
    gen = np.random.default_rng() if rng is None else rng
    rand_first, rand_second = _mode_flags(mode)
    lx = len(tensors)
    carrier = np.eye(1, dtype=complex)  # bottom boundary
    approx: list[np.ndarray] = []
    res = CarrierResult(mode=mode, lx=lx, rel_error=float("nan"), runtime_s=0.0)

    t0 = perf_counter()
    for i, a in enumerate(tensors):
        _, d, r = a.shape
        theta = np.einsum("rx,xpc->rpc", carrier, a, optimize=True)  # absorb carrier from below
        rc = theta.shape[0]
        mat = theta.reshape(rc * d, r)

        if i == lx - 1:  # top site: no further bond, keep the full carried tensor
            approx.append(theta.reshape(rc, d, r))
            break

        # SVD1: isometry extraction (full rank -> lossless canonicalization)
        k1 = min(rc * d, r)
        first = _trunc(mat, k1, rand_first, gen, oversample, n_power, sketch)
        q1, w1 = first.u, first.s[:, None] * first.vh  # q1: (rc*d, k1), w1: (k1, r)

        # SVD2: compress the up-bond to rank eta -> the carrier going up
        k2 = min(eta, k1, r)
        second = _trunc(w1, k2, rand_second, gen, oversample, n_power, sketch)
        site = (q1 @ second.u).reshape(rc, d, k2)  # left-isometric column tensor
        carrier = second.s[:, None] * second.vh  # (k2, r)
        approx.append(site)

        approx_mat = (q1 @ (second.u * second.s)) @ second.vh
        res.row_errors.append(float(la.norm(mat - approx_mat) / max(la.norm(mat), 1e-300)))
        res.carrier_norms.append(float(la.norm(carrier)))
        res.carrier_conds.append(float(np.linalg.cond(carrier)))
        sv_w1 = la.svdvals(w1)
        total = float(np.sum(sv_w1 ** 2))
        res.tail2.append(float(np.sum(sv_w1[k2:] ** 2)) / max(total, 1e-300))  # discarded fraction
        res.iso_defects.append(isometry_defect_columns(site.reshape(rc * d, k2)))

    res.runtime_s = perf_counter() - t0
    res.rel_error = mps_relative_error(tensors, approx)
    return res
