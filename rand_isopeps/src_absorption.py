"""Successive Randomized Compression (SRC) adapter for R-column absorption.

Wraps the vendored ``randommpomps`` SRC routines (Camaño-Epperly-Tropp,
arXiv:2504.06475) so they can be used on the same synthetic MPO/MPS that
``mpo_mps_absorb`` builds. Unlike the zip-up / randomized-local baselines, SRC
never forms the inflated product MPS (bond ``D_mpo * D_mps``): it sketches the
product ``H psi`` site-by-site and compresses on the fly. That is exactly the
"structured from the beginning" R-column absorption the project is aiming for.

This module is *implemented and pluggable* but not wired into any default
experiment yet (the absorption experiment's default methods are unchanged).

Index conventions. Our local absorption tensors (see ``mpo_mps_absorb``) use::

    MPS site : (left_bond, phys, right_bond)          ends have bond 1
    MPO site : (left_bond, phys_out, phys_in, right_bond)

The vendored code uses::

    MPS : (phys, right) | (left, phys, right) | (left, phys)
    MPO : (phys_out, right, phys_in) | (left, phys_out, right, phys_in) | (left, phys_out, phys_in)

so the converters below squeeze the size-1 end bonds and reorder the MPO axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .randommpomps import MPO, MPS, Cutoff, FixedDimension, random_contraction, random_contraction_inc


# --- convert our (mpo_mps_absorb) tensors to the vendored convention ---

def our_mps_to_camano(mps: list[np.ndarray]) -> MPS:
    n = len(mps)
    out: list[np.ndarray] = []
    for i, a in enumerate(mps):  # a: (left, phys, right)
        if i == 0:
            out.append(a[0])              # (phys, right)
        elif i == n - 1:
            out.append(a[:, :, 0])        # (left, phys)
        else:
            out.append(a)                 # (left, phys, right)
    return MPS(out)


def our_mpo_to_camano(mpo: list[np.ndarray]) -> MPO:
    n = len(mpo)
    out: list[np.ndarray] = []
    for i, w in enumerate(mpo):  # w: (left, out, in, right)
        if i == 0:
            out.append(w[0].transpose(0, 2, 1))    # (out, right, in)
        elif i == n - 1:
            out.append(w[:, :, :, 0])              # (left, out, in)
        else:
            out.append(w.transpose(0, 1, 3, 2))    # (left, out, right, in)
    return MPO(out)


def camano_mps_to_vector(mps: MPS) -> np.ndarray:
    """Dense state vector with physical ordering (d0, d1, ..., d_{n-1})."""
    state = mps[0]  # (d0, R0)
    for site in mps[1:-1]:
        state = np.tensordot(state, site, axes=(-1, 0))  # (..., d_i, R_i)
        state = state.reshape(-1, state.shape[-1])
    state = np.tensordot(state, mps[-1], axes=(-1, 0))   # (..., d_{n-1})
    return state.reshape(-1)


@dataclass
class SRCResult:
    vector: np.ndarray
    runtime_s: float
    final_max_bond: int
    method: str


def _max_bond(mps: MPS) -> int:
    n = len(mps)
    best = 0
    for i in range(n):
        t = mps[i]
        if i == 0:
            best = max(best, t.shape[1])
        elif i == n - 1:
            best = max(best, t.shape[0])
        else:
            best = max(best, t.shape[0], t.shape[2])
    return int(best)


def src_absorb(
    mpo: list[np.ndarray],
    mps: list[np.ndarray],
    target_bond: int | None = None,
    cutoff: float | None = None,
    sketchincrement: int = 1,
    incremental: bool = False,
    seed: int | None = None,
) -> SRCResult:
    """Compress ``H psi`` with SRC, returning the dense result vector.

    Exactly one of ``target_bond`` (fixed output bond) or ``cutoff`` (adaptive
    truncation) should be given; ``target_bond`` takes precedence. ``incremental``
    selects ``random_contraction_inc`` (needs the optional C++ incremental-QR
    build for stability; the default ``random_contraction`` is pure numpy).
    SRC draws Gaussian sketches from the global numpy RNG, so the state is seeded
    and restored here for reproducibility without disturbing the caller's RNG.
    """
    h = our_mpo_to_camano(mpo)
    psi = our_mps_to_camano(mps)
    if target_bond is not None:
        stop = FixedDimension(int(target_bond))
    elif cutoff is not None:
        stop = Cutoff(cutoff)
    else:
        raise ValueError("provide target_bond or cutoff")

    contract = random_contraction_inc if incremental else random_contraction

    state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
        t0 = perf_counter()
        out = contract(h, psi, stop=stop, sketchincrement=sketchincrement)
        runtime = perf_counter() - t0
    finally:
        np.random.set_state(state)

    return SRCResult(
        vector=camano_mps_to_vector(out),
        runtime_s=runtime,
        final_max_bond=_max_bond(out),
        method="src_inc" if incremental else "src",
    )
