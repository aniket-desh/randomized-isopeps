"""Method dispatch for the disentangler ablation (experiment 5).

Compares ways to build the local isometric tensor-ring decomposition of a
local tensor ``B``, isolating the role of the disentangler before the second
SVD. All methods produce an approximation ``B_hat`` of the *same* input so the
reconstruction error, the spectrum entering the second SVD, and the runtime are
directly comparable.

Methods (cf. the experiment table):

* ``A_two_svd``            two deterministic SVDs, no disentangler (baseline).
* ``B_disentangle``        SVD1, alternating-min disentangler, deterministic SVD2.
* ``B_disentangle_riem``   as B but the disentangler is found by pymanopt (O(k1)).
* ``C_disentangle_rsvd``   disentangler, then a *randomized* final SVD2.
* ``D_sketched_disentangle`` disentangler found from a *sketched* objective, then exact SVD2.
* ``E_als_random``         ALS on the ring cores from a random start.
* ``F_als_warmstart``      ALS warm-started from the greedy SVD+disentangler cores.

The disentangler is a gauge on the shared bond ``k1``: ``Q`` multiplies the
residual ``vtilde`` and its inverse is absorbed into the first isometry
(``U1 -> U1 Q^H``), so the represented tensor before truncation is unchanged and
the first factor stays isometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la

from .als_ring import als_local_ring
from .disentangler import (
    cut_forward,
    cut_inverse,
    cut_spectrum,
    disentangle_altmin,
    disentangle_riemannian,
    renyi_half_entropy,
    tail_energy,
)
from .randomized_svd import (
    SketchKind,
    isometry_defect_columns,
    isometry_defect_rows,
    relative_frobenius_error,
    rsvd_truncate,
    svd_truncate,
)
from .tn_shapes import MosesDims

METHODS = (
    "A_two_svd",
    "B_disentangle",
    "B_disentangle_riem",
    "C_disentangle_rsvd",
    "D_sketched_disentangle",
    "E_als_random",
    "F_als_warmstart",
)


@dataclass
class MethodResult:
    method: str
    dims: MosesDims
    b_hat: np.ndarray
    spectrum_no_d: np.ndarray  # singular values entering SVD2 at Q = I
    spectrum_used: np.ndarray  # singular values entering SVD2 actually used
    u1_eff: np.ndarray
    vh2: np.ndarray | None
    runtime_s: float
    iters: int

    def metrics(self, b: np.ndarray) -> dict[str, float | str | int]:
        k2 = self.dims.k2
        second_iso = (
            isometry_defect_rows(self.vh2) if self.vh2 is not None else float("nan")
        )
        return {
            "method": self.method,
            "rel_error": relative_frobenius_error(b, self.b_hat),
            "tail_used": tail_energy(self.spectrum_used, k2),
            "tail_no_d": tail_energy(self.spectrum_no_d, k2),
            "renyi_used": renyi_half_entropy(self.spectrum_used),
            "renyi_no_d": renyi_half_entropy(self.spectrum_no_d),
            "first_iso_defect": isometry_defect_columns(self.u1_eff),
            "second_iso_defect": second_iso,
            "runtime_s": self.runtime_s,
            "iters": self.iters,
        }


def _first_svd(b: np.ndarray, dims: MosesDims) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic first SVD: returns U1 (n1 x k1) and vtilde = S1 Vh1 (k1 x n2 n3)."""
    first = svd_truncate(b.reshape(dims.n1, dims.n2 * dims.n3), dims.k1)
    vtilde = first.s[:, None] * first.vh
    return first.u, vtilde


def _reconstruct(u1_eff: np.ndarray, m2_hat: np.ndarray, dims: MosesDims) -> np.ndarray:
    vtilde_hat = cut_inverse(m2_hat, dims)
    return (u1_eff @ vtilde_hat).reshape(dims.tensor_shape)


def run_method(
    method: str,
    b: np.ndarray,
    dims: MosesDims,
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    dis_maxiter: int = 50,
    als_maxiter: int = 100,
    rng: np.random.Generator | None = None,
) -> MethodResult:
    gen = np.random.default_rng() if rng is None else rng

    # --- ALS methods operate on B directly ---
    if method == "E_als_random":
        t0 = perf_counter()
        als = als_local_ring(b, dims, init="random", maxiter=als_maxiter, rng=gen)
        runtime = perf_counter() - t0
        spec = cut_spectrum(_first_svd(b, dims)[1], dims)
        return MethodResult(method, dims, als.b_hat, spec, spec, als.u1, None, runtime, als.iters)

    # everything else starts from the deterministic first SVD
    t0 = perf_counter()
    u1, vtilde = _first_svd(b, dims)
    spectrum_no_d = cut_spectrum(vtilde, dims)

    if method == "A_two_svd":
        m2 = cut_forward(vtilde, dims)
        second = svd_truncate(m2, dims.k2)
        b_hat = _reconstruct(u1, (second.u * second.s) @ second.vh, dims)
        runtime = perf_counter() - t0
        return MethodResult(method, dims, b_hat, spectrum_no_d, spectrum_no_d, u1, second.vh, runtime, 0)

    if method == "F_als_warmstart":
        # greedy SVD + disentangler to seed the cores, then ALS polish
        dis = disentangle_altmin(vtilde, dims, maxiter=dis_maxiter, rng=gen)
        u1_g = u1 @ dis.q.conj().T
        vtilde_g = dis.q @ vtilde
        second = svd_truncate(cut_forward(vtilde_g, dims), dims.k2)
        p1 = second.u * second.s
        p2 = second.vh.T
        als = als_local_ring(
            b, dims, init="warmstart", u1_init=u1_g, p1_init=p1, p2_init=p2,
            maxiter=als_maxiter, rng=gen,
        )
        runtime = perf_counter() - t0
        spec_used = cut_spectrum(vtilde_g, dims)
        return MethodResult(method, dims, als.b_hat, spectrum_no_d, spec_used, als.u1, None, runtime, als.iters)

    # --- disentangler methods B / B_riem / C / D ---
    if method == "D_sketched_disentangle":
        dis = disentangle_altmin(
            vtilde, dims, maxiter=dis_maxiter, sketch=sketch,
            oversample=oversample, n_power=n_power, rng=gen,
        )
    elif method == "B_disentangle_riem":
        dis = disentangle_riemannian(vtilde, dims, maxiter=4 * dis_maxiter)
    else:  # B_disentangle, C_disentangle_rsvd
        dis = disentangle_altmin(vtilde, dims, maxiter=dis_maxiter, rng=gen)

    u1_eff = u1 @ dis.q.conj().T
    vtilde_g = dis.q @ vtilde
    m2 = cut_forward(vtilde_g, dims)
    spectrum_used = la.svdvals(m2)

    if method == "C_disentangle_rsvd":
        second = rsvd_truncate(m2, dims.k2, oversample=oversample, n_power=n_power, rng=gen, sketch=sketch)
    else:  # B, B_riem, D all finish with an exact second SVD
        second = svd_truncate(m2, dims.k2)

    b_hat = _reconstruct(u1_eff, (second.u * second.s) @ second.vh, dims)
    runtime = perf_counter() - t0
    return MethodResult(method, dims, b_hat, spectrum_no_d, spectrum_used, u1_eff, second.vh, runtime, dis.iters)
