"""Alternating least squares for the local isometric tensor ring.

This optimizes the local tensor-ring cores directly, as an alternative to the
greedy two-SVD(+disentangler) construction. The block-isoPEPS paper notes one
can polish the tensor-ring cores with ALS, but in their experience it did not
significantly improve the greedy method; experiments E/F test that synthetically.

Model. With the same factor structure the greedy method produces, the local
tensor is approximated as

    B ~= U1 . A^{-1}(P1 P2^T),

where ``U1`` is an isometry of shape ``(n1, k1)`` on the left index, ``P1`` has
shape ``(eta*n2, k2)`` and ``P2`` shape ``(chi*n3, k2)`` parameterize a rank-k2
factorization across the second-SVD cut, and ``A^{-1}`` is the reshuffle from
``rand_isopeps.disentangler`` mapping ``(eta*n2, chi*n3)`` back to ``(k1, n2*n3)``.

Each sweep solves three convex subproblems in closed form:

* ``P1`` and ``P2`` are linear least squares against the projected target
  ``Mhat = A(U1^T B)`` (since ``U1`` has orthonormal columns).
* ``U1`` is an orthogonal Procrustes / isometric update from the SVD of
  ``B Z^T`` with ``Z = A^{-1}(P1 P2^T)``.

The objective ``||B - U1 A^{-1}(P1 P2^T)||_F`` decreases monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.linalg as la

from .disentangler import cut_forward, cut_inverse
from .randomized_svd import isometry_defect_columns, relative_frobenius_error
from .tn_shapes import MosesDims

_RIDGE = 1e-12


@dataclass
class ALSResult:
    dims: MosesDims
    u1: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    b_hat: np.ndarray
    iters: int
    runtime_s: float
    init: str
    errors: list[float] = field(default_factory=list)

    def metrics(self, b: np.ndarray) -> dict[str, float | str | int]:
        return {
            "rel_error": relative_frobenius_error(b, self.b_hat),
            "first_iso_defect": isometry_defect_columns(self.u1),
            "als_iters": self.iters,
            "runtime_s": self.runtime_s,
        }


def _ls_factor(target: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Least-squares factor: argmin_F ||target - F fixed^T||_F = target fixed (fixed^T fixed)^-1."""
    gram = fixed.T @ fixed
    gram += _RIDGE * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    return la.solve(gram, (target @ fixed).T, assume_a="pos").T


def _reconstruct(u1: np.ndarray, p1: np.ndarray, p2: np.ndarray, dims: MosesDims) -> np.ndarray:
    z = cut_inverse(p1 @ p2.T, dims)
    return (u1 @ z).reshape(dims.tensor_shape)


def als_local_ring(
    b: np.ndarray,
    dims: MosesDims,
    init: str = "random",
    u1_init: np.ndarray | None = None,
    p1_init: np.ndarray | None = None,
    p2_init: np.ndarray | None = None,
    maxiter: int = 100,
    tol: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> ALSResult:
    """Run isometric tensor-ring ALS on a local tensor ``b``.

    ``init='random'`` starts from a random isometry and random rank-k2 factors
    (experiment E: discover the ring from scratch). ``init='warmstart'`` uses the
    supplied ``u1_init/p1_init/p2_init`` from the greedy SVD+disentangler method
    (experiment F: polish a good solution).
    """
    if b.shape != dims.tensor_shape:
        raise ValueError(f"expected tensor shape {dims.tensor_shape}, got {b.shape}")
    gen = np.random.default_rng() if rng is None else rng
    b_mat = b.reshape(dims.n1, dims.n2 * dims.n3)
    k1, k2 = dims.k1, dims.k2

    if init == "warmstart":
        if u1_init is None or p1_init is None or p2_init is None:
            raise ValueError("warmstart requires u1_init, p1_init, p2_init")
        u1 = np.array(u1_init, copy=True)
        p1 = np.array(p1_init, copy=True)
        p2 = np.array(p2_init, copy=True)
    elif init == "random":
        a = gen.standard_normal((dims.n1, k1))
        u1, _ = np.linalg.qr(a)
        u1 = u1[:, :k1]
        p1 = gen.standard_normal((dims.eta * dims.n2, k2))
        p2 = gen.standard_normal((dims.chi * dims.n3, k2))
    else:
        raise ValueError(f"unknown init: {init}")

    t0 = perf_counter()
    errors: list[float] = []
    prev = np.inf
    iters = 0
    for iters in range(1, maxiter + 1):
        # P1, P2 least-squares given the current isometry U1
        mhat = cut_forward(u1.T @ b_mat, dims)
        p1 = _ls_factor(mhat, p2)
        p2 = _ls_factor(mhat.T, p1)
        # isometric Procrustes update for U1 given the factors
        z = cut_inverse(p1 @ p2.T, dims)
        uu, _, vh = la.svd(b_mat @ z.T, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        u1 = uu[:, :k1] @ vh
        b_hat = _reconstruct(u1, p1, p2, dims)
        err = relative_frobenius_error(b, b_hat)
        errors.append(err)
        if abs(prev - err) <= tol:
            break
        prev = err

    runtime = perf_counter() - t0
    b_hat = _reconstruct(u1, p1, p2, dims)
    return ALSResult(
        dims=dims, u1=u1, p1=p1, p2=p2, b_hat=b_hat,
        iters=iters, runtime_s=runtime, init=init, errors=errors,
    )
