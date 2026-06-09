"""Tensor disentangler for the local Moses-Move second SVD.

This implements the disentangler of Wei, Dektor, Shen, Wen, Yang, "Numerical
Optimization for Tensor Disentanglement" (docs/disentangling.pdf), specialized
to the reshape that defines the block-Moses *second* truncated SVD.

Setting. After the first SVD, the local decomposition has a residual factor
``vtilde`` of shape ``(k1, n2*n3)`` with ``k1 = eta*chi``. The second SVD
compresses the reshuffled matrix ``A(vtilde)`` of shape ``(eta*n2, chi*n3)`` to
rank ``k2 = eta``. The disentangler is an orthogonal/unitary ``Q in O(k1)``
acting on the shared bond index ``k1``; it is a gauge transformation (its
inverse is absorbed into the neighboring first isometry, see
``rand_isopeps.local_methods``) so it does not change the represented tensor
before truncation, but it does change the singular spectrum seen by the second
SVD. We choose ``Q`` to make that spectrum decay faster, i.e. to minimize the
rank-``k2`` truncation tail (Eq. 2.1 of the paper)::

    c_k(Q) = sum_{i>k} sigma_i(A(Q vtilde))^2 .

Two optimizers are provided:

* ``disentangle_altmin`` -- the closed-form alternating minimization
  (Algorithm 5): rank-k truncated SVD then an orthogonal Procrustes update
  ``Q = U V^H``. This is the default workhorse: deterministic, monotone, and
  free of step-size tuning. A ``sketch`` argument swaps the inner truncation
  for a randomized SVD, giving the "sketched disentangler objective" used by
  experiment D.
* ``disentangle_riemannian`` -- a Riemannian optimizer on O(k1) via pymanopt
  (the tool used in the paper), driven by the closed-form Euclidean gradient of
  Theorem 3.1. pymanopt is an optional dependency; the function raises a clear
  error if it is missing.

The reshuffle operators ``A`` / ``A^{-1}`` here match the transpose-reshape used
in ``local_ring_decomp`` exactly, so a disentangler of the identity reproduces
the existing two-SVD path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from .randomized_svd import SketchKind, rsvd_truncate
from .tn_shapes import MosesDims


# --- reshuffle operators A : (k1, n2 n3) -> (eta n2, chi n3) and its inverse ---
#
# These are pure permutations/reshapes (norm preserving). ``A`` groups the bond
# legs so the cut matches the second SVD; ``A^{-1}`` undoes it. Mirrors
# local_ring_decomp's ``v4.transpose(0, 2, 1, 3)`` reshape.

def cut_forward(vtilde: np.ndarray, dims: MosesDims) -> np.ndarray:
    """A(vtilde): map (k1, n2*n3) to the second-SVD matrix (eta*n2, chi*n3)."""
    v4 = vtilde.reshape(dims.eta, dims.chi, dims.n2, dims.n3)
    return v4.transpose(0, 2, 1, 3).reshape(dims.eta * dims.n2, dims.chi * dims.n3)


def cut_inverse(m: np.ndarray, dims: MosesDims) -> np.ndarray:
    """A^{-1}(m): map (eta*n2, chi*n3) back to (k1, n2*n3)."""
    m4 = m.reshape(dims.eta, dims.n2, dims.chi, dims.n3)
    return m4.transpose(0, 2, 1, 3).reshape(dims.k1, dims.n2 * dims.n3)


# --- spectral diagnostics on the second-SVD matrix ---

def cut_spectrum(vtilde: np.ndarray, dims: MosesDims) -> np.ndarray:
    """Singular values of the matrix entering the second SVD, A(vtilde)."""
    return la.svdvals(cut_forward(vtilde, dims))


def tail_energy(singular: np.ndarray, k: int) -> float:
    """Discarded energy of a rank-k truncation: sum_{i>k} sigma_i^2."""
    if k >= singular.shape[0]:
        return 0.0
    return float(np.sum(singular[k:] ** 2))


def renyi_half_entropy(singular: np.ndarray) -> float:
    """Renyi-1/2 entanglement entropy S_{1/2} = 2 ln(sum_i sigma_i / ||sigma||).

    Computed from the normalized spectrum (sum of squares = 1) so it is a pure
    measure of how the singular weight is distributed across the cut. A flatter
    spectrum gives a larger entropy; a good disentangler lowers it.
    """
    s = np.asarray(singular, dtype=float)
    norm_sq = float(np.sum(s ** 2))
    if norm_sq <= 0:
        return 0.0
    p = s ** 2 / norm_sq  # eigenvalues of the reduced density matrix
    return float(2.0 * np.log(np.sum(np.sqrt(p))))


@dataclass
class DisentangleResult:
    q: np.ndarray  # optimized disentangler, shape (k1, k1)
    iters: int
    tail_initial: float  # tail energy at Q = I (no disentangler)
    tail_final: float  # tail energy at the optimized Q
    optimizer: str


def _orthogonal_procrustes(c: np.ndarray) -> np.ndarray:
    """argmax over unitary Q of Re tr(Q^H C): Q = U V^H from svd(C)."""
    u, _, vh = la.svd(c, full_matrices=False, check_finite=False)
    return u @ vh


def disentangle_altmin(
    vtilde: np.ndarray,
    dims: MosesDims,
    k: int | None = None,
    maxiter: int = 50,
    tol: float = 1e-10,
    sketch: SketchKind | None = None,
    oversample: int = 8,
    n_power: int = 1,
    rng: np.random.Generator | None = None,
) -> DisentangleResult:
    """Alternating-minimization disentangler (Algorithm 5).

    Starts from Q = I and alternates a rank-``k`` truncation of ``A(Q vtilde)``
    with an orthogonal Procrustes update for Q until ``||Q_{j+1} - Q_j||_F`` is
    below ``tol``. When ``sketch`` is set, the inner rank-``k`` truncation uses a
    randomized SVD (the "sketched objective" of experiment D); the returned Q is
    then meant to be followed by an *exact* final second SVD.
    """
    k = dims.k2 if k is None else k
    x = vtilde
    k1 = x.shape[0]
    q = np.eye(k1, dtype=x.dtype)
    gen = np.random.default_rng() if rng is None else rng

    s0 = la.svdvals(cut_forward(x, dims))
    tail0 = tail_energy(s0, k)

    iters = 0
    for iters in range(1, maxiter + 1):
        y = cut_forward(q @ x, dims)
        # rank-k truncation M_k = SVD_k(A(Q X)), exact or sketched
        if sketch is None:
            uu, ss, vh = la.svd(y, full_matrices=False, check_finite=False, lapack_driver="gesdd")
            kk = min(k, ss.shape[0])
            mk = (uu[:, :kk] * ss[:kk]) @ vh[:kk, :]
        else:
            res = rsvd_truncate(y, k, oversample=oversample, n_power=n_power, rng=gen, sketch=sketch)
            mk = (res.u * res.s) @ res.vh
        # Procrustes: argmin_Q ||Q X - A^{-1}(M_k)||  ->  Q = U V^H of (A^{-1}(M_k) X^H)
        target = cut_inverse(mk, dims)
        c = target @ x.conj().T
        q_new = _orthogonal_procrustes(c)
        delta = float(np.linalg.norm(q_new - q))
        q = q_new
        if delta <= tol:
            break

    tail_final = tail_energy(la.svdvals(cut_forward(q @ x, dims)), k)
    label = "altmin" if sketch is None else f"altmin-sketch-{sketch}"
    return DisentangleResult(q=q, iters=iters, tail_initial=tail0, tail_final=tail_final, optimizer=label)


def _euclidean_gradient_ck(q: np.ndarray, x: np.ndarray, dims: MosesDims, k: int) -> tuple[float, np.ndarray]:
    """Cost c_k(Q) and Euclidean gradient (Theorem 3.1) for the tail objective.

    For phi(t) = t^2 on the tail (i > k) and 0 otherwise, phi'(sigma_i) = 2
    sigma_i on the tail. The gradient is A^{-1}(U phi'(Sigma) V^T) X^T.
    """
    y = cut_forward(q @ x, dims)
    u, s, vh = la.svd(y, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    phip = np.zeros_like(s)
    if k < s.shape[0]:
        phip[k:] = 2.0 * s[k:]
    cost = float(np.sum(s[k:] ** 2)) if k < s.shape[0] else 0.0
    grad = cut_inverse((u * phip) @ vh, dims) @ x.conj().T
    return cost, grad


def disentangle_riemannian(
    vtilde: np.ndarray,
    dims: MosesDims,
    k: int | None = None,
    maxiter: int = 200,
    optimizer: str = "conjugate",
) -> DisentangleResult:
    """Riemannian disentangler on O(k1) via pymanopt (the paper's tool).

    Minimizes c_k(Q) over the special-orthogonal manifold using the closed-form
    Euclidean gradient of Theorem 3.1. Requires a real ``vtilde``. Raises
    ImportError if pymanopt is not installed.
    """
    k = dims.k2 if k is None else k
    if np.iscomplexobj(vtilde):
        raise ValueError("disentangle_riemannian requires a real vtilde (use SpecialOrthogonalGroup)")
    try:
        import pymanopt
        from pymanopt.manifolds import SpecialOrthogonalGroup
        from pymanopt.optimizers import ConjugateGradient, SteepestDescent
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("disentangle_riemannian needs pymanopt (`pip install pymanopt`)") from exc

    x = np.ascontiguousarray(vtilde, dtype=float)
    k1 = x.shape[0]
    manifold = SpecialOrthogonalGroup(k1)

    @pymanopt.function.numpy(manifold)
    def cost(q):
        return _euclidean_gradient_ck(q, x, dims, k)[0]

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(q):
        return _euclidean_gradient_ck(q, x, dims, k)[1]

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    opt = (ConjugateGradient if optimizer == "conjugate" else SteepestDescent)(
        max_iterations=maxiter, verbosity=0
    )
    q0 = np.eye(k1)
    result = opt.run(problem, initial_point=q0)
    q = np.asarray(result.point)

    tail0 = tail_energy(la.svdvals(cut_forward(x, dims)), k)
    tail_final = tail_energy(la.svdvals(cut_forward(q @ x, dims)), k)
    return DisentangleResult(
        q=q, iters=int(result.iterations) if hasattr(result, "iterations") else maxiter,
        tail_initial=tail0, tail_final=tail_final, optimizer=f"riemannian-{optimizer}",
    )
