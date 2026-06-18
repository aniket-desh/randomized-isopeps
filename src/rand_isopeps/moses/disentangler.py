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

from rand_isopeps.linalg.randomized_svd import SketchKind, SketchSpec, as_spec, rsvd_truncate
from rand_isopeps.tn_shapes import MosesDims


# --- reshuffle operators A : (k1, n2 n3) -> (eta n2, chi n3) and its inverse ---
#
# These are pure permutations/reshapes (norm preserving). ``A`` groups the bond
# legs so the cut matches the second SVD; ``A^{-1}`` undoes it. Mirrors
# local_ring_decomp's ``v4.transpose(0, 2, 1, 3)`` reshape.
#
# Cross-checked against Dektor's reference (github.com/adektor/disentangle-python):
# with X = vtilde.reshape(eta, chi, n2, n3), ``cut_forward`` is their
# ``ten_to_mat(X, svd_legs=[0,2])`` and ``Q @ vtilde`` is their
# ``Q @ ten_to_mat(X, dis_legs=[0,1])`` -- bit-identical, and the alt-min /
# Euclidean-gradient below match their trunc_error optimizer to machine
# precision (see log.md 2026-06-16).

def cut_forward(vtilde: np.ndarray, dims: MosesDims) -> np.ndarray:
    """A(vtilde): map (k1, n2*n3) to the second-SVD matrix (eta*n2, chi*n3)."""
    v4 = vtilde.reshape(dims.eta, dims.chi, dims.n2, dims.n3)
    return v4.transpose(0, 2, 1, 3).reshape(dims.eta * dims.n2, dims.chi * dims.n3)


def cut_inverse(m: np.ndarray, dims: MosesDims) -> np.ndarray:
    """A^{-1}(m): map (eta*n2, chi*n3) back to (k1, n2*n3)."""
    m4 = m.reshape(dims.eta, dims.n2, dims.chi, dims.n3)
    return m4.transpose(0, 2, 1, 3).reshape(dims.k1, dims.n2 * dims.n3)


def sketch_for_second_svd(kind: str, dims: MosesDims) -> "str | SketchSpec":
    """Build the sketch argument for the second-SVD matrix ``M = A(vtilde)``.

    Single source of truth for the structured-sketch parameters so the
    correctness-critical Khatri-Rao factor order lives in one place: ``M``'s right
    (column) index is the product ``chi*n3 = chi*(chi*eta)`` with ``chi`` outer and
    ``n3`` inner, so the Khatri-Rao sketch *must* use ``factor_dims=(chi, n3)`` in
    that order. ``"exact"`` and the simple kinds pass through unchanged.
    """
    if kind == "khatri_rao":
        return SketchSpec("khatri_rao", factor_dims=(dims.chi, dims.n3), base="spherical")
    if kind == "sparsestack":
        return SketchSpec("sparsestack", zeta=4)
    return kind  # "exact", "gaussian", "countsketch", "rademacher"


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


# --- gauge diagnostics: the disentangler must stay an exact unitary/orthogonal ---
#
# The disentangler is a gauge: U1 V = (U1 Q^H)(Q V) holds *exactly* only when
# Q^H Q = I. A sketched search may propose an approximate update, but the Q
# actually inserted into the tensor network must be retracted to a unitary
# (the Procrustes/polar step does this). These quantify how well that holds.

def unitary_defect(q: np.ndarray) -> float:
    """||Q^H Q - I||_F: distance of Q from the orthogonal/unitary group."""
    gram = q.conj().T @ q
    return float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=gram.dtype)))


def gauge_defect(u1: np.ndarray, vtilde: np.ndarray, q: np.ndarray) -> float:
    """Pre-truncation gauge error ||U1 V - (U1 Q^H)(Q V)||_F = ||U1 (Q^H Q - I) V||_F.

    Bounded by ``unitary_defect(q) * ||V||_F`` when U1 is an isometry, so it is at
    machine precision exactly when Q is. Zero (to roundoff) for Q = I.
    """
    k1 = q.shape[1]
    inner = (q.conj().T @ q) - np.eye(k1, dtype=q.dtype)
    return float(np.linalg.norm(u1 @ (inner @ vtilde)))


@dataclass
class DisentangleResult:
    q: np.ndarray  # optimized disentangler, shape (k1, k1)
    iters: int
    tail_initial: float  # tail energy at Q = I (no disentangler)
    tail_final: float  # tail energy at the optimized Q
    optimizer: str
    # optional fields (appended with defaults; existing constructors unaffected)
    converged: bool = True
    sketch: str | None = None
    extra: dict[str, float] | None = None  # accepted/rejected steps, surrogate tail, ...


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


def disentangle_altmin_sketched(
    vtilde: np.ndarray,
    dims: MosesDims,
    k: int | None = None,
    maxiter: int = 50,
    tol: float = 1e-10,
    sketch: "SketchKind | SketchSpec" = "gaussian",
    oversample: int = 8,
    n_power: int = 1,
    rng: np.random.Generator | None = None,
    *,
    fresh_sketch: bool = True,
    validate: bool = True,
    accept_tol: float = 0.0,
    max_rejects: int = 5,
) -> DisentangleResult:
    """Validated sketched alternating-minimization disentangler.

    Like :func:`disentangle_altmin` but the inner rank-``k`` truncation is a
    randomized SVD, and -- crucially -- each Procrustes candidate is **accepted
    only if it lowers the EXACT tail** ``c_k(Q) = sum_{i>k} sigma_i^2(A(QV))``
    (``validate=True``). This validation is required, not optional: optimizing a
    *fixed-sketch* surrogate over the manifold ``O(k1)`` is not covered by the
    OSI/OSE guarantees (Camano-Epperly-Meyer-Tropp). Those bounds are
    per-*fixed*-subspace -- once ``Q`` is chosen as a function of the realized
    sketch the quantifier order is reversed, the continuum ``O(k1)`` defeats any
    union bound, and the sketch surrogate can be driven down while the true
    ``c_k`` stalls or rises. Accepting only on the exact tail makes the kept
    sequence exactly non-increasing in the true objective regardless of sketch
    noise (the approximate-monotonicity argument, with exact validation).

    ``fresh_sketch=True`` draws an independent sketch each iteration (the
    held-out-sketch mitigation); ``fresh_sketch=False`` freezes one sketch (the
    overfitting-stress mode). ``validate=False`` accepts every candidate (also a
    stress mode). The returned ``Q`` is always exactly unitary (Procrustes/polar).
    """
    k = dims.k2 if k is None else k
    x = vtilde
    k1 = x.shape[0]
    q = np.eye(k1, dtype=x.dtype)
    gen = np.random.default_rng() if rng is None else rng
    spec = as_spec(sketch)
    # frozen-sketch mode: reuse the SAME sketch draw every iteration
    frozen_seed = None if fresh_sketch else int(gen.integers(0, 2**31 - 1))

    def _exact_tail(qq: np.ndarray) -> float:
        return tail_energy(la.svdvals(cut_forward(qq @ x, dims)), k)

    tail0 = _exact_tail(q)
    c_cur = tail0
    surrogate = float("nan")
    accepted = rejected = consecutive_rejects = 0
    converged = False
    iters = 0
    for iters in range(1, maxiter + 1):
        y = cut_forward(q @ x, dims)
        inner_rng = np.random.default_rng(frozen_seed) if frozen_seed is not None else gen
        res = rsvd_truncate(y, k, oversample=oversample, n_power=n_power, rng=inner_rng, sketch=spec)
        mk = (res.u * res.s) @ res.vh
        q_cand = _orthogonal_procrustes(cut_inverse(mk, dims) @ x.conj().T)

        if validate:
            c_cand = _exact_tail(q_cand)
            if c_cand <= c_cur - accept_tol:
                delta = float(np.linalg.norm(q_cand - q))
                q, c_cur = q_cand, c_cand
                accepted += 1
                consecutive_rejects = 0
                if delta <= tol:
                    converged = True
                    break
            else:
                rejected += 1
                consecutive_rejects += 1
                if consecutive_rejects >= max_rejects:
                    converged = True  # stalled at a (validated) local minimum
                    break
        else:  # unvalidated: accept everything (overfitting-stress mode)
            delta = float(np.linalg.norm(q_cand - q))
            q = q_cand
            accepted += 1
            if delta <= tol:
                converged = True
                break

    tail_final = _exact_tail(q)
    # Re-evaluate the sketch's tail estimate at the FINAL gauge so it pairs with
    # tail_final (the in-loop ``surrogate`` was taken at the pre-Procrustes q, one
    # step earlier). exact - surrogate is then the genuine sketch bias at q: a
    # frozen sketch that has been overfit reports a low surrogate vs the high
    # exact tail (positive bias).
    y_final = cut_forward(q @ x, dims)
    rng_final = np.random.default_rng(frozen_seed) if frozen_seed is not None else gen
    res_final = rsvd_truncate(y_final, k, oversample=oversample, n_power=n_power, rng=rng_final, sketch=spec)
    surrogate = float(np.linalg.norm(y_final - (res_final.u * res_final.s) @ res_final.vh) ** 2)
    extra = {
        "accepted": float(accepted),
        "rejected": float(rejected),
        "surrogate_final": surrogate,
        "exact_minus_surrogate": tail_final - surrogate,
        "unitary_defect": unitary_defect(q),
    }
    label = f"altmin-sketch-{spec.kind}{'' if validate else '-noval'}{'' if fresh_sketch else '-fixed'}"
    return DisentangleResult(
        q=q, iters=iters, tail_initial=tail0, tail_final=tail_final, optimizer=label,
        converged=converged, sketch=spec.kind, extra=extra,
    )


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
