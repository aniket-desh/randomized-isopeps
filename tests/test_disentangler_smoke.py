"""Smoke test for the alternating-minimization tensor disentangler (numpy only).

Builds a disentanglable local tensor, extracts the first-SVD residual `vtilde`
the disentangler consumes (mirroring moses.local_methods._first_svd), runs
disentangle_altmin, and checks that the returned gauge Q is an exact orthogonal
matrix and that the result fields are sane. No quimb/pymanopt required.
"""

import numpy as np

from rand_isopeps.tn_shapes import MosesDims
from rand_isopeps.synthetic.tensors import make_disentanglable_local
from rand_isopeps.linalg.randomized_svd import svd_truncate
from rand_isopeps.moses.disentangler import (
    disentangle_altmin,
    unitary_defect,
    cut_forward,
    tail_energy,
)


def _first_svd_vtilde(b, dims):
    """vtilde = S1 Vh1 of the first SVD, exactly as moses.local_methods builds it."""
    first = svd_truncate(b.reshape(dims.n1, dims.n2 * dims.n3), dims.k1)
    return first.s[:, None] * first.vh  # (k1, n2*n3)


def test_disentangle_altmin_smoke():
    rng = np.random.default_rng(2024)
    dims = MosesDims(chi=2, eta=2, d=2, p=1)  # tiny: k1 = 4

    b = make_disentanglable_local(dims, rng, entangle=1.0, noise=1e-6)
    vtilde = _first_svd_vtilde(b, dims)
    assert vtilde.shape == (dims.k1, dims.n2 * dims.n3)

    res = disentangle_altmin(vtilde, dims, maxiter=50, tol=1e-12, rng=np.random.default_rng(1))

    # Q is square (k1 x k1) and an exact orthogonal/unitary matrix.
    q = res.q
    assert q.shape == (dims.k1, dims.k1)
    defect = unitary_defect(q)
    assert defect < 1e-8
    eye = np.eye(dims.k1, dtype=q.dtype)
    assert np.linalg.norm(q.conj().T @ q - eye) < 1e-8

    # Result fields are sane / finite.
    assert isinstance(res.iters, int)
    assert 1 <= res.iters <= 50
    assert np.isfinite(res.tail_initial)
    assert np.isfinite(res.tail_final)
    assert res.tail_initial >= 0.0
    assert res.tail_final >= 0.0
    # disentangler should not make the rank-k2 tail worse than the identity gauge.
    assert res.tail_final <= res.tail_initial + 1e-9

    # cross-check tail_final against a direct recomputation at the returned gauge.
    direct_tail = tail_energy(np.linalg.svd(cut_forward(q @ vtilde, dims), compute_uv=False), dims.k2)
    assert np.isclose(res.tail_final, direct_tail, atol=1e-10, rtol=1e-6)


def test_disentangle_altmin_reproducible():
    rng = np.random.default_rng(2024)
    dims = MosesDims(chi=2, eta=2, d=2, p=1)
    b = make_disentanglable_local(dims, rng, entangle=1.0, noise=1e-6)
    vtilde = _first_svd_vtilde(b, dims)

    r1 = disentangle_altmin(vtilde, dims, maxiter=50, rng=np.random.default_rng(0))
    r2 = disentangle_altmin(vtilde, dims, maxiter=50, rng=np.random.default_rng(0))
    assert np.allclose(r1.q, r2.q)
    assert r1.iters == r2.iters
