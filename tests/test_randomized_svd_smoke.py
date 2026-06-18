"""Smoke tests for deterministic and randomized truncated SVD.

Builds a small low-rank matrix with a known numerical rank, then checks the
factor shapes and reconstruction error of svd_truncate (deterministic) and
rsvd_truncate (randomized Halko-Martinsson-Tropp). The randomized result is
approximate, so we only require it to be finite and within a loose factor of
the deterministic error.
"""

import numpy as np

from rand_isopeps.linalg.randomized_svd import (
    rsvd_truncate,
    svd_truncate,
    relative_frobenius_error,
    orthonormalize,
)


def _low_rank_matrix(m, n, true_rank, rng, tail=1e-9):
    """A = U diag(s) Vh with `true_rank` dominant singular values + a tiny tail."""
    u = orthonormalize(rng.standard_normal((m, min(m, n))))
    v = orthonormalize(rng.standard_normal((n, min(m, n))))
    k = min(m, n)
    s = np.full(k, tail, dtype=float)
    # dominant block: well-separated O(1) singular values
    s[:true_rank] = np.linspace(1.0, 0.3, true_rank)
    a = (u * s) @ v.conj().T
    return a, s


def test_svd_truncate_shapes_and_error():
    rng = np.random.default_rng(0)
    m, n, true_rank, k = 60, 40, 4, 4
    a, _ = _low_rank_matrix(m, n, true_rank, rng)

    res = svd_truncate(a, k)

    assert res.u.shape == (m, k)
    assert res.vh.shape == (k, n)
    assert res.s.shape == (k,)
    assert len(res.s) == k

    err = relative_frobenius_error(a, res.reconstruct())
    assert np.isfinite(err)
    # rank-4 truncation of an (essentially) rank-4 matrix: error is the tail only
    assert err < 1e-6


def test_rsvd_truncate_shapes_and_error():
    rng = np.random.default_rng(12345)
    m, n, true_rank, k = 60, 40, 4, 4
    a, _ = _low_rank_matrix(m, n, true_rank, rng)

    det = svd_truncate(a, k)
    det_err = relative_frobenius_error(a, det.reconstruct())

    # seeded rng -> deterministic randomized run
    rnd = rsvd_truncate(a, k, oversample=8, n_power=2, rng=np.random.default_rng(7))

    assert rnd.u.shape == (m, k)
    assert rnd.vh.shape == (k, n)
    assert rnd.s.shape == (k,)
    assert len(rnd.s) == k

    rnd_err = relative_frobenius_error(a, rnd.reconstruct())
    assert np.isfinite(rnd_err)
    # randomized is approximate: require finite and within a loose factor of the
    # deterministic error (plus an absolute floor so a near-zero det_err is ok).
    assert rnd_err <= 50.0 * det_err + 1e-6


def test_rsvd_seed_is_reproducible():
    rng = np.random.default_rng(99)
    a, _ = _low_rank_matrix(60, 40, 5, rng)
    r1 = rsvd_truncate(a, 5, rng=np.random.default_rng(3))
    r2 = rsvd_truncate(a, 5, rng=np.random.default_rng(3))
    assert np.allclose(r1.s, r2.s)
    assert np.allclose(r1.reconstruct(), r2.reconstruct())
