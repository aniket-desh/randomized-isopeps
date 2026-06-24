"""rMPS sketch sanity: isotropy, the chi_sk=1 Kronecker limit, and the SketchSpec wiring."""

import numpy as np

from rand_isopeps.linalg.randomized_svd import rsvd_truncate
from rand_isopeps.linalg.rmps_sketch import (
    kron_test_matrix,
    rmps_cores,
    rmps_test_matrix,
    rmps_to_vector,
    rmps_vector,
)
from rand_isopeps.linalg.sketches import SketchSpec


def _isotropy_defect(dims, chi_sk, m, seed, complex_valued=True):
    """||E[omega omega^*] - I||_F estimated from m raw (unnormalized) probes."""
    rng = np.random.default_rng(seed)
    omega = rmps_test_matrix(dims, m, chi_sk, rng, normalize=False, complex_valued=complex_valued)
    cov = (omega @ omega.conj().T) / m
    n = int(np.prod(dims))
    return float(np.linalg.norm(cov - np.eye(n)))


def test_isotropy_holds_for_all_chi_sk():
    # E[omega omega^*] = I (Eq. 1.4) regardless of the sketch bond; Monte-Carlo noise ~ 1/sqrt(m).
    dims = (2, 2, 2)
    for chi in (1, 2, 4):
        defect = _isotropy_defect(dims, chi, m=40000, seed=chi)
        assert defect < 0.15, (chi, defect)


def test_isotropy_real_field():
    assert _isotropy_defect((2, 3, 2), 2, m=40000, seed=11, complex_valued=False) < 0.2


def test_chi_sk_one_is_kronecker_of_gaussians():
    # chi_sk = 1 -> all bonds trivial -> omega = g_1 (x) ... (x) g_t (Gaussian-Kronecker).
    dims = (2, 3, 2)
    cores = rmps_cores(dims, 1, np.random.default_rng(7))
    assert all(c.shape[0] == 1 and c.shape[2] == 1 for c in cores)  # every bond is 1
    factors = [c.reshape(d) for c, d in zip(cores, dims)]
    kron = factors[0]
    for f in factors[1:]:
        kron = np.kron(kron, f)
    assert np.allclose(rmps_to_vector(cores), kron)


def test_kron_alias_matches_rmps_chi1():
    dims = (3, 2, 2)
    a = kron_test_matrix(dims, 4, np.random.default_rng(2))
    b = rmps_test_matrix(dims, 4, 1, np.random.default_rng(2))
    assert np.allclose(a, b)


def test_core_bond_structure_and_single_factor():
    dims = (2, 3, 4, 2)
    cores = rmps_cores(dims, 5, np.random.default_rng(0))
    assert cores[0].shape == (1, 2, 5) and cores[-1].shape == (5, 2, 1)
    assert cores[1].shape == (5, 3, 5) and cores[2].shape == (5, 4, 5)
    # single factor: just a unit-variance Gaussian vector, no bond.
    one = rmps_cores((6,), 4, np.random.default_rng(0))
    assert len(one) == 1 and one[0].shape == (1, 6, 1)


def test_reproducible_with_seed():
    dims = (2, 2, 3)
    a = rmps_vector(dims, 3, np.random.default_rng(123))
    b = rmps_vector(dims, 3, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_normalization_scales_by_sqrt_ell():
    dims, ell = (2, 2, 2), 5
    raw = rmps_test_matrix(dims, ell, 2, np.random.default_rng(9), normalize=False)
    norm = rmps_test_matrix(dims, ell, 2, np.random.default_rng(9), normalize=True)
    assert np.allclose(norm, raw / np.sqrt(ell))


def test_rmps_kind_wired_into_rsvd():
    # SketchSpec(kind="rmps") must route through rsvd_truncate and recover a low-rank matrix.
    rng = np.random.default_rng(3)
    factor_dims = (4, 4)  # n = 16
    n = int(np.prod(factor_dims))
    # rank-3 matrix: rmps SVD should nail it with modest oversampling
    a = rng.standard_normal((20, 3)) @ rng.standard_normal((3, n))
    spec = SketchSpec(kind="rmps", factor_dims=factor_dims, chi_sk=4)
    res = rsvd_truncate(a, k=3, oversample=8, n_power=1, rng=rng, sketch=spec)
    approx = res.reconstruct()
    rel = np.linalg.norm(a - approx) / np.linalg.norm(a)
    assert rel < 1e-8, rel
    assert res.method == "rsvd-rmps"
