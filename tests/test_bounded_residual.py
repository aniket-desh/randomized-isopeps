"""Direct bounded-residual whole-column factorization acceptance gates."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from rand_isopeps.column.bounded_residual import (
    block_mps_to_matrix,
    bounded_residual_column_qr,
    materialize_q,
    score_projection_error,
)
from rand_isopeps.column.operator import ColumnOperator, random_column_operator
from rand_isopeps.linalg.rmps_sketch import rmps_test_matrix
from rand_isopeps.moses.disentangler import cut_forward, cut_inverse, tail_energy


def _op(seed=0, lx=3, out_dim=8, in_dim=2, bond=2, complex_valued=True):
    return random_column_operator(
        lx, in_dim, out_dim, bond, np.random.default_rng(seed),
        ensemble="gaussian", complex_valued=complex_valued,
    )


def _projector(q):
    return q @ q.conj().T


def test_matrix_free_sampling_matches_dense_rmps_oracle():
    op = _op(seed=1, out_dim=4)
    ell, chi_sk, seed = 3, 2, 17
    result = bounded_residual_column_qr(
        op, ell=ell, eta=64, kappa=64, chi_sk=chi_sk, ndis=0,
        rng=np.random.default_rng(seed),
    )
    sampled = block_mps_to_matrix(result.sampled_cores)
    omega = rmps_test_matrix(
        op.input_dims, ell, chi_sk, np.random.default_rng(seed),
        normalize=True, complex_valued=True,
    )
    assert np.allclose(sampled, op.materialize() @ omega, atol=1e-11)
    assert result.matrix_mps_products == ell
    assert result.passes == 1


@pytest.mark.parametrize("complex_valued", [False, True])
def test_no_truncation_reconstructs_sampled_range_and_returns_true_residual(complex_valued):
    op = _op(seed=2, out_dim=4, complex_valued=complex_valued)
    result = bounded_residual_column_qr(
        op, ell=3, eta=64, kappa=64, chi_sk=2, ndis=5,
        rng=np.random.default_rng(9), dense_oracle_max_elements=10_000_000,
    )
    assert result.reconstruction_error < 1e-10
    assert result.residual_consistency_error < 1e-10
    assert result.delta_local < 1e-10
    assert result.delta_global < 1e-10

    q = materialize_q(result.q_cores)
    r = result.residual_operator.materialize()
    c = op.materialize()
    assert np.allclose(r, q.conj().T @ c, atol=1e-10)
    assert result.projection_error_dense == pytest.approx(
        np.linalg.norm(c - q @ r) / np.linalg.norm(c), rel=1e-11
    )
    assert result.projection_error_dense + 1e-12 >= result.spectral_tail_dense
    assert result.q_flat_rank == q.shape[1]


def test_kappa_one_reduces_to_plain_tt_sweep():
    from rand_isopeps.column.structured_qr import _tt_left_canonical_sweep

    op = _op(seed=3, out_dim=4)
    eta = 4
    result = bounded_residual_column_qr(
        op, ell=3, eta=eta, kappa=1, chi_sk=2, ndis=9,
        rng=np.random.default_rng(12),
    )
    y = block_mps_to_matrix(result.sampled_cores)
    q_plain, _, _, _ = _tt_left_canonical_sweep(y, op.output_dims, eta)
    q_direct = materialize_q(result.q_cores)
    assert all(dim == 1 for dim in result.residual_dims[:-1])
    assert q_direct.shape == q_plain.shape
    assert np.allclose(_projector(q_direct), _projector(q_plain), atol=1e-10)


def test_full_eta_matches_dense_qr_range():
    op = _op(seed=4, out_dim=4)
    result = bounded_residual_column_qr(
        op, ell=3, eta=128, kappa=1, chi_sk=2, ndis=0,
        rng=np.random.default_rng(13),
    )
    y = block_mps_to_matrix(result.sampled_cores)
    q_dense = la.qr(y, mode="economic")[0]
    q_direct = materialize_q(result.q_cores)
    assert result.reconstruction_error < 1e-10
    assert np.allclose(_projector(q_direct), _projector(q_dense), atol=1e-10)


def test_seed_is_deterministic_including_complex_gauges():
    op = _op(seed=5, out_dim=8)
    kwargs = dict(ell=4, eta=4, kappa=2, chi_sk=3, ndis=6)
    a = bounded_residual_column_qr(op, rng=np.random.default_rng(99), **kwargs)
    b = bounded_residual_column_qr(op, rng=np.random.default_rng(99), **kwargs)
    for xa, xb in zip(a.q_cores + a.sample_residual_cores,
                      b.q_cores + b.sample_residual_cores):
        assert np.array_equal(xa, xb)
    assert a.reconstruction_error == b.reconstruction_error


def test_eta_and_kappa_improve_on_controlled_case():
    # This seed has genuine composite headroom at the first two cuts.  The test
    # checks the requested cap semantics on one fixed sampled problem; it does not
    # assert a universal monotonicity theorem for greedy TT rounding.
    op = _op(seed=8, out_dim=8, bond=1)

    def error(eta, kappa):
        return bounded_residual_column_qr(
            op, ell=4, eta=eta, kappa=kappa, chi_sk=2, ndis=0,
            rng=np.random.default_rng(31),
        ).reconstruction_error

    eta_errors = [error(eta, 1) for eta in (2, 4, 8)]
    kappa_errors = [error(4, kappa) for kappa in (1, 2, 3)]
    assert np.all(np.diff(eta_errors) <= 1e-10), eta_errors
    assert np.all(np.diff(kappa_errors) <= 1e-10), kappa_errors


def test_nonproduct_gauge_is_necessary_for_spectrum_improvement():
    # A product gauge A (x) B becomes left/right unitary multiplication after the
    # Moses reshuffle, so it cannot change singular values.  A generic full gauge
    # can.  Construct a rank-eta favorable cut, scramble it by a known non-product
    # unitary, and compare both cases exactly.
    class Dims:
        eta, chi, n2, n3 = 2, 2, 3, 2
        k1, k2 = 4, 2

    rng = np.random.default_rng(123)
    left = rng.standard_normal((Dims.eta * Dims.n2, Dims.k2))
    right = rng.standard_normal((Dims.k2, Dims.chi * Dims.n3))
    favorable = left @ right
    v_good = cut_inverse(favorable, Dims)
    full, _ = la.qr(rng.standard_normal((Dims.k1, Dims.k1)))
    v_scrambled = full.conj().T @ v_good
    initial = tail_energy(la.svdvals(cut_forward(v_scrambled, Dims)), Dims.k2)
    recovered = tail_energy(la.svdvals(cut_forward(full @ v_scrambled, Dims)), Dims.k2)
    assert recovered < 1e-20 * max(initial, 1.0)
    assert initial > 1e-6

    for _ in range(8):
        a, _ = la.qr(rng.standard_normal((Dims.eta, Dims.eta)))
        b, _ = la.qr(rng.standard_normal((Dims.chi, Dims.chi)))
        product = np.kron(a, b)
        product_tail = tail_energy(
            la.svdvals(cut_forward(product @ v_scrambled, Dims)), Dims.k2
        )
        assert product_tail == pytest.approx(initial, rel=1e-12, abs=1e-12)


def test_actual_dimensions_and_power_iteration_counters_are_reported():
    op = _op(seed=11, lx=4, out_dim=5, bond=2)
    result = bounded_residual_column_qr(
        op, ell=3, eta=4, kappa=3, chi_sk=2, n_power=1, ndis=2,
        rng=np.random.default_rng(7), dense_oracle_max_elements=1,
    )
    assert len(result.cuts) == op.lx
    assert all(c.composite_dim <= 12 for c in result.cuts)
    assert all(c.q_up <= 4 and c.kappa_actual <= 3 for c in result.cuts if c.kind == "internal")
    assert result.max_q_vertical <= 4
    assert result.matrix_mps_products == 9
    assert result.passes == 3
    assert result.contraction_count > result.matrix_mps_products
    assert result.peak_allocated_bytes > 0
    assert np.isnan(result.delta_global)  # dense oracle deliberately disabled
    assert isinstance(result.residual_operator, ColumnOperator)
    assert result.max_residual_vertical >= result.max_sample_residual_vertical


def test_fresh_rmps_projection_score_calibrates_to_dense_error():
    op = _op(seed=21, out_dim=4)
    result = bounded_residual_column_qr(
        op, ell=3, eta=2, kappa=1, chi_sk=2, ndis=0,
        rng=np.random.default_rng(41),
    )
    score = score_projection_error(
        op, result, n_probes=512, chi_score=4,
        rng=np.random.default_rng(991), n_bootstrap=400,
    )
    assert score.estimate == pytest.approx(result.projection_error_dense, rel=0.12)
    assert score.ci_low <= result.projection_error_dense <= score.ci_high
    assert score.matrix_mps_products == 3 * score.n_probes
    assert score.contraction_count == 6 * score.n_probes * op.lx
    assert score.contraction_flops_estimate > 0
    assert score.peak_mps_bond >= score.chi_score


def test_fresh_projection_score_is_seed_deterministic():
    from dataclasses import asdict

    op = _op(seed=22, out_dim=4, complex_valued=False)
    result = bounded_residual_column_qr(
        op, ell=3, eta=3, kappa=2, chi_sk=2, ndis=2,
        rng=np.random.default_rng(42),
    )
    kwargs = dict(n_probes=12, chi_score=3, n_bootstrap=20)
    a = score_projection_error(op, result, rng=np.random.default_rng(992), **kwargs)
    b = score_projection_error(op, result, rng=np.random.default_rng(992), **kwargs)
    da, db = asdict(a), asdict(b)
    da.pop("runtime_s")
    db.pop("runtime_s")
    assert da == db


def test_cached_dense_reference_matches_uncached_oracle():
    op = _op(seed=41, out_dim=4)
    reference = op.materialize()
    singular = la.svdvals(reference, check_finite=False)
    kwargs = dict(
        ell=4, eta=2, kappa=2, chi_sk=2, sketch_kind="rmps",
        n_power=0, ndis=1,
    )
    uncached = bounded_residual_column_qr(
        op, rng=np.random.default_rng(91), **kwargs
    )
    cached = bounded_residual_column_qr(
        op, rng=np.random.default_rng(91), reference=reference,
        reference_singular_values=singular, **kwargs
    )
    assert np.isclose(cached.projection_error_dense, uncached.projection_error_dense)
    assert np.isclose(cached.spectral_tail_dense, uncached.spectral_tail_dense)
    assert np.isclose(cached.projection_excess_dense, uncached.projection_excess_dense)
