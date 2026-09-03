import numpy as np

from rand_isopeps.campaign.gpu_pilot import _power_output
from rand_isopeps.column.bounded_residual import (
    _range_block_mps,
    block_mps_to_matrix,
)
from rand_isopeps.column.operator import random_column_operator
from rand_isopeps.compression.mpo_mps_absorb import (
    canonicalize_mps_exact,
    max_mps_bond,
    mps_to_vector,
)
from rand_isopeps.linalg.rmps_sketch import rmps_cores


def test_exact_canonicalization_preserves_state_and_removes_redundant_bonds():
    rng = np.random.default_rng(3)
    mps = []
    for site in range(7):
        left = 1 if site == 0 else 16
        right = 1 if site == 6 else 16
        shape = (left, 2, right)
        mps.append(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    expected = mps_to_vector(mps)
    rounded = canonicalize_mps_exact(mps)

    assert max_mps_bond(rounded) <= 8
    assert np.allclose(mps_to_vector(rounded), expected, rtol=1e-12, atol=1e-12)


def test_power_chain_matches_dense_result_at_failed_pilot_scale():
    operator = random_column_operator(7, 2, 2, 8, np.random.default_rng(4))
    probe = canonicalize_mps_exact(
        rmps_cores(
            operator.input_dims,
            16,
            np.random.default_rng(5),
            complex_valued=True,
        )
    )

    result = _power_output(operator, probe, n_power=1)
    matrix = operator.materialize()
    vector = mps_to_vector(probe)
    expected = matrix @ (matrix.conj().T @ (matrix @ vector))

    assert max_mps_bond(result) <= 8
    assert np.allclose(mps_to_vector(result), expected, rtol=1e-11, atol=1e-11)


def test_bounded_range_rounds_each_powered_sample_before_stacking():
    operator = random_column_operator(7, 2, 2, 8, np.random.default_rng(6))
    sampled, products, _, _ = _range_block_mps(
        operator,
        ell=3,
        chi_sk=16,
        n_power=1,
        rng=np.random.default_rng(7),
        complex_valued=True,
    )

    matrix = operator.materialize()
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(3):
        probe = rmps_cores(
            operator.input_dims,
            16,
            rng,
            complex_valued=True,
        )
        vector = mps_to_vector(probe)
        expected.append(matrix @ (matrix.conj().T @ (matrix @ vector)))
    expected = np.column_stack(expected) / np.sqrt(3)

    assert products == 3
    assert max_mps_bond(sampled) <= 24
    assert np.allclose(block_mps_to_matrix(sampled), expected, rtol=1e-11, atol=1e-11)
