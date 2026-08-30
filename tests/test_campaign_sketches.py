import numpy as np

from rand_isopeps.campaign.sketches import (
    _apply_walsh_matrix,
    _psd_nystrom,
    _relative_nuclear_error,
    run_gaussian_limit,
    walsh_subspace,
)


def _task(kind, problem, method, replicates=2):
    return {
        "problem": {"kind": kind, **problem},
        "method": method,
        "measurement": {"replicates": replicates},
        "seeds": {"problem": 1, "sketch": 2, "score": 3},
    }


def test_low_rank_nuclear_error_matches_dense_residual():
    subspace = walsh_subspace(4, 3)
    epsilon = 1e-3
    rng = np.random.default_rng(8)
    omega = rng.standard_normal((16, 5))
    matrix = subspace @ subspace.T + epsilon * np.eye(16)
    vectors, values = _psd_nystrom(
        omega, _apply_walsh_matrix(omega, subspace, epsilon), 1e-12
    )
    approximation = (vectors * values[None, :]) @ vectors.T
    expected = np.sum(np.abs(np.linalg.eigvalsh(matrix - approximation)))
    expected /= np.trace(matrix)
    np.testing.assert_allclose(
        _relative_nuclear_error(subspace, epsilon, vectors, values),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_walsh_benchmarks_emit_one_row_per_trial():
    variance = _task(
        "walsh_variance",
        {
            "tensor_order": 4,
            "rank": 2,
            "epsilon": 1e-3,
            "samples": 40,
            "batch_size": 8,
        },
        {"name": "rmps", "chi_sk": 2},
    )
    variance_rows = run_gaussian_limit(variance)
    assert len(variance_rows) == 2
    assert all(row["normalized_quadratic_variance"] > 0 for row in variance_rows)
    for row in variance_rows:
        reconstructed = row["quadratic_sample_m2"]
        reconstructed /= (row["samples"] - 1) * row["trace_value"] ** 2
        assert reconstructed == row["normalized_quadratic_variance"]

    nystrom = _task(
        "walsh_nystrom",
        {
            "tensor_order": 4,
            "rank": 2,
            "epsilon": 1e-3,
            "ridge": 1e-12,
            "embedding_dim": 4,
        },
        {"name": "mps_gram_nystrom", "chi_sk": 2},
    )
    nystrom_rows = run_gaussian_limit(nystrom)
    assert len(nystrom_rows) == 2
    assert all(0 <= row["relative_nuclear_error"] for row in nystrom_rows)
