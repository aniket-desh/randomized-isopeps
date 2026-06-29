"""Matrix-free column QR == dense path at small Lx (the large-Lx OOM fix).

These lock the principle "dense reference validates at small Lx; matrix-free probes
evaluate the scaling regime": the matrix-free sketch/adjoint must reproduce the dense
products to roundoff, and the probe-estimated eps_proj must track the exact one.
See reports/incidents/2026-06-29-oom-materialize-large-lx.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from rand_isopeps.column.operator import (
    assert_dense_safe,
    mpo_frobenius_norm,
    random_column_operator,
)
from rand_isopeps.column.structured_qr import (
    structured_column_qr,
    structured_column_qr_matrix_free,
)
from rand_isopeps.linalg.rmps_sketch import rmps_cores, rmps_to_vector


def _op(lx=4, in_dim=3, out_dim=2, bond=5, seed=0, ensemble="decay"):
    return random_column_operator(lx, in_dim, out_dim, bond,
                                  np.random.default_rng(seed), ensemble=ensemble)


def test_mpo_frobenius_norm_matches_dense():
    op = _op(seed=1)
    assert mpo_frobenius_norm(op.cores) == pytest.approx(float(np.linalg.norm(op.materialize())), rel=1e-10)


def test_adjoint_apply_matches_dense_transpose():
    # C^* y, matrix-free, equals the dense conjugate-transpose product.
    op = _op(seed=2)
    c = op.materialize()
    rng = np.random.default_rng(7)
    # a probe over the OUTPUT legs (what C^* consumes)
    y_cores = rmps_cores(op.output_dims, 4, rng)
    y = rmps_to_vector(y_cores)
    from rand_isopeps.compression.mpo_mps_absorb import mps_to_vector
    got = mps_to_vector(op.rmatvec_mps(y_cores))
    assert np.allclose(got, c.conj().T @ y, atol=1e-10)


@pytest.mark.parametrize("n_power", [0, 1, 2])
def test_matrix_free_sketch_column_matches_dense(n_power):
    # one sampled column C (C^* C)^q omega, both ways, same probe -> roundoff agreement.
    from rand_isopeps.column.structured_qr import _sketch_column_mf
    op = _op(seed=3)
    c = op.materialize()
    rng = np.random.default_rng(11)
    omega_cores = rmps_cores(op.input_dims, 6, rng)
    omega = rmps_to_vector(omega_cores)
    dense = c @ omega
    for _ in range(n_power):
        dense = c @ (c.conj().T @ dense)
    mf = _sketch_column_mf(op, omega_cores, n_power, complex_valued=True)
    assert np.allclose(mf, dense, atol=1e-9)


@pytest.mark.parametrize("ensemble", ["decay", "gaussian"])
def test_matrix_free_qr_tracks_dense(ensemble):
    # final_bond/isometry identical structure; eps_proj estimate close to exact.
    op = _op(lx=4, bond=5, seed=4, ensemble=ensemble)
    kw = dict(ell=10, eta_q=6, chi_sk=8, sketch_kind="rmps", n_power=1)
    dense = structured_column_qr(op, rng=np.random.default_rng(20), **kw)
    mf = structured_column_qr_matrix_free(op, rng=np.random.default_rng(21),
                                          score_probes=400, **kw)
    assert mf.final_bond == dense.final_bond
    assert mf.delta_local < 1e-8 and mf.delta_global < 1e-8
    # probe estimate is unbiased; 400 probes -> within ~25% relative (or both tiny)
    if dense.eps_proj > 1e-6:
        assert mf.eps_proj == pytest.approx(dense.eps_proj, rel=0.25)
    else:
        assert mf.eps_proj < 1e-3


def test_assert_dense_safe_trips_before_oom():
    # lx=8 boundary-like column: 2^8 x 8^8 ~ 69 GB -> must raise, not allocate.
    with pytest.raises(MemoryError):
        assert_dense_safe(2 ** 8, 8 ** 8, max_gb=2.0)
    # small column is fine
    assert_dense_safe(2 ** 4, 8 ** 4, max_gb=2.0)
