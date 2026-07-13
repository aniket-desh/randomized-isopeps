"""State-level insertion oracle for the executed bounded residual (needs quimb)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.column.bounded_residual import (  # noqa: E402
    apply_boundary_factorization,
    bounded_residual_column_qr,
)
from rand_isopeps.column.from_quimb import from_quimb_column  # noqa: E402


def test_full_range_boundary_insertion_preserves_the_state():
    psi = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=4)
    op = from_quimb_column(psi, 0, split="right", normalize=False)
    result = bounded_residual_column_qr(
        op, ell=op.n_in, eta=64, kappa=1, chi_sk=3, ndis=0,
        rng=np.random.default_rng(22), dense_oracle_max_elements=10_000_000,
    )
    moved = apply_boundary_factorization(psi, 0, result, split="right")
    overlap = (psi.H | moved).contract()
    fidelity = abs(overlap) ** 2 / (abs(psi.norm()) ** 2 * abs(moved.norm()) ** 2)
    assert result.projection_error_dense < 1e-10
    assert abs(1.0 - fidelity) < 1e-10
    assert set(moved.outer_inds()) == set(psi.outer_inds())
