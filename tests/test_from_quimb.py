"""Validate the real-isoTNS column extractor (the PEPS bridge). Skipped without quimb."""

import os

import numpy as np
import pytest

# Some user-site quimb installations enable numba's source cache even when the
# package directory is not a cacheable import locator.  Disabling that optional
# cache changes no numerics and keeps collection portable.
os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
pytest.importorskip("quimb")

from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column  # noqa: E402
from rand_isopeps.column.operator import ColumnOperator  # noqa: E402
from rand_isopeps.linalg.rmps_sketch import rmps_cores, rmps_to_vector  # noqa: E402
from rand_isopeps.real_isotns.moses_move import random_isotns  # noqa: E402


def _isotns(seed=0):
    # tiny canonical isoTNS (centre at column 0 after the Moses-sweep canonicalisation)
    return random_isotns(3, 3, bond=2, phys=2, chi=4, eta=2, cutoff=1e-10, Ndis=8, seed=seed)


def test_extract_returns_valid_column_operator():
    op = from_quimb_column(_isotns(0), 0, split="right")
    assert isinstance(op, ColumnOperator)
    assert op.lx == 3
    assert op.n_in == int(np.prod(op.input_dims))
    assert op.n_out == int(np.prod(op.output_dims))
    assert op.mpo_bond >= 1
    # boundary MPO bonds must be 1 (ColumnOperator would have raised otherwise)
    assert op.cores[0].shape[0] == 1 and op.cores[-1].shape[3] == 1


def test_gate1_matvec_matches_materialize():
    # the decisive correctness gate: the matrix-free MPO-MPS product must agree with the
    # dense materialization, i.e. the leg extraction + row ordering are consistent.
    op = from_quimb_column(_isotns(1), 0, split="right")
    mat = op.materialize()
    probe = rmps_cores(op.input_dims, 3, np.random.default_rng(0))
    free = op.matvec(probe)
    dense = mat @ rmps_to_vector(probe)
    assert np.linalg.norm(free - dense) / max(np.linalg.norm(dense), 1e-30) < 1e-11


def test_center_column_is_nontrivial():
    # the orthogonality-centre column carries entanglement -> its column map is NOT an
    # isometry (a non-flat singular spectrum), unlike an off-centre column.
    op = from_quimb_column(_isotns(2), 0, split="right")
    s = np.linalg.svd(op.materialize(), compute_uv=False)
    s = s[s > 1e-12]
    assert s.shape[0] >= 2
    assert s[-1] / s[0] < 0.99  # genuine spread, not a flat isometry spectrum


def test_reproducible_with_seed():
    a = np.linalg.svd(from_quimb_column(_isotns(5), 0, split="right").materialize(), compute_uv=False)
    b = np.linalg.svd(from_quimb_column(_isotns(5), 0, split="right").materialize(), compute_uv=False)
    assert np.allclose(a, b)


def test_find_center_column_picks_the_non_isometric_boundary():
    # random_isotns Moses-sweeps the centre to column 0; the opposite boundary is an
    # off-centre isometry. find_center_column must pick the centre (col 0, split right).
    psi = _isotns(3)
    j, split = find_center_column(psi)
    assert (j, split) == (0, "right")
    # and the OTHER boundary really is a flat isometry (the thing we must not extract)
    s = np.linalg.svd(from_quimb_column(psi, psi.Ly - 1, split="left").materialize(), compute_uv=False)
    s = s[s > 1e-12]
    assert s[-1] / s[0] > 1.0 - 1e-6  # flat == isometric


def test_split_left_off_lattice_raises():
    # pushing the residual off the left boundary (column 0, split="left") is undefined
    with pytest.raises(ValueError):
        from_quimb_column(_isotns(0), 0, split="left")
