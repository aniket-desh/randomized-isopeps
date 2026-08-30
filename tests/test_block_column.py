from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.block_state import (
    BlockPeps,
    dense_block,
    validate_block_state,
)
from rand_isopeps.real_isotns.block_column import (
    block_column_move,
    extract_block_column,
)


def _random_block(lx: int, ly: int, center: tuple[int, int], seed: int) -> BlockPeps:
    rng = np.random.default_rng(seed)
    peps = qtn.PEPS.rand(lx, ly, bond_dim=2, phys_dim=2, seed=seed)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = rng.standard_normal((*tensor.shape, 2))
    data = data + 1j * rng.standard_normal(data.shape)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    return BlockPeps(peps, "alpha", center)


def _projector(block: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(block, mode="reduced")
    return basis @ basis.conj().T


@pytest.mark.parametrize(
    "ly,column,center,split",
    [
        (2, 0, (1, 0), "right"),
        (2, 1, (0, 1), "left"),
        (3, 1, (1, 1), "right"),
        (3, 1, (0, 1), "left"),
    ],
)
def test_full_range_block_column_move_preserves_dense_subspace(
    ly, column, center, split
):
    state = _random_block(2, ly, center, seed=10 + ly)
    before = dense_block(state)
    operator, layout = extract_block_column(state, column, split=split)

    assert operator.input_dims[layout.block_row] == (
        layout.toward_dims[layout.block_row] * state.size
    )
    assert all(
        operator.input_dims[row] == layout.toward_dims[row]
        for row in range(state.peps.Lx)
        if row != layout.block_row
    )

    moved, metrics = block_column_move(
        state,
        column,
        split=split,
        ell=operator.n_in,
        eta=64,
        kappa=64,
        chi_sk=8,
        ndis=0,
        absorption_max_bond=None,
        absorption_cutoff=0.0,
        rng=np.random.default_rng(91),
    )
    after = dense_block(moved)

    assert moved.center == layout.next_center
    owners = [tensor for tensor in moved.peps.tensors if "alpha" in tensor.inds]
    assert len(owners) == 1
    assert "alpha" not in moved.peps[moved.peps.site_tag_id.format(*center)].inds
    assert after == pytest.approx(before, abs=2e-10)
    assert np.linalg.norm(_projector(after) - _projector(before)) < 2e-10
    assert metrics["block_size"] == 2
    assert metrics["column"] == column
    assert metrics["q_rank"] == np.linalg.matrix_rank(operator.materialize())


def test_truncated_block_column_move_preserves_both_block_slices():
    state = _random_block(2, 2, (1, 0), seed=33)
    moved, metrics = block_column_move(
        state,
        0,
        split="right",
        ell=3,
        eta=2,
        kappa=1,
        chi_sk=1,
        ndis=0,
        absorption_max_bond=2,
        absorption_cutoff=0.0,
        rng=np.random.default_rng(4),
    )

    validate_block_state(moved)
    block = dense_block(moved)
    owners = [
        tensor for tensor in moved.peps.tensors if moved.block_ind in tensor.inds
    ]
    assert moved.center == (1, 1)
    assert moved.size == 2
    assert len(owners) == 1
    assert block.shape[1] == 2
    assert np.linalg.matrix_rank(block) == 2
    assert np.all(np.isfinite(block))
    assert np.all(np.linalg.norm(block, axis=0) > 0.0)
    for name in ("projection_error", "isometry_defect", "isometry_bound"):
        assert np.isfinite(metrics[name])
