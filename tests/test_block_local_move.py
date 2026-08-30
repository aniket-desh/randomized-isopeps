from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.block_state import BlockPeps, dense_block
from rand_isopeps.real_isotns.block_local_move import block_local_column_move
from rand_isopeps.real_isotns.moses_move import (
    _riemannian_disentangler,
    moses_move,
)


def _random_block(ly: int, column: int, size: int, seed: int) -> BlockPeps:
    rng = np.random.default_rng(seed)
    peps = qtn.PEPS.rand(2, ly, bond_dim=2, phys_dim=2, seed=seed)
    center = (1, column)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = rng.standard_normal((*tensor.shape, size))
    data = data + 1j * rng.standard_normal(data.shape)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    return BlockPeps(peps, "alpha", center)


def _projector(block):
    basis, _ = np.linalg.qr(block, mode="reduced")
    return basis @ basis.conj().T


@pytest.mark.parametrize(
    "ly,column,split,next_column",
    [
        (2, 0, "right", 1),
        (2, 1, "left", 0),
        (3, 1, "right", 2),
        (3, 1, "left", 0),
    ],
)
def test_full_rank_local_block_move_preserves_subspace_and_moves_alpha(
    ly, column, split, next_column
):
    state = _random_block(ly, column, 2, seed=70 + ly)
    before = dense_block(state)
    moved, record = block_local_column_move(
        state,
        column,
        split=split,
        chi=64,
        eta=64,
        cutoff=0.0,
        ndis=0,
        absorption_max_bond=None,
        absorption_cutoff=0.0,
    )

    after = dense_block(moved)
    assert moved.center == (0, next_column)
    assert record["backend"] == "local_shared_q"
    assert sum("alpha" in tensor.inds for tensor in moved.peps.tensors) == 1
    assert after == pytest.approx(before, abs=2e-10)
    assert np.linalg.norm(_projector(after) - _projector(before)) < 2e-10


def test_size_one_block_move_matches_the_existing_local_move():
    base = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=81)
    scalar = base.copy()
    moses_move(
        scalar,
        0,
        64,
        64,
        0.0,
        0,
        sweep="up",
        split="right",
        renorm=False,
        absorb_max_bond=None,
        absorb_cutoff=0.0,
    )

    block_peps = base.copy()
    center = (1, 0)
    tensor = block_peps[block_peps.site_tag_id.format(*center)]
    tensor.modify(data=tensor.data[..., None], inds=(*tensor.inds, "alpha"))
    block = BlockPeps(block_peps, "alpha", center)
    moved, _ = block_local_column_move(
        block,
        0,
        split="right",
        chi=64,
        eta=64,
        cutoff=0.0,
        ndis=0,
        absorption_max_bond=None,
        absorption_cutoff=0.0,
    )

    scalar_dense = np.asarray(scalar.to_dense()).reshape(-1)
    assert dense_block(moved)[:, 0] == pytest.approx(scalar_dense, abs=2e-10)


def test_riemannian_renyi_optimizer_lowers_entropy_on_a_real_cut():
    pytest.importorskip("pymanopt")
    rng = np.random.default_rng(2)
    data = rng.standard_normal((2, 2, 2, 3))
    tensor = qtn.Tensor(data.astype(complex), inds=("a", "b", "c", "d"))

    gauge = _riemannian_disentangler(
        tensor,
        dis_bonds=("a", "b"),
        svd_bonds=("b", "c"),
        maxiter=15,
    )

    def entropy(q):
        cut = (q @ data.reshape(4, -1)).reshape(2, 2, 2, 3)
        singular = np.linalg.svd(
            cut.transpose(1, 2, 0, 3).reshape(4, 6),
            compute_uv=False,
        )
        return 2.0 * np.log(np.sum(singular) / np.linalg.norm(singular))

    assert gauge.T @ gauge == pytest.approx(np.eye(4), abs=1e-12)
    assert np.linalg.det(gauge) == pytest.approx(1.0, abs=1e-12)
    assert entropy(gauge) < entropy(np.eye(4)) - 1e-3


def test_riemannian_renyi_is_wired_through_a_real_block_move():
    pytest.importorskip("pymanopt")
    rng = np.random.default_rng(91)
    peps = qtn.PEPS.rand(
        2, 2, bond_dim=2, phys_dim=2, seed=91, dtype="float64"
    )
    center = (1, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = np.stack((tensor.data, rng.standard_normal(tensor.shape)), axis=-1)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    state = BlockPeps(peps, "alpha", center)
    before = dense_block(state)

    moved, record = block_local_column_move(
        state,
        0,
        split="right",
        chi=8,
        eta=4,
        cutoff=0.0,
        ndis=3,
        disentangler="riemannian_renyi",
        absorption_max_bond=None,
        absorption_cutoff=0.0,
    )

    assert moved.center == (0, 1)
    assert record["local_error_squared"] == pytest.approx(0.0, abs=1e-12)
    assert dense_block(moved) == pytest.approx(before, abs=2e-10)
