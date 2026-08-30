from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.block_state import (
    BlockPeps,
    block_gram,
    dense_block,
    move_block_center,
    orthonormalize_block,
    rotate_block,
    rotate_block_lattice,
)


def _product_block(seed=4, lx=2, ly=2):
    rng = np.random.default_rng(seed)
    vectors = {}
    for x in range(lx):
        for y in range(ly):
            value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            vectors[(x, y)] = value / np.linalg.norm(value)
    peps = qtn.PEPS.product_state(vectors)
    center = (0, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    center_data = rng.standard_normal((*tensor.shape, 2))
    center_data = center_data + 1j * rng.standard_normal(center_data.shape)
    tensor.modify(data=center_data, inds=(*tensor.inds, "alpha"))
    return BlockPeps(peps, "alpha", center)


def test_center_gram_matches_dense_block_and_qr():
    state = _product_block()
    dense = dense_block(state)
    assert block_gram(state) == pytest.approx(dense.conj().T @ dense, abs=1e-12)

    orthogonal, factor = orthonormalize_block(state)
    dense_orthogonal = dense_block(orthogonal)
    assert block_gram(orthogonal) == pytest.approx(np.eye(2), abs=1e-12)
    assert dense_orthogonal.conj().T @ dense_orthogonal == pytest.approx(
        np.eye(2), abs=1e-12
    )
    assert dense_orthogonal @ factor == pytest.approx(dense, abs=1e-12)


def test_rotation_changes_only_the_block_basis():
    state = _product_block()
    dense = dense_block(state)
    rotation = np.asarray([[0.0, 1.0], [1.0j, 0.0]])
    rotated = rotate_block(state, rotation)
    assert dense_block(rotated) == pytest.approx(dense @ rotation, abs=1e-12)


def test_exact_qr_moves_the_block_center_without_changing_the_states():
    state = _product_block()
    before = dense_block(state)
    moved = move_block_center(state, (1, 0))

    assert moved.center == (1, 0)
    assert dense_block(moved) == pytest.approx(before, abs=1e-12)
    assert dense_block(state) == pytest.approx(before, abs=1e-12)


def test_four_lattice_rotations_restore_block_and_metadata():
    state = _product_block(lx=2, ly=3)
    before = dense_block(state)
    rotated = rotate_block_lattice(state, turns=4)

    assert (rotated.peps.Lx, rotated.peps.Ly) == (2, 3)
    assert rotated.center == state.center
    assert dense_block(rotated) == pytest.approx(before, abs=1e-12)
    assert dense_block(state) == pytest.approx(before, abs=1e-12)


def test_validation_rejects_a_second_block_owner():
    state = _product_block()
    other = state.peps[state.peps.site_tag_id.format(1, 1)]
    other.modify(data=np.stack([other.data, other.data], axis=-1), inds=(*other.inds, "alpha"))
    with pytest.raises(ValueError, match="exactly one"):
        BlockPeps(state.peps, "alpha", state.center)
