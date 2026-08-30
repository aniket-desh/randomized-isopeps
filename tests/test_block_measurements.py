from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.block_measurements import (
    boundary_block_energies,
    boundary_block_gram,
    boundary_projected_hamiltonian,
    dense_ritz_rotate_block,
    dense_subspace_residuals,
    ritz_rotate_block,
)
from rand_isopeps.physics.block_state import BlockPeps, dense_block, rotate_block
from rand_isopeps.physics.dense import sparse_hamiltonian
from rand_isopeps.real_isotns.tebd2 import tfi_ham


def _product_block(seed=41):
    rng = np.random.default_rng(seed)
    vectors = {}
    for x in range(2):
        for y in range(2):
            vector = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            vectors[(x, y)] = vector / np.linalg.norm(vector)
    peps = qtn.PEPS.product_state(vectors)
    center = (0, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = rng.standard_normal((*tensor.shape, 2))
    data = data + 1j * rng.standard_normal(data.shape)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    return BlockPeps(peps, "alpha", center)


def _block_gauge(seed=53):
    rng = np.random.default_rng(seed)
    gauge = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return gauge + 2.0 * np.eye(2)


def test_boundary_gram_and_projected_hamiltonian_match_dense_oracles():
    state = _product_block()
    block = dense_block(state)
    ham = tfi_ham(2, 2, g=2.7)
    hamiltonian = sparse_hamiltonian(ham, 2, 2)

    assert boundary_block_gram(state) == pytest.approx(
        block.conj().T @ block, abs=1e-11
    )
    assert boundary_projected_hamiltonian(state, ham) == pytest.approx(
        block.conj().T @ (hamiltonian @ block), abs=1e-10
    )
    expected_energies = [
        np.vdot(vector, hamiltonian @ vector).real / np.vdot(vector, vector).real
        for vector in block.T
    ]
    assert boundary_block_energies(state, ham, max_bond=16) == pytest.approx(
        expected_energies, abs=1e-10
    )

    rotated, metrics = ritz_rotate_block(state, ham)
    rotated_block = dense_block(rotated)
    assert rotated_block.conj().T @ rotated_block == pytest.approx(
        np.eye(2), abs=1e-10
    )
    dense_metrics = dense_subspace_residuals(rotated, hamiltonian)
    assert dense_metrics["energies"] == pytest.approx(metrics["energies"], abs=1e-10)


def test_dense_ritz_is_invariant_to_block_gauge():
    state = _product_block()
    gauged = rotate_block(state, _block_gauge())
    ham = tfi_ham(2, 2, g=2.7)
    hamiltonian = sparse_hamiltonian(ham, 2, 2)

    rotated, metrics = dense_ritz_rotate_block(state, hamiltonian)
    gauged_rotated, gauged_metrics = dense_ritz_rotate_block(gauged, hamiltonian)

    assert gauged_metrics["energies"] == pytest.approx(
        metrics["energies"], abs=1e-10
    )
    assert gauged_metrics["residual_norms"] == pytest.approx(
        metrics["residual_norms"], abs=1e-10
    )
    overlap = dense_block(rotated).conj().T @ dense_block(gauged_rotated)
    assert np.abs(overlap) == pytest.approx(np.eye(2), abs=1e-9)


def test_boundary_ritz_is_invariant_to_block_gauge():
    state = _product_block()
    gauged = rotate_block(state, _block_gauge())
    ham = tfi_ham(2, 2, g=2.7)

    rotated, metrics = ritz_rotate_block(state, ham)
    gauged_rotated, gauged_metrics = ritz_rotate_block(gauged, ham)

    assert gauged_metrics["energies"] == pytest.approx(
        metrics["energies"], abs=1e-10
    )
    assert max(metrics["projected_residual_norms"]) < 1e-10
    assert max(gauged_metrics["projected_residual_norms"]) < 1e-10
    overlap = dense_block(rotated).conj().T @ dense_block(gauged_rotated)
    assert np.abs(overlap) == pytest.approx(np.eye(2), abs=1e-9)
