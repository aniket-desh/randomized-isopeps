from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qu = pytest.importorskip("quimb")
qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.block_state import BlockPeps, dense_block
from rand_isopeps.real_isotns.block_gate import apply_block_gate


def _random_block(seed=8):
    rng = np.random.default_rng(seed)
    peps = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=seed)
    center = (0, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = rng.standard_normal((*tensor.shape, 2))
    data = data + 1j * rng.standard_normal(data.shape)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    return BlockPeps(peps, "alpha", center)


@pytest.mark.parametrize("move_to", [(0, 0), (1, 0)])
def test_full_rank_block_gate_matches_dense_evolution_and_moves_alpha(move_to):
    state = _random_block()
    before = dense_block(state)
    where = ((0, 0), (1, 0))
    local = qu.pauli("X") & qu.pauli("X")
    local = local + 0.7 * (qu.pauli("Z") & qu.pauli("Z"))
    gate_matrix = qu.expm(-0.03 * local)
    full_gate = qu.pkron(gate_matrix, [2] * 4, [0, 2])

    info = {}
    evolved = apply_block_gate(
        state, gate_matrix, where, move_to=move_to, info=info
    )
    full = apply_block_gate(
        state, gate_matrix, where, move_to=move_to, reduced=False
    )

    assert evolved.center == move_to
    assert "alpha" in evolved.peps[evolved.peps.site_tag_id.format(*move_to)].inds
    assert dense_block(evolved) == pytest.approx(full_gate @ before, abs=1e-10)
    assert dense_block(evolved) == pytest.approx(dense_block(full), abs=1e-10)
    assert dense_block(state) == pytest.approx(before, abs=1e-12)
    assert info["discarded_weight"] == pytest.approx(0.0, abs=1e-12)


def test_truncated_block_gate_reports_its_discarded_weight():
    state = _random_block()
    gate = qu.expm(-0.03 * (qu.pauli("X") & qu.pauli("X")))
    info = {}
    full_info = {}

    evolved = apply_block_gate(
        state,
        gate,
        ((0, 0), (1, 0)),
        move_to=(1, 0),
        max_bond=1,
        cutoff=0.0,
        info=info,
    )
    full = apply_block_gate(
        state,
        gate,
        ((0, 0), (1, 0)),
        move_to=(1, 0),
        max_bond=1,
        cutoff=0.0,
        info=full_info,
        reduced=False,
    )

    assert evolved.center == (1, 0)
    assert info["bond_dimension"] == 1
    assert np.isfinite(info["discarded_weight"])
    assert info["discarded_weight"] > 0.0
    assert info["discarded_weight"] == pytest.approx(
        full_info["discarded_weight"], rel=1e-12
    )
    assert dense_block(evolved) == pytest.approx(dense_block(full), abs=1e-10)
