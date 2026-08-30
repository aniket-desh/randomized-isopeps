from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics.dense import (
    bond_first_order_imaginary_step,
    bond_hamiltonians,
)
from rand_isopeps.real_isotns.physics_loop import tebd_iteration
from rand_isopeps.real_isotns.tebd2 import tfi_ham


def _product_peps(lx: int, ly: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sites = {}
    for site in np.ndindex(lx, ly):
        value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        sites[site] = value / np.linalg.norm(value)
    return qtn.PEPS.product_state(sites)


def _dense(psi):
    vector = np.asarray(psi.to_dense()).reshape(-1)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("direction", [-1, 1])
def test_first_order_full_peps_matches_dense_gate_order(direction):
    psi = _product_peps(2, 2, seed=8)
    ham = tfi_ham(2, 2, g=2.1)
    found, metrics = tebd_iteration(
        psi,
        ham,
        0.02,
        direction=direction,
        column_backend="none",
        gate_options={"max_bond": None, "cutoff": 0.0},
        trotter_order=1,
    )
    expected = bond_first_order_imaginary_step(
        bond_hamiltonians(ham, 2, 2),
        _dense(psi),
        0.02,
        ly=2,
        direction=direction,
    )
    fidelity = abs(np.vdot(expected, _dense(found))) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert metrics["trotter_order"] == 1


def test_trotter_order_is_validated():
    psi = _product_peps(1, 2)
    with pytest.raises(ValueError, match="trotter_order"):
        tebd_iteration(psi, tfi_ham(1, 2), 0.1, trotter_order=3)
