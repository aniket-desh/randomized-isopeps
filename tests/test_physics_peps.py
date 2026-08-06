from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.physics import bond_hamiltonians, bond_trotter_imaginary_step
from rand_isopeps.real_isotns.column_bridge import extract_column, validate_peps_structure
from rand_isopeps.real_isotns.global_move import rmps_column_move
from rand_isopeps.real_isotns.physics_loop import tebd_iteration
from rand_isopeps.real_isotns.tebd2 import tfi_ham


def _unit_vector(psi):
    vector = np.asarray(psi.to_dense()).reshape(-1)
    return vector / np.linalg.norm(vector)


def _product_peps(lx, ly, seed=3):
    rng = np.random.default_rng(seed)
    sites = {}
    for i in range(lx):
        for j in range(ly):
            value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            sites[(i, j)] = value / np.linalg.norm(value)
    return qtn.PEPS.product_state(sites)


def test_full_range_rmps_move_crosses_interior_and_returns_standard_peps():
    original = qtn.PEPS.rand(2, 3, bond_dim=2, phys_dim=2, seed=4)
    reference = _unit_vector(original)
    state = original
    rng = np.random.default_rng(22)
    for j, split in ((0, "right"), (1, "right"), (2, "left"), (1, "left")):
        column = extract_column(state, j, split=split)
        state, metrics = rmps_column_move(
            state,
            j,
            split=split,
            ell=column.n_in,
            eta=64,
            kappa=64,
            chi_sk=3,
            ndis=0,
            absorption_max_bond=None,
            absorption_cutoff=0.0,
            rng=rng,
        )
        validate_peps_structure(state)
        assert metrics["absorption_bond_after"] <= metrics["absorption_bond_before"]
    assert abs(np.vdot(reference, _unit_vector(state))) ** 2 == pytest.approx(1.0, abs=1e-10)


def test_full_rank_local_iteration_matches_untruncated_peps_product():
    psi = _product_peps(2, 2)
    ham = tfi_ham(2, 2, g=2.5)
    full, _ = tebd_iteration(
        psi,
        ham,
        0.01,
        column_backend="none",
        gate_options={"max_bond": None, "cutoff": 0.0},
    )
    dense_full = bond_trotter_imaginary_step(
        bond_hamiltonians(ham, 2, 2), _unit_vector(psi), 0.01, ly=2
    )
    assert abs(np.vdot(dense_full, _unit_vector(full))) ** 2 == pytest.approx(
        1.0, abs=1e-12
    )
    local, metrics = tebd_iteration(
        psi,
        ham,
        0.01,
        column_backend="local",
        gate_options={"max_bond": None, "cutoff": 0.0},
        column_options={
            "chi": 64,
            "eta": 64,
            "cutoff": 0.0,
            "ndis": 0,
            "absorption_max_bond": None,
            "absorption_cutoff": 0.0,
        },
    )
    fidelity = abs(np.vdot(_unit_vector(full), _unit_vector(local))) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-10)
    assert metrics["column_moves"] == 2


def test_truncated_rmps_iteration_is_repeatable_and_respects_absorption_cap():
    psi = _product_peps(2, 3)
    ham = tfi_ham(2, 3, g=3.0)
    options = {
        "ell": 2,
        "eta": 2,
        "kappa": 2,
        "chi_sk": 3,
        "ndis": 0,
        "absorption_max_bond": 3,
        "absorption_cutoff": 1e-10,
    }
    outputs = []
    for _ in range(2):
        state = psi
        rng = np.random.default_rng(7)
        for _ in range(2):
            state, metrics = tebd_iteration(
                state,
                ham,
                0.01,
                column_backend="rmps",
                gate_options={"max_bond": 3, "cutoff": 1e-10},
                column_options=options,
                rng=rng,
            )
            validate_peps_structure(state)
            assert all(
                record["absorption_bond_after"] <= 3
                for record in metrics["columns"]
            )
        outputs.append(_unit_vector(state))
    assert abs(np.vdot(outputs[0], outputs[1])) ** 2 == pytest.approx(1.0, abs=1e-12)
