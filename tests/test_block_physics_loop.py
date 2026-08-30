from __future__ import annotations

import copy
import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest
import scipy.sparse.linalg as spla

qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.campaign.physics_block import block_oracle_metrics
from rand_isopeps.physics import bond_hamiltonians, exact_imaginary_step
from rand_isopeps.physics.block_measurements import (
    boundary_block_gram,
    orthonormalize_block_from_gram,
)
from rand_isopeps.physics.block_state import BlockPeps, dense_block
from rand_isopeps.physics.dense import (
    block_first_order_imaginary_step,
    normalize_state,
)
from rand_isopeps.real_isotns.block_physics_loop import block_tebd_iteration
from rand_isopeps.real_isotns.tebd2 import tfi_ham


def _product_block(seed=52, lx=2, ly=2):
    rng = np.random.default_rng(seed)
    vectors = {}
    for x in range(lx):
        for y in range(ly):
            vector = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            vectors[(x, y)] = vector / np.linalg.norm(vector)
    peps = qtn.PEPS.product_state(vectors)
    center = (0, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    data = rng.standard_normal((*tensor.shape, 2))
    data = data + 1j * rng.standard_normal(data.shape)
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    state = BlockPeps(peps, "alpha", center)
    return orthonormalize_block_from_gram(state)[0]


def _projector(block):
    basis, _ = np.linalg.qr(block, mode="reduced")
    return basis @ basis.conj().T


def _split_bond_order(lx, ly, direction):
    columns = range(ly) if direction == 1 else range(ly - 1, -1, -1)
    rows = range(lx - 1) if direction == 1 else range(lx - 2, -1, -1)
    vertical = [
        ((row, column), (row + 1, column))
        for column in columns
        for row in rows
    ]

    rotated_columns = range(lx) if direction == 1 else range(lx - 1, -1, -1)
    rotated_rows = (
        range(ly - 1) if direction == 1 else range(ly - 2, -1, -1)
    )
    horizontal = []
    for column in rotated_columns:
        for row in rotated_rows:
            rotated = ((row, column), (row + 1, column))
            horizontal.append(
                tuple((y, ly - 1 - x) for x, y in rotated)
            )
    return vertical + horizontal


def _dense_split_step(block, bonds, tau, lx, ly, direction):
    out = np.asarray(block)
    lookup = {frozenset(where): matrix for where, matrix in bonds.items()}
    for where in _split_bond_order(lx, ly, direction):
        out = spla.expm_multiply((-float(tau)) * lookup[frozenset(where)], out)
    return out


def _options(ell=None):
    return {
        "gate_options": {"max_bond": None, "cutoff": 0.0},
        "column_options": {
            "ell": ell,
            "chi": 64,
            "eta": 64,
            "kappa": 64,
            "chi_sk": 8,
            "cutoff": 0.0,
            "ndis": 0,
            "absorption_max_bond": None,
            "absorption_cutoff": 0.0,
        },
        "gram_options": {"max_bond": None, "cutoff": 0.0},
    }


@pytest.mark.parametrize(
    ("backend", "expected_qr_moves"),
    [("rmps_shared_q", 7), ("local_shared_q", 3)],
)
def test_full_range_block_iteration_matches_dense_split_order(
    backend, expected_qr_moves
):
    lx, ly = 3, 2
    state = _product_block(lx=lx, ly=ly)
    before = dense_block(state)
    ham = tfi_ham(lx, ly, g=2.8)
    tau = 0.012
    reference = _dense_split_step(
        before, bond_hamiltonians(ham, lx, ly), tau, lx, ly, 1
    )

    evolved, metrics = block_tebd_iteration(
        state,
        ham,
        tau,
        direction=1,
        backend=backend,
        rng=np.random.default_rng(17),
        **_options(),
    )

    assert evolved.center == (lx - 1, ly - 1)
    assert metrics["trotter_order"] == 1
    assert metrics["next_direction"] == -1
    assert metrics["gate_order"] == "forward"
    assert metrics["sweep_schedule"] == "alternating_forward_reverse"
    assert metrics["center_qr_moves"] == expected_qr_moves
    assert metrics["gate_count"] == len(_split_bond_order(lx, ly, 1))
    assert np.linalg.norm(_projector(dense_block(evolved)) - _projector(reference)) < 2e-9
    assert boundary_block_gram(evolved) == pytest.approx(np.eye(2), abs=2e-9)

    repeated_reference = _dense_split_step(
        reference, bond_hamiltonians(ham, lx, ly), tau, lx, ly, -1
    )
    repeated, repeated_metrics = block_tebd_iteration(
        evolved,
        ham,
        tau,
        direction=metrics["next_direction"],
        backend=backend,
        rng=np.random.default_rng(18),
        **_options(),
    )
    assert repeated.center == (0, 0)
    assert repeated_metrics["direction"] == -1
    assert repeated_metrics["next_direction"] == 1
    assert repeated_metrics["gate_order"] == "reverse"
    assert np.linalg.norm(
        _projector(dense_block(repeated)) - _projector(repeated_reference)
    ) < 2e-9


def test_rng_state_restart_matches_uninterrupted_second_macrostep():
    state = _product_block(seed=61)
    ham = tfi_ham(2, 2, g=3.1)
    options = _options(ell=2)
    options["column_options"].update({"eta": 2, "kappa": 2, "chi_sk": 2})

    rng = np.random.default_rng(23)
    first, first_record = block_tebd_iteration(
        state, ham, 0.01, direction=1, rng=rng, **options
    )
    saved_state = first.copy()
    saved_rng = copy.deepcopy(rng.bit_generator.state)
    uninterrupted, _ = block_tebd_iteration(
        first,
        ham,
        0.01,
        direction=first_record["next_direction"],
        rng=rng,
        **options,
    )

    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = saved_rng
    resumed, _ = block_tebd_iteration(
        saved_state,
        ham,
        0.01,
        direction=first_record["next_direction"],
        rng=resumed_rng,
        **options,
    )

    assert dense_block(resumed) == pytest.approx(dense_block(uninterrupted), abs=1e-11)
    assert resumed_rng.bit_generator.state == rng.bit_generator.state


def test_column_checkpoint_resume_matches_uninterrupted_iteration():
    state = _product_block(seed=63)
    ham = tfi_ham(2, 2, g=3.1)
    options = _options(ell=2)
    options["column_options"].update({"eta": 2, "kappa": 2, "chi_sk": 2})

    reference_rng = np.random.default_rng(29)
    reference, reference_record = block_tebd_iteration(
        state,
        ham,
        0.01,
        direction=1,
        rng=reference_rng,
        **options,
    )

    interrupted_rng = np.random.default_rng(29)
    saved = {}

    def interrupt(current, progress):
        saved["state"] = current.copy()
        saved["progress"] = copy.deepcopy(progress)
        saved["rng_state"] = copy.deepcopy(interrupted_rng.bit_generator.state)
        raise InterruptedError("column boundary")

    with pytest.raises(InterruptedError, match="column boundary"):
        block_tebd_iteration(
            state,
            ham,
            0.01,
            direction=1,
            rng=interrupted_rng,
            progress_callback=interrupt,
            **options,
        )

    assert saved["progress"]["phase"] == "vertical"
    assert saved["progress"]["pass"]["completed_columns"] == 1
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = saved["rng_state"]
    resumed, resumed_record = block_tebd_iteration(
        saved["state"],
        ham,
        0.01,
        direction=1,
        rng=resumed_rng,
        resume=saved["progress"],
        **options,
    )

    assert dense_block(resumed) == pytest.approx(dense_block(reference), abs=1e-11)
    assert resumed_rng.bit_generator.state == reference_rng.bit_generator.state
    for key in (
        "direction",
        "next_direction",
        "gate_count",
        "center_qr_moves",
        "column_moves",
    ):
        assert resumed_record[key] == reference_record[key]


def test_block_oracle_metrics_match_the_ordered_dense_subspace():
    state = _product_block(seed=64)
    initial = dense_block(state)
    ham = tfi_ham(2, 2, g=2.7)
    bonds = bond_hamiltonians(ham, 2, 2)
    tau = 0.01
    evolved, _ = block_tebd_iteration(
        state,
        ham,
        tau,
        direction=1,
        backend="rmps_shared_q",
        rng=np.random.default_rng(37),
        **_options(),
    )
    ordered = block_first_order_imaginary_step(
        bonds,
        initial,
        tau,
        lx=2,
        ly=2,
        direction=1,
    )
    exact = exact_imaginary_step(
        sum(bonds.values()),
        initial,
        tau,
    )

    metrics = block_oracle_metrics(evolved, exact, ordered)

    assert metrics[
        "projector_frobenius_error_to_ordered_first_order"
    ] < 2e-9
    assert metrics["projector_frobenius_error_to_exact_evolution"] > 0.0
    assert len(metrics["principal_angles_to_ordered_first_order"]) == 2


@pytest.mark.parametrize("direction", [-1, 1])
def test_block_dense_oracle_uses_the_vertical_then_horizontal_order(direction):
    state = dense_block(_product_block(seed=71, lx=3, ly=2))
    bonds = bond_hamiltonians(tfi_ham(3, 2, g=2.4), 3, 2)
    expected = normalize_state(
        _dense_split_step(state, bonds, 0.007, 3, 2, direction)
    )

    found = block_first_order_imaginary_step(
        bonds,
        state,
        0.007,
        lx=3,
        ly=2,
        direction=direction,
    )

    assert found == pytest.approx(expected, abs=1e-12)


def test_full_rank_local_block_iteration_matches_dense_ordered_gate_subspace():
    state = _product_block(seed=68)
    before = dense_block(state)
    ham = tfi_ham(2, 2, g=2.9)
    tau = 0.015
    reference = _dense_split_step(
        before, bond_hamiltonians(ham, 2, 2), tau, 2, 2, 1
    )

    evolved, metrics = block_tebd_iteration(
        state,
        ham,
        tau,
        direction=1,
        backend="local_shared_q",
        rng=np.random.default_rng(19),
        **_options(),
    )

    assert evolved.center == (1, 1)
    assert metrics["backend"] == "local_shared_q"
    assert metrics["columns"][0]["vertical_sweep"] == "up"
    assert np.linalg.norm(
        _projector(dense_block(evolved)) - _projector(reference)
    ) < 2e-9
    assert boundary_block_gram(evolved) == pytest.approx(np.eye(2), abs=2e-9)
