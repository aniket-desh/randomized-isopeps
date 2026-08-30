import json
import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

from experiments.paper_campaign.physics_manifests import physics_task
from rand_isopeps.campaign import finalize_task
from rand_isopeps.campaign.checkpoint import load_checkpoint
from rand_isopeps.campaign.physics_experiments import (
    _block_energy_record,
    _dense_initial,
    _dense_reorthogonalize,
    _energy_record,
    _eigenvalue_error_fields,
    _external_references,
    _initial_block,
    _measurement_convergence,
    _peps_measurement,
    _references,
    _uses_block_path,
    run_physics,
)
from rand_isopeps.physics.block_measurements import boundary_block_gram
from rand_isopeps.physics.block_state import block_gram, dense_block


def test_eigenvalue_errors_preserve_signed_relative_definition():
    fields = _eigenvalue_error_fields([-9.0, -7.5], [-10.0, -8.0])

    assert fields["eigenvalue_errors"] == [1.0, 0.5]
    assert fields["relative_eigenvalue_errors"] == [-0.1, -0.0625]
    assert fields["absolute_relative_eigenvalue_errors"] == [0.1, 0.0625]


def test_dektor_single_state_uses_the_shared_block_path():
    assert _uses_block_path({"states": 1, "study": "dektor_reproduction"})
    assert _uses_block_path({"states": 2, "study": "bond_sweep"})
    assert not _uses_block_path({"states": 1, "study": "bond_sweep"})


def test_dense_first_order_block_task_uses_the_order_one_runner():
    task = finalize_task(physics_task(
        991,
        0,
        (2, 2),
        "tfim@3.5",
        "dense_first_order",
        4,
        4,
        [[0.01, 1]],
        p=2,
        trotter_order=1,
        study="block_correctness",
        problem_overrides={"regime": "dense_oracle"},
    ))

    rows = run_physics(task)

    assert rows[-1]["trotter_order"] == 1
    assert len(rows[-1]["energies"]) == 2


def test_manifest_reference_preserves_explicit_source():
    values, source = _references(
        None,
        {
            "reference_energies": [-42.0, -40.0],
            "reference_source": "dektor_table_2",
        },
        {},
        2,
    )

    assert values == [-42.0, -40.0]
    assert source == "dektor_table_2"


def test_initial_block_center_is_the_global_orthogonality_center():
    state = _initial_block(
        {
            "lx": 2,
            "ly": 2,
            "states": 2,
            "bond": 2,
            "chi": 4,
            "eta": 4,
            "initialization_ndis": 0,
        },
        seed=7,
    )
    center = block_gram(state)
    boundary = boundary_block_gram(state)

    assert state.center == (0, 0)
    assert center == pytest.approx(np.eye(2), abs=1e-11)
    assert boundary == pytest.approx(center, abs=1e-11)


def test_parent_sized_initialization_nests_dense_subspaces():
    common = {
        "lx": 2,
        "ly": 2,
        "bond": 2,
        "chi": 4,
        "eta": 4,
        "initialization_ndis": 0,
        "initialization_parent_states": 3,
    }
    blocks = {
        states: dense_block(_initial_block({**common, "states": states}, seed=23))
        for states in (1, 2, 3)
    }
    parent = blocks[3]
    for states in (1, 2):
        child = blocks[states]
        child_projector = child @ child.conj().T
        parent_prefix = parent[:, :states]
        prefix_projector = parent_prefix @ parent_prefix.conj().T
        assert child_projector == pytest.approx(prefix_projector, abs=1e-11)


def test_initial_block_rejects_parent_smaller_than_requested_block():
    with pytest.raises(ValueError, match="at least states"):
        _initial_block(
            {
                "lx": 2,
                "ly": 2,
                "states": 3,
                "initialization_parent_states": 2,
                "bond": 2,
                "chi": 4,
                "eta": 4,
                "initialization_ndis": 0,
            },
            seed=23,
        )


def test_dektor_heisenberg_rejects_unrestricted_reference(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps({"records": [{
            "hamiltonian": "heis",
            "lx": 6,
            "ly": 6,
            "energies": [-1.0, -0.5],
            "validation_passed": True,
            "reference_metadata": {"symmetry_sector": "unrestricted"},
        }]}),
        encoding="utf-8",
    )
    problem = {
        "hamiltonian": "heis",
        "lx": 6,
        "ly": 6,
        "study": "dektor_reproduction",
    }

    with pytest.raises(ValueError, match="total-Sz sectors"):
        _external_references(path, problem, 2)


def test_dektor_heisenberg_accepts_a_validated_sector_prefix(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps({"records": [{
            "hamiltonian": "heis",
            "lx": 6,
            "ly": 6,
            "energies": [-1.0, -0.5],
            "validation_passed": True,
            "reference_source": "dektor_table_2",
            "reference_metadata": {"symmetry_sector": [0.0, 1.0]},
        }]}),
        encoding="utf-8",
    )
    problem = {
        "hamiltonian": "heis",
        "lx": 6,
        "ly": 6,
        "study": "dektor_reproduction",
    }

    energies, source = _external_references(path, problem, 1)

    assert energies == [-1.0]
    assert source == "dektor_table_2"


def test_external_reference_fails_closed_and_preserves_source(tmp_path):
    path = tmp_path / "references.json"
    record = {
        "hamiltonian": "tfim@3.5",
        "lx": 6,
        "ly": 6,
        "energies": [-4.0],
        "validation_passed": True,
        "reference_source": "dmrg_bond_1000",
    }
    path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    problem = {"hamiltonian": "tfim@3.5", "lx": 6, "ly": 6}

    with pytest.raises(ValueError, match="too few states"):
        _external_references(path, problem, 2)
    energies, source = _external_references(path, problem, 1)
    assert energies == [-4.0]
    assert source == "dmrg_bond_1000"


def test_required_external_reference_rejects_a_partial_artifact(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps({"records": [{
            "hamiltonian": "tfim@3.5",
            "lx": 4,
            "ly": 4,
            "energies": [-4.0],
            "validation_passed": True,
        }]}),
        encoding="utf-8",
    )
    problem = {"hamiltonian": "tfim@3.5", "lx": 2, "ly": 2}

    with pytest.raises(ValueError, match="no matching validated cell"):
        _references(
            object(),
            problem,
            {"reference_path": str(path)},
            1,
            require_external=True,
        )


def test_shared_dense_initializer_densifies_the_same_block(monkeypatch):
    problem = {
        "lx": 2,
        "ly": 2,
        "states": 2,
        "bond": 2,
        "chi": 4,
        "eta": 4,
        "initialization_ndis": 0,
        "initialization": "shared_block_oracle",
    }
    state = _initial_block(problem, seed=11)
    monkeypatch.setattr(
        "rand_isopeps.campaign.physics_block.initial_block",
        lambda _problem, _seed: state,
    )

    found = _dense_initial(problem, 16, {}, seed=11)
    assert found == pytest.approx(dense_block(state), abs=1e-12)


def test_shared_dense_path_uses_qr_without_ritz_rotation():
    state = np.asarray([[1.0, 1.0], [1.0, -2.0], [0.5, 0.25]])
    expected = np.linalg.qr(state, mode="reduced")[0]
    found = _dense_reorthogonalize(
        {"states": 2, "initialization": "shared_block_oracle"},
        np.diag([1.0, 3.0, 7.0]),
        state,
        7.0,
    )
    assert found == pytest.approx(expected, abs=1e-12)


def test_dense_multistate_measurement_is_invariant_to_block_gauge():
    rng = np.random.default_rng(29)
    block = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
    gauge = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    gauge += 2.0 * np.eye(2)
    hamiltonian = np.diag(np.linspace(-2.0, 3.0, 8))

    measured = _energy_record(
        hamiltonian, block, 3.0, [-2.0, -1.0], "test"
    )
    gauged = _energy_record(
        hamiltonian, block @ gauge, 3.0, [-2.0, -1.0], "test"
    )

    assert gauged["energies"] == pytest.approx(measured["energies"], abs=1e-12)
    assert gauged["residual_norms"] == pytest.approx(
        measured["residual_norms"], abs=1e-12
    )
    assert measured["measurement_method"] == "dense_generalized_ritz"


def test_large_measurements_use_2b_and_expose_convergence(monkeypatch):
    monkeypatch.setattr(
        "rand_isopeps.real_isotns.tebd2.energy",
        lambda _state, _ham, max_bond: {16: -3.0, 32: -3.0002}[max_bond],
    )
    measurement = {
        "measurement_bonds": [16, 32],
        "measurement_convergence_tolerance": 1e-4,
    }
    _, single = _peps_measurement(
        object(), object(), None, None, [-3.1], "test", {}, measurement
    )

    monkeypatch.setattr(
        "rand_isopeps.physics.block_measurements.ritz_rotate_block",
        lambda state, _ham, max_bond, cutoff: (
            state,
            {
                "energies": {
                    16: [-3.0, -2.0],
                    32: [-3.00005, -2.0002],
                }[max_bond],
                "projected_residual_norms": [0.0, 0.0],
            },
        ),
    )
    block_state = type("BlockState", (), {"size": 2})()
    _, block = _block_energy_record(
        block_state,
        object(),
        None,
        [-3.1, -2.1],
        "test",
        {},
        measurement,
    )

    assert single["energies"] == [-3.0002]
    assert single["measurement_converged"] is False
    assert block["energies"] == [-3.00005, -2.0002]
    assert block["measurement_energy_differences"] == pytest.approx(
        [5e-5, 2e-4]
    )
    assert block["measurement_converged"] is False
    assert block["measurement_method"] == "paired_boundary_generalized_ritz"


def test_measurement_convergence_accepts_every_state_within_tolerance():
    result = _measurement_convergence(
        (8, 16), [[-2.0, -1.0], [-2.00001, -1.00002]], 3e-5
    )
    assert result["measurement_converged"] is True


def test_partial_block_checkpoint_preserves_dense_oracles(tmp_path):
    task = finalize_task(physics_task(
        990,
        0,
        (2, 2),
        "tfim@3.5",
        "peps_sketch",
        4,
        4,
        [[0.005, 1]],
        p=2,
        trotter_order=1,
        study="block_correctness",
        method_overrides={
            "ell": 4,
            "eta": 4,
            "kappa": 4,
            "chi_sk": 4,
            "gate_bond": None,
            "cutoff": 0.0,
            "absorption_bond": None,
            "absorption_cutoff": 0.0,
        },
    ))
    checkpoint_path = tmp_path / "block.pkl"
    calls = 0

    def stop_after_first_column():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(InterruptedError, match="block-sweep column"):
        run_physics(
            task,
            checkpoint_path=checkpoint_path,
            stop_requested=stop_after_first_column,
        )

    checkpoint = load_checkpoint(checkpoint_path, task["task_id"])
    assert checkpoint["partial_iteration"]["progress"]["phase"] == "vertical"
    assert checkpoint["oracle_exact"] == pytest.approx(
        checkpoint["oracle_trotter"], abs=1e-12
    )

    rows = run_physics(task, checkpoint_path=checkpoint_path)

    assert len(rows) == 2
    assert "projector_frobenius_error_to_exact_evolution" in rows[-1]
    assert "projector_frobenius_error_to_ordered_first_order" in rows[-1]
