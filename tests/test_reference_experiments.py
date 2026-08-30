from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import pytest

from experiments.paper_campaign.reference_manifests import _dmrg_task
from rand_isopeps.campaign.checkpoint import load_checkpoint
from rand_isopeps.campaign.reference_experiments import _target_sectors, run_reference


def _task(method, *, states=1, residual_required=True):
    return {
        "task_id": "reference-test",
        "problem": {
            "hamiltonian": "heis",
            "lx": 1,
            "ly": 2,
            "states": states,
        },
        "method": method,
        "measurement": {
            "residual_required": residual_required,
            "residual_tolerance": 1e-8,
            "overlap_tolerance": 1e-8,
        },
        "seeds": {"problem": 1},
        "backend": "numpy",
    }


def test_exact_reference_targets_each_requested_total_sz_sector():
    result = run_reference(_task({
        "name": "exact_diagonalization",
        "target_sectors": [
            {"state_index": 0, "target_sz": 0},
            {"state_index": 1, "target_sz": 1},
        ],
        "sector_tolerance": 1e-10,
    }, states=2))[0]

    assert result["validation_passed"] is True
    assert result["reference_source"] == "exact_sparse_sector_targeted"
    assert result["reference_metadata"]["symmetry_sector"] == [0.0, 1.0]
    assert [row["total_sz"] for row in result["records"]] == [0.0, 1.0]
    assert result["energies"] == pytest.approx([-3.0, 1.0], abs=1e-10)


def test_empty_target_sector_contract_means_unrestricted():
    assert _target_sectors({}, {"target_sectors": []}, 2) is None


def test_dmrg_reference_caller_can_skip_exact_residual_materialization():
    result = run_reference(_task({
        "name": "dmrg_reference",
        "bond_dims": [2],
        "max_sweeps": 2,
        "restarts": 1,
        "target_sectors": [{"state_index": 0, "target_sz": 1}],
        "sector_penalty_shift": 20.0,
        "sector_tolerance": 1e-8,
        "residual_max_bond": None,
        "residual_max_bytes": None,
    }, residual_required=False))[0]

    assert result["validation_passed"] is True
    assert result["residual_required"] is False
    assert result["residual_norms"] == [None]
    assert result["records"][0]["residual_method"] == "not_computed"
    assert result["records"][0]["total_sz"] == pytest.approx(1.0, abs=1e-10)


def test_nonsector_block_reference_manifest_bounds_projector_states():
    task = _dmrg_task(
        0,
        6,
        6,
        "tfim@3.5",
        2,
        1000,
        convergence_pair=(500, 1000),
    )

    assert task["method"]["projector_state_bond"] == 64
    assert task["method"]["projector_state_tolerance"] == 1e-4


def test_reference_checkpoint_resumes_after_a_completed_sweep(tmp_path):
    task = _task({
        "name": "dmrg_reference",
        "bond_dims": [2],
        "max_sweeps": 2,
        "restarts": 2,
        "target_sectors": [{"state_index": 0, "target_sz": 1}],
        "sector_penalty_shift": 20.0,
        "sector_tolerance": 1e-8,
        "residual_max_bond": None,
        "residual_max_bytes": None,
    }, residual_required=False)
    checkpoint_path = tmp_path / "reference.pkl"
    calls = 0

    def stop_after_first_sweep():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(InterruptedError, match="dmrg sweep"):
        run_reference(
            task,
            checkpoint_path=checkpoint_path,
            stop_requested=stop_after_first_sweep,
        )

    checkpoint = load_checkpoint(checkpoint_path, task["task_id"])
    assert checkpoint["iteration"] == 0
    assert checkpoint["dmrg_restart"]["state_index"] == 0
    assert checkpoint["dmrg_restart"]["candidates"] == []
    active = checkpoint["dmrg_restart"]["active_restart"]
    assert active["completed_sweeps"] == 1
    assert len(active["energy_history"]) == 1
    assert active["direction_history"] == ["R"]
    assert active["bond_dimension_history"] == [2]

    result = run_reference(task, checkpoint_path=checkpoint_path)[0]

    assert result["validation_passed"] is True
    completed = load_checkpoint(checkpoint_path, task["task_id"])
    assert "dmrg_restart" not in completed
