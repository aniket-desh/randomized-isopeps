import json

import pytest

from rand_isopeps.campaign.aggregate import (
    MissingDataError,
    isometry_summary,
    paired_median_bands,
    read_result_records,
    trajectory_groups,
    validated_references,
)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_recursive_loader_retains_failures_and_deduplicates(tmp_path):
    rows = [
        {
            "task_id": "a",
            "status": "ok",
            "value": 1.0,
            "problem": {"lx": 4},
            "task_method": {"name": "peps_sketch"},
        },
        {
            "task_id": "a",
            "status": "failed",
            "error": "expected",
            "problem": {"lx": 4},
            "task_method": {"name": "peps_sketch"},
        },
    ]
    _write(tmp_path / "one" / "tasks" / "a.jsonl", rows)
    _write(tmp_path / "copy" / "tasks" / "a.jsonl", rows)
    _write(tmp_path / "manifest.jsonl", [{"task_id": "manifest", "method": {}}])
    loaded = read_result_records(tmp_path)
    assert [row["status"] for row in loaded] == ["ok", "failed"]
    assert loaded[0]["lx"] == 4
    assert loaded[0]["method"] == "peps_sketch"
    assert all(row["_source_path"].endswith("a.jsonl") for row in loaded)


def test_recursive_loader_rejects_conflicting_task_ids(tmp_path):
    _write(tmp_path / "one.jsonl", [{"task_id": "a", "status": "ok", "value": 1}])
    _write(tmp_path / "two.jsonl", [{"task_id": "a", "status": "ok", "value": 2}])
    with pytest.raises(ValueError, match="conflicting immutable records"):
        read_result_records(tmp_path)


def test_recursive_loader_reports_missing_results(tmp_path):
    with pytest.raises(MissingDataError, match="no immutable task records"):
        read_result_records(tmp_path)


def test_paired_bands_use_only_common_problem_seeds():
    rows = []
    for group, values in {"a": {1: 1.0, 2: 3.0, 3: 100.0}, "b": {1: 2.0, 2: 4.0}}.items():
        for seed, value in values.items():
            rows.append(
                {
                    "status": "ok",
                    "group": group,
                    "x": 1,
                    "value": value,
                    "seeds": {"problem": seed},
                }
            )
    bands = paired_median_bands(rows, "group", "x", "value", groups=("a", "b"))
    assert bands["a"]["median"] == [2.0]
    assert bands["b"]["median"] == [3.0]
    assert bands["a"]["n"] == [2]
    assert bands["a"]["low"][0] < bands["a"]["high"][0]


def test_isometry_and_trajectory_summaries():
    rows = [
        {
            "task_id": "a",
            "status": "ok",
            "method_label": "rmps",
            "route": "right",
            "iteration": step,
            "max_local_isometry_defect": value,
        }
        for step, value in enumerate((1e-12, 2e-12, 3e-12))
    ]
    summary = isometry_summary(rows)
    assert summary[0]["max"] == 3e-12
    assert summary[0]["median"] == 2e-12
    assert [row["iteration"] for row in trajectory_groups(rows)[("a",)]] == [0, 1, 2]

    with pytest.raises(ValueError, match="duplicate trajectory step"):
        trajectory_groups([rows[0], dict(rows[0])])


def test_validated_references_require_two_bonds_and_agreement():
    rows = []
    for bond, energy in ((128, -1.0), (256, -1.0 - 1e-7)):
        rows.append({
            "experiment": "reference",
            "task_id": str(bond),
            "status": "ok",
            "validation_passed": True,
            "hamiltonian": "tfim@3.5",
            "lx": 4,
            "ly": 4,
            "states": 1,
            "reference_tier": "paper_energy",
            "residual_required": False,
            "energy_convergence_pair": [128, 256],
            "energies": [energy],
            "residual_norms": [None],
            "overlap_tolerance": 1e-6,
            "records": [{
                "solver_converged": True,
                "max_bond": bond,
                "max_previous_overlap": 0.0,
                "projector_compression_infidelities": [],
            }],
            "reference_metadata": {"projector_state_tolerance": 1e-4},
            "method_config": {"bond_dims": [32, bond]},
        })
    payload = validated_references(rows, bond_tolerance=1e-6)
    record = payload["records"][0]
    assert record["validated_bonds"] == [128, 256]
    assert record["residual_norms"] == [None]
    assert record["max_previous_overlap"] == 0.0
    assert record["validated_actual_max_bonds"] == [128, 256]
    assert record["validation_contract"] == "nested_bond_energy_orthogonality"
    assert record["validation_passed"] is True

    rows[-1]["records"][0]["max_bond"] = 255
    with pytest.raises(MissingDataError, match="nested-bond"):
        validated_references(rows, bond_tolerance=1e-6)
    rows[-1]["records"][0]["max_bond"] = 256

    rows[-1]["energies"] = [-1.01]
    with pytest.raises(MissingDataError, match="nested-bond"):
        validated_references(rows, bond_tolerance=1e-6)

    rows[-1]["energies"] = [-1.0 - 1e-7]
    rows[-1]["energy_convergence_tolerance"] = 1e-8
    with pytest.raises(MissingDataError, match="nested-bond"):
        validated_references(rows, bond_tolerance=1e-6)

    with pytest.raises(MissingDataError, match="nested-bond"):
        validated_references(rows[1:], bond_tolerance=1e-6)


def test_validated_references_reject_failed_orthogonality_evidence():
    rows = []
    for bond, energy in ((2500, -1.0), (5000, -1.0 - 1e-7)):
        rows.append({
            "experiment": "reference",
            "task_id": f"paper-{bond}",
            "status": "ok",
            "validation_passed": True,
            "hamiltonian": "tfim@3.5",
            "lx": 12,
            "ly": 12,
            "states": 2,
            "reference_tier": "paper_energy",
            "energy_convergence_pair": [2500, 5000],
            "energies": [energy, energy + 0.2],
            "overlap_tolerance": 1e-6,
            "records": [
                {
                    "solver_converged": True,
                    "max_bond": bond,
                    "max_previous_overlap": 0.0,
                    "projector_compression_infidelities": [],
                },
                {
                    "solver_converged": True,
                    "max_bond": bond,
                    "max_previous_overlap": 1e-3,
                    "projector_compression_infidelities": [1e-5],
                },
            ],
            "reference_metadata": {"projector_state_tolerance": 1e-4},
            "method_config": {"bond_dims": [16, bond]},
        })

    with pytest.raises(MissingDataError, match="orthogonality"):
        validated_references(rows, bond_tolerance=1e-6)


def test_validated_references_accept_residual_gated_exact_result():
    row = {
        "experiment": "reference",
        "task_id": "exact",
        "status": "ok",
        "validation_passed": True,
        "hamiltonian": "heis",
        "lx": 4,
        "ly": 4,
        "states": 2,
        "energies": [-1.0, -0.8],
        "residual_norms": [1e-12, 2e-12],
        "reference_source": "exact_sparse_sector_targeted",
        "reference_metadata": {"symmetry_sector": [0.0, 1.0]},
        "method_config": {"name": "exact_diagonalization"},
    }
    record = validated_references(iter([row]))["records"][0]
    assert record["validated_bonds"] == []
    assert record["source_task_ids"] == ["exact"]
    assert record["reference_source"] == "exact_sparse_sector_targeted"
    assert record["reference_metadata"]["symmetry_sector"] == [0.0, 1.0]


def test_validated_references_reject_mixed_campaign_revisions():
    rows = [
        {
            "experiment": "reference",
            "status": "ok",
            "campaign_revision": revision,
            "runtime_source_fingerprint": "source-a",
        }
        for revision in ("campaign-a", "campaign-b")
    ]

    with pytest.raises(ValueError, match="cannot mix campaign"):
        validated_references(rows)


def test_validated_references_filters_to_the_requested_campaign_revision():
    current = {
        "experiment": "reference",
        "task_id": "current-exact",
        "status": "ok",
        "validation_passed": True,
        "campaign_revision": "campaign-current",
        "runtime_source_fingerprint": "source-current",
        "hamiltonian": "tfim@3.5",
        "lx": 4,
        "ly": 4,
        "states": 1,
        "energies": [-1.0],
        "method_config": {"name": "exact_diagonalization"},
    }
    old = {
        **current,
        "task_id": "old-exact",
        "campaign_revision": "campaign-old",
        "runtime_source_fingerprint": "source-old",
    }

    payload = validated_references(
        [old, current],
        expected_task_ids=["current-exact"],
        expected_campaign_revision="campaign-current",
        expected_source_fingerprint="source-current",
    )

    record = payload["records"][0]
    assert record["source_task_ids"] == ["current-exact"]
    assert record["campaign_revision"] == "campaign-current"
    assert record["runtime_source_fingerprint"] == "source-current"
