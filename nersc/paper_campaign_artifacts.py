"""preflight validation for campaign dependency artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.paper_campaign.build_manifests import builders, stamp_family_tasks
from experiments.paper_campaign.manifest_common import schema_version
from experiments.paper_campaign.table_2_validation import (
    table_2_exact_relative_tolerance,
    validate_table_2_exact_references,
)
from rand_isopeps.campaign import (
    finalize_task,
    manifest_hash,
    read_manifest,
    runtime_source_fingerprint,
)
from rand_isopeps.campaign.aggregate import MissingDataError, read_result_records
from rand_isopeps.campaign.gpu_gate import campaign_code_revision


def _json_object(path: Path, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a json object: {path}")
    return payload


def validate_gpu_gate_artifact(path: Path) -> None:
    """validate promotion evidence that does not require a live GPU."""
    payload = _json_object(path, "gpu gate artifact")
    pilot_tasks, _ = stamp_family_tasks("gpu_pilot", builders["gpu_pilot"]())
    expected = [finalize_task(task) for task in pilot_tasks]
    expected_ids = sorted(task["task_id"] for task in expected)
    checks = (
        payload.get("schema_version") == "gpu_gate_v1",
        payload.get("passed") is True,
        payload.get("campaign_code_revision") == campaign_code_revision(),
        schema_version in payload.get("campaign_schema_versions", ()),
        payload.get("pilot_manifest_hash") == manifest_hash(expected),
        payload.get("expected_task_ids") == expected_ids,
        payload.get("expected_tasks") == len(expected_ids),
        isinstance(payload.get("kernel_rows"), int) and payload["kernel_rows"] > 0,
        isinstance(payload.get("physics_pairs"), int)
        and payload["physics_pairs"] > 0,
        bool(payload.get("devices")),
    )
    if not all(checks):
        raise ValueError(
            "gpu gate does not match the complete current pilot manifest and code"
        )


def _required_reference_tasks(parts: list[dict]) -> list[dict]:
    return [
        task
        for part in parts
        if part.get("requires_reference_artifact")
        for task in read_manifest(part["manifest"])
        if "reference_artifact" in task.get("requirements", ())
    ]


def _table_2_tasks(parts: list[dict]) -> list[dict]:
    return [
        task
        for part in parts
        for task in read_manifest(part["manifest"])
        if "table_2" in task.get("problem", {}).get("dektor_panels", ())
    ]


def requires_reference_preflight(parts: list[dict]) -> bool:
    """report whether selected work depends on external or table 2 references."""
    return any(part["requires_reference_artifact"] for part in parts) or bool(
        _table_2_tasks(parts)
    )


def _matches_cell(record: dict, problem: dict) -> bool:
    try:
        return (
            record.get("hamiltonian") == problem.get("hamiltonian")
            and int(record.get("lx", -1)) == int(problem["lx"])
            and int(record.get("ly", -1)) == int(problem["ly"])
            and record.get("validation_passed") is True
        )
    except (TypeError, ValueError):
        return False


def _has_finite_energies(record: dict, states: int) -> bool:
    energies = record.get("energies")
    if not isinstance(energies, list) or len(energies) < states:
        return False
    try:
        return all(math.isfinite(float(value)) for value in energies[:states])
    except (TypeError, ValueError):
        return False


def _has_large_reference_contract(record: dict) -> bool:
    if record.get("validation_contract") != "nested_bond_energy_orthogonality":
        return False
    bonds = record.get("validated_bonds")
    if not isinstance(bonds, list) or len(bonds) != 2:
        return False
    actual_bonds = record.get("validated_actual_max_bonds")
    if not isinstance(actual_bonds, list) or len(actual_bonds) != 2:
        return False
    try:
        declared = [int(value) for value in bonds]
        actual = [int(value) for value in actual_bonds]
        required_bond = int(record["required_max_bond"])
        minimum_actual = int(record["minimum_actual_max_bond"])
        bond_difference = float(record["max_bond_difference"])
        bond_tolerance = float(record["bond_tolerance"])
        overlap = float(record["max_previous_overlap"])
        overlap_tolerance = float(record["overlap_tolerance"])
        projector_error = float(record["max_projector_compression_infidelity"])
        projector_tolerance = float(record["projector_state_tolerance"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (
        bond_difference,
        bond_tolerance,
        overlap,
        overlap_tolerance,
        projector_error,
        projector_tolerance,
    )
    return (
        all(math.isfinite(value) for value in values)
        and all(found >= required for found, required in zip(actual, declared))
        and required_bond == declared[-1]
        and minimum_actual == actual[-1]
        and bond_tolerance > 0.0
        and overlap_tolerance >= 0.0
        and projector_tolerance >= 0.0
        and bond_difference <= bond_tolerance
        and overlap <= overlap_tolerance
        and projector_error <= projector_tolerance
    )


def _has_heisenberg_sectors(record: dict, states: int, dektor: bool) -> bool:
    sectors = record.get("target_sectors", ())
    try:
        targets = [
            int(sector["target_sz"])
            for sector in sectors
            if isinstance(sector, dict) and "target_sz" in sector
        ]
    except (TypeError, ValueError):
        return False
    if targets[:states] != list(range(states)):
        return False
    if record.get("sector_validation_passed") is not True:
        return False
    if not dektor:
        return True
    measured = record.get("reference_metadata", {}).get("symmetry_sector")
    expected = [float(index) for index in range(states)]
    return isinstance(measured, list) and measured[:states] == expected


def validate_reference_artifact(path: Path, parts: list[dict]) -> None:
    """require a validated reference for every selected physics cell."""
    payload = _json_object(path, "reference artifact")
    if payload.get("schema_version") != "validated_references_v1":
        raise ValueError("reference artifact has an unsupported schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("reference artifact records must be a list")

    _, expected_revision = stamp_family_tasks("references", builders["references"]())
    revisions = {
        record.get("campaign_revision")
        for record in records
        if isinstance(record, dict)
    }
    source_revisions = {
        record.get("runtime_source_fingerprint")
        for record in records
        if isinstance(record, dict)
    }
    if revisions != {expected_revision} or source_revisions != {
        runtime_source_fingerprint()
    }:
        raise ValueError("reference artifact was produced by a different campaign revision")

    table_2_tasks = _table_2_tasks(parts)
    if table_2_tasks:
        tolerances = {
            float(task.get("measurement", {}).get(
                "published_reference_relative_tolerance",
                table_2_exact_relative_tolerance,
            ))
            for task in table_2_tasks
        }
        if len(tolerances) != 1:
            raise ValueError("table 2 tasks declare inconsistent exact-reference tolerances")
        validate_table_2_exact_references(
            records,
            relative_tolerance=next(iter(tolerances)),
        )

    missing = []
    for task in _required_reference_tasks(parts):
        problem = task["problem"]
        states = int(problem.get("states", 1))
        candidates = [
            record
            for record in records
            if isinstance(record, dict)
            and _matches_cell(record, problem)
            and _has_finite_energies(record, states)
        ]
        if int(problem["lx"]) * int(problem["ly"]) > 16:
            candidates = [
                record for record in candidates
                if _has_large_reference_contract(record)
            ]
        if str(problem.get("hamiltonian")) == "heis":
            dektor = problem.get("study") == "dektor_reproduction"
            candidates = [
                record
                for record in candidates
                if _has_heisenberg_sectors(record, states, dektor)
            ]
        if not candidates:
            missing.append(
                f"{problem.get('hamiltonian')} {problem['lx']}x{problem['ly']} "
                f"p={states}"
            )
    if missing:
        cells = ", ".join(sorted(set(missing)))
        raise ValueError(f"reference artifact lacks validated coverage for: {cells}")


def validate_reference_stage1_results(output_root: Path) -> None:
    """require every current exact calibration before paper-scale DMRG."""
    tasks, revision = stamp_family_tasks("references", builders["references"]())
    expected = [
        finalize_task(task)
        for task in tasks
        if int(task["measurement"].get("stage", 0)) == 1
    ]
    expected_by_id = {task["task_id"]: task for task in expected}
    try:
        records = read_result_records(output_root)
    except MissingDataError as exc:
        raise ValueError("reference stage 1 has no result records") from exc
    rows = [row for row in records if row.get("task_id") in expected_by_id]
    by_id = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(expected_by_id) - set(by_id))
    if missing:
        raise ValueError(f"reference stage 1 is missing {len(missing)} current tasks")

    source = runtime_source_fingerprint()
    for task_id, task in expected_by_id.items():
        row = by_id[task_id]
        valid = (
            row.get("status") == "ok"
            and row.get("validation_passed") is True
            and row.get("campaign_revision") == revision
            and row.get("runtime_source_fingerprint") == source
        )
        if task["measurement"].get("require_sector_validation"):
            valid &= row.get("sector_validation_passed") is True
        if not valid:
            raise ValueError(f"reference stage 1 task did not validate: {task_id}")
