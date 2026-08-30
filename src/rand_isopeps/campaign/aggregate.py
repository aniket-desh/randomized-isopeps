"""load immutable campaign records and compute robust paired summaries."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np


class MissingDataError(RuntimeError):
    """report that a requested campaign summary has no usable records."""


_missing = object()


def field(row: Mapping, path: str, default=_missing):
    """read a dotted field from nested campaign data."""
    current = row
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is _missing:
                raise KeyError(path)
            return default
        current = current[part]
    return current


def finite_float(value) -> float | None:
    """convert a scalar to a finite float or return none."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _normalized_record(raw: Mapping, source: Path) -> dict:
    row = dict(raw)
    problem = row.get("problem")
    if isinstance(problem, Mapping):
        for key, value in problem.items():
            row.setdefault(str(key), value)
    measurement = row.get("measurement")
    if isinstance(measurement, Mapping):
        for key, value in measurement.items():
            row.setdefault(str(key), value)
    method = row.get("method_config")
    if not isinstance(method, Mapping) and isinstance(row.get("task_method"), Mapping):
        method = row["task_method"]
    if not isinstance(method, Mapping) and isinstance(row.get("method"), Mapping):
        method = row["method"]
    if isinstance(method, Mapping):
        row["method_config"] = dict(method)
        row["method"] = str(method.get("name", row.get("method", "unknown")))
        row.setdefault("method_label", str(method.get("label", row["method"])))
    row["_source_path"] = str(source)
    return row


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object json in {path} at line {line_number}")
            rows.append(row)
    return rows


def read_result_records(root: str | Path) -> list[dict]:
    """recursively load task jsonl and reject conflicting task copies."""
    source = Path(root)
    paths = [source] if source.is_file() else sorted(source.rglob("*.jsonl"))
    seen: dict[str, str] = {}
    records = []
    for path in paths:
        rows = _jsonl(path)
        if not rows or not any("status" in row for row in rows):
            continue
        if any("status" not in row for row in rows):
            raise ValueError(f"mixed result and non-result rows in {path}")
        task_ids = {str(row.get("task_id", "")) for row in rows}
        if "" in task_ids or len(task_ids) != 1:
            raise ValueError(f"result file must contain exactly one task id: {path}")
        task_id = next(iter(task_ids))
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        previous = seen.get(task_id)
        if previous is not None:
            if previous != payload:
                raise ValueError(f"conflicting immutable records for task {task_id}")
            continue
        seen[task_id] = payload
        records.extend(_normalized_record(row, path) for row in rows)
    if not records:
        raise MissingDataError(f"no immutable task records found under {source}")
    return records


def require_rows(
    rows: Iterable[Mapping],
    predicate: Callable[[Mapping], bool],
    description: str,
) -> list[dict]:
    """select records or raise an explicit missing-data error."""
    selected = [dict(row) for row in rows if predicate(row)]
    if not selected:
        raise MissingDataError(f"missing data for {description}")
    return selected


def _key(row: Mapping, key: str | Sequence[str]):
    if isinstance(key, str):
        return field(row, key)
    return tuple(field(row, part) for part in key)


def median_bands(
    rows: Iterable[Mapping],
    group_key: str | Sequence[str],
    x_key: str,
    value_key: str,
    *,
    low: float = 10.0,
    high: float = 90.0,
    groups: Sequence | None = None,
) -> dict:
    """compute median and percentile bands without weighting failed rows."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        value = finite_float(field(row, value_key, None))
        x = finite_float(field(row, x_key, None))
        if value is None or x is None:
            continue
        try:
            group = _key(row, group_key)
        except KeyError:
            continue
        buckets[(group, x)].append(value)
    order = list(groups) if groups is not None else sorted(
        {group for group, _ in buckets}, key=str
    )
    out = {}
    for group in order:
        xs = sorted(x for candidate, x in buckets if candidate == group)
        if not xs:
            continue
        out[group] = {
            "x": xs,
            "median": [float(np.median(buckets[(group, x)])) for x in xs],
            "low": [float(np.percentile(buckets[(group, x)], low)) for x in xs],
            "high": [float(np.percentile(buckets[(group, x)], high)) for x in xs],
            "n": [len(buckets[(group, x)]) for x in xs],
        }
    return out


def paired_median_bands(
    rows: Iterable[Mapping],
    group_key: str | Sequence[str],
    x_key: str,
    value_key: str,
    *,
    pair_key: str | Sequence[str] = "seeds.problem",
    low: float = 10.0,
    high: float = 90.0,
    groups: Sequence | None = None,
) -> dict:
    """summarize only common paired instances at each x value."""
    observations = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        value = finite_float(field(row, value_key, None))
        x = finite_float(field(row, x_key, None))
        if value is None or x is None:
            continue
        try:
            group = _key(row, group_key)
            pair = _key(row, pair_key)
        except KeyError:
            continue
        observations[(x, group, pair)].append(value)
    order = list(groups) if groups is not None else sorted(
        {group for _, group, _ in observations}, key=str
    )
    xs = sorted({x for x, _, _ in observations})
    values = defaultdict(list)
    for x in xs:
        present = [
            group
            for group in order
            if any(key_x == x and key_group == group for key_x, key_group, _ in observations)
        ]
        if not present:
            continue
        pair_sets = [
            {
                pair
                for key_x, key_group, pair in observations
                if key_x == x and key_group == group
            }
            for group in present
        ]
        common = set.intersection(*pair_sets)
        for group in present:
            for pair in sorted(common, key=str):
                values[(group, x)].append(
                    float(np.median(observations[(x, group, pair)]))
                )
    paired_rows = []
    for (group, x), samples in values.items():
        paired_rows.extend(
            {"group": group, "x": x, "value": value, "status": "ok"}
            for value in samples
        )
    return median_bands(
        paired_rows,
        "group",
        "x",
        "value",
        low=low,
        high=high,
        groups=order,
    )


def isometry_summary(
    rows: Iterable[Mapping],
    group_keys: Sequence[str] = ("method_label", "route"),
    value_key: str = "max_local_isometry_defect",
) -> list[dict]:
    """compute median, 95th percentile, and maximum isometry defects."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        value = finite_float(field(row, value_key, None))
        if value is None:
            continue
        try:
            key = tuple(field(row, name) for name in group_keys)
        except KeyError:
            continue
        buckets[key].append(value)
    summaries = []
    for key in sorted(buckets, key=str):
        values = np.asarray(buckets[key], dtype=float)
        summaries.append(
            {
                **dict(zip(group_keys, key)),
                "n": int(values.size),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95.0)),
                "max": float(np.max(values)),
            }
        )
    return summaries


def trajectory_groups(
    rows: Iterable[Mapping],
    group_keys: Sequence[str] = ("task_id",),
    step_key: str = "iteration",
) -> dict[tuple, list[dict]]:
    """group successful trajectory records and enforce one row per step."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        if finite_float(field(row, step_key, None)) is None:
            continue
        try:
            key = tuple(field(row, name) for name in group_keys)
        except KeyError:
            continue
        groups[key].append(dict(row))
    out = {}
    for key, values in groups.items():
        ordered = sorted(values, key=lambda row: float(field(row, step_key)))
        steps = [float(field(row, step_key)) for row in ordered]
        if len(steps) != len(set(steps)):
            raise ValueError(f"duplicate trajectory step for group {key}")
        out[key] = ordered
    return out


def _converged_bond_pair(by_bond: Mapping, tolerance: float):
    bonds = sorted(by_bond)
    if len(bonds) < 2:
        return None
    lower_bond, upper_bond = bonds[-2:]
    lower_row, lower = by_bond[lower_bond]
    upper_row, upper = by_bond[upper_bond]
    if len(lower) != len(upper):
        return None
    configured = upper_row.get("energy_convergence_pair")
    if configured is not None and [lower_bond, upper_bond] != [
        int(value) for value in configured
    ]:
        return None
    configured_tolerance = finite_float(
        upper_row.get("energy_convergence_tolerance")
    )
    if configured_tolerance is not None and configured_tolerance <= 0.0:
        return None
    effective_tolerance = min(
        tolerance,
        configured_tolerance if configured_tolerance is not None else tolerance,
    )
    difference = max(abs(a - b) for a, b in zip(lower, upper))
    if difference > effective_tolerance:
        return None
    return (
        lower_bond,
        upper_bond,
        lower_row,
        upper_row,
        upper,
        difference,
        effective_tolerance,
    )


def _paper_reference_evidence(
    row: Mapping,
    *,
    required_bond: int,
) -> dict | None:
    records = row.get("records")
    states = int(row.get("states", 0))
    if not isinstance(records, Sequence) or len(records) != states:
        return None
    if any(
        not isinstance(record, Mapping)
        or record.get("solver_converged") is not True
        for record in records
    ):
        return None
    actual_bonds = [finite_float(record.get("max_bond")) for record in records]
    if any(
        value is None or not value.is_integer() or value < int(required_bond)
        for value in actual_bonds
    ):
        return None
    overlaps = [finite_float(record.get("max_previous_overlap")) for record in records]
    overlap_tolerance = finite_float(row.get("overlap_tolerance"))
    if overlap_tolerance is None or any(value is None for value in overlaps):
        return None
    maximum_overlap = max(overlaps, default=0.0)
    if maximum_overlap > overlap_tolerance:
        return None
    infidelities = [
        finite_float(value)
        for record in records
        for value in record.get("projector_compression_infidelities", ())
    ]
    if any(value is None for value in infidelities):
        return None
    metadata = row.get("reference_metadata", {})
    projector_tolerance = finite_float(
        metadata.get("projector_state_tolerance")
        if isinstance(metadata, Mapping) else None
    )
    if projector_tolerance is None:
        return None
    maximum_infidelity = max(infidelities, default=0.0)
    if maximum_infidelity > projector_tolerance:
        return None
    return {
        "max_previous_overlap": float(maximum_overlap),
        "overlap_tolerance": float(overlap_tolerance),
        "max_projector_compression_infidelity": float(maximum_infidelity),
        "projector_state_tolerance": float(projector_tolerance),
        "minimum_actual_max_bond": min(int(value) for value in actual_bonds),
        "required_max_bond": int(required_bond),
    }


def _filter_reference_rows(
    rows: Iterable[Mapping],
    *,
    expected_task_ids: Iterable[str] | None,
    expected_campaign_revision: str | None,
    expected_source_fingerprint: str | None,
) -> list[Mapping]:
    task_ids = (
        {str(value) for value in expected_task_ids}
        if expected_task_ids is not None
        else None
    )
    selected = []
    for row in rows:
        if row.get("experiment") != "reference":
            continue
        if task_ids is not None and str(row.get("task_id")) not in task_ids:
            continue
        if (
            expected_campaign_revision is not None
            and row.get("campaign_revision") != expected_campaign_revision
        ):
            continue
        if (
            expected_source_fingerprint is not None
            and row.get("runtime_source_fingerprint")
            != expected_source_fingerprint
        ):
            continue
        selected.append(row)
    return selected


def _reference_provenance(rows: Sequence[Mapping]) -> tuple[object, object]:
    revisions = {
        (row.get("campaign_revision"), row.get("runtime_source_fingerprint"))
        for row in rows
    }
    if len(revisions) > 1:
        raise ValueError(
            "reference aggregation cannot mix campaign or runtime-source revisions"
        )
    return next(iter(revisions)) if revisions else (None, None)


def _valid_reference_row(row: Mapping) -> bool:
    if row.get("status") != "ok" or row.get("validation_passed") is not True:
        return False
    return not row.get("require_sector_validation") or (
        row.get("sector_validation_passed") is True
    )


def _reference_method(row: Mapping) -> Mapping:
    method = row.get("method_config", row.get("task_method", {}))
    return method if isinstance(method, Mapping) else {}


def _reference_energies(row: Mapping) -> list[float] | None:
    energies = row.get("energies")
    if not isinstance(energies, Sequence) or not energies:
        return None
    values = [finite_float(value) for value in energies]
    if any(value is None for value in values):
        return None
    return values


def _reference_cell(row: Mapping, states: int) -> tuple[str, int, int, int]:
    return (
        str(row.get("hamiltonian")),
        int(row.get("lx", -1)),
        int(row.get("ly", -1)),
        int(row.get("states", states)),
    )


def _paper_reference_groups(rows: Sequence[Mapping]) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        if not _valid_reference_row(row):
            continue
        energies = _reference_energies(row)
        bonds = _reference_method(row).get("bond_dims", ())
        if energies is None or not bonds:
            continue
        tier = str(row.get("reference_tier", "dmrg"))
        cell = _reference_cell(row, len(energies))
        groups[(*cell, tier)][max(int(value) for value in bonds)] = (
            dict(row),
            energies,
        )
    return groups


def _source_task_ids(*rows: Mapping) -> list:
    return [
        value
        for value in dict.fromkeys(row.get("task_id") for row in rows)
        if value is not None
    ]


def _source_manifest_ids(*rows: Mapping) -> list[str]:
    return sorted(
        {
            str(row["manifest_id"])
            for row in rows
            if row.get("manifest_id") is not None
        }
    )


def _paper_reference_candidate(
    key: tuple,
    by_bond: Mapping,
    *,
    bond_tolerance: float,
    campaign_revision,
    source_fingerprint,
) -> dict | None:
    pair = _converged_bond_pair(by_bond, bond_tolerance)
    if pair is None or key[-1] != "paper_energy":
        return None
    lower_bond, upper_bond, lower_row, upper_row, energies, difference, tolerance = pair
    lower_evidence = _paper_reference_evidence(lower_row, required_bond=lower_bond)
    evidence = _paper_reference_evidence(upper_row, required_bond=upper_bond)
    if lower_evidence is None or evidence is None:
        return None
    hamiltonian, lx, ly, states, tier = key
    method = _reference_method(upper_row)
    return {
        "hamiltonian": hamiltonian,
        "lx": lx,
        "ly": ly,
        "states": states,
        "energies": energies,
        "residual_norms": upper_row.get("residual_norms"),
        "validated_bonds": [lower_bond, upper_bond],
        "validated_actual_max_bonds": [
            lower_evidence["minimum_actual_max_bond"],
            evidence["minimum_actual_max_bond"],
        ],
        "max_bond_difference": float(difference),
        "bond_tolerance": float(tolerance),
        "validation_passed": True,
        "reference_source": upper_row.get("reference_source", "dmrg_reference"),
        "reference_metadata": upper_row.get("reference_metadata", {}),
        "reference_tier": tier,
        "reference_method": method,
        "validation_contract": "nested_bond_energy_orthogonality",
        **evidence,
        "target_sectors": upper_row.get(
            "target_sectors", method.get("target_sectors", [])
        ),
        "sector_expectations": upper_row.get("sector_expectations"),
        "sector_variances": upper_row.get("sector_variances"),
        "sector_validation_passed": upper_row.get("sector_validation_passed"),
        "source_task_ids": _source_task_ids(lower_row, upper_row),
        "source_manifest_ids": _source_manifest_ids(lower_row, upper_row),
        "campaign_revision": campaign_revision,
        "runtime_source_fingerprint": source_fingerprint,
    }


def _paper_references(
    rows: Sequence[Mapping],
    *,
    bond_tolerance: float,
    campaign_revision,
    source_fingerprint,
) -> list[dict]:
    candidates = []
    for key, by_bond in sorted(_paper_reference_groups(rows).items()):
        candidate = _paper_reference_candidate(
            key,
            by_bond,
            bond_tolerance=bond_tolerance,
            campaign_revision=campaign_revision,
            source_fingerprint=source_fingerprint,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _exact_reference(
    row: Mapping,
    *,
    bond_tolerance: float,
    campaign_revision,
    source_fingerprint,
) -> tuple[tuple[str, int, int, int], dict] | None:
    if not _valid_reference_row(row):
        return None
    method = _reference_method(row)
    energies = _reference_energies(row)
    if method.get("name") != "exact_diagonalization" or energies is None:
        return None
    key = _reference_cell(row, len(energies))
    record = {
        "hamiltonian": key[0],
        "lx": key[1],
        "ly": key[2],
        "states": key[3],
        "energies": energies,
        "residual_norms": row.get("residual_norms"),
        "validated_bonds": [],
        "max_bond_difference": 0.0,
        "bond_tolerance": float(bond_tolerance),
        "validation_passed": True,
        "reference_source": row.get(
            "reference_source", "exact_diagonalization"
        ),
        "reference_metadata": row.get("reference_metadata", {}),
        "reference_tier": str(row.get("reference_tier", "exact")),
        "reference_method": method,
        "validation_contract": row.get("validation_contract"),
        "target_sectors": row.get(
            "target_sectors", method.get("target_sectors", [])
        ),
        "sector_expectations": row.get("sector_expectations"),
        "sector_variances": row.get("sector_variances"),
        "sector_validation_passed": row.get("sector_validation_passed"),
        "source_task_ids": [row.get("task_id")],
        "source_manifest_ids": _source_manifest_ids(row),
        "campaign_revision": campaign_revision,
        "runtime_source_fingerprint": source_fingerprint,
    }
    return key, record


def _exact_references(
    rows: Sequence[Mapping],
    *,
    bond_tolerance: float,
    campaign_revision,
    source_fingerprint,
) -> dict[tuple[str, int, int, int], dict]:
    exact = {}
    for row in rows:
        selected = _exact_reference(
            row,
            bond_tolerance=bond_tolerance,
            campaign_revision=campaign_revision,
            source_fingerprint=source_fingerprint,
        )
        if selected is not None:
            key, record = selected
            exact[key] = record
    return exact


def validated_references(
    rows: Iterable[Mapping],
    *,
    bond_tolerance: float = 1e-5,
    expected_task_ids: Iterable[str] | None = None,
    expected_campaign_revision: str | None = None,
    expected_source_fingerprint: str | None = None,
) -> dict:
    """select exact or energy-and-orthogonality validated references."""
    if bond_tolerance <= 0.0:
        raise ValueError("bond_tolerance must be positive")
    reference_rows = _filter_reference_rows(
        rows,
        expected_task_ids=expected_task_ids,
        expected_campaign_revision=expected_campaign_revision,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    campaign_revision, source_fingerprint = _reference_provenance(reference_rows)
    records = _paper_references(
        reference_rows,
        bond_tolerance=bond_tolerance,
        campaign_revision=campaign_revision,
        source_fingerprint=source_fingerprint,
    )
    exact = _exact_references(
        reference_rows,
        bond_tolerance=bond_tolerance,
        campaign_revision=campaign_revision,
        source_fingerprint=source_fingerprint,
    )
    records = [
        record
        for record in records
        if (
            record["hamiltonian"],
            record["lx"],
            record["ly"],
            record["states"],
        )
        not in exact
    ]
    records.extend(exact[key] for key in sorted(exact))
    if not records:
        raise MissingDataError(
            "no reference cell passed exact calibration or the declared nested-bond "
            "energy, orthogonality, projector, and sector contract"
        )
    return {"schema_version": "validated_references_v1", "records": records}
