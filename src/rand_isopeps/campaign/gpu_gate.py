"""fail-closed validation for promoting accuracy tasks to cupy."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .aggregate import MissingDataError, field, finite_float
from .manifest import manifest_hash, required_fields


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def campaign_code_revision() -> str:
    """fingerprint the kernels and evolution paths promoted by the gate."""
    package = Path(__file__).resolve().parents[1]
    paths = {
        package / "backend.py",
        package / "campaign" / "gpu_pilot.py",
        package / "column" / "bounded_residual.py",
        package / "column" / "operator.py",
        package / "compression" / "mpo_mps_absorb.py",
        package / "real_isotns" / "column_bridge.py",
        package / "real_isotns" / "physics_loop.py",
        package / "real_isotns" / "tebd2.py",
    }
    paths.update((package / "campaign").glob("physics_*.py"))
    paths.update((package / "real_isotns").glob("block_*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _current_device() -> dict:
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime = int(cp.cuda.runtime.runtimeGetVersion())
    return {
        "name": str(name),
        "compute_capability": [
            int(properties.get("major", -1)),
            int(properties.get("minor", -1)),
        ],
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": runtime,
    }


def _numbers(value) -> list[float]:
    values = (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
    out = [finite_float(item) for item in values]
    return [item for item in out if item is not None]


def _scope_pilot_records(rows: Iterable[Mapping]) -> list[dict]:
    return [
        dict(row)
        for row in rows
        if row.get("experiment") == "gpu_pilot"
        or (
            row.get("experiment") == "physics"
            and field(row, "problem.study", row.get("study")) == "gpu_pilot"
        )
    ]


def _expected_task_coverage(
    records: list[dict],
    expected_tasks: Iterable[Mapping] | None,
) -> dict:
    if expected_tasks is None:
        return {
            "records": records,
            "expected_rows": None,
            "expected_ids": None,
            "expected_ids_hash": None,
            "manifest_hash": None,
        }
    expected_rows = [dict(task) for task in expected_tasks]
    expected_ids = {str(task["task_id"]) for task in expected_rows}
    present = {
        str(row.get("task_id"))
        for row in records
        if str(row.get("task_id")) in expected_ids
    }
    missing = expected_ids - present
    if missing:
        raise MissingDataError(f"missing results for {len(missing)} gpu pilot tasks")
    complete_manifest = all(
        all(name in task for name in required_fields) for task in expected_rows
    )
    return {
        "records": [
            row for row in records if str(row.get("task_id")) in expected_ids
        ],
        "expected_rows": expected_rows,
        "expected_ids": expected_ids,
        "expected_ids_hash": hashlib.sha256(
            _canonical(sorted(expected_ids)).encode("utf-8")
        ).hexdigest(),
        "manifest_hash": manifest_hash(expected_rows) if complete_manifest else None,
    }


def _campaign_schema_versions(
    records: list[dict], expected_rows: list[dict] | None
) -> list[str]:
    if expected_rows is not None and all(
        all(name in task for name in required_fields) for task in expected_rows
    ):
        return sorted({str(task["schema_version"]) for task in expected_rows})
    return sorted({str(row.get("schema_version", "unknown")) for row in records})


def _validated_kernels(records: list[dict]) -> tuple[list[dict], list[float]]:
    kernels = [
        row
        for row in records
        if row.get("experiment") == "gpu_pilot" and row.get("backend") == "cupy"
    ]
    if not kernels:
        raise MissingDataError("missing cupy matrix-free gpu pilot records")
    kernel_errors = [
        error for row in kernels for error in _numbers(row.get("parity_error"))
    ]
    if not kernel_errors or any(row.get("parity_passed") is not True for row in kernels):
        raise RuntimeError("a matrix-free gpu pilot did not pass parity")
    incomplete_power = [
        row
        for row in kernels
        if int(row.get("n_power", 0)) > 0
        and (
            row.get("parity_scope") != "full_power_chain"
            or int(row.get("parity_matrix_products", 0))
            != 1 + 2 * int(row["n_power"])
        )
    ]
    if incomplete_power:
        raise RuntimeError("a powered gpu pilot lacks full power-chain parity evidence")
    return kernels, kernel_errors


def _physics_pair_key(row: Mapping) -> tuple:
    return (
        _canonical(row.get("problem", {})),
        _canonical(row.get("task_method", row.get("method_config", {}))),
        field(row, "seeds.problem", None),
        row.get("iteration"),
    )


def _validated_physics_pairs(records: list[dict]) -> tuple[list[dict], list[float]]:
    physics = [
        row
        for row in records
        if row.get("experiment") == "physics"
        and field(row, "problem.study", row.get("study")) == "gpu_pilot"
    ]
    pairs = defaultdict(dict)
    for row in physics:
        pairs[_physics_pair_key(row)][str(row.get("backend"))] = row
    matched = [pair for pair in pairs.values() if {"numpy", "cupy"} <= set(pair)]
    cupy_count = sum("cupy" in pair for pair in pairs.values())
    if not matched or len(matched) != cupy_count:
        raise MissingDataError("every cupy peps pilot row needs its paired numpy row")
    differences = []
    for pair in matched:
        for metric in (
            "energies",
            "ground_energy_errors",
            "state_infidelity_to_full_trotter",
            "state_infidelity_to_exact_evolution",
        ):
            left = _numbers(pair["numpy"].get(metric))
            right = _numbers(pair["cupy"].get(metric))
            if len(left) != len(right):
                raise RuntimeError(f"cpu/gpu pilot metric mismatch for {metric}")
            differences.extend(abs(a - b) for a, b in zip(left, right))
    return matched, differences


def _device_evidence(kernels: list[dict]) -> list[dict]:
    devices = {
        _canonical(
            {
                "name": row.get("device_name", "unknown"),
                "compute_capability": row.get("compute_capability"),
                "cupy_version": row.get("cupy_version"),
                "cuda_runtime_version": row.get("cuda_runtime_version"),
                "cuda_driver_version": row.get("cuda_driver_version"),
            }
        )
        for row in kernels
    }
    return [json.loads(value) for value in sorted(devices)]


def validate_gpu_pilot(
    rows: Iterable[Mapping],
    *,
    tolerance: float = 1e-10,
    expected_tasks: Iterable[Mapping] | None = None,
) -> dict:
    """require device-kernel parity and paired cpu/gpu peps trajectories."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coverage = _expected_task_coverage(_scope_pilot_records(rows), expected_tasks)
    records = coverage["records"]
    failures = [row for row in records if row.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"{len(failures)} gpu pilot records failed")
    schema_versions = _campaign_schema_versions(records, coverage["expected_rows"])
    kernels, kernel_errors = _validated_kernels(records)
    matched, differences = _validated_physics_pairs(records)
    max_kernel = max(kernel_errors, default=0.0)
    max_physics = max(differences, default=0.0)
    if max_kernel > tolerance or max_physics > tolerance:
        raise RuntimeError(
            f"gpu parity exceeded {tolerance:.3e}: "
            f"kernel={max_kernel:.3e}, physics={max_physics:.3e}"
        )
    expected_ids = coverage["expected_ids"]
    return {
        "schema_version": "gpu_gate_v1",
        "passed": True,
        "tolerance": float(tolerance),
        "kernel_rows": len(kernels),
        "physics_pairs": len(matched),
        "max_kernel_parity_error": float(max_kernel),
        "max_physics_parity_error": float(max_physics),
        "expected_tasks": None if expected_ids is None else len(expected_ids),
        "expected_task_ids": None if expected_ids is None else sorted(expected_ids),
        "expected_task_ids_hash": coverage["expected_ids_hash"],
        "pilot_manifest_hash": coverage["manifest_hash"],
        "campaign_schema_versions": schema_versions,
        "campaign_code_revision": campaign_code_revision(),
        "devices": _device_evidence(kernels),
    }


def require_gpu_gate(task: Mapping, path: str | Path | None) -> None:
    """reject non-pilot cupy work unless a validated gate artifact is supplied."""
    if task.get("backend") != "cupy":
        return
    if (
        task.get("experiment") == "gpu_pilot"
        and field(task, "problem.study", None) != "gpu_crossover"
    ) or field(task, "problem.study", None) == "gpu_pilot":
        return
    if path is None:
        raise RuntimeError("cupy task requires RAND_ISOPEPS_GPU_GATE_PATH")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "gpu_gate_v1" or payload.get("passed") is not True:
        raise RuntimeError("gpu gate artifact is missing a passing gpu_gate_v1 result")
    if payload.get("campaign_code_revision") != campaign_code_revision():
        raise RuntimeError("gpu gate was produced by a different campaign code revision")
    if str(task.get("schema_version")) not in payload.get("campaign_schema_versions", []):
        raise RuntimeError("gpu gate does not cover this campaign schema version")
    current = _current_device()
    covered = any(
        device.get("name") == current["name"]
        and device.get("compute_capability") == current["compute_capability"]
        and device.get("cupy_version") == current["cupy_version"]
        and int(device.get("cuda_runtime_version", -1))
        == current["cuda_runtime_version"]
        for device in payload.get("devices", [])
    )
    if not covered:
        raise RuntimeError(
            "gpu gate does not cover the current device, CuPy, and CUDA runtime"
        )
