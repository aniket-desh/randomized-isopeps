"""single-task dispatcher for local and slurm campaign execution."""

from __future__ import annotations

import math
import os
import signal
from pathlib import Path
from time import monotonic

import numpy as np

from .manifest import finalize_task, runtime_source_fingerprint
from .records import read_records, write_task_record
from .gpu_gate import require_gpu_gate


_stop_requested = False


def _request_stop(_signum, _frame):
    global _stop_requested
    _stop_requested = True


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dispatch(task: dict, checkpoint_path: Path, stop_requested):
    experiment = str(task["experiment"])
    if experiment == "gaussian_limit":
        from .sketches import run_gaussian_limit

        return run_gaussian_limit(task)
    if experiment in {"column_move", "column_moves"}:
        from .column_experiments import run_column_comparison

        return run_column_comparison(task)
    if experiment == "isometry":
        from .column_experiments import run_isometry

        return run_isometry(task)
    if experiment == "physics":
        from .physics_experiments import run_physics

        return run_physics(
            task,
            checkpoint_path=checkpoint_path,
            stop_requested=stop_requested,
        )
    if experiment in {"gpu_pilot", "gpu_crossover"}:
        from .gpu_pilot import run_gpu_pilot

        return run_gpu_pilot(task)
    if experiment == "reference":
        from .reference_experiments import run_reference

        return run_reference(
            task,
            checkpoint_path=checkpoint_path,
            stop_requested=stop_requested,
        )
    raise ValueError(f"unknown campaign experiment: {experiment!r}")


def _task_provenance(task: dict, manifest_id: str) -> dict:
    provenance = {
        "task_id": task["task_id"],
        "manifest_id": str(manifest_id),
        "schema_version": task["schema_version"],
        "experiment": task["experiment"],
        "backend": task["backend"],
        "dtype": task["dtype"],
        "seeds": task["seeds"],
        "problem": task["problem"],
        "task_method": task["method"],
        "measurement": task["measurement"],
        "runtime_source_fingerprint": task["runtime_source_fingerprint"],
        "resources": task.get("resources", {}),
    }
    for field in ("campaign_family", "campaign_revision"):
        if field in task:
            provenance[field] = task[field]
    return provenance


def _failed_result_error(path: Path, task_id: str) -> RuntimeError:
    return RuntimeError(
        f"task {task_id} has an immutable failed result at {path}; "
        "it is terminal for this manifest, so use a new manifest after fixing the task"
    )


def _error_diagnostic(exc: Exception) -> dict:
    message = " ".join(str(exc).split()) or "no error message"
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": message[:2000],
    }


def run_task(
    task: dict,
    *,
    output_root: str | Path,
    manifest_id: str,
    checkpoint_root: str | Path | None = None,
    stop_after_seconds: float | None = None,
) -> Path:
    """run one immutable task, returning only when every result row succeeds."""
    global _stop_requested
    _stop_requested = False
    if stop_after_seconds is not None and stop_after_seconds <= 0.0:
        raise ValueError("stop_after_seconds must be positive")
    deadline = (
        None if stop_after_seconds is None else monotonic() + stop_after_seconds
    )

    def should_stop():
        return _stop_requested or (deadline is not None and monotonic() >= deadline)

    selected = finalize_task(task)
    current_fingerprint = runtime_source_fingerprint()
    if selected.get("runtime_source_fingerprint") != current_fingerprint:
        raise RuntimeError(
            "task runtime_source_fingerprint does not match the current source; "
            "build a new manifest before running it"
        )
    if selected.get("blocked"):
        requirements = ", ".join(selected.get("requirements", []))
        raise RuntimeError(f"blocked task requires {requirements}: {selected['blocked_reason']}")
    require_gpu_gate(selected, os.environ.get("RAND_ISOPEPS_GPU_GATE_PATH"))
    root = Path(output_root)
    result_path = root / str(manifest_id) / "tasks" / f"{selected['task_id']}.jsonl"
    if result_path.exists():
        rows = read_records(result_path)
        matching = rows and all(
            row.get("task_id") == selected["task_id"] for row in rows
        )
        if not matching:
            raise ValueError(
                f"task has a conflicting immutable result with {len(rows)} rows"
            )
        if all(row.get("status") == "ok" for row in rows):
            return result_path
        raise _failed_result_error(result_path, selected["task_id"])
    checkpoint_base = root if checkpoint_root is None else Path(checkpoint_root)
    checkpoint_path = checkpoint_base / str(manifest_id) / "checkpoints" / (
        f"{selected['task_id']}.pkl"
    )
    provenance = _task_provenance(selected, manifest_id)
    old_handlers = {}
    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            old_handlers[number] = signal.signal(number, _request_stop)
    try:
        try:
            raw_rows = _dispatch(selected, checkpoint_path, should_stop)
        except InterruptedError:
            raise
        except Exception as exc:
            record = _json_value({**_error_diagnostic(exc), **provenance})
            result = write_task_record(root, manifest_id, selected, [record])
            checkpoint_path.unlink(missing_ok=True)
            raise _failed_result_error(result, selected["task_id"]) from exc
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    records = []
    for raw in raw_rows:
        payload = dict(raw)
        if "seeds" in payload:
            payload.setdefault("replicate_seeds", payload.pop("seeds"))
        record = {"status": "ok", **payload, **provenance}
        records.append(_json_value(record))
    result = write_task_record(root, manifest_id, selected, records)
    checkpoint_path.unlink(missing_ok=True)
    if any(record.get("status") != "ok" for record in records):
        raise _failed_result_error(result, selected["task_id"])
    return result
