"""deterministic jsonl manifests for independent experiment tasks."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path


required_fields = (
    "schema_version",
    "experiment",
    "backend",
    "dtype",
    "problem",
    "method",
    "seeds",
    "measurement",
)


def canonical_json(value) -> str:
    """return a stable compact json encoding."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@lru_cache(maxsize=1)
def runtime_source_fingerprint() -> str:
    """hash the python sources imported by campaign jobs."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def compute_task_id(task: Mapping) -> str:
    """hash every task field except its derived identifier."""
    payload = {key: value for key, value in task.items() if key != "task_id"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_task(task: Mapping) -> None:
    """reject incomplete tasks and unsafe blocked-task declarations."""
    missing = [field for field in required_fields if field not in task]
    if missing:
        raise ValueError(f"task is missing required fields: {missing}")
    for field in ("problem", "method", "seeds", "measurement"):
        if not isinstance(task[field], Mapping):
            raise ValueError(f"task field {field!r} must be an object")
    requirements = task.get("requirements", [])
    if not isinstance(requirements, list) or not all(
        isinstance(value, str) and value for value in requirements
    ):
        raise ValueError("task requirements must be a list of nonempty strings")
    if task.get("blocked"):
        if not requirements:
            raise ValueError("a blocked task must name at least one requirement")
        if not task.get("blocked_reason"):
            raise ValueError("a blocked task must include blocked_reason")


def finalize_task(task: Mapping) -> dict:
    """validate a task and attach its deterministic identifier."""
    out = copy.deepcopy(dict(task))
    validate_task(out)
    expected = compute_task_id(out)
    supplied = out.get("task_id")
    if supplied is not None and supplied != expected:
        raise ValueError(f"task_id mismatch: expected {expected}, found {supplied}")
    out["task_id"] = expected
    return out


def _set_path(value: dict, dotted_path: str, replacement) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid configuration path: {dotted_path!r}")
    current = value
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise KeyError(f"configuration path does not exist: {dotted_path!r}")
        current = current[part]
    if parts[-1] not in current:
        raise KeyError(f"configuration path does not exist: {dotted_path!r}")
    current[parts[-1]] = replacement


def _get_path(value: Mapping, dotted_path: str):
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"configuration path does not exist: {dotted_path!r}")
        current = current[part]
    return current


def grid_product(base: Mapping, axes: Mapping[str, Sequence]) -> list[dict]:
    """expand a full product over dotted configuration paths."""
    items = [(path, tuple(values)) for path, values in axes.items()]
    if any(not values for _, values in items):
        raise ValueError("grid axes must be nonempty")
    rows = []
    for values in itertools.product(*(values for _, values in items)):
        row = copy.deepcopy(dict(base))
        for (path, _), replacement in zip(items, values):
            _set_path(row, path, replacement)
        rows.append(row)
    return rows


def one_at_a_time(base: Mapping, axes: Mapping[str, Sequence]) -> list[dict]:
    """return the baseline and every single-axis deviation from it."""
    rows = [copy.deepcopy(dict(base))]
    for path, values in axes.items():
        baseline = _get_path(base, path)
        for replacement in values:
            if replacement == baseline:
                continue
            row = copy.deepcopy(dict(base))
            _set_path(row, path, replacement)
            rows.append(row)
    return rows


def manifest_hash(source: str | Path | Sequence[Mapping]) -> str:
    """hash a manifest path or task sequence independently of whitespace."""
    tasks = read_manifest(source) if isinstance(source, (str, Path)) else source
    finalized = [finalize_task(task) for task in tasks]
    return hashlib.sha256(canonical_json(finalized).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_manifest(path: str | Path, tasks: Sequence[Mapping]) -> Path:
    """write a validated manifest atomically, one task per line."""
    finalized = [finalize_task(task) for task in tasks]
    identifiers = [task["task_id"] for task in finalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("manifest contains duplicate tasks")
    text = "".join(canonical_json(task) + "\n" for task in finalized)
    out = Path(path)
    _atomic_text(out, text)
    return out


def read_manifest(path: str | Path) -> list[dict]:
    """read and validate every task in a jsonl manifest."""
    tasks = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            tasks.append(finalize_task(raw))
    identifiers = [task["task_id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("manifest contains duplicate tasks")
    return tasks


def load_manifest(path: str | Path) -> list[dict]:
    """alias with the name used by campaign runners."""
    return read_manifest(path)


def select_task(path: str | Path, index: int, *, allow_blocked: bool = False) -> dict:
    """select one zero-based task and refuse blocked work by default."""
    if index < 0:
        raise IndexError("task index must be nonnegative")
    selected = None
    with Path(path).open(encoding="utf-8") as handle:
        current = -1
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            current += 1
            if current != index:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            selected = finalize_task(raw)
            break
    if selected is None:
        raise IndexError(f"task index {index} is outside the manifest")
    if selected.get("blocked") and not allow_blocked:
        requirements = ", ".join(selected.get("requirements", []))
        raise RuntimeError(
            f"task {selected['task_id']} is blocked by {requirements}: "
            f"{selected['blocked_reason']}"
        )
    return selected
