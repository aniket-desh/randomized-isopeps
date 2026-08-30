"""atomic immutable result records and replaceable checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .manifest import canonical_json, finalize_task


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


def _atomic_create_text(path: Path, text: str) -> bool:
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
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            if path.read_text(encoding="utf-8") == text:
                return False
            raise ValueError(f"conflicting result already exists: {path}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _records_text(records: Sequence[Mapping]) -> str:
    return "".join(canonical_json(dict(record)) + "\n" for record in records)


def write_records(path: str | Path, records: Sequence[Mapping]) -> bool:
    """write immutable jsonl results; identical retries are no-ops."""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("a result file must contain at least one record")
    if any("task_id" not in row for row in rows):
        raise ValueError("every result record must include task_id")
    task_ids = {row["task_id"] for row in rows}
    if len(task_ids) != 1:
        raise ValueError("one result file cannot mix task ids")
    text = _records_text(rows)
    out = Path(path)
    return _atomic_create_text(out, text)


def write_record(path: str | Path, record: Mapping) -> bool:
    """write one immutable result record."""
    return write_records(path, [record])


def read_records(path: str | Path) -> list[dict]:
    """read jsonl result records."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on result line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"result line {line_number} is not an object")
            rows.append(row)
    return rows


def record_is_complete(path: str | Path, task_id: str | None = None) -> bool:
    """return true when a result exists and every row completed successfully."""
    out = Path(path)
    if not out.exists():
        return False
    rows = read_records(out)
    if not rows or any(row.get("status") != "ok" for row in rows):
        return False
    return task_id is None or all(row.get("task_id") == task_id for row in rows)


def replace_checkpoint(path: str | Path, checkpoint: Mapping) -> Path:
    """atomically replace a task checkpoint with its latest state."""
    out = Path(path)
    _atomic_text(out, canonical_json(dict(checkpoint)) + "\n")
    return out


def write_task_record(
    output_root: str | Path,
    manifest_id: str,
    task: Mapping,
    rows: Sequence[Mapping],
) -> Path:
    """write one task under ``<root>/<manifest>/tasks/<task_id>.jsonl``."""
    finalized = finalize_task(task)
    task_id = finalized["task_id"]
    normalized = []
    for raw in rows:
        row = dict(raw)
        supplied = row.setdefault("task_id", task_id)
        if supplied != task_id:
            raise ValueError("result task_id does not match its manifest task")
        normalized.append(row)
    path = Path(output_root) / str(manifest_id) / "tasks" / f"{task_id}.jsonl"
    write_records(path, normalized)
    return path
