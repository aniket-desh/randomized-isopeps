"""atomic trajectory checkpoints for resumable array tasks."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    """atomically replace a trusted local checkpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=out.parent,
        prefix=f".{out.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return out


def load_checkpoint(path: str | Path, task_id: str) -> dict | None:
    """load a task checkpoint and reject cross-configuration resumes."""
    source = Path(path)
    if not source.exists():
        return None
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("task_id") != task_id:
        raise ValueError("checkpoint does not match the selected task")
    return payload
