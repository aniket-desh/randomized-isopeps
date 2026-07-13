"""Versioned, conflict-safe row storage for resumable paper experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def canonical_json(value) -> str:
    """Stable JSON encoding used by configuration and row hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_commit(root: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_identity(
    config: dict,
    seed_hierarchy: dict,
    *,
    method: str,
    schema_version: str,
    root: str | Path | None = None,
) -> dict[str, str]:
    """Metadata and a deterministic resume key for one fully specified row."""
    commit = git_commit(root)
    config_hash = content_hash(config)
    key_payload = {
        "schema_version": schema_version,
        "git_commit": commit,
        "config_hash": config_hash,
        "seed_hierarchy": seed_hierarchy,
        "method": method,
    }
    return {
        "run_uuid": str(uuid.uuid4()),
        "run_key": content_hash(key_payload),
        "config_hash": config_hash,
        "git_commit": commit,
        "schema_version": schema_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": str(os.getpid()),
        "seed_hierarchy": canonical_json(seed_hierarchy),
    }


def validate_unique_rows(rows: list[dict[str, object]], key: str = "run_key") -> dict[str, dict]:
    """Deduplicate identical rows and reject a key with conflicting payloads."""
    unique: dict[str, dict] = {}
    for raw in rows:
        row = {str(k): str(v) for k, v in raw.items()}
        run_key = row.get(key, "")
        if not run_key:
            raise ValueError(f"row is missing {key!r}")
        old = unique.get(run_key)
        if old is not None and canonical_json(old) != canonical_json(row):
            raise ValueError(f"conflicting duplicate payload for {key}={run_key}")
        unique[run_key] = row
    return unique


class VersionedCsvStore:
    """Append-only CSV with full-key resume and conflicting-duplicate refusal."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.rows: dict[str, dict] = {}
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
            self.rows = validate_unique_rows(existing)
            old_fields = list(existing[0]) if existing else self.fieldnames
            if old_fields != self.fieldnames:
                raise ValueError(
                    f"schema columns changed for {self.path}; use a new schema version/path"
                )

    def contains(self, run_key: str) -> bool:
        return run_key in self.rows

    def append(self, row: dict[str, object]) -> bool:
        missing = [field for field in self.fieldnames if field not in row]
        extra = [field for field in row if field not in self.fieldnames]
        if missing or extra:
            raise ValueError(f"row/schema mismatch: missing={missing}, extra={extra}")
        normalized = {field: str(row[field]) for field in self.fieldnames}
        run_key = normalized["run_key"]
        old = self.rows.get(run_key)
        if old is not None:
            if canonical_json(old) != canonical_json(normalized):
                raise ValueError(f"conflicting duplicate payload for run_key={run_key}")
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)
            handle.flush()
        self.rows[run_key] = normalized
        return True


def write_manifest(path: str | Path, manifest: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_hash"] = content_hash(payload)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out
