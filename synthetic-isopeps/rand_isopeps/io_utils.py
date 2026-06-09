"""small file IO helpers for experiments."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out
