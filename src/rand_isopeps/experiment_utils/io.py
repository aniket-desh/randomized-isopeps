"""small file IO helpers for experiments."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def repo_root() -> Path:
    """Best-effort repo root: walk up from CWD to a ``pyproject.toml`` / ``.git``.

    Experiments are launched from the repo root (``python experiments/<suite>/
    scripts/<name>.py``), so anchoring outputs here keeps every suite's raw data
    and figures under one ignored ``outputs/`` tree regardless of the exact CWD.
    Falls back to the CWD if no marker is found.
    """
    cur = Path.cwd().resolve()
    for parent in (cur, *cur.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return cur


def output_paths(suite: str, slug: str) -> tuple[str, str]:
    """Return ``(csv_path, pdf_path)`` under ``<repo>/outputs/<suite>/{data,figures}``.

    ``suite`` is the experiment-suite name, e.g. ``"synthetic_kernels"`` or
    ``"real_moses_move"``. The whole ``outputs/`` tree is gitignored.
    """
    base = repo_root() / "outputs" / suite
    csv_path = base / "data" / f"{slug}.csv"
    pdf_path = base / "figures" / f"{slug}.pdf"
    return str(csv_path), str(pdf_path)


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
