"""small file IO helpers for experiments."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def experiment_suite_dir(experiment_file: str | Path) -> Path:
    """Directory of the experiment suite that owns ``experiment_file``.

    Experiment scripts live at ``<suite>/experiments/<name>.py``, so the suite
    root is the parent of the ``experiments`` directory. Anchoring output paths
    here keeps results under e.g. ``synthetic/outputs`` no matter what working
    directory the script is launched from.
    """
    return Path(experiment_file).resolve().parents[1]


def output_paths(experiment_file: str | Path, slug: str) -> tuple[str, str]:
    """Return ``(csv_path, pdf_path)`` under the suite's ``outputs`` tree."""
    base = experiment_suite_dir(experiment_file)
    csv_path = base / "outputs" / "data" / f"{slug}.csv"
    pdf_path = base / "outputs" / "figures" / f"{slug}.pdf"
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
