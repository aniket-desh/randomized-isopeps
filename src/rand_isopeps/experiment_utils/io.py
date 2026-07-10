"""small file IO helpers for experiments."""

from __future__ import annotations

import csv
import os
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
    """A per-process-unique run slug: timestamp + PID. The PID matters -- Slurm array tasks that
    start in the same wall-clock second would otherwise collide on the same output path and
    interleave their CSV appends (observed 2026-07-09: exp09 tasks 2 & 3 merged a row)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


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


class IncrementalCsvWriter:
    """Append rows to a CSV as a sweep runs, flushing after each batch.

    A long parallel sweep that crashes (e.g. the 108 GB OOM) loses everything when
    the CSV is only written at the end. Used as a context manager, this writes the
    header on the first non-empty batch and flushes each write, so a crash keeps all
    completed trials on disk::

        with IncrementalCsvWriter(path) as w:
            for batch in run_parallel_stream(trial, tasks, workers):
                w.write(batch)

    The fieldnames are fixed from the first batch's first row, so every row must
    share the same keys (true for these experiments).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None
        self._writer = None

    def __enter__(self) -> "IncrementalCsvWriter":
        ensure_dir(self.path.parent)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        return self

    def write(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(rows[0].keys()))
            self._writer.writeheader()
        self._writer.writerows(rows)
        self._fh.flush()

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
