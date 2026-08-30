"""run one zero-based task from a paper campaign manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

repo_root = Path(__file__).resolve().parents[2]
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rand_isopeps.campaign import manifest_hash, select_task


def _array_index(value: int | None) -> int:
    if value is not None:
        return value
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("pass --task-index or set SLURM_ARRAY_TASK_ID")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="optional scratch root for replaceable trajectory checkpoints",
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=float,
        help="checkpoint and exit before the scheduler wall-time limit",
    )
    args = parser.parse_args()

    task = select_task(args.manifest, _array_index(args.task_index))
    manifest_id = manifest_hash(args.manifest)
    try:
        from rand_isopeps.campaign.runner import run_task
    except ModuleNotFoundError as exc:
        if exc.name != "rand_isopeps.campaign.runner":
            raise
        raise RuntimeError(
            "the campaign executor is unavailable; install the current package revision"
        ) from exc

    try:
        result = run_task(
            task,
            output_root=args.output_root,
            manifest_id=manifest_id,
            checkpoint_root=args.checkpoint_root,
            stop_after_seconds=args.stop_after_seconds,
        )
    except InterruptedError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75) from exc
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
