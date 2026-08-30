"""build deterministic manifests for the paper experiment campaign."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rand_isopeps.campaign import manifest_hash, write_manifest

from experiments.paper_campaign.column_manifests import build_column_moves, build_isometry
from experiments.paper_campaign.gpu_manifests import build_gpu_crossover, build_gpu_pilot
from experiments.paper_campaign.physics_manifests import build_physics
from experiments.paper_campaign.reference_manifests import build_references
from experiments.paper_campaign.sketch_manifests import build_gaussian_limit


builders: dict[str, Callable[[], list[dict]]] = {
    "gaussian_limit": build_gaussian_limit,
    "column_moves": build_column_moves,
    "isometry": build_isometry,
    "physics": build_physics,
    "gpu_pilot": build_gpu_pilot,
    "gpu_crossover": build_gpu_crossover,
    "references": build_references,
}


def stamp_family_tasks(family: str, tasks: list[dict]) -> tuple[list[dict], str]:
    """bind every task to one full-family campaign revision."""
    revision = manifest_hash(tasks)
    stamped = []
    for task_spec in tasks:
        selected = copy.deepcopy(task_spec)
        selected.pop("task_id", None)
        selected["campaign_family"] = family
        selected["campaign_revision"] = revision
        stamped.append(selected)
    return stamped, revision


def build_selected(output_dir: str | Path, families: Iterable[str]) -> list[Path]:
    """write selected campaign manifests and return their paths."""
    output = Path(output_dir)
    paths = []
    for family in families:
        tasks, revision = stamp_family_tasks(family, builders[family]())
        path = write_manifest(output / f"{family}.jsonl", tasks)
        print(
            f"{family}: {len(tasks)} tasks, revision={revision}, "
            f"manifest={manifest_hash(path)}"
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(builders),
        help="build one family; repeat the flag to build several",
    )
    args = parser.parse_args()
    build_selected(args.output_dir, args.family or tuple(builders))


if __name__ == "__main__":
    main()
