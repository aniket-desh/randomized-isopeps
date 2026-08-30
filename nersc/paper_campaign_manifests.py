"""content-addressed manifest planning for Perlmutter arrays."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.paper_campaign.manifest_common import schema_version
from rand_isopeps.campaign import (
    manifest_hash,
    read_manifest,
    runtime_source_fingerprint,
    write_manifest,
)


def write_immutable_manifest(path: Path, tasks: list[dict]) -> Path:
    """create one content-addressed manifest without replacing other content."""
    expected = manifest_hash(tasks)
    if path.exists():
        if manifest_hash(path) != expected:
            raise FileExistsError(f"refusing to replace conflicting manifest: {path}")
        return path
    write_manifest(path, tasks)
    return path


def _requires_gpu_gate(task: dict) -> bool:
    if task.get("backend") != "cupy":
        return False
    study = task.get("problem", {}).get("study")
    pilot = task.get("experiment") == "gpu_pilot" and study != "gpu_crossover"
    return not pilot and study != "gpu_pilot"


def split_manifest(path: Path, output_dir: Path, max_array_size: int) -> list[dict]:
    """split a family by resources and the cluster array-size limit."""
    if max_array_size < 1:
        raise ValueError("max_array_size must be positive")
    groups = defaultdict(list)
    for task in read_manifest(path):
        resources = task.get("resources", {})
        hardware = str(resources.get("hardware", ""))
        if hardware not in {"cpu", "gpu"}:
            raise ValueError(f"task has unsupported hardware: {hardware!r}")
        stage = int(task.get("measurement", {}).get("stage", 0))
        resource_key = json.dumps(resources, sort_keys=True, separators=(",", ":"))
        groups[(hardware, stage, resource_key)].append(task)

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    hardware_counts = defaultdict(int)
    for hardware, stage, resource_key in sorted(groups):
        tasks = groups[(hardware, stage, resource_key)]
        resources = json.loads(resource_key)
        resource_index = hardware_counts[hardware]
        hardware_counts[hardware] += 1
        for part_index, start in enumerate(range(0, len(tasks), max_array_size)):
            selected = tasks[start : start + max_array_size]
            part_hash = manifest_hash(selected)
            destination = output_dir / (
                f"{path.stem}.{hardware}.{resource_index:02d}.{part_index:03d}."
                f"{part_hash}.jsonl"
            )
            write_immutable_manifest(destination, selected)
            parts.append({
                "family": path.stem,
                "hardware": hardware,
                "stage": stage,
                "resources": resources,
                "manifest": destination.resolve(),
                "manifest_hash": part_hash,
                "count": len(selected),
                "cpus": int(resources.get("cpus", 1)),
                "requires_gpu_gate": any(map(_requires_gpu_gate, selected)),
                "requires_reference_artifact": any(
                    "reference_artifact" in task.get("requirements", ())
                    for task in selected
                ),
            })
    return parts


def _largest_task_key(task: dict) -> tuple:
    problem = task.get("problem", {})
    method = task.get("method", {})
    lx = int(problem.get("lx", problem.get("column_size", 1)))
    ly = int(problem.get("ly", 1))
    bond_dims = tuple(int(value) for value in method.get("bond_dims", ()))
    numerical_scale = max(
        int(problem.get("chi", 0)),
        int(problem.get("eta", 0)),
        int(problem.get("mpo_bond", 0)),
        int(method.get("chi", 0)),
        int(method.get("eta", 0)),
        int(method.get("ell", 0)),
        int(method.get("chi_sk", 0)),
        max(bond_dims, default=0),
    )
    return (
        lx * ly,
        int(problem.get("states", 1)),
        numerical_scale,
        str(task.get("task_id", "")),
    )


def select_largest_tasks(parts: list[dict], output_dir: Path) -> list[dict]:
    """select one maximum-scale task from each complete resource class."""
    groups = defaultdict(list)
    for part in parts:
        key = (
            part["family"],
            part["hardware"],
            int(part.get("stage", 0)),
            json.dumps(part["resources"], sort_keys=True, separators=(",", ":")),
        )
        groups[key].extend(read_manifest(part["manifest"]))

    selected = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (key, tasks) in enumerate(sorted(groups.items())):
        family, hardware, stage, resource_key = key
        task = max(tasks, key=_largest_task_key)
        digest = manifest_hash([task])
        path = output_dir / f"{family}.{hardware}.{index:02d}.{digest}.jsonl"
        write_immutable_manifest(path, [task])
        resources = json.loads(resource_key)
        problem = task.get("problem", {})
        selected.append({
            "family": family,
            "hardware": hardware,
            "stage": stage,
            "resources": resources,
            "manifest": path.resolve(),
            "manifest_hash": digest,
            "count": 1,
            "cpus": int(resources.get("cpus", 1)),
            "requires_gpu_gate": _requires_gpu_gate(task),
            "requires_reference_artifact": (
                "reference_artifact" in task.get("requirements", ())
            ),
            "selected_task": {
                "task_id": task["task_id"],
                "study": problem.get("study"),
                "lx": problem.get("lx"),
                "ly": problem.get("ly"),
                "states": problem.get("states", 1),
                "method": task.get("method", {}).get("name"),
            },
        })
    return selected


def stage_manifests(parts: list[dict], output_root: Path) -> list[dict]:
    """stage immutable array manifests beside durable campaign results."""
    staged = []
    destination_root = output_root / "manifests"
    for part in parts:
        source = Path(part["manifest"])
        destination = destination_root / source.name
        tasks = read_manifest(source)
        if manifest_hash(tasks) != part["manifest_hash"]:
            raise ValueError(f"split manifest changed before staging: {source}")
        write_immutable_manifest(destination, tasks)
        staged.append({**part, "manifest": destination.resolve()})
    return staged


def stage_family_manifests(
    manifests: list[Path], output_root: Path
) -> dict[str, Path]:
    """stage each complete family for validation and provenance."""
    staged = {}
    destination_root = output_root / "manifests"
    for source in manifests:
        tasks = read_manifest(source)
        digest = manifest_hash(tasks)
        destination = destination_root / f"{source.stem}.{digest}.jsonl"
        write_immutable_manifest(destination, tasks)
        staged[source.stem] = destination.resolve()
    return staged


def validate_manifest_plan(manifests: list[Path], parts: list[dict]) -> None:
    """require an exact, revision-bound partition before submission."""
    family_tasks = {path.stem: read_manifest(path) for path in manifests}
    expected = {
        family: {task["task_id"] for task in tasks}
        for family, tasks in family_tasks.items()
    }
    expected_revisions = {
        family: {
            (task.get("campaign_family"), task.get("campaign_revision"))
            for task in tasks
        }
        for family, tasks in family_tasks.items()
    }
    for family, values in expected_revisions.items():
        if len(values) != 1:
            raise ValueError("a family manifest has inconsistent campaign provenance")
        stamped_family, stamped_revision = next(iter(values))
        if stamped_family != family or not stamped_revision:
            raise ValueError("a family manifest lacks campaign provenance")

    observed = defaultdict(list)
    revision = runtime_source_fingerprint()
    for part in parts:
        tasks = read_manifest(part["manifest"])
        if len(tasks) != part["count"]:
            raise ValueError(f"manifest count mismatch: {part['manifest']}")
        if manifest_hash(tasks) != part["manifest_hash"]:
            raise ValueError(f"manifest hash mismatch: {part['manifest']}")
        for task in tasks:
            if task["schema_version"] != schema_version:
                raise ValueError(f"unsupported campaign schema: {task['schema_version']!r}")
            if task.get("runtime_source_fingerprint") != revision:
                raise ValueError(f"task source revision is stale: {task['task_id']}")
            provenance = (task.get("campaign_family"), task.get("campaign_revision"))
            if provenance not in expected_revisions[part["family"]]:
                raise ValueError(f"task campaign revision mismatch: {part['manifest']}")
            if task.get("resources", {}) != part["resources"]:
                raise ValueError(f"mixed resource class: {part['manifest']}")
            observed[part["family"]].append(task["task_id"])

    if set(observed) != set(expected):
        raise ValueError("split plan does not cover every selected family")
    for family, identifiers in observed.items():
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"split plan duplicates tasks in {family}")
        if set(identifiers) != expected[family]:
            raise ValueError(f"split plan does not exactly cover {family}")
