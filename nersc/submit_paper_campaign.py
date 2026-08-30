#!/usr/bin/env python3
"""build paper manifests and optionally submit zero-based slurm arrays."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
requested_walltime_hours = 24.0
for source in (repo_root, repo_root / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.paper_campaign.build_manifests import builders, build_selected
from nersc.paper_campaign_artifacts import (
    requires_reference_preflight,
    validate_gpu_gate_artifact,
    validate_reference_artifact,
    validate_reference_stage1_results,
)
from nersc.paper_campaign_manifests import (
    select_largest_tasks,
    split_manifest,
    stage_family_manifests,
    stage_manifests,
    validate_manifest_plan,
    write_immutable_manifest,
)


def sbatch_command(
    part: dict,
    *,
    output_root: Path,
    gpu_gate: Path,
    reference_artifact: Path | None = None,
    cpu_array_throttle: int,
    gpu_array_throttle: int,
) -> list[str]:
    """construct one explicit zero-based array submission."""
    hardware = part["hardware"]
    throttle = (
        cpu_array_throttle if hardware == "cpu" else gpu_array_throttle
    )
    if throttle < 1:
        raise ValueError("array throttle must be positive")
    script = repo_root / "nersc" / "jobs" / f"paper_campaign_{hardware}.slurm"
    reference_artifact = reference_artifact or output_root / "references.json"
    grace = int(part["resources"].get("stop_grace_seconds", 1800))
    exports = [
        "ALL",
        f"MANIFEST={part['manifest']}",
        f"OUTPUT_ROOT={output_root.resolve()}",
        f"REPO_ROOT={repo_root}",
        f"RAND_ISOPEPS_GPU_GATE_PATH={gpu_gate.resolve()}",
        f"STOP_GRACE_SECONDS={grace}",
    ]
    if part.get("requires_reference_artifact"):
        exports.append(
            f"RAND_ISOPEPS_REFERENCE_PATH={reference_artifact.resolve()}"
        )
    command = [
        "sbatch",
        f"--array=0-{part['count'] - 1}%{throttle}",
        f"--export={','.join(exports)}",
        f"--chdir={repo_root}",
    ]
    if hardware == "cpu":
        command.append(f"--cpus-per-task={part['cpus']}")
    command.append(str(script))
    return command


def _resource_summary(parts: list[dict], args) -> None:
    totals = defaultdict(int)
    classes = defaultdict(int)
    for part in parts:
        hardware = part["hardware"]
        totals[hardware] += part["count"]
        key = (
            part["family"],
            hardware,
            int(part.get("stage", 0)),
            part["cpus"],
            int(part["resources"].get("gpus", 0)),
            int(part["resources"].get("stop_grace_seconds", 1800)),
        )
        classes[key] += part["count"]
    for (family, hardware, stage, cpus, gpus, grace), count in sorted(classes.items()):
        stage_label = f", stage={stage}" if stage else ""
        print(
            f"resource class: {family} {hardware}, {count} tasks, "
            f"cpus={cpus}, gpus={gpus}, stop_grace={grace}s{stage_label}"
        )
    for part in parts:
        if "selected_task" not in part:
            continue
        selected = part["selected_task"]
        print(
            f"pilot task: {part['family']} {part['hardware']}, "
            f"study={selected['study']}, "
            f"lattice={selected['lx']}x{selected['ly']}, "
            f"states={selected['states']}, method={selected['method']}, "
            f"task_id={selected['task_id']}"
        )
    concurrent = {
        hardware: sum(
            min(
                part["count"],
                args.cpu_array_throttle
                if hardware == "cpu"
                else args.gpu_array_throttle,
            )
            for part in parts
            if part["hardware"] == hardware
        )
        for hardware in ("cpu", "gpu")
    }
    print(
        "worst-case simultaneous shared allocations across independent arrays: "
        f"{concurrent['cpu']} cpu, {concurrent['gpu']} gpu"
    )
    node_equivalents = {"cpu": 0.0, "gpu": 0.0}
    charge_ceilings = {"cpu": 0.0, "gpu": 0.0}
    for part in parts:
        hardware = part["hardware"]
        slots = min(
            part["count"],
            args.cpu_array_throttle
            if hardware == "cpu"
            else args.gpu_array_throttle,
        )
        if hardware == "cpu":
            fraction = math.ceil(part["cpus"] / 2) / 128
        else:
            fraction = int(part["resources"].get("gpus", 0)) / 4
        node_equivalents[hardware] += slots * fraction
        charge_ceilings[hardware] += (
            part["count"] * fraction * requested_walltime_hours
        )
    print(
        "worst-case charge rate at those per-array throttles: "
        f"{node_equivalents['cpu']:.2f} cpu node-hours/hour, "
        f"{node_equivalents['gpu']:.2f} gpu node-hours/hour"
    )
    print(
        f"requested-walltime ceiling if every task uses all "
        f"{requested_walltime_hours:g} hours: "
        f"{charge_ceilings['cpu']:.2f} cpu node-hours, "
        f"{charge_ceilings['gpu']:.2f} gpu node-hours"
    )
    print(
        f"slurm plan: {totals['cpu']} cpu tasks in "
        f"{sum(part['hardware'] == 'cpu' for part in parts)} arrays; "
        f"{totals['gpu']} gpu tasks in "
        f"{sum(part['hardware'] == 'gpu' for part in parts)} arrays"
    )


def _preflight_dependencies(
    parser: argparse.ArgumentParser,
    parts: list[dict],
    gpu_gate: Path,
    references: Path,
) -> None:
    if any(part["requires_gpu_gate"] for part in parts):
        if not gpu_gate.is_file():
            parser.error(
                "non-pilot cupy tasks require a validated gpu gate; submit the "
                "gpu_pilot family, run validate_gpu_pilot.py, then retry"
            )
        try:
            validate_gpu_gate_artifact(gpu_gate)
        except ValueError as exc:
            parser.error(str(exc))
    if requires_reference_preflight(parts):
        if not references.is_file():
            parser.error(
                "selected physics tasks require --reference-artifact from validated "
                "reference tasks, including the exact table 2 cross-check"
            )
        try:
            validate_reference_artifact(references, parts)
        except ValueError as exc:
            parser.error(str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", choices=tuple(builders))
    parser.add_argument("--hardware", choices=("cpu", "gpu"))
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=repo_root / "experiments" / "paper_campaign" / "manifests",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign"),
    )
    parser.add_argument("--max-array-size", type=int, default=1000)
    parser.add_argument(
        "--cpu-array-throttle",
        type=int,
        default=1,
        help="simultaneous elements in each CPU array, not a campaign-global cap",
    )
    parser.add_argument(
        "--gpu-array-throttle",
        type=int,
        default=1,
        help="simultaneous elements in each GPU array, not a campaign-global cap",
    )
    parser.add_argument("--gpu-gate", type=Path)
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--reference-stage", type=int, choices=(1, 2))
    parser.add_argument(
        "--pilot-largest",
        action="store_true",
        help="select one maximum-scale task from each selected resource class",
    )
    parser.add_argument("--submit", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not str(args.output_root.resolve()).startswith("/global/cfs/cdirs/"):
        parser.error("--output-root must be a durable cfs path")
    if args.cpu_array_throttle < 1 or args.gpu_array_throttle < 1:
        parser.error("array throttles must be positive")

    families = tuple(args.family or builders)
    if args.reference_stage is not None and families != ("references",):
        parser.error("--reference-stage requires --family references")
    if args.submit and "gpu_pilot" in families and args.hardware is not None:
        parser.error("gpu_pilot submission must include paired cpu and gpu tasks")
    if args.submit and "references" in families and args.reference_stage is None:
        parser.error("reference submission requires --reference-stage 1 or 2")
    manifests = build_selected(args.manifest_dir, families)
    parts = [
        part
        for manifest in manifests
        for part in split_manifest(
            manifest,
            args.manifest_dir / "slurm",
            args.max_array_size,
        )
    ]
    try:
        validate_manifest_plan(manifests, parts)
    except ValueError as exc:
        parser.error(str(exc))
    if args.hardware is not None:
        parts = [part for part in parts if part["hardware"] == args.hardware]
    if args.reference_stage is not None:
        parts = [
            part
            for part in parts
            if part["family"] == "references"
            and int(part.get("stage", 0)) == args.reference_stage
        ]
        if not parts:
            parser.error("the selected reference stage has no tasks")
    if args.pilot_largest:
        parts = select_largest_tasks(
            parts,
            args.manifest_dir / "pilot_largest",
        )

    gpu_gate = args.gpu_gate or args.output_root / "gpu_gate.json"
    references = args.reference_artifact or args.output_root / "references.json"
    staged_families = {}
    if args.submit:
        if args.reference_stage == 2:
            try:
                validate_reference_stage1_results(args.output_root)
            except ValueError as exc:
                parser.error(str(exc))
        _preflight_dependencies(parser, parts, gpu_gate, references)
        staged_families = stage_family_manifests(manifests, args.output_root)
        parts = stage_manifests(parts, args.output_root)

    _resource_summary(parts, args)
    print(
        f"dependencies: gpu_gate={gpu_gate.resolve()}, "
        f"references={references.resolve()}"
    )
    for family, snapshot in sorted(staged_families.items()):
        print(f"immutable family manifest: {family}={snapshot}")
    for part in parts:
        command = sbatch_command(
            part,
            output_root=args.output_root,
            gpu_gate=gpu_gate,
            reference_artifact=references,
            cpu_array_throttle=args.cpu_array_throttle,
            gpu_array_throttle=args.gpu_array_throttle,
        )
        print(shlex.join(command))
        if args.submit:
            subprocess.run(command, cwd=repo_root, check=True)
    if not args.submit:
        print("dry run only; pass --submit to execute these commands")


if __name__ == "__main__":
    main()
