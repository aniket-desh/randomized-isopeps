import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nersc.submit_paper_campaign as launch
import pytest
from experiments.paper_campaign.build_manifests import stamp_family_tasks
from experiments.paper_campaign.table_2_validation import (
    table_2_exact_relative_tolerance,
    table_2_references,
)
from nersc.submit_paper_campaign import (
    sbatch_command,
    requires_reference_preflight,
    select_largest_tasks,
    split_manifest,
    stage_manifests,
    validate_gpu_gate_artifact,
    validate_reference_artifact,
    validate_reference_stage1_results,
    validate_manifest_plan,
)
from rand_isopeps.campaign import (
    finalize_task,
    manifest_hash,
    read_manifest,
    runtime_source_fingerprint,
    write_manifest,
)
from rand_isopeps.campaign.gpu_gate import campaign_code_revision


def _task(index: int, hardware: str, *, pilot: bool = True) -> dict:
    return {
        "schema_version": "paper_campaign_v1",
        "experiment": "gpu_pilot" if pilot else "reference",
        "backend": "cupy" if hardware == "gpu" else "numpy",
        "dtype": "complex128",
        "problem": {"index": index},
        "method": {"name": "test"},
        "seeds": {"problem": index},
        "measurement": {"primary_metric": "error"},
        "runtime_source_fingerprint": runtime_source_fingerprint(),
        "resources": {
            "hardware": hardware,
            "cpus": index + 1,
            "gpus": hardware == "gpu",
        },
    }


def test_split_manifest_preserves_tasks_and_separates_resources(tmp_path):
    source = tmp_path / "mixed.jsonl"
    tasks = [
        _task(0, "cpu"),
        _task(1, "gpu"),
        _task(2, "cpu"),
        _task(3, "gpu", pilot=False),
    ]
    write_manifest(source, tasks)

    parts = split_manifest(source, tmp_path / "slurm", max_array_size=1)

    assert [(part["hardware"], part["cpus"], part["count"]) for part in parts] == [
        ("cpu", 1, 1),
        ("cpu", 3, 1),
        ("gpu", 2, 1),
        ("gpu", 4, 1),
    ]
    assert [part["requires_gpu_gate"] for part in parts] == [False, False, False, True]
    original_ids = {task["task_id"] for task in read_manifest(source)}
    split_ids = {
        task["task_id"]
        for part in parts
        for task in read_manifest(part["manifest"])
    }
    assert split_ids == original_ids
    for part in parts:
        assert part["manifest_hash"] in part["manifest"].name
        assert {
            tuple(sorted(task["resources"].items()))
            for task in read_manifest(part["manifest"])
        } == {tuple(sorted(part["resources"].items()))}


def test_largest_pilot_selects_one_task_per_resource_class(tmp_path):
    tasks = [_task(index, "cpu") for index in range(3)]
    for index, task_spec in enumerate(tasks):
        task_spec["resources"]["cpus"] = 4
        task_spec["problem"].update(lx=index + 2, ly=index + 2)
    tasks, _ = stamp_family_tasks("pilot", tasks)
    source = write_manifest(tmp_path / "pilot.jsonl", tasks)
    parts = split_manifest(source, tmp_path / "slurm", max_array_size=2)

    selected = select_largest_tasks(parts, tmp_path / "largest")

    assert len(selected) == 1
    assert selected[0]["count"] == 1
    assert selected[0]["selected_task"]["task_id"]
    assert read_manifest(selected[0]["manifest"])[0]["problem"]["lx"] == 4


def test_sbatch_command_is_zero_based_and_bound_to_one_launcher(tmp_path):
    part = {
        "hardware": "gpu",
        "manifest": tmp_path / "gpu.jsonl",
        "count": 7,
        "cpus": 8,
        "resources": {"hardware": "gpu", "cpus": 8, "gpus": 1},
    }
    command = sbatch_command(
        part,
        output_root=tmp_path / "results",
        gpu_gate=tmp_path / "gpu_gate.json",
        cpu_array_throttle=5,
        gpu_array_throttle=3,
    )

    assert "--array=0-6%3" in command
    assert "RAND_ISOPEPS_GPU_GATE_PATH=" in next(
        value for value in command if value.startswith("--export=")
    )
    assert command[-1].endswith("nersc/jobs/paper_campaign_gpu.slurm")


def test_split_plan_validates_and_stages_immutable_snapshots(tmp_path):
    tasks, _ = stamp_family_tasks("pilot", [_task(0, "cpu")])
    source = write_manifest(tmp_path / "pilot.jsonl", tasks)
    parts = split_manifest(source, tmp_path / "slurm", max_array_size=10)

    validate_manifest_plan([source], parts)
    staged = stage_manifests(parts, tmp_path / "durable")

    assert staged[0]["manifest"].parent == (tmp_path / "durable" / "manifests")
    assert read_manifest(staged[0]["manifest"])[0]["task_id"] == read_manifest(source)[0]["task_id"]


def test_reference_manifest_splits_into_ordered_stages(tmp_path):
    tasks, _ = stamp_family_tasks("references", launch.builders["references"]())
    source = write_manifest(tmp_path / "references.jsonl", tasks)
    parts = split_manifest(source, tmp_path / "slurm", max_array_size=1000)

    expected = {
        stage: sum(
            int(task["measurement"]["stage"]) == stage for task in tasks
        )
        for stage in (1, 2)
    }
    observed = {
        stage: sum(part["count"] for part in parts if part["stage"] == stage)
        for stage in (1, 2)
    }
    assert observed == expected


def test_reference_stage2_gate_accepts_complete_current_stage1(tmp_path):
    tasks, revision = stamp_family_tasks("references", launch.builders["references"]())
    stage1 = [
        finalize_task(task)
        for task in tasks
        if int(task["measurement"]["stage"]) == 1
    ]
    for index, task in enumerate(stage1):
        states = int(task["problem"].get("states", 1))
        record = {
            "task_id": task["task_id"],
            "status": "ok",
            "validation_passed": True,
            "campaign_revision": revision,
            "runtime_source_fingerprint": runtime_source_fingerprint(),
            "energies": [-1.0 - state for state in range(states)],
            "energy_convergence_tolerance": 1e-5,
        }
        if task["measurement"].get("require_sector_validation"):
            record["sector_validation_passed"] = True
        (tmp_path / f"{index:03d}.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "old-revision.jsonl").write_text(
        json.dumps({
            "task_id": "old-reference-task",
            "experiment": "reference",
            "status": "failed",
            "campaign_revision": "old-revision",
            "runtime_source_fingerprint": "old-source",
        })
        + "\n",
        encoding="utf-8",
    )

    validate_reference_stage1_results(tmp_path)


def test_reference_preflight_accepts_sector_prefix_for_p1(tmp_path):
    task_spec = _task(0, "cpu")
    task_spec.update(
        experiment="physics",
        problem={
            "hamiltonian": "heis",
            "lx": 6,
            "ly": 6,
            "states": 1,
            "study": "dektor_reproduction",
        },
        requirements=["reference_artifact"],
    )
    tasks, _ = stamp_family_tasks("physics", [task_spec])
    source = write_manifest(tmp_path / "physics.jsonl", tasks)
    parts = split_manifest(source, tmp_path / "slurm", max_array_size=10)
    _, reference_revision = stamp_family_tasks(
        "references", launch.builders["references"]()
    )
    artifact = tmp_path / "references.json"
    artifact.write_text(json.dumps({
        "schema_version": "validated_references_v1",
        "records": [{
            "hamiltonian": "heis",
            "lx": 6,
            "ly": 6,
            "states": 2,
            "energies": [-1.0, -0.8],
            "validation_passed": True,
            "validation_contract": "nested_bond_energy_orthogonality",
            "validated_bonds": [500, 1000],
            "validated_actual_max_bonds": [500, 1000],
            "minimum_actual_max_bond": 1000,
            "required_max_bond": 1000,
            "max_bond_difference": 1e-7,
            "bond_tolerance": 1e-5,
            "max_previous_overlap": 0.0,
            "overlap_tolerance": 1e-6,
            "max_projector_compression_infidelity": 0.0,
            "projector_state_tolerance": 0.1,
            "sector_validation_passed": True,
            "target_sectors": [
                {"state_index": 0, "target_sz": 0},
                {"state_index": 1, "target_sz": 1},
            ],
            "reference_metadata": {"symmetry_sector": [0.0, 1.0]},
            "campaign_revision": reference_revision,
            "runtime_source_fingerprint": runtime_source_fingerprint(),
        }],
    }), encoding="utf-8")

    validate_reference_artifact(artifact, parts)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["records"][0]["validated_actual_max_bonds"][-1] = 999
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks validated coverage"):
        validate_reference_artifact(artifact, parts)


def test_reference_preflight_gates_table_2_against_exact_cells(tmp_path):
    task_spec = _task(0, "cpu")
    task_spec.update(
        experiment="physics",
        problem={
            "hamiltonian": "tfim@3.5",
            "lx": 4,
            "ly": 4,
            "states": 2,
            "study": "dektor_reproduction",
            "dektor_panels": ["table_2"],
        },
        measurement={
            "primary_metric": "energy_error",
            "published_reference_relative_tolerance": (
                table_2_exact_relative_tolerance
            ),
        },
        requirements=[],
    )
    tasks, _ = stamp_family_tasks("physics", [task_spec])
    source = write_manifest(tmp_path / "physics.jsonl", tasks)
    parts = split_manifest(source, tmp_path / "slurm", max_array_size=10)
    assert requires_reference_preflight(parts)

    _, reference_revision = stamp_family_tasks(
        "references", launch.builders["references"]()
    )
    records = [
        {
            "hamiltonian": hamiltonian,
            "lx": 4,
            "ly": 4,
            "states": 2,
            "energies": energies,
            "validation_passed": True,
            "reference_tier": "exact",
            "campaign_revision": reference_revision,
            "runtime_source_fingerprint": runtime_source_fingerprint(),
        }
        for hamiltonian, energies in table_2_references().items()
    ]
    artifact = tmp_path / "references.json"
    payload = {"schema_version": "validated_references_v1", "records": records}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    validate_reference_artifact(artifact, parts)

    records[0]["energies"][0] += 0.01
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with exact diagonalization"):
        validate_reference_artifact(artifact, parts)


def test_gpu_gate_preflight_binds_complete_current_pilot(tmp_path):
    pilot_tasks, _ = stamp_family_tasks(
        "gpu_pilot", launch.builders["gpu_pilot"]()
    )
    expected = [finalize_task(task_spec) for task_spec in pilot_tasks]
    gate = tmp_path / "gpu_gate.json"
    payload = {
        "schema_version": "gpu_gate_v1",
        "passed": True,
        "campaign_code_revision": campaign_code_revision(),
        "campaign_schema_versions": ["paper_campaign_v1"],
        "pilot_manifest_hash": manifest_hash(expected),
        "expected_task_ids": sorted(task_spec["task_id"] for task_spec in expected),
        "expected_tasks": len(expected),
        "kernel_rows": 1,
        "physics_pairs": 1,
        "devices": [{"name": "test-a100"}],
    }
    gate.write_text(json.dumps(payload), encoding="utf-8")

    validate_gpu_gate_artifact(gate)
    payload["campaign_code_revision"] = "stale"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete current pilot"):
        validate_gpu_gate_artifact(gate)


def test_main_is_dry_run_without_submit(monkeypatch, tmp_path, capsys):
    tasks, _ = stamp_family_tasks("pilot", [_task(0, "cpu")])
    manifest = write_manifest(tmp_path / "pilot.jsonl", tasks)
    monkeypatch.setattr(launch, "build_selected", lambda _output, _families: [manifest])

    def unexpected_submit(*_args, **_kwargs):
        raise AssertionError("dry run called sbatch")

    monkeypatch.setattr(launch.subprocess, "run", unexpected_submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_paper_campaign.py",
            "--family",
            "gpu_pilot",
            "--manifest-dir",
            str(tmp_path),
        ],
    )
    launch.main()

    output = capsys.readouterr().out
    assert "worst-case charge rate" in output
    assert "across independent arrays" in output
    assert "dry run only" in output


def test_resource_summary_sums_independent_array_throttles(capsys):
    parts = [
        {
            "family": family,
            "hardware": "cpu",
            "count": 5,
            "cpus": 32,
            "resources": {"hardware": "cpu", "cpus": 32, "gpus": 0},
        }
        for family in ("a", "b")
    ]

    launch._resource_summary(
        parts,
        SimpleNamespace(cpu_array_throttle=2, gpu_array_throttle=1),
    )

    output = capsys.readouterr().out
    assert "across independent arrays: 4 cpu, 0 gpu" in output
    assert "0.50 cpu node-hours/hour" in output
    assert "30.00 cpu node-hours, 0.00 gpu node-hours" in output


def test_submit_refuses_promoted_gpu_without_gate(monkeypatch, tmp_path):
    tasks, _ = stamp_family_tasks("physics", [_task(0, "gpu", pilot=False)])
    manifest = write_manifest(
        tmp_path / "physics.jsonl",
        tasks,
    )
    monkeypatch.setattr(launch, "build_selected", lambda _output, _families: [manifest])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_paper_campaign.py",
            "--family",
            "physics",
            "--hardware",
            "gpu",
            "--manifest-dir",
            str(tmp_path),
            "--gpu-gate",
            str(tmp_path / "missing-gate.json"),
            "--submit",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        launch.main()


def test_submit_refuses_an_unpaired_gpu_pilot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_paper_campaign.py",
            "--family",
            "gpu_pilot",
            "--hardware",
            "gpu",
            "--submit",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        launch.main()


def test_campaign_slurm_scripts_declare_shared_resources_and_resume():
    root = Path(__file__).resolve().parents[1]
    cpu = (root / "nersc/jobs/paper_campaign_cpu.slurm").read_text()
    gpu = (root / "nersc/jobs/paper_campaign_gpu.slurm").read_text()
    common = (root / "nersc/jobs/paper_campaign_task.sh").read_text()

    for script in (cpu, gpu):
        assert "#SBATCH --array=0-0" in script
        assert "#SBATCH --qos=shared" in script
        assert "#SBATCH --time-min" not in script
        assert "#SBATCH --signal=" not in script
        assert "#SBATCH --requeue" in script
        assert "#SBATCH --licenses=scratch,cfs" in script
    assert "#SBATCH --account=m4926\n" in cpu
    assert "#SBATCH --account=m4926_g\n" in gpu
    assert "--gpus-per-task=1" in gpu
    assert "#SBATCH --constraint=gpu&hbm40g" in gpu
    assert "physical_threads=$(( (SLURM_CPUS_PER_TASK + 1) / 2 ))" in common
    assert "--cpu-bind=cores" in common
    assert '--checkpoint-root "$checkpoint_root"' in common
    assert "SLURM_JOB_END_TIME" in common
    assert '--stop-after-seconds "$stop_after"' in common
    assert "status == 75" in common
    assert 'scontrol requeue "$SLURM_JOB_ID"' in common
    assert 'case "$OUTPUT_ROOT" in' in common
