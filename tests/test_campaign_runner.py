import pytest

from rand_isopeps.campaign import finalize_task, read_records, runtime_source_fingerprint
from rand_isopeps.campaign.runner import _dispatch, run_task


def _task():
    return {
        "schema_version": "test_v1",
        "experiment": "test",
        "backend": "numpy",
        "dtype": "complex128",
        "problem": {"kind": "test"},
        "method": {"name": "test"},
        "seeds": {"problem": 1},
        "measurement": {"primary_metric": "value"},
        "runtime_source_fingerprint": runtime_source_fingerprint(),
        "campaign_family": "test",
        "campaign_revision": "revision",
        "resources": {"hardware": "cpu", "cpus": 1, "gpus": 0},
        "requirements": [],
    }


def test_runner_dispatches_gpu_crossover_through_gpu_pilot(tmp_path, monkeypatch):
    task = {**_task(), "experiment": "gpu_crossover"}
    observed = []
    monkeypatch.setattr(
        "rand_isopeps.campaign.gpu_pilot.run_gpu_pilot",
        lambda selected: observed.append(selected) or [{"value": 1.0}],
    )

    rows = _dispatch(task, tmp_path / "checkpoint.pkl", lambda: False)

    assert rows == [{"value": 1.0}]
    assert observed == [task]


def test_runner_separates_replaceable_checkpoints_from_final_records(
    tmp_path, monkeypatch
):
    output = tmp_path / "results"
    checkpoints = tmp_path / "scratch"

    def dispatch(_task, checkpoint_path, _stop_requested):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("temporary", encoding="utf-8")
        return [{"value": 1.0}]

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    result = run_task(
        _task(),
        output_root=output,
        checkpoint_root=checkpoints,
        manifest_id="manifest",
    )
    assert result.is_file()
    assert not tuple(checkpoints.rglob("*.pkl"))


def test_runner_exposes_application_wall_clock_stop(tmp_path, monkeypatch):
    times = iter((0.0, 2.0))
    monkeypatch.setattr("rand_isopeps.campaign.runner.monotonic", lambda: next(times))

    def dispatch(_task, _checkpoint_path, stop_requested):
        assert stop_requested() is True
        raise InterruptedError("checkpointed")

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    with pytest.raises(InterruptedError, match="checkpointed"):
        run_task(
            _task(),
            output_root=tmp_path,
            manifest_id="manifest",
            stop_after_seconds=1.0,
        )


def test_runner_keeps_manifest_provenance_authoritative(tmp_path, monkeypatch):
    task = finalize_task(_task())
    replacements = {
        "task_id": "wrong",
        "manifest_id": "wrong",
        "schema_version": "wrong",
        "experiment": "wrong",
        "backend": "cupy",
        "dtype": "float32",
        "seeds": {"problem": 999},
        "problem": {"kind": "wrong"},
        "task_method": {"name": "wrong"},
        "measurement": {"primary_metric": "wrong"},
        "runtime_source_fingerprint": "wrong",
        "campaign_family": "wrong",
        "campaign_revision": "wrong",
        "resources": {"hardware": "gpu", "cpus": 99, "gpus": 1},
        "value": 1.0,
    }
    monkeypatch.setattr(
        "rand_isopeps.campaign.runner._dispatch",
        lambda *_args: [replacements],
    )

    result = run_task(task, output_root=tmp_path, manifest_id="manifest")
    row = read_records(result)[0]

    assert row["value"] == 1.0
    assert row["task_id"] == task["task_id"]
    assert row["manifest_id"] == "manifest"
    assert row["schema_version"] == task["schema_version"]
    assert row["experiment"] == task["experiment"]
    assert row["backend"] == task["backend"]
    assert row["dtype"] == task["dtype"]
    assert row["seeds"] == task["seeds"]
    assert row["replicate_seeds"] == replacements["seeds"]
    assert row["problem"] == task["problem"]
    assert row["task_method"] == task["method"]
    assert row["measurement"] == task["measurement"]
    assert row["runtime_source_fingerprint"] == task["runtime_source_fingerprint"]
    assert row["campaign_family"] == task["campaign_family"]
    assert row["campaign_revision"] == task["campaign_revision"]
    assert row["resources"] == task["resources"]


def test_runner_rejects_stale_source_before_reuse_or_dispatch(tmp_path, monkeypatch):
    calls = 0

    def dispatch(*_args):
        nonlocal calls
        calls += 1
        return [{"value": 1.0}]

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    task = _task()
    run_task(task, output_root=tmp_path, manifest_id="manifest")
    monkeypatch.setattr(
        "rand_isopeps.campaign.runner.runtime_source_fingerprint",
        lambda: "different-source",
    )

    with pytest.raises(RuntimeError, match="build a new manifest"):
        run_task(task, output_root=tmp_path, manifest_id="manifest")
    assert calls == 1


def test_runner_reuses_only_successful_immutable_results(tmp_path, monkeypatch):
    calls = 0

    def dispatch(*_args):
        nonlocal calls
        calls += 1
        return [{"value": 1.0}]

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    first = run_task(_task(), output_root=tmp_path, manifest_id="manifest")
    second = run_task(_task(), output_root=tmp_path, manifest_id="manifest")

    assert first == second
    assert calls == 1


def test_runner_persists_failure_then_requires_a_new_manifest(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    calls = 0

    def dispatch(_task, checkpoint_path, _stop_requested):
        nonlocal calls
        calls += 1
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("temporary", encoding="utf-8")
        return [
            {"method": "good", "status": "ok", "value": 1.0},
            {"method": "bad", "status": "failed", "error": "diagnostic"},
        ]

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    match = "immutable failed result.*terminal for this manifest.*new manifest"
    with pytest.raises(RuntimeError, match=match):
        run_task(
            _task(),
            output_root=tmp_path,
            checkpoint_root=checkpoints,
            manifest_id="manifest",
        )

    paths = tuple((tmp_path / "manifest" / "tasks").glob("*.jsonl"))
    assert len(paths) == 1
    assert [row["status"] for row in read_records(paths[0])] == ["ok", "failed"]
    assert not tuple(checkpoints.rglob("*.pkl"))

    with pytest.raises(RuntimeError, match=match):
        run_task(
            _task(),
            output_root=tmp_path,
            checkpoint_root=checkpoints,
            manifest_id="manifest",
        )
    assert calls == 1


def test_runner_persists_dispatch_exception_as_terminal_failure(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    calls = 0

    def dispatch(_task, checkpoint_path, _stop_requested):
        nonlocal calls
        calls += 1
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("temporary", encoding="utf-8")
        raise ValueError("bad\nconfiguration")

    monkeypatch.setattr("rand_isopeps.campaign.runner._dispatch", dispatch)
    match = "immutable failed result.*terminal for this manifest.*new manifest"
    with pytest.raises(RuntimeError, match=match):
        run_task(
            _task(),
            output_root=tmp_path,
            checkpoint_root=checkpoints,
            manifest_id="manifest",
        )

    paths = tuple((tmp_path / "manifest" / "tasks").glob("*.jsonl"))
    assert len(paths) == 1
    assert read_records(paths[0])[0]["error_type"] == "ValueError"
    assert read_records(paths[0])[0]["error"] == "bad configuration"
    assert not tuple(checkpoints.rglob("*.pkl"))

    with pytest.raises(RuntimeError, match=match):
        run_task(
            _task(),
            output_root=tmp_path,
            checkpoint_root=checkpoints,
            manifest_id="manifest",
        )
    assert calls == 1
