import pytest

from rand_isopeps.campaign import (
    finalize_task,
    read_records,
    record_is_complete,
    replace_checkpoint,
    write_record,
    write_task_record,
)


def _task():
    return {
        "schema_version": "test_v1",
        "experiment": "test",
        "backend": "numpy",
        "dtype": "float64",
        "problem": {"kind": "column"},
        "method": {"name": "rmps"},
        "seeds": {"problem": 1},
        "measurement": {"metric": "error"},
    }


def test_records_are_atomic_immutable_and_idempotent(tmp_path):
    path = tmp_path / "record.jsonl"
    record = {"task_id": "abc", "status": "ok", "error": 0.1}
    assert write_record(path, record) is True
    assert write_record(path, record) is False
    assert read_records(path) == [record]
    assert record_is_complete(path, "abc") is True
    assert not tuple(tmp_path.glob(".record.jsonl.*.tmp"))

    with pytest.raises(ValueError, match="conflicting result"):
        write_record(path, {**record, "error": 0.2})
    assert read_records(path) == [record]


def test_task_records_use_manifest_and_task_directories(tmp_path):
    task = finalize_task(_task())
    path = write_task_record(
        tmp_path,
        "manifest_hash",
        task,
        [{"status": "ok", "error": 0.0}],
    )
    assert path == (
        tmp_path
        / "manifest_hash"
        / "tasks"
        / f"{task['task_id']}.jsonl"
    )
    assert read_records(path)[0]["task_id"] == task["task_id"]


def test_checkpoints_are_replaceable(tmp_path):
    path = tmp_path / "checkpoint.json"
    replace_checkpoint(path, {"step": 1})
    replace_checkpoint(path, {"step": 2})
    assert path.read_text(encoding="utf-8") == '{"step":2}\n'
