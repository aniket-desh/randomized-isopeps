import json

import pytest

from rand_isopeps.campaign import (
    compute_task_id,
    finalize_task,
    load_manifest,
    manifest_hash,
    one_at_a_time,
    select_task,
    write_manifest,
)


def _task(method_name="rmps"):
    return {
        "schema_version": "test_v1",
        "experiment": "test",
        "backend": "numpy",
        "dtype": "complex128",
        "problem": {"size": 3, "kind": "column"},
        "method": {"name": method_name, "chi_sk": 4},
        "seeds": {"problem": 1, "sketch": 2},
        "measurement": {"metric": "error"},
    }


def test_task_ids_are_deterministic_and_content_addressed():
    task = _task()
    reordered = dict(reversed(tuple(task.items())))
    assert compute_task_id(task) == compute_task_id(reordered)
    assert finalize_task(task)["task_id"] == compute_task_id(task)
    assert compute_task_id(task) != compute_task_id(_task("gaussian"))


def test_one_at_a_time_never_combines_deviations():
    base = {"method": {"eta": 4, "chi_sk": 8, "n_power": 0}}
    rows = one_at_a_time(
        base,
        {
            "method.eta": (2, 4, 8),
            "method.chi_sk": (4, 8, 16),
            "method.n_power": (0, 1),
        },
    )
    assert len(rows) == 6
    for row in rows[1:]:
        differences = sum(
            row["method"][key] != base["method"][key]
            for key in base["method"]
        )
        assert differences == 1


def test_jsonl_loading_selection_and_hashing(tmp_path):
    tasks = [_task("rmps"), _task("gaussian")]
    path = write_manifest(tmp_path / "manifest.jsonl", tasks)
    assert [task["method"]["name"] for task in load_manifest(path)] == [
        "rmps",
        "gaussian",
    ]
    assert select_task(path, 1)["method"]["name"] == "gaussian"
    assert manifest_hash(path) == manifest_hash(tasks)
    with pytest.raises(IndexError):
        select_task(path, 2)

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(f"\n{lines[0]}\n\n{lines[1]}\n", encoding="utf-8")
    assert select_task(path, 1)["method"]["name"] == "gaussian"


def test_blocked_task_selection_fails_closed(tmp_path):
    task = {
        **_task(),
        "blocked": True,
        "requirements": ["block_peps"],
        "blocked_reason": "block evolution is unavailable",
    }
    path = write_manifest(tmp_path / "blocked.jsonl", [task])
    with pytest.raises(RuntimeError, match="block_peps"):
        select_task(path, 0)
    assert select_task(path, 0, allow_blocked=True)["blocked"] is True


def test_manifest_rejects_a_tampered_task_id(tmp_path):
    task = finalize_task(_task())
    task["task_id"] = "bad"
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task_id mismatch"):
        load_manifest(path)
