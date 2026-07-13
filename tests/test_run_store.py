from __future__ import annotations

import csv

import pytest

from rand_isopeps.experiment_utils.run_store import (
    VersionedCsvStore,
    content_hash,
    validate_unique_rows,
)


def test_content_hash_is_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_versioned_store_resumes_and_rejects_conflicts(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ["run_key", "value"]
    store = VersionedCsvStore(path, fields)
    assert store.append({"run_key": "abc", "value": 1}) is True
    assert store.append({"run_key": "abc", "value": 1}) is False
    with pytest.raises(ValueError, match="conflicting"):
        store.append({"run_key": "abc", "value": 2})
    with path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    resumed = VersionedCsvStore(path, fields)
    assert resumed.contains("abc")


def test_duplicate_validator_fails_before_plotting():
    assert len(validate_unique_rows([
        {"run_key": "x", "value": 1}, {"run_key": "x", "value": 1}
    ])) == 1
    with pytest.raises(ValueError, match="conflicting"):
        validate_unique_rows([
            {"run_key": "x", "value": 1}, {"run_key": "x", "value": 9}
        ])
