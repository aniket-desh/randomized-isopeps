import pytest

from rand_isopeps.campaign import column_experiments, column_methods
from rand_isopeps.campaign.column_cases import physical_state
from rand_isopeps.column.from_quimb import _center_probe_seed
from experiments.paper_campaign.manifest_common import column_methods as campaign_methods


def test_large_column_measurements_use_an_increasing_bond_pair(monkeypatch):
    def measured(_reference, _moved, _hamiltonian, bond):
        return {
            "state_infidelity": bond * 1e-6,
            "norm_drift": bond * 1e-7,
        }

    monkeypatch.setattr(column_methods, "_boundary_metrics", measured)
    result = column_methods.physical_state_metrics(
        object(),
        object(),
        None,
        {"lx": 5, "ly": 4},
        {
            "measurement_bonds": [32, 64],
            "measurement_convergence_tolerance": 1e-6,
        },
    )

    assert result["measurement_bonds"] == [32, 64]
    assert result["state_infidelity"] == 64e-6
    assert result["measurement_converged"] is False


def test_each_isometry_move_gets_distinct_reproducible_streams():
    seeds = {"problem": 1, "sketch": 2, "score": 3}
    first = column_experiments._move_seeds(seeds, 0)
    second = column_experiments._move_seeds(seeds, 1)

    assert first == column_experiments._move_seeds(seeds, 0)
    assert first["sketch"] != second["sketch"]
    assert first["score"] != second["score"]


def test_boundary_probe_seed_is_process_independent():
    assert _center_probe_seed(0, "right") == 0
    assert _center_probe_seed(0, "left") == 1
    assert _center_probe_seed(7, "right") == 14
    assert _center_probe_seed(7, "left") == 15


def test_unconverged_physical_preparation_is_rejected():
    with pytest.raises(RuntimeError, match="preparation did not converge"):
        column_experiments._require_converged_preparation({
            "preparation": {"converged": False, "sweeps": 12}
        })


def test_isometry_violation_marks_the_task_row_failed(monkeypatch):
    monkeypatch.setattr(
        column_experiments,
        "_column_isometry_defect",
        lambda _state, _column, _split: 1.0,
    )
    rows = column_experiments.run_isometry({
        "problem": {
            "lx": 2,
            "ly": 2,
            "state": "random_raw",
            "bond": 2,
            "route": "boundary_right",
            "regime": "truncated",
        },
        "method": {
            "name": "sequential_moses",
            "label": "local_det_ndis0",
            "chi": 2,
            "eta": 2,
            "ndis": 0,
        },
        "measurement": {
            "isometry_tolerance": 1e-10,
            "deterministic_repeats": 1,
        },
        "seeds": {"problem": 47, "sketch": 53, "score": 59},
    })

    assert rows[0]["isometry_passed"] is False
    assert rows[0]["status"] == "failed"
    assert "exceeds" in rows[0]["error"]


def test_paired_methods_share_score_probes_but_not_sketches():
    task = {
        "measurement": {"sketch_repeats": 2},
        "seeds": {
            "sketch": 5,
            "score": 7,
            "method": {
                "a": {"sketch": 11},
                "b": {"sketch": 13},
            },
        },
    }
    method_a = {"name": "global_rmps", "label": "a"}
    method_b = {"name": "global_rmps", "label": "b"}
    repeats_a = [seeds for _, seeds in column_experiments._method_repeats(task, method_a)]
    repeats_b = [seeds for _, seeds in column_experiments._method_repeats(task, method_b)]

    assert [row["score"] for row in repeats_a] == [row["score"] for row in repeats_b]
    assert [row["sketch"] for row in repeats_a] != [row["sketch"] for row in repeats_b]


def test_riemannian_comparator_executes_on_a_physical_column():
    prepared = physical_state(
        {
            "lx": 2,
            "ly": 2,
            "state": "random_raw",
            "bond": 2,
        },
        seed=41,
    )
    assert {
        tensor.data.dtype.name for tensor in prepared["state"].tensors
    } == {"float64"}
    method = next(
        method
        for method in campaign_methods(eta=2, include_dense=False)
        if method["name"] == "sequential_moses_riemannian"
    )
    moved, metrics = column_methods.move_physical_column(
        prepared["state"],
        prepared["column"],
        prepared["center"],
        prepared["split"],
        {**method, "chi": 2, "cutoff": 1e-10},
        {"sketch": 43},
        {},
    )

    assert moved is not None
    assert metrics["method"] == "sequential_moses_riemannian"
    assert metrics["local_svd_count"] > 0


def test_full_rank_isometry_preserves_the_declared_optimizer():
    column = type("Column", (), {"n_out": 8, "n_in": 8})()
    method = column_experiments._full_rank_method(
        {
            "name": "sequential_moses_riemannian",
            "ndis": 30,
            "disentangler": "riemannian_renyi",
        },
        column,
        max_dimension=16,
    )

    assert method["eta"] == 8
    assert method["ndis"] == 30
    assert method["disentangler"] == "riemannian_renyi"
