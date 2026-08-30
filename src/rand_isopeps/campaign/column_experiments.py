"""column-comparison and isometry experiment tasks."""

from __future__ import annotations

import copy

import numpy as np

from rand_isopeps.real_isotns.column_bridge import validate_peps_structure

from .seeds import derive_seed

from .column_cases import physical_state
from .column_methods import (
    is_local_method,
    move_physical_column,
    physical_state_metrics,
    run_synthetic_method,
)


def method_configs(task: dict) -> list[dict]:
    """normalize a single method or a paired bundle into configurations."""
    method = task["method"]
    if method.get("name") != "paired_bundle":
        return [dict(method)]
    configs = [dict(config) for config in method.get("configs", [])]
    if not configs:
        raise ValueError("paired_bundle requires nonempty method.configs")
    if any(not str(config.get("name", "")) for config in configs):
        raise ValueError("bundled method names must be nonempty")
    labels = [str(config.get("label", index)) for index, config in enumerate(configs)]
    if len(labels) != len(set(labels)):
        raise ValueError("bundled method labels must be unique")
    return configs


def _task_with_method(task: dict, method: dict) -> dict:
    out = copy.deepcopy(task)
    out["method"] = dict(method)
    return out


def _move_seeds(seeds: dict, move_index: int) -> dict:
    return {
        **seeds,
        "sketch": derive_seed(seeds["sketch"], "move", int(move_index)),
        "score": derive_seed(seeds["score"], "move", int(move_index)),
    }


def _method_repeats(task: dict, method: dict):
    name = str(method["name"])
    randomized = name not in {
        "sequential_moses",
        "sequential_moses_riemannian",
        "local_moses",
    }
    count = int(task["measurement"].get(
        "sketch_repeats" if randomized else "deterministic_repeats", 1
    ))
    label = str(method.get("label", name))
    method_stream = task["seeds"].get("method", {}).get(label, {})
    root_sketch = int(method_stream.get("sketch", task["seeds"]["sketch"]))
    root_score = int(task["seeds"]["score"])
    for repeat in range(count):
        seeds = dict(task["seeds"])
        seeds["sketch"] = derive_seed(root_sketch, "repeat", repeat)
        seeds["score"] = derive_seed(root_score, "repeat", repeat)
        yield repeat, seeds


def _column_isometry_defect(psi, j: int, split: str) -> float:
    from rand_isopeps.real_isotns.column_bridge import extract_column

    column = extract_column(psi, int(j), split=split)
    matrix = column.materialize()
    gram = matrix.conj().T @ matrix
    return float(np.linalg.norm(gram - np.eye(gram.shape[0])))


def _require_converged_preparation(case: dict) -> None:
    preparation = case.get("preparation", {})
    if preparation.get("converged") is False:
        sweeps = preparation.get("sweeps", "unknown")
        raise RuntimeError(
            f"physical-state preparation did not converge after {sweeps} sweeps"
        )


def run_column_comparison(task: dict) -> list[dict]:
    """run paired methods on one synthetic or prepared physical column."""
    problem = task["problem"]
    methods = method_configs(task)
    if str(problem.get("source", "synthetic")) == "synthetic":
        rows = []
        for method in methods:
            for repeat, seeds in _method_repeats(task, method):
                configured = _task_with_method(task, method)
                configured["seeds"] = seeds
                try:
                    result = run_synthetic_method(configured)
                    status, error = "ok", None
                except (MemoryError, ValueError, RuntimeError) as exc:
                    result = {}
                    status, error = "failed", f"{type(exc).__name__}: {exc}"
                rows.append({
                    **result,
                    "method": str(method["name"]),
                    "method_label": str(method.get("label", method["name"])),
                    "method_config": method,
                    "sketch_repeat": repeat,
                    "status": status,
                    "error": error,
                })
        return rows

    case = physical_state(problem, int(task["seeds"]["problem"]))
    _require_converged_preparation(case)
    rows = []
    for method in methods:
        base = {
            "method": str(method["name"]),
            "method_label": str(method.get("label", method["name"])),
            "method_config": method,
            "state": str(problem.get("state", "random")),
            "lx": int(problem["lx"]),
            "ly": int(problem["ly"]),
            "center": int(case["center"]),
            "split": case["split"],
            "preparation": case["preparation"],
        }
        for repeat, seeds in _method_repeats(task, method):
            try:
                moved, method_metrics = move_physical_column(
                    case["state"], case["column"], case["center"], case["split"],
                    method, seeds, task["measurement"],
                )
                validate_peps_structure(moved)
                rows.append({
                    **base,
                    **physical_state_metrics(
                        case["state"],
                        moved,
                        case["hamiltonian"],
                        problem,
                        task["measurement"],
                    ),
                    **method_metrics,
                    "sketch_repeat": repeat,
                    "status": "ok",
                    "error": None,
                })
            except (MemoryError, ValueError, RuntimeError) as exc:
                rows.append({
                    **base,
                    "sketch_repeat": repeat,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return rows


def _route_boundary(route: str, ly: int) -> int:
    if route == "interior_left":
        return ly - 1
    if route in {"boundary", "boundary_right", "interior", "interior_right", "round_trip"}:
        return 0
    raise ValueError(f"unknown isometry route: {route!r}")


def _move_sequence(center: int, ly: int, route: str):
    if ly < 2:
        raise ValueError("isometry routes require at least two columns")
    expected = _route_boundary(route, ly)
    if center != expected:
        raise ValueError(f"route {route!r} must start at column {expected}")
    if route in {"boundary", "boundary_right"}:
        return [(0, "right")]
    if route in {"interior", "interior_right"}:
        if ly < 3:
            raise ValueError("an interior move requires at least three columns")
        return [(0, "right"), (1, "right")]
    if route == "interior_left":
        if ly < 3:
            raise ValueError("an interior move requires at least three columns")
        return [(ly - 1, "left"), (ly - 2, "left")]
    if route == "round_trip":
        return (
            [(j, "right") for j in range(ly - 1)]
            + [(j, "left") for j in range(ly - 1, 0, -1)]
        )
    raise ValueError(f"unknown isometry route: {route!r}")


def _full_rank_method(method: dict, column, max_dimension: int | None = None) -> dict:
    configured = dict(method)
    rank = min(int(column.n_out), int(column.n_in))
    if max_dimension is not None and rank > int(max_dimension):
        raise ValueError(
            f"full-rank oracle dimension {rank} exceeds cap {int(max_dimension)}"
        )
    configured.update({
        "eta": rank,
        "cutoff": 0.0,
        "ndis": int(configured.get("ndis", 0)),
        "absorption_bond": None,
        "absorption_cutoff": 0.0,
    })
    if is_local_method(str(configured["name"])):
        configured["chi"] = rank
    else:
        configured.update({"ell": rank, "kappa": rank, "chi_sk": max(rank, 1)})
    return configured


def _center_at_boundary(
    state,
    center: int,
    target: int,
    seeds: dict,
    measurement: dict,
    *,
    full_rank: bool,
):
    from rand_isopeps.real_isotns.column_bridge import extract_column

    out = state.copy()
    direction = 1 if target > center else -1
    while center != target:
        split = "right" if direction == 1 else "left"
        column = extract_column(out, center, split=split)
        if full_rank:
            method = _full_rank_method(
                {"name": "sequential_moses"},
                column,
                int(measurement.get("full_rank_max_dimension", 256)),
            )
        else:
            rank = int(measurement.get("recenter_rank", 8))
            method = {
                "name": "sequential_moses",
                "eta": rank,
                "chi": rank,
                "cutoff": 1e-10,
            }
        out, _ = move_physical_column(
            out, column, center, split, method, seeds, measurement
        )
        center += direction
    validate_peps_structure(out)
    return out, center


def run_isometry(task: dict) -> list[dict]:
    """exercise each route and fail explicit arrow-isometry violations."""
    problem = task["problem"]
    route = str(problem.get("route", task["measurement"].get("route", "boundary")))
    tolerance = float(task["measurement"].get("isometry_tolerance", 1e-10))
    if tolerance < 0.0:
        raise ValueError("isometry_tolerance must be nonnegative")
    base = physical_state(problem, int(task["seeds"]["problem"]))
    _require_converged_preparation(base)
    target = _route_boundary(route, int(base["state"].Ly))
    centered, center = _center_at_boundary(
        base["state"],
        int(base["center"]),
        target,
        task["seeds"],
        task["measurement"],
        full_rank=problem.get("regime") == "full_rank_oracle",
    )
    sequence = _move_sequence(center, int(centered.Ly), route)
    rows = []
    for method in method_configs(task):
        for repeat, seeds in _method_repeats(task, method):
            state = centered.copy()
            defects = []
            move_rows = []
            try:
                for move_index, (j, split) in enumerate(sequence):
                    from rand_isopeps.real_isotns.column_bridge import extract_column

                    move_seeds = _move_seeds(seeds, move_index)
                    column = extract_column(state, j, split=split)
                    configured = (
                        _full_rank_method(
                            method,
                            column,
                            int(task["measurement"].get("full_rank_max_dimension", 256)),
                        )
                        if problem.get("regime") == "full_rank_oracle"
                        else method
                    )
                    state, metrics = move_physical_column(
                        state,
                        column,
                        j,
                        split,
                        configured,
                        move_seeds,
                        task["measurement"],
                    )
                    validate_peps_structure(state)
                    defects.append(_column_isometry_defect(state, j, split))
                    move_rows.append({
                        **metrics,
                        "executed_method_config": configured,
                        "move_index": move_index,
                        "sketch_seed": move_seeds["sketch"],
                        "score_seed": move_seeds["score"],
                    })
                final_center = sequence[-1][0] + (1 if sequence[-1][1] == "right" else -1)
                next_split = "right" if final_center < state.Ly - 1 else "left"
                extract_column(state, final_center, split=next_split)
                state_metrics = physical_state_metrics(
                    centered,
                    state,
                    base["hamiltonian"],
                    problem,
                    task["measurement"],
                )
                max_defect = max(defects, default=0.0)
                passed = max_defect <= tolerance
                rows.append({
                    "method": str(method["name"]),
                    "method_label": str(method.get("label", method["name"])),
                    "method_config": method,
                    "regime": str(problem.get("regime", "truncated")),
                    "route": route,
                    "state": str(problem.get("state", "random")),
                    "lx": int(problem["lx"]),
                    "ly": int(problem["ly"]),
                    "sketch_repeat": repeat,
                    "move_count": len(sequence),
                    "local_isometry_defects": defects,
                    "max_local_isometry_defect": max_defect,
                    "peps_structure_valid": True,
                    "next_column_extractable": True,
                    "isometry_passed": bool(passed),
                    "moves": move_rows,
                    "status": "ok" if passed else "failed",
                    "error": (
                        None
                        if passed
                        else (
                            f"isometry defect {max_defect:.3e} exceeds "
                            f"{tolerance:.3e}"
                        )
                    ),
                    **state_metrics,
                })
            except (MemoryError, ValueError, RuntimeError) as exc:
                rows.append({
                    "method": str(method["name"]),
                    "method_label": str(method.get("label", method["name"])),
                    "method_config": method,
                    "regime": str(problem.get("regime", "truncated")),
                    "route": route,
                    "state": str(problem.get("state", "random")),
                    "lx": int(problem["lx"]),
                    "ly": int(problem["ly"]),
                    "sketch_repeat": repeat,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return rows
