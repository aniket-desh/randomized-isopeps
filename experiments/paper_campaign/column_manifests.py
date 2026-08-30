"""column-move and isometry campaign manifests."""

from __future__ import annotations

import itertools

from rand_isopeps.campaign import one_at_a_time

from .manifest_common import (
    bundle,
    bundle_seeds,
    column_methods,
    global_column_methods,
    rmps_method,
    task,
)


def column_problem(column_size: int, state: str, chi: int = 8) -> dict:
    """describe one raw or prepared physical column."""
    raw = state == "random_raw"
    return {
        "kind": "prepared_column",
        "source": "physical",
        "lx": column_size,
        "ly": 4,
        "column_size": column_size,
        "lattice": [column_size, 4],
        "physical_dim": 2,
        "bond": 2,
        "source_bond": 2,
        "chi": chi,
        "prep_eta": max(8, chi),
        "initialization": "random_peps" if raw else "random_isotns",
        "state": state,
    }


def _method_comparisons() -> list[dict]:
    tasks = []
    states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
    for problem_index, (column_size, state, replicate) in enumerate(
        itertools.product(range(2, 8), states, range(8))
    ):
        include_dense = state == "random_raw" or column_size <= 4
        methods = bundle(column_methods(eta=4, include_dense=include_dense))
        tasks.append(task(
            "column_moves",
            column_problem(column_size, state),
            methods,
            bundle_seeds(problem_index, methods["names"], replicate),
            {
                "sketch_repeats": 4,
                "deterministic_repeats": 1,
                "score_probes": 16,
                "metrics": [
                    "state_infidelity",
                    "relative_frobenius_error",
                    "projection_error",
                    "isometry_error",
                    "retained_weight",
                ],
                "primary_metric": "state_infidelity",
                "study": "method_comparison",
                "dense_global_controls": include_dense,
                "measurement_bonds": [32, 64],
                "measurement_convergence_tolerance": 1e-5,
            },
            dtype="float64",
            resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
        ))
    return tasks


def _rmps_tuning() -> list[dict]:
    baseline = rmps_method(eta=4, ell=8, chi_sk=8)
    axes = {
        "eta": (2, 4, 8),
        "ell": (4, 6, 8, 12),
        "chi_sk": (1, 2, 4, 8, 16, 32),
        "kappa": (1, 2, 4),
        "n_power": (0, 1),
    }
    methods = one_at_a_time(baseline, axes)
    for index, method in enumerate(methods):
        differences = [key for key in axes if method[key] != baseline[key]]
        method["label"] = (
            "rmps_baseline"
            if index == 0
            else f"rmps_{differences[0]}_{method[differences[0]]}"
        )
    paired = bundle(methods)
    states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
    tasks = []
    for index, (column_size, state, replicate) in enumerate(
        itertools.product((4, 6, 7), states, range(8)), start=10_000
    ):
        tasks.append(task(
            "column_moves",
            column_problem(column_size, state),
            paired,
            bundle_seeds(index, paired["names"], replicate),
            {
                "sketch_repeats": 4,
                "score_probes": 16,
                "metrics": [
                    "state_infidelity",
                    "projection_error",
                    "isometry_error",
                ],
                "primary_metric": "state_infidelity",
                "study": "one_at_a_time",
                "baseline": baseline,
                "measurement_bonds": [32, 64],
                "measurement_convergence_tolerance": 1e-5,
            },
            dtype="float64",
            resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
        ))
    return tasks


def _controlled_spectra() -> list[dict]:
    tasks = []
    decays = {
        "controlled_exp": (2.0, 4.0, 8.0),
        "controlled_power": (0.5, 1.0, 2.0),
    }
    methods = bundle(global_column_methods(eta=4))
    axes = (
        (column_size, family, decay, replicate)
        for column_size in range(2, 8)
        for family, values in decays.items()
        for decay in values
        for replicate in range(6)
    )
    for index, (column_size, family, decay, replicate) in enumerate(
        axes, start=20_000
    ):
        problem = {
            "kind": "column_operator",
            "source": "synthetic",
            "lx": column_size,
            "column_size": column_size,
            "in_dims": [2] * column_size,
            "out_dims": [2] * column_size,
            "family": family,
            "decay": decay,
            "state": f"{family}@{decay:g}",
        }
        tasks.append(task(
            "column_moves",
            problem,
            methods,
            bundle_seeds(index, methods["names"], replicate),
            {
                "sketch_repeats": 4,
                "score_probes": 16,
                "metrics": [
                    "projection_error",
                    "projection_excess",
                    "spectral_floor",
                    "isometry_defect",
                ],
                "primary_metric": "projection_excess",
                "study": "controlled_spectrum",
                "global_only": True,
            },
            dtype="float64",
            resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
        ))
    return tasks


def build_column_moves() -> list[dict]:
    """compare methods, tune rmps, and stress controlled spectra."""
    return _method_comparisons() + _rmps_tuning() + _controlled_spectra()


def build_isometry() -> list[dict]:
    """measure all justified methods without oversized full-rank work."""
    tasks = []
    states = ("random_raw", "tfim@3.5", "tfim@3.04", "heis")
    routes = ("boundary_right", "interior_right", "interior_left", "round_trip")
    lattices = (
        ((2, 3), ("full_rank_oracle", "truncated")),
        ((3, 3), ("rank_saturated", "truncated")),
        ((4, 4), ("rank_saturated", "truncated")),
    )
    axes = (
        (lattice, state, regime, route, replicate)
        for lattice, regimes in lattices
        for state, regime, route, replicate in itertools.product(
            states, regimes, routes, range(4)
        )
    )
    for problem_index, (lattice, state, regime, route, replicate) in enumerate(axes):
        eta = 8 if regime == "rank_saturated" else 4
        include_dense = state == "random_raw" or lattice[0] <= 3
        methods = bundle(column_methods(eta=eta, include_dense=include_dense))
        problem = {
            **column_problem(lattice[0], state),
            "ly": lattice[1],
            "lattice": list(lattice),
            "regime": regime,
            "rank_cap": eta if regime == "rank_saturated" else None,
            "route": route,
        }
        tasks.append(task(
            "isometry",
            problem,
            methods,
            bundle_seeds(problem_index, methods["names"], replicate),
            {
                "sketch_repeats": 4,
                "deterministic_repeats": 1,
                "isometry_tolerance": 1e-10,
                "full_rank_max_dimension": 256,
                "recenter_rank": eta,
                "dense_global_controls": include_dense,
                "metrics": [
                    "local_isometry_defects",
                    "max_local_isometry_defect",
                    "isometry_passed",
                    "state_infidelity",
                    "norm_drift",
                ],
                "primary_metric": "max_local_isometry_defect",
                "measurement_bonds": [32, 64],
                "measurement_convergence_tolerance": 1e-5,
            },
            dtype="float64",
            resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
        ))
    return tasks
