"""shared configuration and measurement helpers for physics campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from rand_isopeps.physics.measurements import low_energy_references


def value(problem: dict, method: dict, name: str, default=None):
    return method.get(name, problem.get(name, default))


def schedule(problem: dict, method: dict) -> list[float]:
    stages = value(problem, method, "stages")
    if stages:
        values = []
        for stage in stages:
            if isinstance(stage, dict):
                tau, count = float(stage["tau"]), int(stage["iterations"])
            else:
                tau, count = float(stage[0]), int(stage[1])
            values.extend([tau] * count)
        return values
    return [float(value(problem, method, "tau", 0.1))] * int(
        value(problem, method, "iterations", 50)
    )


def uses_block_path(problem: dict) -> bool:
    return (
        int(problem.get("states", 1)) > 1
        or str(problem.get("study", "")) == "dektor_reproduction"
    )


def site_vectors(lx: int, ly: int, seed: int):
    rng = np.random.default_rng(int(seed))
    vectors = {}
    for site in np.ndindex(lx, ly):
        vector = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        vectors[site] = vector / np.linalg.norm(vector)
    return vectors


def external_references(path, problem: dict, count: int):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    for record in records:
        matches = (
            record.get("hamiltonian")
            == problem.get("hamiltonian", problem.get("state"))
            and int(record.get("lx", -1)) == int(problem["lx"])
            and int(record.get("ly", -1)) == int(problem["ly"])
        )
        if not matches:
            continue
        if record.get("validation_passed") is not True:
            raise ValueError("matching reference record has not passed validation")
        is_dektor_heisenberg = (
            str(problem.get("study", "")) == "dektor_reproduction"
            and str(problem.get("hamiltonian", "")) in {"heis", "heisenberg"}
        )
        if is_dektor_heisenberg:
            expected = [float(index) for index in range(count)]
            sectors = record.get("reference_metadata", {}).get("symmetry_sector")
            if not isinstance(sectors, list) or sectors[:count] != expected:
                raise ValueError(
                    "Dektor Heisenberg references require validated "
                    f"total-Sz sectors {expected}"
                )
        energies = [float(value) for value in record["energies"]]
        if len(energies) < count:
            raise ValueError("matching reference record contains too few states")
        metadata = record.get("reference_metadata", {})
        source = record.get(
            "reference_source",
            metadata.get("reference_source", metadata.get("method", "artifact")),
        )
        return energies[:count], str(source)
    return None


def references(
    h,
    problem: dict,
    measurement: dict,
    count: int,
    *,
    require_external: bool = False,
):
    reference_path = measurement.get("reference_path") or os.environ.get(
        "RAND_ISOPEPS_REFERENCE_PATH"
    )
    if require_external:
        if reference_path is None:
            raise RuntimeError(
                "reference_artifact requirement needs a reference artifact path"
            )
        external = external_references(reference_path, problem, count)
        if external is None:
            raise ValueError(
                "reference_artifact requirement has no matching validated cell"
            )
        return external
    supplied = problem.get("reference_energies")
    if supplied is not None:
        values = [float(value) for value in supplied]
        if len(values) < count:
            raise ValueError("reference_energies does not contain every requested state")
        return values[:count], str(problem.get("reference_source", "manifest"))
    external = external_references(reference_path, problem, count)
    if external is not None:
        return external
    if h is not None:
        return low_energy_references(h, count, seed=17), "exact_sparse"
    return None, "unavailable"


def eigenvalue_error_fields(energies, references) -> dict:
    count = len(energies)
    if references is None:
        empty = [None] * count
        return {
            "eigenvalue_errors": empty.copy(),
            "relative_eigenvalue_errors": empty.copy(),
            "absolute_relative_eigenvalue_errors": empty,
            "ground_energy_errors": [None] * count,
        }
    errors = [
        float(energy - references[index])
        for index, energy in enumerate(energies)
    ]
    relative = [
        None if references[index] == 0.0 else float(error / references[index])
        for index, error in enumerate(errors)
    ]
    return {
        "eigenvalue_errors": errors,
        "relative_eigenvalue_errors": relative,
        "absolute_relative_eigenvalue_errors": [
            None if value is None else abs(value) for value in relative
        ],
        "ground_energy_errors": errors,
    }


def measurement_settings(problem: dict, measurement: dict):
    configured = measurement.get(
        "measurement_bonds", problem.get("measurement_bonds")
    )
    if configured is None:
        base = int(problem.get("measurement_bond", 64))
        bonds = (base, 2 * base)
    else:
        bonds = tuple(int(value) for value in configured)
    if len(bonds) != 2 or bonds[0] < 1 or bonds[1] != 2 * bonds[0]:
        raise ValueError("measurement_bonds must be [B, 2B]")
    tolerance = float(
        measurement.get(
            "measurement_convergence_tolerance",
            problem.get("measurement_convergence_tolerance", 1e-6),
        )
    )
    if tolerance < 0.0:
        raise ValueError("measurement_convergence_tolerance must be nonnegative")
    cutoff = float(
        measurement.get(
            "measurement_cutoff", problem.get("measurement_cutoff", 1e-10)
        )
    )
    return bonds, cutoff, tolerance


def measurement_convergence(bonds, estimates, tolerance: float) -> dict:
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("energy estimates must have shape (2, states)")
    differences = np.abs(values[1] - values[0])
    return {
        "measurement_bonds": [int(value) for value in bonds],
        "measurement_energies_by_bond": {
            str(bond): [float(value) for value in row]
            for bond, row in zip(bonds, values)
        },
        "measurement_energy_differences": [
            float(value) for value in differences
        ],
        "measurement_convergence_tolerance": float(tolerance),
        "measurement_converged": bool(np.all(differences <= tolerance)),
    }


def hamiltonian(problem: dict):
    from rand_isopeps.real_isotns.tebd2 import ham_from_spec

    name = str(problem.get("hamiltonian", problem.get("state", "tfim@3.5")))
    operator = ham_from_spec(name, int(problem["lx"]), int(problem["ly"]))
    if operator is None:
        raise ValueError("physics experiments require a Hamiltonian")
    return operator
