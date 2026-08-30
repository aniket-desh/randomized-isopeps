"""high-bond mps reference tasks for large physics cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from rand_isopeps.physics import sparse_hamiltonian
from rand_isopeps.physics.mps_reference import dmrg_low_energies

from .checkpoint import load_checkpoint, save_checkpoint


def _target_sectors(problem: dict, method: dict, count: int):
    values = method.get("target_sectors", problem.get("target_sectors"))
    if values is None:
        return None
    values = tuple(values)
    if not values:
        return None
    if values and isinstance(values[0], dict):
        indexed = {
            int(value["state_index"]): float(value["target_sz"])
            for value in values
        }
        if len(indexed) != len(values) or set(indexed) != set(range(count)):
            raise ValueError("target_sectors must index every requested state once")
        sectors = tuple(indexed[index] for index in range(count))
    else:
        sectors = tuple(float(value) for value in values)
    if len(sectors) != int(count):
        raise ValueError("target_sectors must contain one value per state")
    return sectors


def _lowest_eigenpairs(matrix, count: int, seed: int):
    dimension = int(matrix.shape[0])
    if count < 1 or count > dimension:
        raise ValueError("the requested eigenvalue count is out of range")
    if dimension <= max(count + 1, 4) or count == dimension:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return values[:count], vectors[:, :count]
    rng = np.random.default_rng(int(seed))
    values, vectors = spla.eigsh(
        matrix,
        k=int(count),
        which="SA",
        v0=rng.standard_normal(dimension),
    )
    order = np.argsort(values.real)
    return values[order].real, vectors[:, order]


def _sector_indices(n_sites: int, target: float) -> np.ndarray:
    down_spins = n_sites / 2.0 - float(target)
    rounded = int(round(down_spins))
    if abs(down_spins - rounded) > 1e-12 or not 0 <= rounded <= n_sites:
        raise ValueError(f"invalid total-Sz sector {target} for {n_sites} spins")
    return np.fromiter(
        (
            basis
            for basis in range(2**n_sites)
            if basis.bit_count() == rounded
        ),
        dtype=np.int64,
    )


def _sector_eigenpairs(matrix, n_sites: int, sectors, seed: int):
    counts = {sector: sectors.count(sector) for sector in set(sectors)}
    solved = {}
    for offset, (sector, count) in enumerate(sorted(counts.items())):
        indices = _sector_indices(n_sites, sector)
        values, vectors = _lowest_eigenpairs(
            matrix[indices][:, indices], count, seed + offset
        )
        solved[sector] = (indices, values, vectors)
    used = {sector: 0 for sector in counts}
    values = []
    vectors = []
    for sector in sectors:
        index = used[sector]
        indices, sector_values, sector_vectors = solved[sector]
        vector = np.zeros(matrix.shape[0], dtype=sector_vectors.dtype)
        vector[indices] = sector_vectors[:, index]
        values.append(float(sector_values[index]))
        vectors.append(vector)
        used[sector] += 1
    return np.asarray(values), np.column_stack(vectors)


def _reference_hamiltonian(problem: dict):
    from rand_isopeps.real_isotns.tebd2 import ham_from_spec

    hamiltonian = ham_from_spec(
        str(problem["hamiltonian"]), int(problem["lx"]), int(problem["ly"])
    )
    if hamiltonian is None:
        raise ValueError("a reference task requires a Hamiltonian")
    return hamiltonian


def _exact_records(matrix, values, vectors, sectors):
    records = []
    for index, energy in enumerate(values):
        residual = matrix @ vectors[:, index] - energy * vectors[:, index]
        overlaps = np.abs(vectors[:, :index].conj().T @ vectors[:, index])
        target = None if sectors is None else sectors[index]
        records.append({
            "state_index": index,
            "converged": True,
            "solver_converged": True,
            "energy": float(energy),
            "residual_norm": float(np.linalg.norm(residual)),
            "max_previous_overlap": float(np.max(overlaps, initial=0.0)),
            "target_total_sz": target,
            "total_sz": target,
            "total_sz_variance": 0.0 if target is not None else None,
            "sector_validated": True if target is not None else None,
        })
    return records


def _run_exact_reference(task, hamiltonian, count, sectors):
    problem, method = task["problem"], task["method"]
    matrix = sparse_hamiltonian(
        hamiltonian, int(problem["lx"]), int(problem["ly"])
    )
    seed = int(task["seeds"]["problem"])
    if sectors is None:
        values, vectors = _lowest_eigenpairs(matrix, count, seed)
    else:
        values, vectors = _sector_eigenpairs(
            matrix, int(problem["lx"]) * int(problem["ly"]), sectors, seed
        )
    records = _exact_records(matrix, values, vectors, sectors)
    metadata = {
        "backend": "numpy",
        "method": "exact_sparse",
        "symmetry_sector": "unrestricted" if sectors is None else list(sectors),
        "sector_tolerance": (
            None
            if sectors is None
            else float(method.get("sector_tolerance", 1e-10))
        ),
    }
    return vectors, records, metadata


def _save_reference_checkpoint(
    checkpoint_path,
    task_id,
    states,
    records,
    restart_progress=None,
):
    if checkpoint_path is None:
        return
    payload = {
        "task_id": task_id,
        "iteration": len(records),
        "state": states,
        "rows": records,
    }
    if restart_progress is not None:
        payload["dmrg_restart"] = restart_progress
    save_checkpoint(checkpoint_path, payload)


def _dmrg_callbacks(task_id, checkpoint_path, stop_requested):
    def save(states, records, restart_progress=None):
        _save_reference_checkpoint(
            checkpoint_path,
            task_id,
            states,
            records,
            restart_progress,
        )

    def checkpoint_callback(states, records, _metadata):
        save(states, records)
        if stop_requested():
            raise InterruptedError("checkpointed after campaign wall-clock stop")

    def restart_callback(states, records, _metadata, restart_progress):
        save(states, records, restart_progress)
        if stop_requested():
            raise InterruptedError("checkpointed after a completed dmrg restart")

    def sweep_callback(states, records, _metadata, restart_progress):
        save(states, records, restart_progress)
        if stop_requested():
            raise InterruptedError("checkpointed after a completed dmrg sweep")

    return checkpoint_callback, restart_callback, sweep_callback


def _dmrg_residual_settings(task):
    residual_required = bool(task["measurement"].get("residual_required", True))
    residual_max_bond = task["method"].get("residual_max_bond", 512)
    if residual_required and residual_max_bond is None:
        raise ValueError(
            "residual_max_bond is required when residual_required is true"
        )
    return residual_required, residual_max_bond


def _dmrg_options(task, sectors, checkpoint, callbacks):
    problem, method = task["problem"], task["method"]
    residual_required, residual_max_bond = _dmrg_residual_settings(task)
    return {
        "bond_dims": tuple(
            int(value) for value in method.get("bond_dims", (64, 128, 256))
        ),
        "cutoff": float(method.get("cutoff", 1e-10)),
        "tolerance": float(method.get("tolerance", 1e-8)),
        "max_sweeps": int(method.get("max_sweeps", 12)),
        "projector_bond": int(method.get("projector_bond", 512)),
        "projector_state_bond": method.get("projector_state_bond"),
        "projector_state_tolerance": float(
            method.get("projector_state_tolerance", 0.1)
        ),
        "residual_max_bond": residual_max_bond,
        "residual_max_bytes": method.get("residual_max_bytes", 8 * 1024**3),
        "compute_residual": residual_required,
        "penalty_shift": method.get("penalty_shift"),
        "target_sectors": sectors,
        "sector_penalty_shift": method.get("sector_penalty_shift"),
        "sector_tolerance": float(method.get("sector_tolerance", 1e-6)),
        "backend": str(task.get("backend", "numpy")),
        "restarts": int(method.get("restarts", 3)),
        "sweep_sequence": str(method.get("sweep_sequence", "RL")),
        "seed": int(task["seeds"]["problem"]),
        "initial_states": () if checkpoint is None else checkpoint["state"],
        "initial_records": () if checkpoint is None else checkpoint["rows"],
        "initial_restart_progress": (
            None if checkpoint is None else checkpoint.get("dmrg_restart")
        ),
        "checkpoint_callback": callbacks[0],
        "restart_callback": callbacks[1],
        "sweep_callback": callbacks[2],
    }


def _run_dmrg_reference(
    task,
    hamiltonian,
    count,
    sectors,
    checkpoint_path,
    stop_requested,
):
    _dmrg_residual_settings(task)
    checkpoint = (
        load_checkpoint(checkpoint_path, task["task_id"])
        if checkpoint_path is not None
        else None
    )
    callbacks = _dmrg_callbacks(
        task["task_id"], checkpoint_path, stop_requested
    )
    if stop_requested():
        _save_reference_checkpoint(
            checkpoint_path,
            task["task_id"],
            [] if checkpoint is None else checkpoint["state"],
            [] if checkpoint is None else checkpoint["rows"],
            None if checkpoint is None else checkpoint.get("dmrg_restart"),
        )
        raise InterruptedError("checkpointed before the next dmrg solve")
    problem = task["problem"]
    return dmrg_low_energies(
        hamiltonian,
        int(problem["lx"]),
        int(problem["ly"]),
        count,
        **_dmrg_options(task, sectors, checkpoint, callbacks),
    )


def _reference_passed(records, measurement):
    residual_tolerance = float(measurement.get("residual_tolerance", 1e-5))
    overlap_tolerance = float(measurement.get("overlap_tolerance", 1e-6))
    residual_required = bool(measurement.get("residual_required", True))
    return all(
        record["converged"]
        and (
            not residual_required
            or record["residual_norm"] <= residual_tolerance
        )
        and record["max_previous_overlap"] <= overlap_tolerance
        for record in records
    )


def _reference_result(task, records, metadata, passed):
    problem = task["problem"]
    residual_required = bool(task["measurement"].get("residual_required", True))
    sector_targeted = metadata["symmetry_sector"] != "unrestricted"
    reference_source = (
        f"{metadata['method']}_sector_targeted"
        if sector_targeted
        else f"{metadata['method']}_unrestricted"
    )
    return {
        "hamiltonian": str(problem["hamiltonian"]),
        "lx": int(problem["lx"]),
        "ly": int(problem["ly"]),
        "states": int(problem.get("states", 1)),
        "energies": [record["energy"] for record in records],
        "residual_norms": [record["residual_norm"] for record in records],
        "sector_expectations": [record.get("total_sz") for record in records],
        "sector_variances": [
            record.get("total_sz_variance") for record in records
        ],
        "sector_validation_passed": all(
            record.get("sector_validated") is not False for record in records
        ),
        "records": records,
        "reference_source": reference_source,
        "residual_required": residual_required,
        "reference_metadata": metadata,
        "validation_passed": passed,
        "status": "ok" if passed else "failed",
        "error": (
            None
            if passed
            else "reference convergence, residual, sector, or orthogonality gate failed"
        ),
    }


def run_reference(
    task: dict,
    *,
    checkpoint_path: str | Path | None = None,
    stop_requested=lambda: False,
):
    """compute and validate one low-energy reference cell."""
    problem, method = task["problem"], task["method"]
    hamiltonian = _reference_hamiltonian(problem)
    count = int(problem.get("states", 1))
    sectors = _target_sectors(problem, method, count)
    if method["name"] == "exact_diagonalization":
        states, records, metadata = _run_exact_reference(
            task, hamiltonian, count, sectors
        )
    elif method["name"] == "dmrg_reference":
        states, records, metadata = _run_dmrg_reference(
            task,
            hamiltonian,
            count,
            sectors,
            checkpoint_path,
            stop_requested,
        )
    else:
        raise ValueError(f"unknown reference method: {method['name']!r}")
    passed = _reference_passed(records, task["measurement"])
    _save_reference_checkpoint(
        checkpoint_path, task["task_id"], states, records
    )
    return [_reference_result(task, records, metadata, passed)]
