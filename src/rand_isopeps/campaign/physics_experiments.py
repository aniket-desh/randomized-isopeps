"""dispatch resumable dense, single-peps, and shared-block trajectories."""

from __future__ import annotations

import os
from pathlib import Path

from rand_isopeps.physics import (
    bond_hamiltonians,
    checkerboard_layers,
    sparse_hamiltonian,
)

from .physics_block import (
    block_energy_record as _block_energy_record,
    block_trajectory as _block_trajectory,
    initial_block as _initial_block,
)
from .physics_common import (
    eigenvalue_error_fields as _eigenvalue_error_fields,
    external_references as _external_references,
    hamiltonian as _hamiltonian,
    measurement_convergence as _measurement_convergence,
    measurement_settings as _measurement_settings,
    references as _references,
    schedule as _schedule,
    site_vectors as _site_vectors,
    uses_block_path as _uses_block_path,
    value as _value,
)
from .physics_dense import (
    dense_initial as _dense_initial,
    dense_product as _dense_product,
    dense_reorthogonalize as _dense_reorthogonalize,
    dense_trajectory as _dense_trajectory,
    energy_record as _energy_record,
)
from .physics_single import (
    device_peps as _device_peps,
    peps_measurement as _peps_measurement,
    peps_options as _peps_options,
    peps_trajectory as _peps_trajectory,
)


def run_physics(
    task: dict,
    *,
    checkpoint_path: str | Path | None = None,
    stop_requested=lambda: False,
) -> list[dict]:
    """run one dense, single-state, or shared-block peps trajectory."""
    problem, method = task["problem"], task["method"]
    if "reference_artifact" in task.get("requirements", ()) and not (
        task["measurement"].get("reference_path")
        or os.environ.get("RAND_ISOPEPS_REFERENCE_PATH")
    ):
        raise RuntimeError(
            "physics task requires RAND_ISOPEPS_REFERENCE_PATH from validated references"
        )
    hamiltonian = _hamiltonian(problem)
    lx, ly = int(problem["lx"]), int(problem["ly"])
    n_sites = lx * ly
    dense_limit = int(task["measurement"].get("dense_oracle_max_sites", 16))
    reference_limit = int(task["measurement"].get("reference_max_sites", 16))
    needs_dense = str(method["name"]).startswith("dense_")
    if needs_dense and n_sites > dense_limit:
        raise ValueError("dense trajectory exceeds dense_oracle_max_sites")
    h_limit = max(dense_limit if needs_dense else 0, reference_limit)
    h = sparse_hamiltonian(hamiltonian, lx, ly) if n_sites <= h_limit else None
    oracle_h = h if n_sites <= dense_limit else None
    bonds = (
        bond_hamiltonians(hamiltonian, lx, ly)
        if oracle_h is not None else None
    )
    layers = (
        checkerboard_layers(hamiltonian, lx, ly)
        if oracle_h is not None else None
    )
    references, source = _references(
        h if n_sites <= reference_limit else None,
        problem,
        task["measurement"],
        int(problem.get("states", 1)),
        require_external="reference_artifact" in task.get("requirements", ()),
    )
    vectors = _site_vectors(lx, ly, int(task["seeds"]["problem"]))
    schedule = _schedule(problem, method)
    if needs_dense:
        return _dense_trajectory(
            task,
            h,
            bonds,
            layers,
            schedule,
            vectors,
            references,
            source,
            checkpoint_path,
            stop_requested,
        )
    if _uses_block_path(problem):
        return _block_trajectory(
            task,
            hamiltonian,
            oracle_h,
            bonds,
            schedule,
            references,
            source,
            checkpoint_path,
            stop_requested,
        )
    return _peps_trajectory(
        task,
        hamiltonian,
        oracle_h,
        bonds,
        schedule,
        vectors,
        references,
        source,
        checkpoint_path,
        stop_requested,
    )
