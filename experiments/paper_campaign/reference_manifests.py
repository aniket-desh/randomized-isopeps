"""independent exact and DMRG reference manifests."""

from __future__ import annotations

from .manifest_common import method_seeds, task
from .physics_manifests import build_physics


def _physics_cells() -> dict[tuple[int, int, str], int]:
    cells = {}
    for physics in build_physics():
        problem = physics["problem"]
        if problem["study"] in {"correctness_ladder", "block_correctness", "gpu_pilot"}:
            continue
        key = (int(problem["lx"]), int(problem["ly"]), str(problem["hamiltonian"]))
        cells[key] = max(cells.get(key, 1), int(problem.get("states", 1)))
    return cells


def _target_sectors(hamiltonian: str, states: int) -> list[dict]:
    if hamiltonian != "heis":
        return []
    sectors = (0, 1)
    if states > len(sectors):
        raise ValueError("the Heisenberg reference contract defines only alpha0 and alpha1")
    return [
        {"state_index": index, "target_sz": sectors[index]}
        for index in range(states)
    ]


def _progression(maximum: int) -> list[int]:
    values = [value for value in (16, 32, 64, 128, 256, 512, 1000, 2500, 5000) if value <= maximum]
    if maximum not in values:
        values.append(maximum)
    return values


def _dmrg_task(
    problem_index: int,
    lx: int,
    ly: int,
    hamiltonian: str,
    states: int,
    maximum_bond: int,
    *,
    convergence_pair: tuple[int, int],
) -> dict:
    sectors = _target_sectors(hamiltonian, states)
    problem = {
        "lx": lx,
        "ly": ly,
        "hamiltonian": hamiltonian,
        "states": states,
        "study": "mps_reference",
        "reference_tier": "paper_energy",
    }
    measurement = {
        "primary_metric": "energy",
        "validation_contract": "nested_bond_energy_orthogonality",
        "residual_required": False,
        "overlap_tolerance": 1e-6,
        "energy_convergence_pair": list(convergence_pair),
        "energy_convergence_tolerance": 1e-5,
        "require_sector_validation": bool(sectors),
        "stage": 2,
    }
    requirements = ["quimb"]
    if sectors:
        requirements.append("sector_validation")
    method = {
        "name": "dmrg_reference",
        "bond_dims": _progression(maximum_bond),
        "cutoff": 1e-10,
        "tolerance": 1e-8,
        "max_sweeps": 16,
        "projector_bond": min(2 * maximum_bond, 512),
        "restarts": 3,
        "target_sectors": sectors,
        "sector_penalty_shift": 100.0,
        "sector_tolerance": 1e-8,
    }
    if states > 1 and not sectors:
        method.update(
            projector_state_bond=64,
            projector_state_tolerance=1e-4,
        )
    return task(
        "reference",
        problem,
        method,
        method_seeds(50_000 + problem_index, maximum_bond, 0),
        measurement,
        resources={
            "hardware": "cpu",
            "cpus": 128,
            "gpus": 0,
            "stop_grace_seconds": 7200,
        },
        requirements=requirements,
    )


def build_references() -> list[dict]:
    """build exact small cells plus staged residual and paper-scale DMRG cells."""
    tasks = []
    for problem_index, ((lx, ly, hamiltonian), states) in enumerate(
        sorted(_physics_cells().items())
    ):
        sectors = _target_sectors(hamiltonian, states)
        if lx * ly <= 16:
            requirements = ["scipy"]
            if sectors:
                requirements.append("sector_validation")
            tasks.append(task(
                "reference",
                {
                    "lx": lx,
                    "ly": ly,
                    "hamiltonian": hamiltonian,
                    "states": states,
                    "study": "exact_reference",
                    "reference_tier": "exact",
                },
                {
                    "name": "exact_diagonalization",
                    "target_sectors": sectors,
                    "sector_penalty_shift": 100.0,
                    "sector_tolerance": 1e-10,
                },
                method_seeds(50_000 + problem_index, 0, 0),
                {
                    "primary_metric": "energy",
                    "validation_contract": "exact_residual",
                    "residual_tolerance": 1e-10,
                    "overlap_tolerance": 1e-10,
                    "require_sector_validation": bool(sectors),
                    "stage": 1,
                },
                dtype="float64",
                resources={"hardware": "cpu", "cpus": 64, "gpus": 0},
                requirements=requirements,
            ))
            continue

        paper_pair = (
            (500, 1000)
            if hamiltonian == "heis"
            else (2500, 5000)
            if hamiltonian.startswith("tfim@")
            else (512, 1000)
        )
        for maximum_bond in paper_pair:
            tasks.append(_dmrg_task(
                problem_index,
                lx,
                ly,
                hamiltonian,
                states,
                maximum_bond,
                convergence_pair=paper_pair,
            ))
    return tasks
