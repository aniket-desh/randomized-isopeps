"""small GPU promotion gate and optional crossover manifests."""

from __future__ import annotations

import itertools

from .manifest_common import hamiltonian, method_seeds, task
from .physics_manifests import physics_task


def _column_task(index: int, case: tuple, replicate: int, backend: str) -> dict:
    size, mpo_bond, ell, chi_sk, n_power = case
    return task(
        "gpu_pilot",
        {
            "kind": "column_operator",
            "source": "synthetic",
            "lx": size,
            "in_dim": 2,
            "out_dim": 2,
            "mpo_bond": mpo_bond,
            "family": "gaussian",
            "replicate": replicate,
        },
        {
            "name": "global_rmps",
            "ell": ell,
            "chi_sk": chi_sk,
            "n_power": n_power,
        },
        method_seeds(index, 0, replicate),
        {
            "repeats": 5,
            "metrics": ["wall_seconds", "parity_error", "gpu_memory_bytes"],
            "primary_metric": "wall_seconds",
            "parity_tolerance": 1e-10,
        },
        backend=backend,
        resources={
            "hardware": "gpu" if backend == "cupy" else "cpu",
            "cpus": 32 if backend == "cupy" else 4,
            "gpus": 1 if backend == "cupy" else 0,
        },
        requirements=["cupy"] if backend == "cupy" else [],
    )


def _physics_pair(index: int, size: int, ham: str, backend: str) -> dict:
    task_spec = physics_task(
        index,
        0,
        (size, size),
        ham,
        "peps_sketch",
        chi=8 if ham.startswith("tfim@") else 12,
        eta=16 if ham.startswith("tfim@") else 36,
        schedule=[[0.03, 3]],
        study="gpu_pilot",
    )
    task_spec.update(
        backend=backend,
        requirements=["cupy"] if backend == "cupy" else [],
        resources={
            "hardware": "gpu" if backend == "cupy" else "cpu",
            "cpus": 32 if backend == "cupy" else 8,
            "gpus": 1 if backend == "cupy" else 0,
        },
    )
    task_spec["measurement"].update({
        "metrics": [
            "wall_seconds",
            "energy",
            "state_infidelity_to_full_trotter",
        ],
        "primary_metric": "energy",
        "parity_tolerance": 1e-10,
    })
    return task_spec


def build_gpu_pilot() -> list[dict]:
    """build the small mandatory CPU/CuPy promotion gate."""
    cases = (
        (5, 2, 8, 4, 0),
        (6, 4, 20, 8, 1),
        (7, 8, 40, 16, 0),
        (7, 8, 40, 16, 1),
    )
    tasks = []
    for index, (case, replicate) in enumerate(
        itertools.product(cases, range(2)), start=30_000
    ):
        for backend in ("numpy", "cupy"):
            tasks.append(_column_task(index, case, replicate, backend))
    for index, (size, ham) in enumerate(itertools.product(
        (2, 4), (hamiltonian("tfim", 3.5), hamiltonian("heisenberg"))
    ), start=40_000):
        for backend in ("numpy", "cupy"):
            tasks.append(_physics_pair(index, size, ham, backend))
    return tasks


def build_gpu_crossover() -> list[dict]:
    """build the broad optional timing grid after the parity gate passes."""
    tasks = []
    axes = itertools.product(
        (5, 6, 7),
        (2, 4, 8),
        (8, 20, 40),
        (4, 8, 16),
        (0, 1),
        range(2),
    )
    for index, values in enumerate(axes, start=60_000):
        case = values[:5]
        replicate = values[5]
        for backend in ("numpy", "cupy"):
            selected = _column_task(index, case, replicate, backend)
            selected["experiment"] = "gpu_crossover"
            selected["problem"]["study"] = "gpu_crossover"
            if backend == "cupy":
                selected["requirements"].append("gpu_parity")
            tasks.append(selected)
    return tasks
