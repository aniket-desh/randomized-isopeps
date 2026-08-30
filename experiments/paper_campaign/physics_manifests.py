"""accuracy-first physics campaign manifests."""

from __future__ import annotations

import itertools

from .manifest_common import hamiltonian, method_seeds, task
from .table_2_validation import (
    table_2_exact_relative_tolerance,
    table_2_path,
    table_2_references,
)


def physics_method(
    name: str,
    eta: int,
    lattice_size: int,
    overrides: dict | None = None,
) -> dict:
    """build one local, rmps, or dense evolution method."""
    if name == "peps_sketch":
        method = {
            "name": name,
            "label": "global_rmps_bounded",
            "column_backend": "rmps",
            "eta": eta,
            "ell": eta + 4,
            "chi_sk": max(8, lattice_size),
            "kappa": 2,
            "n_power": 0,
        }
    elif name == "peps_local":
        method = {
            "name": name,
            "label": "local_det",
            "column_backend": "local_det",
            "eta": eta,
        }
    else:
        method = {"name": name, "label": name, "eta": eta}
    method.update(overrides or {})
    return method


def physics_measurement(primary_metric: str = "energy_error") -> dict:
    """return the common accuracy measurements and convergence contract."""
    return {
        "metrics": [
            "energy",
            "energy_error",
            "energy_variance",
            "state_infidelity",
            "isometry_error",
            "truncation_error",
        ],
        "primary_metric": primary_metric,
        "reference": "exact_diagonalization_or_dmrg",
        "measurement_bonds": [64, 128],
        "measurement_convergence_tolerance": 1e-5,
    }


def physics_task(
    problem_index: int,
    replicate: int,
    lattice: tuple[int, int],
    hamiltonian_name: str,
    method_name: str,
    chi: int,
    eta: int,
    schedule: list[list[float | int]],
    *,
    p: int = 1,
    trotter_order: int = 2,
    study: str,
    method_overrides: dict | None = None,
    problem_overrides: dict | None = None,
    measurement_overrides: dict | None = None,
) -> dict:
    """build one physics trajectory with explicit backend dependencies."""
    block_correctness = study == "block_correctness"
    allowed_block_method = method_name in {"peps_local", "peps_sketch"}
    allowed_block_method |= block_correctness and method_name.startswith("dense_")
    blocked = p > 1 and not allowed_block_method
    dektor = study == "dektor_reproduction"
    block_path = p > 1 or dektor
    use_gpu = method_name == "peps_sketch" and p == 1 and not dektor and study not in {
        "correctness_ladder",
        "block_correctness",
        "gpu_pilot",
    }
    requirements = ["block_peps"] if block_path else []
    if use_gpu:
        requirements.extend(("cupy", "gpu_parity"))

    overrides = dict(problem_overrides or {})
    embedded_reference = "reference_energies" in overrides
    no_artifact_studies = {"correctness_ladder", "block_correctness", "gpu_pilot"}
    if study not in no_artifact_studies and not embedded_reference:
        requirements.append("reference_artifact")
    if block_correctness and p > 1:
        initialization = "shared_block_oracle"
    elif p == 1 and study in {"correctness_ladder", "gpu_pilot"}:
        initialization = "random_product"
    else:
        initialization = "random_isotns"
    problem = {
        "kind": "imaginary_time",
        "lx": lattice[0],
        "ly": lattice[1],
        "lattice": list(lattice),
        "physical_dim": 2,
        "hamiltonian": hamiltonian_name,
        "chi": chi,
        "eta": eta,
        "states": p,
        "initialization": initialization,
        "stages": [
            {"tau": float(tau), "iterations": int(iterations)}
            for tau, iterations in schedule
        ],
        "schedule": schedule,
        "trotter_order": trotter_order,
        "reference": "exact_diagonalization" if lattice[0] * lattice[1] <= 16 else "dmrg",
        "study": study,
        **overrides,
    }
    method_index = {
        "dense_exact": 0,
        "dense_strang": 1,
        "peps_full": 2,
        "peps_local": 3,
        "peps_sketch": 4,
        "dense_first_order": 5,
    }[method_name]
    measurement = physics_measurement(
        "low_lying_energy_error" if p > 1 else "energy_error"
    )
    measurement["ritz_interval"] = 1 if lattice[0] * lattice[1] <= 16 else 5
    measurement.update(measurement_overrides or {})
    if use_gpu:
        cpus = 32
    elif block_path:
        cpus = 32 if max(lattice) <= 4 else 64 if max(lattice) <= 6 else 128
    else:
        cpus = 8
    stop_grace = 7200 if block_path and max(lattice) >= 6 else 1800
    return task(
        "physics",
        problem,
        physics_method(method_name, eta, max(lattice), method_overrides),
        method_seeds(problem_index, method_index, replicate),
        measurement,
        backend="cupy" if use_gpu else "numpy",
        dtype="complex128" if initialization == "random_product" else "float64",
        resources={
            "hardware": "gpu" if use_gpu else "cpu",
            "cpus": cpus,
            "gpus": 1 if use_gpu else 0,
            "stop_grace_seconds": stop_grace,
        },
        requirements=requirements,
        blocked=blocked,
        blocked_reason="the method has no validated shared-q block move" if blocked else None,
    )


def _physics_correctness() -> list[dict]:
    tasks = []
    modes = ("dense_exact", "dense_strang", "peps_full", "peps_local", "peps_sketch")
    axes = itertools.product(
        ((2, 2), (2, 3), (3, 3)),
        (hamiltonian("tfim", 3.5), hamiltonian("heisenberg")),
        (0.1, 0.05, 0.025, 0.0125),
        range(2),
    )
    for index, (lattice, ham, tau, replicate) in enumerate(axes):
        for mode in modes:
            tasks.append(physics_task(
                index,
                replicate,
                lattice,
                ham,
                mode,
                chi=8,
                eta=8,
                schedule=[[tau, 1]],
                study="correctness_ladder",
            ))
    return tasks


def _block_correctness() -> list[dict]:
    tasks = []
    cells = ((2, 2, 2), (2, 3, 2), (2, 2, 3))
    for index, ((lx, ly, p), ham) in enumerate(itertools.product(
        cells, (hamiltonian("tfim", 3.5), hamiltonian("heisenberg"))
    ), start=500):
        common = dict(
            problem_index=index,
            replicate=0,
            lattice=(lx, ly),
            hamiltonian_name=ham,
            chi=32,
            eta=32,
            schedule=[[0.1, 2], [0.03, 2]],
            p=p,
            trotter_order=1,
            study="block_correctness",
        )
        for method_name in ("dense_exact", "dense_first_order"):
            tasks.append(physics_task(
                method_name=method_name,
                problem_overrides={"regime": "dense_oracle"},
                **common,
            ))
        for regime in ("full_rank_oracle", "truncated"):
            if regime == "full_rank_oracle":
                method_overrides = {
                    "eta": 32,
                    "ell": 32,
                    "chi": 32,
                    "kappa": 32,
                    "gate_bond": None,
                    "cutoff": 0.0,
                    "absorption_bond": None,
                    "absorption_cutoff": 0.0,
                    "moses_cutoff": 0.0,
                    "label": "full_rank_oracle",
                }
            else:
                method_overrides = {
                    "eta": 8,
                    "ell": 12,
                    "chi": 8,
                    "kappa": 2,
                    "gate_bond": 8,
                    "absorption_bond": 8,
                    "label": "truncated",
                }
            for method_name in ("peps_local", "peps_sketch"):
                tasks.append(physics_task(
                    method_name=method_name,
                    method_overrides=method_overrides,
                    problem_overrides={"regime": regime},
                    **common,
                ))
    return tasks


def dektor_cells() -> list[dict]:
    """return the deduplicated Dektor reproduction cells and panel uses."""
    cells = {}

    def add(lattice, ham, chi, eta, states, panel):
        key = (*lattice, ham, chi, eta, states)
        if key not in cells:
            cells[key] = {
                "lattice": lattice,
                "hamiltonian": ham,
                "chi": chi,
                "eta": eta,
                "states": states,
                "dektor_panels": [],
            }
        cells[key]["dektor_panels"].append(panel)

    for p in (1, 2, 3):
        add((6, 6), hamiltonian("tfim", 3.5), 8, 16, p, "figure_2")
    for size, g, p in itertools.product((4, 6, 8, 10, 12), (3.0, 3.5), (1, 2)):
        add((size, size), hamiltonian("tfim", g), 12, 20, p, "figure_3")
    for g, (chi, eta) in itertools.product(
        (1.0, 2.0, 3.0, 3.5), ((4, 8), (12, 20))
    ):
        add((4, 4), hamiltonian("tfim", g), chi, eta, 2, "table_2")
    for size, p in itertools.product((4, 6, 8), (1, 2)):
        add((size, size), hamiltonian("heisenberg"), 12, 36, p, "figure_4")
    return list(cells.values())


def _dektor_initialization_plan(cells: list[dict]) -> dict[tuple, tuple[int, int]]:
    """pair block initializers and retain the largest requested state count."""
    keys = [
        (
            *cell["lattice"],
            cell["hamiltonian"],
            int(cell["chi"]),
            int(cell["eta"]),
        )
        for cell in cells
    ]
    unique_keys = list(dict.fromkeys(keys))
    parent_states = {
        key: max(
            int(cell["states"])
            for cell, candidate in zip(cells, keys, strict=True)
            if candidate == key
        )
        for key in unique_keys
    }
    return {
        key: (1_000 + index, parent_states[key])
        for index, key in enumerate(unique_keys)
    }


def _physics_dektor() -> list[dict]:
    tasks = []
    cells = dektor_cells()
    initialization_plan = _dektor_initialization_plan(cells)
    for cell in cells:
        p = int(cell["states"])
        initialization_key = (
            *cell["lattice"],
            cell["hamiltonian"],
            int(cell["chi"]),
            int(cell["eta"]),
        )
        problem_index, parent_states = initialization_plan[initialization_key]
        problem_overrides = {
            "dektor_panels": list(cell["dektor_panels"]),
            "initialization_parent_states": parent_states,
        }
        references = table_2_references().get(cell["hamiltonian"])
        if cell["lattice"] == (4, 4) and references is not None:
            problem_overrides.update(
                reference_energies=references[:p],
                reference_source="dektor_table_2",
            )
        plot_role = (
            "low_energy"
            if cell["lattice"] == (4, 4)
            and cell["hamiltonian"] == "tfim@3.5"
            and cell["chi"] == 12
            and p == 2
            else None
        )
        measurement_overrides = {}
        if plot_role:
            measurement_overrides["plot_role"] = plot_role
        if "table_2" in cell["dektor_panels"]:
            measurement_overrides["published_reference_relative_tolerance"] = (
                table_2_exact_relative_tolerance
            )
        for replicate, method_name in itertools.product(
            range(3), ("peps_local", "peps_sketch")
        ):
            method_overrides = (
                {
                    "ndis": 30,
                    "disentangler": "riemannian_renyi",
                    "label": "local_riemannian_ndis30",
                }
                if method_name == "peps_local"
                else None
            )
            tasks.append(physics_task(
                problem_index,
                replicate,
                cell["lattice"],
                cell["hamiltonian"],
                method_name,
                cell["chi"],
                cell["eta"],
                [[0.1, 50]],
                p=p,
                trotter_order=1,
                study="dektor_reproduction",
                method_overrides=method_overrides,
                problem_overrides=problem_overrides,
                measurement_overrides=measurement_overrides or None,
            ))
    return tasks


def _physics_bond_and_hamiltonian_sweeps() -> list[dict]:
    cells = []
    cells.extend(
        ((size, size), hamiltonian("tfim", g), chi, eta)
        for size, g, (chi, eta) in itertools.product(
            (4, 6, 8), (3.0, 3.5), ((4, 8), (8, 16), (12, 20))
        )
    )
    cells.extend(
        ((size, size), hamiltonian("heisenberg"), chi, eta)
        for size, (chi, eta) in itertools.product((4, 6, 8), ((8, 24), (12, 36)))
    )
    cells.extend(
        ((size, size), hamiltonian("xxz", delta), 8, 16)
        for size, delta in itertools.product((4, 6), (0.5, 1.0, 1.5))
    )
    cells.append(((4, 4), hamiltonian("compass"), 8, 16))
    schedule = [[0.1, 10], [0.03, 20], [0.01, 20]]
    tasks = []
    for index, (lattice, ham, chi, eta) in enumerate(cells, start=2_000):
        study = (
            "hamiltonian_robustness"
            if ham.startswith("xxz@") or ham == "compass"
            else "bond_sweep"
        )
        plot_role = (
            "physics_trajectory"
            if (lattice, ham, chi, eta) == ((4, 4), "tfim@3.5", 8, 16)
            else None
        )
        for replicate, method_name in itertools.product(
            range(3), ("peps_local", "peps_sketch")
        ):
            tasks.append(physics_task(
                index,
                replicate,
                lattice,
                ham,
                method_name,
                chi,
                eta,
                schedule,
                study=study,
                measurement_overrides={"plot_role": plot_role} if plot_role else None,
            ))
    return tasks


def _physics_parameter_sweeps() -> list[dict]:
    lattice = (4, 4)
    ham = hamiltonian("tfim", 3.5)
    schedule = [[0.1, 5], [0.03, 10], [0.01, 10]]
    baseline = {"ell": 12, "chi_sk": 8, "kappa": 2}
    configs = [("baseline", None, None, baseline)]
    for axis, values in (
        ("ell", (8, 16, 24)),
        ("chi_sk", (1, 2, 4, 16, 32)),
        ("kappa", (1, 4)),
    ):
        for value in values:
            configs.append((f"{axis}_{value}", axis, value, {**baseline, axis: value}))

    tasks = []
    for p, replicate in itertools.product((1, 2), range(2)):
        for label, axis, value, config in configs:
            tasks.append(physics_task(
                3_000 + p,
                replicate,
                lattice,
                ham,
                "peps_sketch",
                chi=8,
                eta=8,
                schedule=schedule,
                p=p,
                trotter_order=1,
                study="sketch_parameter_sweep",
                method_overrides={**config, "label": f"rmps_{label}"},
                measurement_overrides={
                    "plot_role": "physics_sweep",
                    "sweep_axis": axis or "baseline",
                    "sweep_value": value,
                },
            ))
    for p, replicate, method_name in itertools.product(
        (1, 2, 3), range(2), ("peps_local", "peps_sketch")
    ):
        tasks.append(physics_task(
            3_100,
            replicate,
            lattice,
            ham,
            method_name,
            chi=8,
            eta=8,
            schedule=schedule,
            p=p,
            trotter_order=1,
            study="block_size_sweep",
            problem_overrides={"initialization_parent_states": 3},
            measurement_overrides={
                "plot_role": "physics_sweep",
                "sweep_axis": "states",
                "sweep_value": p,
            },
        ))
    bond_configs = [
        ("baseline", None, None, 8, 16),
        ("chi_4", "chi", 4, 4, 16),
        ("chi_12", "chi", 12, 12, 16),
        ("eta_8", "eta", 8, 8, 8),
        ("eta_24", "eta", 24, 8, 24),
    ]
    for replicate, method_name, (label, axis, value, chi, eta) in itertools.product(
        range(2), ("peps_local", "peps_sketch"), bond_configs
    ):
        tasks.append(physics_task(
            3_200,
            replicate,
            lattice,
            ham,
            method_name,
            chi=chi,
            eta=eta,
            schedule=schedule,
            p=1,
            study="bond_parameter_sweep",
            method_overrides={
                "label": f"{method_name}_{label}",
                **({"ell": 20} if method_name == "peps_sketch" else {}),
            },
            measurement_overrides={
                "plot_role": "physics_sweep",
                "sweep_axis": axis or "baseline",
                "sweep_value": value,
            },
        ))
    return tasks


def build_physics() -> list[dict]:
    """build correctness, reproduction, robustness, and bounded ablations."""
    return (
        _physics_correctness()
        + _block_correctness()
        + _physics_dektor()
        + _physics_bond_and_hamiltonian_sweeps()
        + _physics_parameter_sweeps()
    )
