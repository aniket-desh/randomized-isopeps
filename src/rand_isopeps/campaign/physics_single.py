"""single-state peps initialization, measurement, and evolution."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.backend import synchronize
from rand_isopeps.physics import (
    bond_trotter_imaginary_step,
    dense_state_vector,
    exact_imaginary_step,
    local_term_norm_bound,
)
from rand_isopeps.physics.dense import bond_first_order_imaginary_step
from rand_isopeps.physics.measurements import state_infidelity
from rand_isopeps.real_isotns.physics_loop import tebd_iteration

from .checkpoint import load_checkpoint, save_checkpoint
from .physics_common import (
    eigenvalue_error_fields,
    measurement_convergence,
    measurement_settings,
    value,
)
from .physics_dense import energy_record


def peps_options(problem: dict, method: dict):
    name = str(method["name"])
    if name in {"peps_full", "full_peps"}:
        return "none", {"max_bond": None, "cutoff": 0.0}, {}
    gate = {
        "max_bond": value(
            problem, method, "gate_bond", value(problem, method, "eta", 8)
        ),
        "cutoff": float(value(problem, method, "cutoff", 1e-10)),
    }
    common = {
        "eta": int(value(problem, method, "eta", 8)),
        "ndis": int(value(problem, method, "ndis", 0)),
        "absorption_max_bond": value(
            problem,
            method,
            "absorption_bond",
            value(problem, method, "eta", 8),
        ),
        "absorption_cutoff": float(
            value(problem, method, "absorption_cutoff", 1e-10)
        ),
    }
    if name in {"local_moses", "peps_local", "sequential_moses"}:
        return "local", gate, {
            "chi": int(value(problem, method, "chi", 8)),
            **common,
        }
    if name in {"global_rmps", "peps_sketch", "rmps"}:
        return "rmps", gate, {
            "ell": int(value(problem, method, "ell", 8)),
            "kappa": int(value(problem, method, "kappa", 2)),
            "chi_sk": int(value(problem, method, "chi_sk", 8)),
            **common,
        }
    raise ValueError(f"unsupported peps physics method: {name!r}")


def device_peps(state, backend: str):
    if backend == "numpy":
        return state
    if backend != "cupy":
        raise ValueError(f"unsupported backend: {backend!r}")
    import cupy as cp

    state.apply_to_arrays(cp.asarray)
    return state


def peps_measurement(
    state,
    hamiltonian,
    h,
    norm_bound,
    references,
    source,
    problem,
    measurement,
):
    from rand_isopeps.real_isotns.tebd2 import energy

    if h is not None:
        vector, log_norm = dense_state_vector(state)
        return vector, {
            **energy_record(h, vector, norm_bound, references, source),
            "state_log10_norm": float(log_norm),
            "residual_method": "dense_peps_oracle",
        }
    bonds, _, tolerance = measurement_settings(problem, measurement)
    estimates = [[
        energy(state, hamiltonian, max_bond=bond)
    ] for bond in bonds]
    convergence = measurement_convergence(bonds, estimates, tolerance)
    energies = estimates[-1]
    return None, {
        "energies": energies,
        **eigenvalue_error_fields(energies, references),
        "residual_norms": [None],
        "relative_residuals": [None],
        "variances": [None],
        "residual_identity_errors": [None],
        "reference_source": source,
        "reference_energies": references,
        "state_log10_norm": None,
        "residual_method": "unavailable_large_peps",
        "measurement_method": "paired_boundary_rayleigh",
        **convergence,
    }


def initial_peps(task: dict, vectors):
    import quimb.tensor as qtn

    problem = task["problem"]
    initialization = str(problem.get("initialization", "random_product"))
    if initialization == "random_product":
        state = qtn.PEPS.product_state(vectors)
    elif initialization == "random_raw":
        state = qtn.PEPS.rand(
            int(problem["lx"]),
            int(problem["ly"]),
            bond_dim=int(problem.get("bond", 2)),
            phys_dim=2,
            dtype="float64",
            seed=int(task["seeds"]["problem"]),
        )
        state.normalize()
    elif initialization == "random_isotns":
        from rand_isopeps.real_isotns.moses_move import random_isotns

        state = random_isotns(
            int(problem["lx"]),
            int(problem["ly"]),
            bond=int(problem.get("bond", 2)),
            phys=2,
            chi=int(problem.get("chi", 8)),
            eta=int(problem.get("eta", 8)),
            cutoff=float(problem.get("cutoff", 1e-10)),
            Ndis=int(problem.get("initialization_ndis", 0)),
            seed=int(task["seeds"]["problem"]),
        )
    else:
        raise ValueError(f"unknown peps initialization: {initialization!r}")
    return device_peps(state, str(task.get("backend", "numpy")))


def peps_trajectory(
    task: dict,
    hamiltonian,
    h,
    bonds,
    schedule,
    vectors,
    references,
    reference_source,
    checkpoint_path,
    stop_requested,
):
    problem, method = task["problem"], task["method"]
    if int(problem.get("states", 1)) != 1:
        raise ValueError("p>1 peps tasks require the block-column correctness gate")
    norm_bound = local_term_norm_bound(hamiltonian)
    checkpoint = (
        load_checkpoint(checkpoint_path, task["task_id"])
        if checkpoint_path else None
    )
    if checkpoint is None:
        state = initial_peps(task, vectors)
        sketch_rng = np.random.default_rng(int(task["seeds"]["sketch"]))
        oracle_exact = dense_state_vector(state)[0] if h is not None else None
        oracle_trotter = None if oracle_exact is None else oracle_exact.copy()
        _, initial = peps_measurement(
            state,
            hamiltonian,
            h,
            norm_bound,
            references,
            reference_source,
            problem,
            task["measurement"],
        )
        rows = [{"iteration": 0, **initial}]
        start = 0
    else:
        state = checkpoint["state"]
        sketch_rng = np.random.default_rng()
        sketch_rng.bit_generator.state = checkpoint["rng_state"]
        oracle_exact = checkpoint.get("oracle_exact")
        oracle_trotter = checkpoint.get("oracle_trotter")
        rows = checkpoint["rows"]
        start = int(checkpoint["iteration"])
    column_backend, gate_options, column_options = peps_options(problem, method)
    order = int(value(problem, method, "trotter_order", 2))
    for iteration in range(start + 1, len(schedule) + 1):
        tau = schedule[iteration - 1]
        direction = 1 if iteration % 2 else -1
        synchronize(state.arrays)
        started = perf_counter()
        state, update = tebd_iteration(
            state,
            hamiltonian,
            tau,
            direction=direction,
            column_backend=column_backend,
            gate_options=gate_options,
            column_options=column_options,
            rng=sketch_rng,
            inplace=False,
            trotter_order=order,
        )
        synchronize(state.arrays)
        update["synchronized_update_runtime_s"] = float(
            perf_counter() - started
        )
        if h is not None:
            oracle_exact = exact_imaginary_step(h, oracle_exact, tau)
            oracle_trotter = (
                bond_first_order_imaginary_step(
                    bonds,
                    oracle_trotter,
                    tau,
                    ly=int(problem["ly"]),
                    direction=direction,
                )
                if order == 1
                else bond_trotter_imaginary_step(
                    bonds,
                    oracle_trotter,
                    tau,
                    ly=int(problem["ly"]),
                    direction=direction,
                )
            )
        vector, measured = peps_measurement(
            state,
            hamiltonian,
            h,
            norm_bound,
            references,
            reference_source,
            problem,
            task["measurement"],
        )
        if vector is not None:
            measured.update({
                "state_infidelity_to_exact_evolution": state_infidelity(
                    vector, oracle_exact
                ),
                "state_infidelity_to_full_trotter": state_infidelity(
                    vector, oracle_trotter
                ),
                "trotter_infidelity": state_infidelity(
                    oracle_trotter, oracle_exact
                ),
            })
        rows.append({"iteration": iteration, "tau": tau, **measured, **update})
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "task_id": task["task_id"],
                "iteration": iteration,
                "state": state,
                "rows": rows,
                "rng_state": sketch_rng.bit_generator.state,
                "oracle_exact": oracle_exact,
                "oracle_trotter": oracle_trotter,
            })
        if stop_requested():
            raise InterruptedError("checkpointed after scheduler stop request")
    return rows
