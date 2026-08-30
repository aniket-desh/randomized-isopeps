"""dense oracle initialization and imaginary-time trajectories."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.physics import (
    bond_trotter_imaginary_step,
    exact_imaginary_step,
    local_term_norm_bound,
    normalize_state,
    rayleigh_ritz,
)
from rand_isopeps.physics.dense import bond_first_order_imaginary_step
from rand_isopeps.physics.measurements import energy_metrics

from .checkpoint import load_checkpoint, save_checkpoint
from .physics_common import eigenvalue_error_fields, hamiltonian, value


def dense_product(vectors: dict, lx: int, ly: int):
    state = np.asarray([1.0 + 0.0j])
    for site in np.ndindex(lx, ly):
        state = np.kron(state, vectors[site])
    return normalize_state(state)


def dense_initial(problem: dict, dimension: int, vectors: dict, seed: int):
    states = int(problem.get("states", 1))
    lx, ly = int(problem["lx"]), int(problem["ly"])
    if str(problem.get("initialization", "")) == "shared_block_oracle":
        from rand_isopeps.physics.block_state import dense_block

        from .physics_block import initial_block

        block = dense_block(initial_block(problem, seed))
        if block.shape != (dimension, states):
            raise ValueError("the shared block initializer has the wrong shape")
        return block
    if states == 1:
        return dense_product(vectors, lx, ly)
    rng = np.random.default_rng(int(seed))
    block = rng.standard_normal((dimension, states))
    block = block + 1j * rng.standard_normal(block.shape)
    return np.linalg.qr(block, mode="reduced")[0]


def dense_reorthogonalize(problem: dict, h, state, norm_bound):
    if int(problem.get("states", 1)) <= 1:
        return state
    if str(problem.get("initialization", "")) == "shared_block_oracle":
        return np.linalg.qr(state, mode="reduced")[0]
    return rayleigh_ritz(h, state, h_norm_bound=norm_bound)["vectors"]


def energy_record(h, state, norm_bound, references, source: str):
    block = np.asarray(state)
    if block.ndim == 2 and block.shape[1] > 1:
        vectors = rayleigh_ritz(
            h, block, h_norm_bound=norm_bound
        )["vectors"]
        metrics = energy_metrics(h, vectors, h_norm_bound=norm_bound)
        metrics["measurement_method"] = "dense_generalized_ritz"
    else:
        metrics = energy_metrics(h, block, h_norm_bound=norm_bound)
        metrics["measurement_method"] = "dense_state_rayleigh"
    metrics.update(eigenvalue_error_fields(metrics["energies"], references))
    metrics["reference_source"] = source
    metrics["reference_energies"] = references
    return metrics


def dense_trajectory(
    task: dict,
    h,
    bonds,
    layers,
    schedule,
    vectors,
    references,
    reference_source,
    checkpoint_path,
    stop_requested,
):
    problem, method = task["problem"], task["method"]
    norm_bound = local_term_norm_bound(hamiltonian(problem))
    checkpoint = (
        load_checkpoint(checkpoint_path, task["task_id"])
        if checkpoint_path else None
    )
    if checkpoint is None:
        state = dense_initial(
            problem,
            h.shape[0],
            vectors,
            int(task["seeds"]["problem"]),
        )
        rows = [{
            "iteration": 0,
            **energy_record(h, state, norm_bound, references, reference_source),
        }]
        start = 0
    else:
        state = checkpoint["state"]
        rows = checkpoint["rows"]
        start = int(checkpoint["iteration"])
    name = str(method["name"])
    order = int(value(problem, method, "trotter_order", 2))
    for iteration in range(start + 1, len(schedule) + 1):
        tau = schedule[iteration - 1]
        direction = 1 if iteration % 2 else -1
        started = perf_counter()
        if name == "dense_exact":
            state = exact_imaginary_step(h, state, tau)
        elif order == 1:
            state = bond_first_order_imaginary_step(
                bonds,
                state,
                tau,
                ly=int(problem["ly"]),
                direction=direction,
            )
        elif name in {"dense_checkerboard", "dense_strang_layers"}:
            from rand_isopeps.physics import trotter_imaginary_step

            state = trotter_imaginary_step(layers, state, tau)
        else:
            state = bond_trotter_imaginary_step(
                bonds,
                state,
                tau,
                ly=int(problem["ly"]),
                direction=direction,
            )
        state = dense_reorthogonalize(problem, h, state, norm_bound)
        rows.append({
            "iteration": iteration,
            "tau": tau,
            "trotter_order": 0 if name == "dense_exact" else order,
            "update_runtime_s": float(perf_counter() - started),
            **energy_record(h, state, norm_bound, references, reference_source),
        })
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "task_id": task["task_id"],
                "iteration": iteration,
                "state": state,
                "rows": rows,
            })
        if stop_requested():
            raise InterruptedError("checkpointed after scheduler stop request")
    return rows
