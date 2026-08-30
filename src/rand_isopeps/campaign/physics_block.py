"""shared-block peps initialization, measurement, and evolution."""

from __future__ import annotations

import numpy as np

from rand_isopeps.physics import exact_imaginary_step
from rand_isopeps.physics.dense import block_first_order_imaginary_step
from rand_isopeps.physics.measurements import subspace_metrics

from .checkpoint import load_checkpoint, save_checkpoint
from .physics_common import (
    eigenvalue_error_fields,
    measurement_convergence,
    measurement_settings,
    value,
)


def initial_block(problem: dict, seed: int):
    from rand_isopeps.physics.block_state import BlockPeps, orthonormalize_block
    from rand_isopeps.real_isotns.moses_move import random_isotns

    size = int(problem["states"])
    parent_states = int(problem.get("initialization_parent_states", size))
    if size < 1 or parent_states < size:
        raise ValueError("initialization_parent_states must be at least states")
    rng = np.random.default_rng(int(seed))
    dektor = str(problem.get("study", "")) == "dektor_reproduction"
    peps = random_isotns(
        int(problem["lx"]),
        int(problem["ly"]),
        bond=int(problem.get("bond", 2)),
        phys=2,
        chi=int(problem.get("chi", 8)),
        eta=int(problem.get("eta", 8)),
        cutoff=float(problem.get("cutoff", 1e-10)),
        Ndis=int(problem.get("initialization_ndis", 30 if dektor else 0)),
        disentangler=str(problem.get(
            "initialization_disentangler",
            "riemannian_renyi" if dektor else "altmin",
        )),
        seed=int(seed),
    )
    center = (0, 0)
    tensor = peps[peps.site_tag_id.format(*center)]
    parent_data = rng.standard_normal((*tensor.shape, parent_states))
    data = parent_data[..., :size]
    tensor.modify(data=data, inds=(*tensor.inds, "alpha"))
    return orthonormalize_block(BlockPeps(peps, "alpha", center))[0]


def block_energy_record(
    state,
    hamiltonian,
    h,
    references,
    source,
    problem,
    measurement,
):
    from rand_isopeps.physics.block_measurements import (
        dense_ritz_rotate_block,
        ritz_rotate_block,
    )

    if h is not None:
        state, measured = dense_ritz_rotate_block(state, h)
        measured["measurement_method"] = "dense_generalized_ritz"
    else:
        bonds, cutoff, tolerance = measurement_settings(problem, measurement)
        results = [
            ritz_rotate_block(
                state,
                hamiltonian,
                max_bond=bond,
                cutoff=cutoff,
            )
            for bond in bonds
        ]
        state, measured = results[-1]
        estimates = [entry[1]["energies"] for entry in results]
        convergence = measurement_convergence(bonds, estimates, tolerance)
        measured.update({
            "residual_norms": [None] * state.size,
            "relative_residuals": [None] * state.size,
            "variances": [None] * state.size,
            "residual_identity_errors": [None] * state.size,
            "subspace_residual_norm": None,
            "relative_subspace_residual": None,
            "measurement_method": "paired_boundary_generalized_ritz",
            "measurement_bond": bonds[-1],
            "measurement_cutoff": cutoff,
            **convergence,
        })
    measured.update(eigenvalue_error_fields(measured["energies"], references))
    measured["reference_source"] = source
    measured["reference_energies"] = references
    return state, measured


def block_oracle_metrics(state, oracle_exact, oracle_trotter):
    """compare a block subspace with matched dense evolution oracles."""
    from rand_isopeps.physics.block_state import dense_block

    candidate = dense_block(state)
    comparisons = {
        "to_exact_evolution": subspace_metrics(oracle_exact, candidate),
        "to_ordered_first_order": subspace_metrics(oracle_trotter, candidate),
        "trotter": subspace_metrics(oracle_exact, oracle_trotter),
    }
    return {
        f"{name}_{suffix}": value
        for suffix, metrics in comparisons.items()
        for name, value in metrics.items()
    }


def _block_backend(task):
    problem, method = task["problem"], task["method"]
    if str(task.get("backend", "numpy")) != "numpy":
        raise ValueError("block peps is currently a cpu path")
    method_name = str(method["name"])
    if method_name in {"peps_sketch", "global_rmps", "rmps_shared_q"}:
        backend = "rmps_shared_q"
    elif method_name in {"peps_local", "local_moses", "local_shared_q"}:
        backend = "local_shared_q"
    else:
        raise ValueError(f"unsupported shared-q block method: {method_name!r}")
    if int(value(problem, method, "trotter_order", 1)) != 1:
        raise ValueError("the validated block peps schedule is first order")
    return backend


def _block_options(task, column_backend):
    problem, method = task["problem"], task["method"]
    dektor = str(problem.get("study", "")) == "dektor_reproduction"
    gate = {
        "max_bond": value(problem, method, "gate_bond", problem.get("eta")),
        "cutoff": float(value(problem, method, "cutoff", 1e-10)),
    }
    column = {
        "chi": int(value(problem, method, "chi", problem["chi"])),
        "ell": int(value(problem, method, "ell", int(problem["eta"]) + 4)),
        "eta": int(value(problem, method, "eta", problem["eta"])),
        "kappa": int(value(problem, method, "kappa", 2)),
        "chi_sk": int(
            value(problem, method, "chi_sk", max(8, int(problem["lx"])))
        ),
        "ndis": int(value(
            problem,
            method,
            "ndis",
            30 if dektor and column_backend == "local_shared_q" else 0,
        )),
        "disentangler": str(value(
            problem,
            method,
            "disentangler",
            "riemannian_renyi"
            if dektor and column_backend == "local_shared_q"
            else "altmin",
        )),
        "cutoff": float(value(
            problem, method, "moses_cutoff", 1e-6 if dektor else 1e-10
        )),
        "absorption_max_bond": value(
            problem, method, "absorption_bond", problem["eta"]
        ),
        "absorption_cutoff": float(value(
            problem, method, "absorption_cutoff", 1e-6 if dektor else 1e-10
        )),
    }
    gram = {
        "max_bond": problem.get("measurement_bond"),
        "cutoff": float(problem.get("measurement_cutoff", 0.0)),
        "validate_boundary": bool(problem.get("validate_boundary_gram", False)),
    }
    return gate, column, gram


def _new_block_trajectory(
    task,
    hamiltonian,
    h,
    references,
    reference_source,
):
    problem = task["problem"]
    state = initial_block(problem, int(task["seeds"]["problem"]))
    rng = np.random.default_rng(int(task["seeds"]["sketch"]))
    oracle_exact = None
    oracle_trotter = None
    if h is not None:
        from rand_isopeps.physics.block_state import dense_block

        oracle_exact = dense_block(state)
        oracle_trotter = oracle_exact.copy()
    state, measured = block_energy_record(
        state,
        hamiltonian,
        h,
        references,
        reference_source,
        problem,
        task["measurement"],
    )
    if oracle_exact is not None:
        measured.update(block_oracle_metrics(state, oracle_exact, oracle_trotter))
    return state, rng, 1, oracle_exact, oracle_trotter, [
        {"iteration": 0, **measured}
    ], 0, None


def _restore_block_trajectory(checkpoint, h):
    state = checkpoint["state"]
    rng = np.random.default_rng()
    rng.bit_generator.state = checkpoint["rng_state"]
    oracle_exact = checkpoint.get("oracle_exact")
    oracle_trotter = checkpoint.get("oracle_trotter")
    if h is not None and (oracle_exact is None or oracle_trotter is None):
        raise ValueError("dense block oracles are missing from the checkpoint")
    return (
        state,
        rng,
        int(checkpoint["direction"]),
        oracle_exact,
        oracle_trotter,
        checkpoint["rows"],
        int(checkpoint["iteration"]),
        checkpoint.get("partial_iteration"),
    )


def _save_block_checkpoint(
    checkpoint_path,
    task_id,
    state,
    rows,
    rng,
    direction,
    oracle_exact,
    oracle_trotter,
    iteration,
    partial=None,
):
    if not checkpoint_path:
        return
    payload = {
        "task_id": task_id,
        "iteration": int(iteration),
        "state": state,
        "rows": rows,
        "rng_state": rng.bit_generator.state,
        "direction": int(direction),
        "oracle_exact": oracle_exact,
        "oracle_trotter": oracle_trotter,
    }
    if partial is not None:
        payload["partial_iteration"] = partial
    save_checkpoint(checkpoint_path, payload)


def _block_resume_progress(partial, iteration, tau):
    if partial is None:
        return None
    if int(partial.get("iteration", -1)) != iteration:
        raise ValueError("block checkpoint has a stale partial iteration")
    if float(partial.get("tau", -1.0)) != float(tau):
        raise ValueError("block checkpoint has a stale partial tau")
    return partial["progress"]


def _unmeasured_block_record(state, references, reference_source):
    return {
        "energies": [None] * state.size,
        "ground_energy_errors": [None] * state.size,
        "eigenvalue_errors": [None] * state.size,
        "relative_eigenvalue_errors": [None] * state.size,
        "absolute_relative_eigenvalue_errors": [None] * state.size,
        "residual_norms": [None] * state.size,
        "measurement_converged": None,
        "reference_source": reference_source,
        "reference_energies": references,
    }


def _measure_block_iteration(
    state,
    iteration,
    schedule_length,
    interval,
    hamiltonian,
    h,
    references,
    reference_source,
    problem,
    measurement,
):
    if iteration % interval != 0 and iteration != schedule_length:
        return state, _unmeasured_block_record(
            state, references, reference_source
        )
    return block_energy_record(
        state,
        hamiltonian,
        h,
        references,
        reference_source,
        problem,
        measurement,
    )


def _advance_block_oracles(
    h,
    bonds,
    oracle_exact,
    oracle_trotter,
    tau,
    problem,
    direction,
):
    if h is None:
        return oracle_exact, oracle_trotter
    exact = exact_imaginary_step(h, oracle_exact, tau)
    trotter = block_first_order_imaginary_step(
        bonds,
        oracle_trotter,
        tau,
        lx=int(problem["lx"]),
        ly=int(problem["ly"]),
        direction=direction,
    )
    return exact, trotter


def _evolve_block_iteration(
    task,
    state,
    hamiltonian,
    rng,
    rows,
    direction,
    oracle_exact,
    oracle_trotter,
    iteration,
    tau,
    partial_iteration,
    column_backend,
    gate_options,
    column_options,
    gram_options,
    checkpoint_path,
    stop_requested,
):
    from rand_isopeps.real_isotns.block_physics_loop import block_tebd_iteration

    resume = _block_resume_progress(partial_iteration, iteration, tau)

    def checkpoint_progress(current_state, progress):
        _save_block_checkpoint(
            checkpoint_path,
            task["task_id"],
            current_state,
            rows,
            rng,
            direction,
            oracle_exact,
            oracle_trotter,
            iteration - 1,
            partial={
                "iteration": iteration,
                "tau": float(tau),
                "progress": progress,
            },
        )
        if stop_requested():
            raise InterruptedError(
                "checkpointed after a completed block-sweep column"
            )

    state, update = block_tebd_iteration(
        state,
        hamiltonian,
        tau,
        direction=direction,
        backend=column_backend,
        gate_options=gate_options,
        column_options=column_options,
        gram_options=gram_options,
        rng=rng,
        inplace=False,
        resume=resume,
        progress_callback=checkpoint_progress if checkpoint_path else None,
    )
    return state, update, int(update["next_direction"])


def _record_block_iteration(
    task,
    state,
    hamiltonian,
    h,
    bonds,
    references,
    reference_source,
    rows,
    rng,
    direction,
    step_direction,
    oracle_exact,
    oracle_trotter,
    iteration,
    tau,
    schedule_length,
    interval,
    update,
    checkpoint_path,
    stop_requested,
):
    problem = task["problem"]
    oracle_exact, oracle_trotter = _advance_block_oracles(
        h,
        bonds,
        oracle_exact,
        oracle_trotter,
        tau,
        problem,
        step_direction,
    )
    state, measured = _measure_block_iteration(
        state,
        iteration,
        schedule_length,
        interval,
        hamiltonian,
        h,
        references,
        reference_source,
        problem,
        task["measurement"],
    )
    if oracle_exact is not None:
        measured.update(block_oracle_metrics(state, oracle_exact, oracle_trotter))
    rows.append({
        "iteration": iteration,
        "tau": tau,
        **measured,
        **update,
    })
    _save_block_checkpoint(
        checkpoint_path,
        task["task_id"],
        state,
        rows,
        rng,
        direction,
        oracle_exact,
        oracle_trotter,
        iteration,
    )
    if stop_requested():
        raise InterruptedError("checkpointed after scheduler stop request")
    return state, oracle_exact, oracle_trotter


def block_trajectory(
    task: dict,
    hamiltonian,
    h,
    bonds,
    schedule,
    references,
    reference_source,
    checkpoint_path,
    stop_requested,
):
    column_backend = _block_backend(task)
    checkpoint = (
        load_checkpoint(checkpoint_path, task["task_id"])
        if checkpoint_path else None
    )
    if checkpoint is None:
        trajectory = _new_block_trajectory(
            task,
            hamiltonian,
            h,
            references,
            reference_source,
        )
    else:
        trajectory = _restore_block_trajectory(checkpoint, h)
    (
        state,
        rng,
        direction,
        oracle_exact,
        oracle_trotter,
        rows,
        start,
        partial_iteration,
    ) = trajectory
    gate_options, column_options, gram_options = _block_options(
        task, column_backend
    )
    interval = max(1, int(task["measurement"].get("ritz_interval", 1)))

    if checkpoint is None:
        _save_block_checkpoint(
            checkpoint_path,
            task["task_id"],
            state,
            rows,
            rng,
            direction,
            oracle_exact,
            oracle_trotter,
            start,
        )
    if stop_requested():
        raise InterruptedError("checkpointed before the next block sweep")

    for iteration in range(start + 1, len(schedule) + 1):
        tau = schedule[iteration - 1]
        step_direction = direction
        state, update, direction = _evolve_block_iteration(
            task,
            state,
            hamiltonian,
            rng,
            rows,
            direction,
            oracle_exact,
            oracle_trotter,
            iteration,
            tau,
            partial_iteration,
            column_backend,
            gate_options,
            column_options,
            gram_options,
            checkpoint_path,
            stop_requested,
        )
        partial_iteration = None
        state, oracle_exact, oracle_trotter = _record_block_iteration(
            task,
            state,
            hamiltonian,
            h,
            bonds,
            references,
            reference_source,
            rows,
            rng,
            direction,
            step_direction,
            oracle_exact,
            oracle_trotter,
            iteration,
            tau,
            len(schedule),
            interval,
            update,
            checkpoint_path,
            stop_requested,
        )
    return rows
