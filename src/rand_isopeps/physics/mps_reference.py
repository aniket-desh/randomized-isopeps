"""independent snake-mps references for square-lattice low energies."""

from __future__ import annotations

import warnings

import numpy as np

from rand_isopeps.backend import to_numpy


def _operator_schmidt(term, *, tolerance: float = 1e-13):
    matrix = np.asarray(term)
    local_dim = int(round(np.sqrt(matrix.shape[0])))
    reshaped = matrix.reshape(
        local_dim, local_dim, local_dim, local_dim
    ).transpose(0, 2, 1, 3).reshape(local_dim**2, local_dim**2)
    left, singular_values, right = np.linalg.svd(reshaped, full_matrices=False)
    scale = max(float(singular_values[0]), np.finfo(float).tiny)
    factors = []
    for index, value in enumerate(singular_values):
        if value <= tolerance * scale:
            continue
        root = np.sqrt(value)
        factors.append((
            root * left[:, index].reshape(local_dim, local_dim),
            root * right[index].reshape(local_dim, local_dim),
        ))
    return factors


def square_lattice_mpo(ham, lx: int, ly: int, *, cutoff: float = 1e-13):
    """build an exact row-major mpo from a local square-lattice hamiltonian."""
    import quimb.tensor as qtn

    n_sites = int(lx) * int(ly)
    first_term = np.asarray(next(iter(ham.terms.values())))
    local_dim = int(round(np.sqrt(first_term.shape[0])))
    identity = np.eye(local_dim, dtype=first_term.dtype)
    terms = []
    for where, term in ham.terms.items():
        first, second = tuple(where)
        first_index = int(first[0]) * int(ly) + int(first[1])
        second_index = int(second[0]) * int(ly) + int(second[1])
        for left, right in _operator_schmidt(term, tolerance=cutoff):
            arrays = [identity] * n_sites
            arrays[first_index] = left
            arrays[second_index] = right
            terms.append(qtn.MPO_product_operator(arrays))
    if not terms:
        raise ValueError("the Hamiltonian contains no nonzero terms")
    while len(terms) > 1:
        reduced = []
        for index in range(0, len(terms), 2):
            if index + 1 == len(terms):
                reduced.append(terms[index])
                continue
            combined = terms[index] + terms[index + 1]
            combined.compress(cutoff=float(cutoff))
            reduced.append(combined)
        terms = reduced
    return terms[0]


def _to_backend(network, backend: str):
    if backend == "numpy":
        return network
    if backend != "cupy":
        raise ValueError(f"unsupported backend: {backend!r}")
    import cupy as cp

    network.apply_to_arrays(cp.asarray)
    return network


def residual_resource_estimate(state, hamiltonian_mpo) -> dict[str, int]:
    """estimate uncompressed H|psi> tensor sizes before materialization."""
    applied_bonds = [
        int(state.bond_size(site, site + 1))
        * int(hamiltonian_mpo.bond_size(site, site + 1))
        for site in range(state.L - 1)
    ]
    residual_bonds = [
        bond + int(state.bond_size(site, site + 1))
        for site, bond in enumerate(applied_bonds)
    ]

    def element_count(bonds):
        return sum(
            (1 if site == 0 else bonds[site - 1])
            * int(state.phys_dim(site))
            * (1 if site == state.L - 1 else bonds[site])
            for site in range(state.L)
        )

    applied_elements = int(element_count(applied_bonds))
    residual_elements = int(element_count(residual_bonds))
    itemsize = np.dtype(np.result_type(state.dtype, hamiltonian_mpo.dtype)).itemsize
    return {
        "residual_applied_bond_estimate": max(applied_bonds, default=1),
        "residual_bond_estimate": max(residual_bonds, default=1),
        "residual_applied_bytes_estimate": int(applied_elements * itemsize),
        "residual_vector_bytes_estimate": int(residual_elements * itemsize),
        "residual_peak_bytes_estimate": int(
            (applied_elements + residual_elements) * itemsize
        ),
    }


def _state_metrics(
    state,
    hamiltonian_mpo,
    *,
    residual_max_bond: int | None = None,
    residual_max_bytes: int | None = None,
):
    norm = float(np.real(to_numpy(state.H @ state)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("the mps norm must be finite and positive")
    resources = residual_resource_estimate(state, hamiltonian_mpo)
    too_wide = (
        residual_max_bond is not None
        and resources["residual_bond_estimate"] > int(residual_max_bond)
    )
    too_large = (
        residual_max_bytes is not None
        and resources["residual_peak_bytes_estimate"] > int(residual_max_bytes)
    )
    if too_wide or too_large:
        raise RuntimeError(
            "residual_resource_gate: exact uncompressed H|psi> exceeds "
            f"bond/bytes limits; estimate={resources}, "
            f"limits={{'bond': {residual_max_bond}, 'bytes': {residual_max_bytes}}}"
        )
    applied = hamiltonian_mpo.apply(state, compress=False)
    energy = float(
        np.real(to_numpy(state.H @ applied)) / norm
    )
    residual = applied + state.multiply(-energy)
    raw_variance = float(np.real(to_numpy(residual.H @ residual)) / norm)
    variance = max(raw_variance, 0.0)
    return {
        "energy": energy,
        "residual_norm": float(np.sqrt(variance)),
        "variance": float(variance),
        "residual_identity_error": float(abs(variance - raw_variance)),
        "residual_method": "exact_uncompressed_residual_mps",
        **resources,
    }


def _expectation(state, operator) -> float:
    norm = float(np.real(to_numpy(state.H @ state)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("the mps norm must be finite and positive")
    value = state.H.expec(operator, state, compress=False)
    return float(np.real(to_numpy(value)) / norm)


def _operator_moments(state, operator) -> tuple[float, float]:
    norm = float(np.real(to_numpy(state.H @ state)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("the mps norm must be finite and positive")
    mean = float(
        np.real(to_numpy(state.H.expec(operator, state, compress=False))) / norm
    )
    squared = operator.apply(operator, compress=False)
    second = float(
        np.real(to_numpy(state.H.expec(squared, state, compress=False))) / norm
    )
    return mean, max(second - mean**2, 0.0)


def _projector_state(state, max_bond: int, tolerance: float, cutoff: float):
    compressed = state.copy()
    if int(compressed.max_bond()) > int(max_bond):
        compressed.compress(max_bond=int(max_bond), cutoff=float(cutoff))
    original_norm = float(np.real(to_numpy(state.H @ state)))
    compressed_norm = float(np.real(to_numpy(compressed.H @ compressed)))
    if original_norm <= 0.0 or compressed_norm <= 0.0:
        raise ValueError("projector states must have positive norm")
    overlap = complex(to_numpy(state.H @ compressed))
    fidelity = min(
        float(abs(overlap) ** 2 / (original_norm * compressed_norm)), 1.0
    )
    infidelity = max(1.0 - fidelity, 0.0)
    if infidelity > float(tolerance):
        raise RuntimeError(
            "projector_resource_gate: bounded projector state exceeds "
            f"infidelity tolerance; max_bond={max_bond}, infidelity={infidelity}"
        )
    return compressed, infidelity


def _normalized_overlap(first, second) -> float:
    first_norm = float(np.real(to_numpy(first.H @ first)))
    second_norm = float(np.real(to_numpy(second.H @ second)))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("overlap states must have positive norm")
    overlap = complex(to_numpy(first.H @ second))
    return float(abs(overlap) / np.sqrt(first_norm * second_norm))


def _penalized_mpo(
    base,
    states,
    shift: float,
    *,
    projector_bond: int,
    projector_state_bond: int,
    projector_state_tolerance: float,
    cutoff: float,
):
    out = base.copy()
    infidelities = []
    for state in states:
        bounded, infidelity = _projector_state(
            state,
            projector_state_bond,
            projector_state_tolerance,
            cutoff,
        )
        projector = bounded.partial_trace_to_mpo(
            range(bounded.L), upper_ind_id=base.lower_ind_id
        )
        projector.multiply_(float(shift), spread_over="all")
        projector.compress(max_bond=int(projector_bond), cutoff=float(cutoff))
        out = out + projector
        out.compress(cutoff=float(cutoff))
        infidelities.append(float(infidelity))
    return out, infidelities


def total_sz_mpo(n_sites: int):
    """build the exact spin-half total-sz mpo."""
    import quimb.tensor as qtn

    return qtn.MPO_ham_heis(int(n_sites), j=0.0, bz=-1.0)


def _sector_hamiltonian(base, total_sz, target: float, shift: float):
    shifted = total_sz.copy()
    if target != 0.0:
        import quimb.tensor as qtn

        identity = qtn.MPO_identity(
            total_sz.L,
            phys_dim=total_sz.phys_dim(),
            dtype=total_sz.dtype,
        )
        identity.multiply_(-float(target), spread_over="all")
        shifted = shifted + _to_backend(identity, total_sz.backend)
    penalty = shifted.apply(shifted, compress=False)
    penalty.multiply_(float(shift), spread_over="all")
    return base + penalty


def _dmrg_sweep(
    solver,
    *,
    direction: str,
    previous_direction: str | None,
    max_bond: int,
    cutoff: float,
) -> float:
    """run one quimb sweep with the same options as ``dmrg.solve``."""
    canonize = not (
        previous_direction is not None
        and direction + previous_direction in {"LR", "RL"}
    )
    options = {
        "canonize": canonize,
        "max_bond": int(max_bond),
        "cutoff": float(cutoff),
        "cutoff_mode": solver.opts["bond_compress_cutoff_mode"],
        "method": solver.opts["bond_compress_method"],
        "verbosity": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = solver.sweep(direction=direction, **options)
    value = float(np.real(to_numpy(energy)))
    solver.energies.append(value)
    solver._compute_post_sweep()
    return value


def _converged_at_final_bond(energies, bonds, final_bond, tolerance):
    """require two energy-stable sweeps at the declared final bond."""
    return bool(
        len(energies) >= 2
        and len(bonds) >= 2
        and [int(value) for value in bonds[-2:]] == [int(final_bond)] * 2
        and abs(float(energies[-2]) - float(energies[-1])) < float(tolerance)
    )


def _dmrg_settings(
    lx,
    ly,
    count,
    bond_dims,
    tolerance,
    max_sweeps,
    projector_state_bond,
    projector_state_tolerance,
    residual_max_bond,
    residual_max_bytes,
    compute_residual,
    target_sectors,
    sector_tolerance,
    restarts,
    sweep_sequence,
):
    if count < 1:
        raise ValueError("count must be positive")
    bonds = tuple(int(value) for value in bond_dims)
    if not bonds or any(value < 1 for value in bonds):
        raise ValueError("bond_dims must contain positive values")
    if int(max_sweeps) < 1:
        raise ValueError("max_sweeps must be positive")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative")
    requested_sequence = str(sweep_sequence).upper()
    if not requested_sequence or set(requested_sequence) - {"L", "R"}:
        raise ValueError("sweep_sequence must contain only L and R")
    sequence = "R" if int(lx) * int(ly) == 2 else requested_sequence
    sectors = (
        None
        if target_sectors is None
        else tuple(float(value) for value in target_sectors)
    )
    if sectors is not None and len(sectors) != int(count):
        raise ValueError("target_sectors must contain one value per state")
    if sector_tolerance <= 0.0:
        raise ValueError("sector_tolerance must be positive")
    if compute_residual and residual_max_bond is None:
        raise ValueError("residual_max_bond is required when computing residuals")
    if residual_max_bond is not None and int(residual_max_bond) < 1:
        raise ValueError("residual_max_bond must be positive")
    if residual_max_bytes is not None and int(residual_max_bytes) < 1:
        raise ValueError("residual_max_bytes must be positive")
    state_bond = 32 if projector_state_bond is None else projector_state_bond
    if int(state_bond) < 1:
        raise ValueError("projector_state_bond must be positive")
    if projector_state_tolerance < 0.0:
        raise ValueError("projector_state_tolerance must be nonnegative")
    return {
        "bond_dims": bonds,
        "tolerance": float(tolerance),
        "max_sweeps": int(max_sweeps),
        "projector_state_bond": int(state_bond),
        "residual_max_bond": residual_max_bond,
        "residual_max_bytes": residual_max_bytes,
        "restart_count": max(1, int(restarts)),
        "sectors": sectors,
        "sweep_sequence": sequence,
        "requested_sweep_sequence": requested_sequence,
    }


def _dmrg_operators(
    ham,
    lx,
    ly,
    backend,
    sectors,
    penalty_shift,
    sector_penalty_shift,
):
    mpo = _to_backend(square_lattice_mpo(ham, lx, ly), backend)
    if penalty_shift is None:
        penalty_shift = 2.0 * sum(
            float(np.linalg.norm(np.asarray(term), 2))
            for term in ham.terms.values()
        ) + 1.0
    if sector_penalty_shift is None:
        sector_penalty_shift = float(penalty_shift)
    total_sz = None
    if sectors is not None:
        total_sz = _to_backend(total_sz_mpo(int(lx) * int(ly)), backend)
    return mpo, total_sz, float(penalty_shift), float(sector_penalty_shift)


def _validate_active_restart(progress, settings):
    candidates = list(progress.get("candidates", ()))
    restart_count = settings["restart_count"]
    if len(candidates) > restart_count:
        raise ValueError("restart progress contains too many candidates")
    for restart, candidate in enumerate(candidates):
        if int(candidate.get("restart", -1)) != restart:
            raise ValueError("restart progress is not a contiguous prefix")
        if candidate.get("converged") is True and not _converged_at_final_bond(
            candidate.get("energy_history", ()),
            candidate.get("bond_dimension_history", ()),
            settings["bond_dims"][-1],
            settings["tolerance"],
        ):
            raise ValueError("restart progress has stale convergence evidence")
    active = progress.get("active_restart")
    if active is None:
        return None
    active = dict(active)
    if int(active.get("restart", -1)) != len(candidates):
        raise ValueError("active restart does not follow completed candidates")
    if len(candidates) >= restart_count:
        raise ValueError("active restart exceeds the requested restart count")
    completed = int(active.get("completed_sweeps", -1))
    histories = [
        list(active.get("energy_history", ())),
        list(active.get("direction_history", ())),
        list(active.get("bond_dimension_history", ())),
    ]
    if not 1 <= completed <= settings["max_sweeps"]:
        raise ValueError("active restart has an invalid sweep count")
    if not all(len(history) == completed for history in histories):
        raise ValueError("active restart histories have inconsistent lengths")
    expected_directions = [
        settings["sweep_sequence"][index % len(settings["sweep_sequence"])]
        for index in range(completed)
    ]
    expected_bonds = [
        settings["bond_dims"][min(index, len(settings["bond_dims"]) - 1)]
        for index in range(completed)
    ]
    if histories[1] != expected_directions:
        raise ValueError("active restart has a stale sweep sequence")
    if [int(value) for value in histories[2]] != expected_bonds:
        raise ValueError("active restart has stale bond progression")
    if active.get("converged") is True and not _converged_at_final_bond(
        histories[0],
        histories[2],
        settings["bond_dims"][-1],
        settings["tolerance"],
    ):
        raise ValueError("active restart has stale convergence evidence")
    if "state" not in active:
        raise ValueError("active restart is missing its state")
    return active


def _resume_dmrg(
    initial_states,
    initial_records,
    initial_restart_progress,
    count,
    settings,
):
    states = list(initial_states)
    records = [dict(record) for record in initial_records]
    if len(states) != len(records) or len(states) > int(count):
        raise ValueError("resumed states and records must have a valid common length")
    if initial_restart_progress is None:
        return states, records, None, None
    progress = dict(initial_restart_progress)
    if len(states) >= int(count):
        raise ValueError("restart progress cannot follow every requested state")
    if int(progress.get("state_index", -1)) != len(states):
        raise ValueError("restart progress does not match the next state")
    active = _validate_active_restart(progress, settings)
    return states, records, progress, active


def _dmrg_metadata(
    mpo,
    settings,
    *,
    penalty_shift,
    projector_bond,
    projector_state_tolerance,
    compute_residual,
    sector_penalty_shift,
    sector_tolerance,
    backend,
):
    sectors = settings["sectors"]
    sequence = settings["sweep_sequence"]
    requested = settings["requested_sweep_sequence"]
    return {
        "method": "dmrg",
        "mpo_bond": int(mpo.max_bond()),
        "penalty_shift": float(penalty_shift),
        "projector_bond": int(projector_bond),
        "projector_state_bond": settings["projector_state_bond"],
        "projector_state_tolerance": float(projector_state_tolerance),
        "residual_method": (
            "exact_uncompressed_residual_mps" if compute_residual else "not_computed"
        ),
        "residual_required": bool(compute_residual),
        "residual_max_bond": (
            None
            if settings["residual_max_bond"] is None
            else int(settings["residual_max_bond"])
        ),
        "residual_max_bytes": (
            None
            if settings["residual_max_bytes"] is None
            else int(settings["residual_max_bytes"])
        ),
        "backend": backend,
        "symmetry_sector": "unrestricted" if sectors is None else list(sectors),
        "sector_penalty_shift": (
            None if sectors is None else float(sector_penalty_shift)
        ),
        "sector_tolerance": None if sectors is None else float(sector_tolerance),
        "sweep_sequence": sequence,
        "requested_sweep_sequence": requested,
        "sweep_sequence_adjustment": (
            "single_bond_right_sweeps" if sequence != requested else None
        ),
        "max_sweeps": settings["max_sweeps"],
        "bond_dims": list(settings["bond_dims"]),
        "convergence_tolerance": settings["tolerance"],
        "convergence_contract": "two_consecutive_final_bond_sweeps",
        "checkpoint_boundary": "completed_dmrg_sweep",
    }


def _effective_mpo(
    mpo,
    total_sz,
    states,
    index,
    settings,
    *,
    penalty_shift,
    sector_penalty_shift,
    projector_bond,
    projector_state_tolerance,
    cutoff,
):
    sectors = settings["sectors"]
    target = None if sectors is None else sectors[index]
    base = (
        mpo
        if target is None
        else _sector_hamiltonian(mpo, total_sz, target, sector_penalty_shift)
    )
    penalty_states = []
    skipped = 0
    for previous_index, previous in enumerate(states):
        previous_sector = None if sectors is None else sectors[previous_index]
        if target is not None and previous_sector != target:
            skipped += 1
        else:
            penalty_states.append(previous)
    if not penalty_states:
        return base, target, penalty_states, skipped, []
    effective, infidelities = _penalized_mpo(
        base,
        penalty_states,
        penalty_shift,
        projector_bond=int(projector_bond),
        projector_state_bond=settings["projector_state_bond"],
        projector_state_tolerance=float(projector_state_tolerance),
        cutoff=float(cutoff),
    )
    return effective, target, penalty_states, skipped, infidelities


def _restart_state(qtn, active, lx, ly, settings, backend, seed, index, restart):
    if active is not None and restart == int(active["restart"]):
        return (
            active["state"],
            int(active["completed_sweeps"]),
            [float(value) for value in active["energy_history"]],
            list(active["direction_history"]),
            [int(value) for value in active["bond_dimension_history"]],
            bool(active.get("converged", False)),
        )
    initial = qtn.MPS_rand_state(
        int(lx) * int(ly),
        bond_dim=min(settings["bond_dims"][0], 16),
        dtype="complex128",
        seed=int(seed) + 1009 * index + restart,
    )
    return _to_backend(initial, backend), 0, [], [], [], False


def _run_restart(
    qtn,
    effective,
    active,
    restart,
    index,
    states,
    records,
    candidates,
    metadata,
    settings,
    *,
    lx,
    ly,
    backend,
    seed,
    cutoff,
    tolerance,
    sweep_callback,
):
    initial, completed, energies, directions, bonds, converged = _restart_state(
        qtn, active, lx, ly, settings, backend, seed, index, restart
    )
    solver = qtn.DMRG2(
        effective,
        bond_dims=settings["bond_dims"],
        cutoffs=float(cutoff),
        p0=initial,
    )
    solver.energies = list(energies)
    sequence = settings["sweep_sequence"]
    for sweep_index in range(completed, settings["max_sweeps"]):
        if converged:
            break
        direction = sequence[sweep_index % len(sequence)]
        previous = None if sweep_index == 0 else sequence[(sweep_index - 1) % len(sequence)]
        max_bond = settings["bond_dims"][
            min(sweep_index, len(settings["bond_dims"]) - 1)
        ]
        energy = _dmrg_sweep(
            solver,
            direction=direction,
            previous_direction=previous,
            max_bond=max_bond,
            cutoff=float(cutoff),
        )
        energies.append(energy)
        directions.append(direction)
        bonds.append(max_bond)
        completed = sweep_index + 1
        converged = _converged_at_final_bond(
            energies,
            bonds,
            settings["bond_dims"][-1],
            tolerance,
        )
        active_progress = {
            "restart": restart,
            "completed_sweeps": completed,
            "energy_history": list(energies),
            "direction_history": list(directions),
            "bond_dimension_history": list(bonds),
            "converged": converged,
            "state": solver.state,
        }
        if sweep_callback is not None:
            sweep_callback(states, records, metadata, {
                "state_index": index,
                "candidates": candidates,
                "active_restart": active_progress,
            })
    return {
        "restart": restart,
        "energy": float(energies[-1]),
        "converged": converged,
        "state": solver.state,
        "sweeps_completed": completed,
        "energy_history": list(energies),
        "direction_history": list(directions),
        "bond_dimension_history": list(bonds),
    }


def _restart_candidates(
    effective,
    progress,
    active,
    index,
    states,
    records,
    metadata,
    settings,
    *,
    lx,
    ly,
    backend,
    seed,
    cutoff,
    tolerance,
    restart_callback,
    sweep_callback,
):
    import quimb.tensor as qtn

    candidates = [] if progress is None else list(progress.get("candidates", ()))
    for restart in range(len(candidates), settings["restart_count"]):
        candidate = _run_restart(
            qtn,
            effective,
            active,
            restart,
            index,
            states,
            records,
            candidates,
            metadata,
            settings,
            lx=lx,
            ly=ly,
            backend=backend,
            seed=seed,
            cutoff=cutoff,
            tolerance=tolerance,
            sweep_callback=sweep_callback,
        )
        candidates.append(candidate)
        active = None
        if restart_callback is not None:
            restart_callback(states, records, metadata, {
                "state_index": index,
                "candidates": candidates,
            })
    return candidates


def _sector_record(
    state,
    total_sz,
    target,
    sector_penalty_shift,
    sector_tolerance,
):
    record = {
        "target_total_sz": target,
        "total_sz": None,
        "total_sz_variance": None,
        "sector_validated": None,
        "sector_penalty_shift": (
            None if target is None else float(sector_penalty_shift)
        ),
    }
    if target is None:
        return record
    total_sz_value, variance = _operator_moments(state, total_sz)
    validated = (
        abs(total_sz_value - target) <= float(sector_tolerance)
        and variance <= float(sector_tolerance) ** 2
    )
    record.update({
        "total_sz": total_sz_value,
        "total_sz_variance": variance,
        "sector_validated": bool(validated),
    })
    return record


def _dmrg_state_record(
    state,
    selected,
    candidates,
    states,
    mpo,
    total_sz,
    target,
    penalty_states,
    skipped_projectors,
    projector_infidelities,
    settings,
    *,
    index,
    compute_residual,
    sector_penalty_shift,
    sector_tolerance,
):
    overlaps = [_normalized_overlap(previous, state) for previous in states]
    if compute_residual:
        metrics = _state_metrics(
            state,
            mpo,
            residual_max_bond=int(settings["residual_max_bond"]),
            residual_max_bytes=settings["residual_max_bytes"],
        )
    else:
        metrics = {
            "energy": _expectation(state, mpo),
            "residual_norm": None,
            "variance": None,
            "residual_identity_error": None,
            "residual_method": "not_computed",
            **residual_resource_estimate(state, mpo),
        }
    sector = _sector_record(
        state,
        total_sz,
        target,
        sector_penalty_shift,
        sector_tolerance,
    )
    return {
        "state_index": index,
        "converged": bool(
            selected["converged"] and sector["sector_validated"] is not False
        ),
        "solver_converged": bool(selected["converged"]),
        "effective_energy": float(selected["energy"]),
        "restart_energies": [float(candidate["energy"]) for candidate in candidates],
        "restart_sweep_counts": [
            candidate.get("sweeps_completed") for candidate in candidates
        ],
        "restart_energy_histories": [
            candidate.get("energy_history") for candidate in candidates
        ],
        "restart_direction_histories": [
            candidate.get("direction_history") for candidate in candidates
        ],
        "restart_bond_dimension_histories": [
            candidate.get("bond_dimension_history") for candidate in candidates
        ],
        "restarts": settings["restart_count"],
        "max_bond": int(state.max_bond()),
        "overlaps": overlaps,
        "max_previous_overlap": max(overlaps, default=0.0),
        "overlap_method": "exact_normalized_mps_overlap",
        "projector_states_used": len(penalty_states),
        "projector_states_skipped": skipped_projectors,
        "projector_compression_infidelities": projector_infidelities,
        **sector,
        **metrics,
    }


def _solve_dmrg_state(
    mpo,
    total_sz,
    states,
    records,
    metadata,
    progress,
    active,
    settings,
    *,
    index,
    lx,
    ly,
    backend,
    seed,
    cutoff,
    tolerance,
    penalty_shift,
    sector_penalty_shift,
    sector_tolerance,
    projector_bond,
    projector_state_tolerance,
    compute_residual,
    restart_callback,
    sweep_callback,
):
    effective, target, penalty_states, skipped, infidelities = _effective_mpo(
        mpo,
        total_sz,
        states,
        index,
        settings,
        penalty_shift=penalty_shift,
        sector_penalty_shift=sector_penalty_shift,
        projector_bond=projector_bond,
        projector_state_tolerance=projector_state_tolerance,
        cutoff=cutoff,
    )
    candidates = _restart_candidates(
        effective,
        progress,
        active,
        index,
        states,
        records,
        metadata,
        settings,
        lx=lx,
        ly=ly,
        backend=backend,
        seed=seed,
        cutoff=cutoff,
        tolerance=tolerance,
        restart_callback=restart_callback,
        sweep_callback=sweep_callback,
    )
    selected = min(candidates, key=lambda item: item["energy"])
    state = selected["state"]
    record = _dmrg_state_record(
        state,
        selected,
        candidates,
        states,
        mpo,
        total_sz,
        target,
        penalty_states,
        skipped,
        infidelities,
        settings,
        index=index,
        compute_residual=compute_residual,
        sector_penalty_shift=sector_penalty_shift,
        sector_tolerance=sector_tolerance,
    )
    return state, record


def dmrg_low_energies(
    ham,
    lx: int,
    ly: int,
    count: int,
    *,
    bond_dims=(64, 128, 256),
    cutoff: float = 1e-10,
    tolerance: float = 1e-8,
    max_sweeps: int = 12,
    projector_bond: int = 512,
    projector_state_bond: int | None = None,
    projector_state_tolerance: float = 0.1,
    residual_max_bond: int | None = 512,
    residual_max_bytes: int | None = 8 * 1024**3,
    compute_residual: bool = True,
    penalty_shift: float | None = None,
    target_sectors=None,
    sector_penalty_shift: float | None = None,
    sector_tolerance: float = 1e-6,
    backend: str = "numpy",
    restarts: int = 3,
    sweep_sequence: str = "RL",
    seed: int = 0,
    initial_states=(),
    initial_records=(),
    initial_restart_progress=None,
    checkpoint_callback=None,
    restart_callback=None,
    sweep_callback=None,
):
    """compute low energies with sequential orthogonality penalties.

    checkpoints can resume completed sweeps, restarts, and target states. an
    individual quimb sweep remains atomic because its local eigensolver and
    moving environments have no stable partial serialization boundary.
    """
    settings = _dmrg_settings(
        lx,
        ly,
        count,
        bond_dims,
        tolerance,
        max_sweeps,
        projector_state_bond,
        projector_state_tolerance,
        residual_max_bond,
        residual_max_bytes,
        compute_residual,
        target_sectors,
        sector_tolerance,
        restarts,
        sweep_sequence,
    )
    mpo, total_sz, penalty_shift, sector_penalty_shift = _dmrg_operators(
        ham,
        lx,
        ly,
        backend,
        settings["sectors"],
        penalty_shift,
        sector_penalty_shift,
    )
    states, records, progress, active = _resume_dmrg(
        initial_states,
        initial_records,
        initial_restart_progress,
        count,
        settings,
    )
    metadata = _dmrg_metadata(
        mpo,
        settings,
        penalty_shift=penalty_shift,
        projector_bond=projector_bond,
        projector_state_tolerance=projector_state_tolerance,
        compute_residual=compute_residual,
        sector_penalty_shift=sector_penalty_shift,
        sector_tolerance=sector_tolerance,
        backend=backend,
    )
    for index in range(len(states), int(count)):
        state, record = _solve_dmrg_state(
            mpo,
            total_sz,
            states,
            records,
            metadata,
            progress,
            active,
            settings,
            index=index,
            lx=lx,
            ly=ly,
            backend=backend,
            seed=seed,
            cutoff=cutoff,
            tolerance=tolerance,
            penalty_shift=penalty_shift,
            sector_penalty_shift=sector_penalty_shift,
            sector_tolerance=sector_tolerance,
            projector_bond=projector_bond,
            projector_state_tolerance=projector_state_tolerance,
            compute_residual=compute_residual,
            restart_callback=restart_callback,
            sweep_callback=sweep_callback,
        )
        records.append(record)
        states.append(state)
        progress = None
        active = None
        if checkpoint_callback is not None:
            checkpoint_callback(states, records, metadata)
    return states, records, metadata
