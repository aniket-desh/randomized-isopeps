import numpy as np
import pytest

from rand_isopeps.physics import sparse_hamiltonian
from rand_isopeps.physics.mps_reference import (
    _sector_hamiltonian,
    _state_metrics,
    _expectation,
    _normalized_overlap,
    _projector_state,
    dmrg_low_energies,
    residual_resource_estimate,
    square_lattice_mpo,
    total_sz_mpo,
)
from rand_isopeps.real_isotns.tebd2 import ham_from_spec


def test_square_lattice_mpo_matches_dense_hamiltonian():
    hamiltonian = ham_from_spec("tfim@3.5", 2, 2)
    expected = sparse_hamiltonian(hamiltonian, 2, 2).toarray()
    actual = np.asarray(square_lattice_mpo(hamiltonian, 2, 2).to_dense())
    assert np.linalg.norm(actual - expected) < 1e-11


def test_mps_residual_uses_exact_uncompressed_second_moment():
    import quimb.tensor as qtn

    hamiltonian = ham_from_spec("tfim@3.5", 2, 2)
    matrix = sparse_hamiltonian(hamiltonian, 2, 2).toarray()
    mpo = square_lattice_mpo(hamiltonian, 2, 2)
    state = qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=17)
    state.multiply_(2.3, spread_over="all")
    vector = np.asarray(state.to_dense()).reshape(-1)
    norm = np.vdot(vector, vector).real
    energy = np.vdot(vector, matrix @ vector).real / norm
    residual = matrix @ vector - energy * vector

    metrics = _state_metrics(state, mpo)

    assert _expectation(state, mpo) == pytest.approx(energy, abs=1e-11)
    assert metrics["energy"] == pytest.approx(energy, abs=1e-11)
    assert metrics["residual_norm"] == pytest.approx(
        np.linalg.norm(residual) / np.sqrt(norm), abs=1e-10
    )
    assert metrics["residual_method"] == "exact_uncompressed_residual_mps"
    assert metrics["residual_applied_bond_estimate"] == max(
        state.bond_size(site, site + 1) * mpo.bond_size(site, site + 1)
        for site in range(state.L - 1)
    )


def test_exact_residual_fails_before_an_oversized_apply():
    import quimb.tensor as qtn

    hamiltonian = ham_from_spec("tfim@3.5", 2, 2)
    mpo = square_lattice_mpo(hamiltonian, 2, 2)
    state = qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=19)
    estimate = residual_resource_estimate(state, mpo)

    with pytest.raises(RuntimeError, match="residual_resource_gate"):
        _state_metrics(
            state,
            mpo,
            residual_max_bond=estimate["residual_bond_estimate"] - 1,
        )


def test_projector_state_compression_fails_closed():
    import quimb.tensor as qtn

    state = qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23)
    bounded, infidelity = _projector_state(state, 2, 1e-12, 0.0)
    assert bounded.max_bond() <= 2
    assert infidelity <= 1e-12
    with pytest.raises(RuntimeError, match="projector_resource_gate"):
        _projector_state(state, 1, 0.0, 0.0)


def test_exact_overlap_is_conjugated_and_normalized_for_complex_states():
    import quimb.tensor as qtn

    first = qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=29)
    second = first.multiply(2.0j)

    assert _normalized_overlap(first, second) == pytest.approx(1.0, abs=1e-12)


def test_total_sz_penalty_targets_the_physical_spin_sector():
    hamiltonian = ham_from_spec("heis", 2, 2)
    base = square_lattice_mpo(hamiltonian, 2, 2)
    total_sz = total_sz_mpo(4)
    target = 1.0
    shift = 23.0

    actual = np.asarray(
        _sector_hamiltonian(base, total_sz, target, shift).to_dense()
    )
    h = np.asarray(base.to_dense())
    sz = np.asarray(total_sz.to_dense())
    expected = h + shift * (sz - target * np.eye(sz.shape[0])) @ (
        sz - target * np.eye(sz.shape[0])
    )

    assert actual == pytest.approx(expected, abs=1e-11)
    assert np.diag(sz)[:5] == pytest.approx([2.0, 1.0, 1.0, 0.0, 1.0])


def test_first_dmrg_state_honors_a_non_ground_symmetry_sector():
    hamiltonian = ham_from_spec("heis", 1, 2)
    _, records, metadata = dmrg_low_energies(
        hamiltonian,
        1,
        2,
        1,
        bond_dims=(2,),
        max_sweeps=2,
        restarts=1,
        target_sectors=(1,),
        sector_penalty_shift=20.0,
        sector_tolerance=1e-8,
        residual_max_bond=64,
        seed=1,
    )

    assert records[0]["total_sz"] == pytest.approx(1.0, abs=1e-10)
    assert records[0]["total_sz_variance"] <= 1e-16
    assert records[0]["sector_validated"] is True
    assert records[0]["energy"] == pytest.approx(1.0, abs=1e-10)
    assert metadata["symmetry_sector"] == [1.0]
    assert metadata["sector_penalty_shift"] == 20.0


def test_distinct_symmetry_sectors_skip_quadratic_projectors():
    hamiltonian = ham_from_spec("heis", 1, 2)
    _, records, _ = dmrg_low_energies(
        hamiltonian,
        1,
        2,
        2,
        bond_dims=(2,),
        max_sweeps=2,
        restarts=1,
        target_sectors=(0, 1),
        sector_penalty_shift=20.0,
        sector_tolerance=1e-8,
        projector_state_bond=1,
        projector_state_tolerance=0.0,
        residual_max_bond=64,
        seed=2,
    )

    assert records[1]["projector_states_used"] == 0
    assert records[1]["projector_states_skipped"] == 1


def test_dmrg_convergence_requires_two_sweeps_at_the_final_bond():
    hamiltonian = ham_from_spec("heis", 1, 4)
    _, records, metadata = dmrg_low_energies(
        hamiltonian,
        1,
        4,
        1,
        bond_dims=(1, 2),
        tolerance=1e9,
        max_sweeps=3,
        restarts=1,
        compute_residual=False,
        seed=3,
    )

    assert records[0]["solver_converged"] is True
    assert records[0]["restart_bond_dimension_histories"] == [[1, 2, 2]]
    assert metadata["convergence_contract"] == "two_consecutive_final_bond_sweeps"


def test_dmrg_resume_reuses_completed_restart_candidates():
    hamiltonian = ham_from_spec("heis", 1, 2)
    saved = {}

    def interrupt_after_restart(_states, _records, _metadata, progress):
        saved.update(progress)
        raise InterruptedError("stop after restart")

    with pytest.raises(InterruptedError, match="after restart"):
        dmrg_low_energies(
            hamiltonian,
            1,
            2,
            1,
            bond_dims=(2,),
            max_sweeps=2,
            restarts=2,
            compute_residual=False,
            seed=31,
            restart_callback=interrupt_after_restart,
        )

    assert saved["state_index"] == 0
    assert len(saved["candidates"]) == 1
    first_energy = saved["candidates"][0]["energy"]
    resumed_progress = []
    _, records, _ = dmrg_low_energies(
        hamiltonian,
        1,
        2,
        1,
        bond_dims=(2,),
        max_sweeps=2,
        restarts=2,
        compute_residual=False,
        seed=31,
        initial_restart_progress=saved,
        restart_callback=lambda _s, _r, _m, progress: resumed_progress.append(
            len(progress["candidates"])
        ),
    )

    assert resumed_progress == [2]
    assert records[0]["restart_energies"][0] == pytest.approx(first_energy)


def test_dmrg_sweep_resume_matches_uninterrupted_state_and_energy():
    hamiltonian = ham_from_spec("heis", 1, 4)
    options = {
        "bond_dims": (2, 4),
        "cutoff": 1e-10,
        "tolerance": 0.0,
        "max_sweeps": 4,
        "restarts": 1,
        "compute_residual": False,
        "sweep_sequence": "RL",
        "seed": 41,
    }
    full_states, full_records, full_metadata = dmrg_low_energies(
        hamiltonian,
        1,
        4,
        1,
        **options,
    )
    saved = {}

    def interrupt_after_two_sweeps(_states, _records, _metadata, progress):
        active = progress["active_restart"]
        if active["completed_sweeps"] == 2:
            saved.update(progress)
            raise InterruptedError("stop after two sweeps")

    with pytest.raises(InterruptedError, match="after two sweeps"):
        dmrg_low_energies(
            hamiltonian,
            1,
            4,
            1,
            sweep_callback=interrupt_after_two_sweeps,
            **options,
        )

    active = saved["active_restart"]
    assert active["completed_sweeps"] == 2
    assert active["direction_history"] == ["R", "L"]
    assert active["bond_dimension_history"] == [2, 4]
    resumed_sweeps = []
    resumed_states, resumed_records, resumed_metadata = dmrg_low_energies(
        hamiltonian,
        1,
        4,
        1,
        initial_restart_progress=saved,
        sweep_callback=lambda _s, _r, _m, progress: resumed_sweeps.append(
            progress["active_restart"]["completed_sweeps"]
        ),
        **options,
    )

    assert resumed_sweeps == [3, 4]
    assert full_records[0]["restart_direction_histories"] == [["R", "L", "R", "L"]]
    assert full_records[0]["restart_bond_dimension_histories"] == [[2, 4, 4, 4]]
    assert resumed_records[0]["effective_energy"] == pytest.approx(
        full_records[0]["effective_energy"], abs=1e-12
    )
    assert resumed_records[0]["restart_energy_histories"][0] == pytest.approx(
        full_records[0]["restart_energy_histories"][0], abs=1e-12
    )
    assert _normalized_overlap(resumed_states[0], full_states[0]) == pytest.approx(
        1.0, abs=1e-12
    )
    assert full_metadata["sweep_sequence"] == "RL"
    assert resumed_metadata["checkpoint_boundary"] == "completed_dmrg_sweep"


def test_staged_sweeps_match_quimb_multi_sweep_solve():
    import quimb.tensor as qtn

    hamiltonian = ham_from_spec("heis", 1, 4)
    staged_states, staged_records, _ = dmrg_low_energies(
        hamiltonian,
        1,
        4,
        1,
        bond_dims=(2, 4),
        cutoff=1e-10,
        tolerance=0.0,
        max_sweeps=4,
        restarts=1,
        compute_residual=False,
        sweep_sequence="RL",
        seed=47,
    )
    mpo = square_lattice_mpo(hamiltonian, 1, 4)
    initial = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        dtype="complex128",
        seed=47,
    )
    solver = qtn.DMRG2(
        mpo,
        bond_dims=(2, 4),
        cutoffs=1e-10,
        p0=initial,
    )
    solver.solve(
        tol=0.0,
        sweep_sequence="RL",
        max_sweeps=4,
        verbosity=0,
    )

    assert staged_records[0]["restart_energy_histories"][0] == pytest.approx(
        np.real(solver.energies), abs=1e-12
    )
    assert _normalized_overlap(staged_states[0], solver.state) == pytest.approx(
        1.0, abs=1e-12
    )
