from __future__ import annotations

import os

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import pytest

pytest.importorskip("quimb")

from rand_isopeps.physics import (
    checkerboard_layers,
    exact_imaginary_step,
    local_term_norm_bound,
    normalize_state,
    rayleigh_residual,
    rayleigh_ritz,
    run_iterations,
    sparse_hamiltonian,
    trotter_imaginary_step,
)
from rand_isopeps.real_isotns.tebd2 import compass_ham, tfi_ham


@pytest.mark.parametrize("builder", [lambda: tfi_ham(2, 3, g=2.1), lambda: compass_ham(2, 3)])
def test_checkerboard_layers_are_disjoint_and_sum_to_hamiltonian(builder):
    ham = builder()
    layers = checkerboard_layers(ham, 2, 3)
    total = sum((entry["matrix"] for entry in layers.values()))
    exact = sparse_hamiltonian(ham, 2, 3)
    assert np.linalg.norm((total - exact).toarray()) < 1e-12
    for entry in layers.values():
        sites = [site for bond in entry["bonds"] for site in bond]
        assert len(sites) == len(set(sites))


def test_rayleigh_residual_matches_variance_and_keeps_layer_cross_terms():
    ham = tfi_ham(2, 2, g=1.7)
    h = sparse_hamiltonian(ham, 2, 2)
    layers = checkerboard_layers(ham, 2, 2)
    rng = np.random.default_rng(14)
    state = normalize_state(rng.standard_normal(16) + 1j * rng.standard_normal(16))
    metrics = rayleigh_residual(h, state, h_norm_bound=local_term_norm_bound(ham))
    assert metrics["variance"] == pytest.approx(metrics["variance_expanded"], abs=1e-12)

    actions = [entry["matrix"] @ state for entry in layers.values() if entry["bonds"]]
    full_h2 = sum(np.vdot(a, b) for a in actions for b in actions).real
    diagonal_only = sum(np.vdot(a, a).real for a in actions)
    assert full_h2 == pytest.approx(np.vdot(h @ state, h @ state).real, abs=1e-12)
    assert abs(full_h2 - diagonal_only) > 1e-6

    values, vectors = np.linalg.eigh(h.toarray())
    eigen = rayleigh_residual(h, vectors[:, 0], h_norm_bound=local_term_norm_bound(ham))
    assert eigen["energy"] == pytest.approx(values[0], abs=1e-11)
    assert eigen["residual_norm"] < 1e-10


def test_dense_imaginary_time_and_block_ritz():
    ham = tfi_ham(2, 2, g=2.5)
    h = sparse_hamiltonian(ham, 2, 2)
    bound = local_term_norm_bound(ham)
    rng = np.random.default_rng(9)
    state = normalize_state(rng.standard_normal(16) + 1j * rng.standard_normal(16))
    energies = []
    for _ in range(3):
        energies.append(rayleigh_residual(h, state, h_norm_bound=bound)["energy"])
        state = exact_imaginary_step(h, state, 0.02)
    energies.append(rayleigh_residual(h, state, h_norm_bound=bound)["energy"])
    assert np.all(np.diff(energies) <= 1e-12)

    _, eigenvectors = np.linalg.eigh(h.toarray())
    mixed = eigenvectors[:, :2] @ np.array([[1.0, 1.0], [1.0, -1.0]])
    ritz = rayleigh_ritz(h, mixed, h_norm_bound=bound)
    assert np.all(ritz["residual_norms"] < 1e-10)


def test_strang_one_step_error_is_cubic_under_step_halving():
    ham = tfi_ham(2, 3, g=2.1)
    h = sparse_hamiltonian(ham, 2, 3)
    layers = checkerboard_layers(ham, 2, 3)
    rng = np.random.default_rng(5)
    state = normalize_state(rng.standard_normal(64) + 1j * rng.standard_normal(64))

    errors = []
    for tau in (0.04, 0.02):
        exact = exact_imaginary_step(h, state, tau)
        trotter = trotter_imaginary_step(layers, state, tau)
        phase = np.vdot(exact, trotter)
        trotter = trotter * np.exp(-1j * np.angle(phase))
        errors.append(np.linalg.norm(exact - trotter))
    assert errors[0] / errors[1] > 7.0


def test_fixed_iteration_loop_streams_initial_and_updated_records():
    streamed = []

    def update(state, _iteration):
        return state + 1, {"updated": True}

    _, history = run_iterations(
        0,
        iterations=2,
        update=update,
        measure=lambda state, _iteration: {"state": state},
        on_record=lambda record: streamed.append(dict(record)),
    )
    assert [record["state"] for record in history] == [0, 1, 2]
    assert streamed == history
