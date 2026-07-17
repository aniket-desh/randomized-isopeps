from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import quimb as qu

from rand_isopeps.real_isotns.tebd2 import ham_from_spec

_SCRIPT = Path(__file__).parents[1] / "experiments/column_sketch/scripts/exp11_invariant_one_move.py"
_SPEC = importlib.util.spec_from_file_location("exp11_invariant_one_move", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
_dense_hamiltonian = _MODULE._dense_hamiltonian
_dense_observables = _MODULE._dense_observables
_scale_safe_dense_state = _MODULE._scale_safe_dense_state
_method_grid = _MODULE._method_grid
_state_cut_diagnostics = _MODULE._state_cut_diagnostics
_boundary_overlap = _MODULE._boundary_overlap
_boundary_observables = _MODULE._boundary_observables
_exact_ground_energy = _MODULE._exact_ground_energy


def test_dense_hamiltonian_embeds_noncontiguous_vertical_terms():
    lx = ly = 2
    g = 1.7
    found = _dense_hamiltonian(ham_from_spec(f"tfim@{g}", lx, ly), lx, ly).toarray()
    dims = [2] * (lx * ly)
    x, z = qu.pauli("X"), qu.pauli("Z")
    wanted = np.zeros_like(found)
    for site in range(lx * ly):
        wanted -= g * qu.ikron(x, dims, site)
    for pair in ((0, 1), (0, 2), (1, 3), (2, 3)):
        wanted -= qu.ikron([z, z], dims, pair)
    assert np.allclose(found, wanted)
    assert _exact_ground_energy(f"tfim@{g}", lx, ly, 4) == __import__("pytest").approx(
        np.linalg.eigvalsh(wanted)[0], abs=1e-10
    )


def test_dense_observables_and_cut_entropies_on_product_state():
    vector = np.zeros(16)
    vector[0] = 1.0
    mags, corrs, parity = _dense_observables(vector, 2, 2)
    assert np.allclose(mags, 1.0)
    assert np.allclose(corrs, 1.0)
    assert parity == 0.0
    cuts = _state_cut_diagnostics(vector, 2, 2)
    assert np.allclose(__import__("json").loads(cuts["prep_row_cut_renyi2"]), 0.0)
    assert np.allclose(__import__("json").loads(cuts["prep_column_cut_renyi2"]), 0.0)

    rng = np.random.default_rng(33)
    random_vector = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    mags, corrs, parity = _dense_observables(random_vector, 2, 2)
    norm = np.vdot(random_vector, random_vector).real
    dims = [2] * 4
    z, x = qu.pauli("Z"), qu.pauli("X")
    expected_mags = [
        (np.vdot(random_vector, qu.ikron(z, dims, site) @ random_vector) / norm).real
        for site in range(4)
    ]
    expected_corrs = [
        (np.vdot(random_vector, qu.ikron([z, z], dims, pair) @ random_vector) / norm).real
        for pair in ((0, 2), (0, 1), (1, 3), (2, 3))
    ]
    expected_parity = (
        np.vdot(random_vector, qu.ikron([x] * 4, dims, tuple(range(4))) @ random_vector)
        / norm
    ).real
    assert np.allclose(mags, expected_mags)
    assert np.allclose(corrs, expected_corrs)
    assert parity == __import__("pytest").approx(expected_parity)


def test_boundary_metrics_match_exact_small_contraction():
    from rand_isopeps.real_isotns.moses_move import random_isotns

    a = random_isotns(2, 2, bond=2, phys=2, chi=2, eta=2, Ndis=1, seed=71)
    b = random_isotns(2, 2, bond=2, phys=2, chi=2, eta=2, Ndis=1, seed=72)
    overlap, norm_a, norm_b = _boundary_overlap(a, b, max_bond=32)
    va = np.asarray(a.to_dense()).reshape(-1)
    vb = np.asarray(b.to_dense()).reshape(-1)
    assert overlap == __import__("pytest").approx(np.vdot(va, vb), abs=1e-10)
    assert norm_a == __import__("pytest").approx(np.vdot(va, va).real, abs=1e-10)
    assert norm_b == __import__("pytest").approx(np.vdot(vb, vb).real, abs=1e-10)

    exact_mags, exact_corrs, _ = _dense_observables(va, 2, 2)
    mags, corrs = _boundary_observables(a, 2, 2, max_bond=32)
    assert np.allclose(mags, exact_mags, atol=1e-10)
    assert np.allclose(corrs, exact_corrs, atol=1e-10)


def test_scale_safe_dense_state_avoids_global_exponent_overflow():
    import pytest
    import quimb.tensor as qtn

    psi = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=17)
    reference = psi.copy()
    reference.exponent = 0.0
    raw = np.asarray(reference.to_dense()).reshape(-1)
    expected = raw / np.linalg.norm(raw)

    psi.exponent = 400.0
    vector, log10_norm = _scale_safe_dense_state(psi)
    assert np.all(np.isfinite(vector))
    assert np.allclose(vector, expected)
    assert log10_norm == pytest.approx(400.0 + np.log10(np.linalg.norm(raw)))
    assert psi.exponent == 400.0


def test_one_at_a_time_grid_avoids_cartesian_explosion():
    methods = [
        "local_det", "local_rsvd2", "global_gaussian", "global_rmps_plain",
        "global_rmps_bounded", "global_kron",
    ]
    common = dict(
        methods=methods, eta_grid=(4,), ell_grid=(), sketch_seeds=4,
        n_power_grid=(0, 1), chi_sk_grid=(1, 2, 4, 8),
        kappa_grid=(1, 2, 3, 4), ell_oversampling_grid=(2, 4, 8),
    )
    op = SimpleNamespace(n_in=65536)
    cartesian = len(_method_grid(SimpleNamespace(**common, grid_mode="cartesian"), op))
    one_at_a_time = len(
        _method_grid(SimpleNamespace(**common, grid_mode="one_at_a_time"), op)
    )
    assert cartesian == 513
    assert one_at_a_time == 117
