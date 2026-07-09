"""Hamiltonian builders + the shared state-spec grammar (ham_from_spec)."""

import numpy as np
import pytest

quimb = pytest.importorskip("quimb")

from rand_isopeps.real_isotns.tebd2 import (  # noqa: E402
    compass_ham,
    ham_from_spec,
    heis_ham,
    tfi_ham,
    xxz_ham,
)

LX, LY = 3, 3
N_BONDS = LX * (LY - 1) + (LX - 1) * LY  # nearest-neighbour bonds on an open 3x3


def _terms(ham):
    return {k: np.asarray(v) for k, v in ham.terms.items()}


@pytest.mark.parametrize("spec", ["tfim@3.5", "heis", "xxz@0.5", "compass"])
def test_spec_builds_all_bonds_hermitian(spec):
    ham = ham_from_spec(spec, LX, LY)
    terms = _terms(ham)
    assert len(terms) == N_BONDS
    for H in terms.values():
        assert np.allclose(H, H.conj().T)


def test_random_spec_is_none_and_unknown_raises():
    assert ham_from_spec("random", LX, LY) is None
    with pytest.raises(ValueError):
        ham_from_spec("kitaev", LX, LY)


def test_tfim_spec_matches_builder():
    a, b = _terms(ham_from_spec("tfim@2.0", LX, LY)), _terms(tfi_ham(LX, LY, g=2.0))
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_xxz_delta_one_is_heisenberg():
    a, b = _terms(xxz_ham(LX, LY, delta=1.0)), _terms(heis_ham(LX, LY))
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_compass_is_bond_direction_anisotropic():
    terms = _terms(compass_ham(LX, LY))
    h = terms[((0, 0), (0, 1))]   # horizontal: -XX (+ any folded H1, none here)
    v = terms[((0, 0), (1, 0))]   # vertical:   -YY
    assert not np.allclose(h, v)
    xx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
    assert np.allclose(h, -xx)
