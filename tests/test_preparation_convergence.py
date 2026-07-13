from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
qtn = pytest.importorskip("quimb.tensor")

from rand_isopeps.real_isotns.tebd2 import (  # noqa: E402
    imaginary_time_converged,
    tfi_ham,
)


def test_converged_preparation_records_every_sweep_and_gate():
    zero = np.array([1.0, 0.0])
    psi = qtn.PEPS.product_state({(i, j): zero for i in range(2) for j in range(2)})
    ham = tfi_ham(2, 2, g=3.5)
    _, prep = imaginary_time_converged(
        psi, ham, taus=(0.1,), chi=2, eta=2, Ndis=1,
        energy_rtol=1.0, stable_sweeps=1, min_sweeps_per_tau=1,
        max_sweeps_per_tau=2, e_max_bond=8,
    )
    assert prep.steps
    assert prep.converged
    assert prep.final_energy == prep.steps[-1].energy
    assert prep.steps[-1].stable_count >= 1
