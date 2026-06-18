"""isoTNS imaginary-time TEBD^2 on a quimb isoPEPS (single-state, p=1).

A minimal ground-state solver built on the Moses Move of ``rand_isopeps.isotns``.
Following Zaletel-Pollmann, TEBD^2 is 1D TEBD applied along the orthogonality
column plus a Moses Move that shifts the orthogonality center to the next column,
swept across the lattice. The orthogonality center is the only place a local SVD
truncation is variationally meaningful (the isometric environment contracts to
identity), so gates are applied there and the Moses Move re-canonicalizes as the
center advances.

We own the imaginary-time loop and the center-sweeping schedule; quimb provides
the tensor primitives: ``LocalHam2D`` (2D Hamiltonian, single-site field folded
into the bond terms), ``PEPS.gate_`` (the local gate + truncating split), and
``compute_local_expectation`` (energy). Passing ``rand=RandSVD(...)`` randomizes
every SVD in the Moses-Move re-canonicalization, so the randomized linear algebra
is exercised inside a real ground-state calculation.

Minimal by design: 1st/2nd-order Trotter via alternating left/right sweeps;
single state (no block index). Verified to lower the energy toward the exact
(ED) ground state of the 2D transverse-field Ising model on small lattices.
"""

from __future__ import annotations

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d_tebd import LocalHam2D

from .moses_move import RandSVD, moses_move


def tfi_ham(Lx: int, Ly: int, j: float = 1.0, g: float = 3.0) -> LocalHam2D:
    """2D transverse-field Ising: H = -j sum_<ab> Z_a Z_b - g sum_a X_a."""
    return LocalHam2D(Lx, Ly, H2=-j * (qu.pauli("Z") & qu.pauli("Z")), H1=-g * qu.pauli("X"))


def _gates(ham: LocalHam2D, tau: float) -> dict:
    """imaginary-time two-site gates exp(-tau * H_bond), reshaped for gate_."""
    d = ham.terms[next(iter(ham.terms))].shape[0]
    dd = int(round(d ** 0.5))
    return {tuple(where): qu.expm(-tau * H).reshape(dd, dd, dd, dd) for where, H in ham.terms.items()}


def _col(site):
    return site[1]


def _touches_center(where, j: int, direction: int) -> bool:
    """Gate (a,b) is applied when the orthogonality center sits at column j:
    a vertical bond inside column j, or the horizontal bond to column j+direction."""
    ca, cb = _col(where[0]), _col(where[1])
    if ca == cb == j:
        return True
    return {ca, cb} == {j, j + direction}


def _sweep(psi, gates, chi, eta, cutoff, Ndis, direction, rand):
    """One TEBD^2 sweep: move the orthogonality center across the lattice in
    ``direction`` (+1 right, -1 left), gating at the center then Moses-moving."""
    cols = range(psi.Ly) if direction == 1 else range(psi.Ly - 1, -1, -1)
    sweep, split = ("up", "right") if direction == 1 else ("up", "left")
    for j in cols:
        for where, G in gates.items():
            if _touches_center(where, j, direction):
                psi.gate_(G, where=where, contract="reduce-split", max_bond=eta, cutoff=cutoff)
        if 0 <= j + direction < psi.Ly:
            moses_move(psi, j, chi, eta, cutoff, Ndis, orientation="col",
                       sweep=sweep, split=split, renorm=True, rand=rand)


def energy(psi, ham: LocalHam2D, max_bond: int = 32) -> float:
    """Energy <psi|H|psi>/<psi|psi> via a boundary-contraction of the 2D network."""
    return float(psi.compute_local_expectation(ham.terms, normalized=True, max_bond=max_bond).real)


def imaginary_time(psi, ham, taus, steps, chi, eta, cutoff=1e-10, Ndis=10,
                   rand=None, e_max_bond=32, record_every=1):
    """Imaginary-time TEBD^2 ground-state search (modifies ``psi`` in place).

    ``taus`` is a single step size or a list of (tau, n_steps) stages (anneal the
    step down for accuracy). Alternates right/left sweeps (2nd-order Trotter).
    Returns ``(psi, energies)`` where ``energies`` is the recorded <H> trajectory.
    """
    schedule = [(taus, steps)] if isinstance(taus, (int, float)) else list(taus)
    energies, direction, k = [], 1, 0
    for tau, nstp in schedule:
        gates = _gates(ham, tau)
        for _ in range(nstp):
            _sweep(psi, gates, chi, eta, cutoff, Ndis, direction, rand)
            direction = -direction
            psi.equalize_norms_(1.0)
            if k % record_every == 0:
                energies.append(energy(psi, ham, e_max_bond))
            k += 1
    energies.append(energy(psi, ham, e_max_bond))
    return psi, energies
