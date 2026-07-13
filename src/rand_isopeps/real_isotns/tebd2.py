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

from dataclasses import dataclass

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d_tebd import LocalHam2D

from .moses_move import RandSVD, moses_move


def tfi_ham(Lx: int, Ly: int, j: float = 1.0, g: float = 3.0) -> LocalHam2D:
    """2D transverse-field Ising: H = -j sum_<ab> Z_a Z_b - g sum_a X_a."""
    return LocalHam2D(Lx, Ly, H2=-j * (qu.pauli("Z") & qu.pauli("Z")), H1=-g * qu.pauli("X"))


def heis_ham(Lx: int, Ly: int, j: float = 1.0) -> LocalHam2D:
    """2D antiferromagnetic Heisenberg (Pauli convention, eq. 35 of Dektor et al.):
    H = j sum_<ab> (X_a X_b + Y_a Y_b + Z_a Z_b)."""
    return LocalHam2D(Lx, Ly, H2=j * ((qu.pauli("X") & qu.pauli("X"))
                                      + (qu.pauli("Y") & qu.pauli("Y"))
                                      + (qu.pauli("Z") & qu.pauli("Z"))))


def xxz_ham(Lx: int, Ly: int, delta: float = 1.0, j: float = 1.0) -> LocalHam2D:
    """2D XXZ: H = j sum_<ab> (X_a X_b + Y_a Y_b + delta Z_a Z_b). delta=1 is Heisenberg;
    delta interpolates the entanglement 'hardness' dial (XY-like -> Ising-like)."""
    return LocalHam2D(Lx, Ly, H2=j * ((qu.pauli("X") & qu.pauli("X"))
                                      + (qu.pauli("Y") & qu.pauli("Y"))
                                      + delta * (qu.pauli("Z") & qu.pauli("Z"))))


def compass_ham(Lx: int, Ly: int, j: float = 1.0) -> LocalHam2D:
    """Square-lattice quantum compass model (the Kitaev-honeycomb analog on our lattice):
    -j X_a X_b on horizontal bonds, -j Y_a Y_b on vertical bonds. Frustrated by bond-
    direction-dependent couplings -- the 'hard' end of the survey ladder."""
    XX, YY = -j * (qu.pauli("X") & qu.pauli("X")), -j * (qu.pauli("Y") & qu.pauli("Y"))
    H2 = {}
    for i in range(Lx):
        for k in range(Ly):
            if k + 1 < Ly:
                H2[((i, k), (i, k + 1))] = XX      # horizontal (along a row)
            if i + 1 < Lx:
                H2[((i, k), (i + 1, k))] = YY      # vertical (along a column)
    return LocalHam2D(Lx, Ly, H2=H2)


def ham_from_spec(spec: str, Lx: int, Ly: int) -> LocalHam2D | None:
    """Build the Hamiltonian named by an experiment state spec; ``None`` for 'random'.

    Grammar (the ``--states`` vocabulary shared by the column_sketch experiments):
      ``random``    -- no Hamiltonian (skip imaginary time; the random isoTNS itself)
      ``tfim@G``    -- transverse-field Ising at field g=G (g_c ~ 3.044)
      ``heis``      -- antiferromagnetic Heisenberg
      ``xxz@D``     -- XXZ at anisotropy delta=D
      ``compass``   -- square-lattice compass (Kitaev analog)
    """
    if spec == "random":
        return None
    kind, _, arg = spec.partition("@")
    if kind == "tfim":
        return tfi_ham(Lx, Ly, g=float(arg))
    if kind == "heis":
        return heis_ham(Lx, Ly)
    if kind == "xxz":
        return xxz_ham(Lx, Ly, delta=float(arg))
    if kind == "compass":
        return compass_ham(Lx, Ly)
    raise ValueError(f"unknown state spec {spec!r} (want random | tfim@G | heis | xxz@D | compass)")


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


@dataclass
class PreparationStep:
    tau: float
    stage: int
    sweep: int
    energy: float
    relative_energy_change: float
    stable_count: int


@dataclass
class PreparationResult:
    steps: list[PreparationStep]
    converged: bool
    final_energy: float
    final_relative_energy_change: float


def imaginary_time_converged(
    psi,
    ham,
    *,
    taus=(0.3, 0.1, 0.03, 0.01),
    chi,
    eta,
    cutoff=1e-10,
    Ndis=10,
    energy_rtol=1e-6,
    stable_sweeps=3,
    min_sweeps_per_tau=3,
    max_sweeps_per_tau=40,
    rand=None,
    e_max_bond=32,
):
    """Imaginary-time preparation with a common, recorded convergence gate.

    Every sweep records its energy.  Each ``tau`` stage continues until the
    relative energy change is below ``energy_rtol`` for ``stable_sweeps``
    consecutive sweeps (after ``min_sweeps_per_tau``), or until the stage cap is
    reached.  The returned ``converged`` flag is true only if the *final* stage
    satisfies that same rule; callers must reject or separately label false
    preparations rather than silently mixing them with converged states.
    """
    if stable_sweeps < 1 or min_sweeps_per_tau < 1 or max_sweeps_per_tau < 1:
        raise ValueError("sweep counts must be positive")
    if min_sweeps_per_tau > max_sweeps_per_tau:
        raise ValueError("min_sweeps_per_tau cannot exceed max_sweeps_per_tau")
    if energy_rtol <= 0:
        raise ValueError("energy_rtol must be positive")

    records: list[PreparationStep] = []
    direction = 1
    previous = energy(psi, ham, e_max_bond)
    final_stage_converged = False
    for stage, tau in enumerate(tuple(float(t) for t in taus)):
        gates = _gates(ham, tau)
        stable = 0
        stage_converged = False
        for sweep in range(int(max_sweeps_per_tau)):
            _sweep(psi, gates, chi, eta, cutoff, Ndis, direction, rand)
            direction = -direction
            psi.equalize_norms_(1.0)
            current = energy(psi, ham, e_max_bond)
            rel = abs(current - previous) / max(abs(current), abs(previous), 1e-15)
            stable = stable + 1 if rel <= energy_rtol else 0
            records.append(PreparationStep(
                tau=tau, stage=stage, sweep=sweep, energy=float(current),
                relative_energy_change=float(rel), stable_count=stable,
            ))
            previous = current
            if sweep + 1 >= min_sweeps_per_tau and stable >= stable_sweeps:
                stage_converged = True
                break
        final_stage_converged = stage_converged

    final_rel = records[-1].relative_energy_change if records else float("inf")
    return psi, PreparationResult(
        steps=records,
        converged=bool(final_stage_converged),
        final_energy=float(previous),
        final_relative_energy_change=float(final_rel),
    )
