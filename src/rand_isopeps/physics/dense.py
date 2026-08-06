"""Exact small-system physics and Rayleigh diagnostics.

States are dense because that is the oracle regime; Hamiltonians remain sparse.
The same functions score small PEPS after converting them to a vector, which
keeps the reference mathematics out of experiment drivers.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


LAYER_NAMES = (
    "horizontal_even",
    "horizontal_odd",
    "vertical_even",
    "vertical_odd",
)


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Return a normalized vector or column block without changing the input."""
    array = np.asarray(state)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("state must have a finite, nonzero norm")
        return array / norm
    if array.ndim != 2:
        raise ValueError("state must be a vector or a matrix of state columns")
    norms = np.linalg.norm(array, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("every state column must have a finite, nonzero norm")
    return array / norms


def _geometry(ham, lx: int | None, ly: int | None) -> tuple[int, int]:
    if lx is None:
        lx = getattr(ham, "Lx", None)
    if ly is None:
        ly = getattr(ham, "Ly", None)
    if lx is None or ly is None:
        raise ValueError("lx and ly are required when the Hamiltonian does not store them")
    lx, ly = int(lx), int(ly)
    if lx < 1 or ly < 1:
        raise ValueError("lattice dimensions must be positive")
    return lx, ly


def _local_dimension(ham) -> int:
    term = np.asarray(next(iter(ham.terms.values())))
    d = int(round(math.sqrt(term.shape[0])))
    if term.shape != (d * d, d * d):
        raise ValueError("expected fused two-site Hamiltonian terms")
    return d


def _embed_term(term, where, lx: int, ly: int, local_dim: int):
    import quimb as qu

    sites = tuple(where) if isinstance(where[0], tuple) else (where,)
    indices = tuple(int(i * ly + j) for i, j in sites)
    return qu.pkron(
        np.asarray(term),
        [local_dim] * (lx * ly),
        indices,
        sparse=True,
    ).tocsr()


def sparse_hamiltonian(ham, lx: int | None = None, ly: int | None = None):
    """Embed ``LocalHam2D.terms`` into a row-major sparse Hamiltonian."""
    lx, ly = _geometry(ham, lx, ly)
    d = _local_dimension(ham)
    size = d ** (lx * ly)
    total = sp.csr_matrix((size, size), dtype=complex)
    for where, term in ham.terms.items():
        total = total + _embed_term(term, where, lx, ly, d)
    return total.tocsr()


def bond_hamiltonians(ham, lx: int | None = None, ly: int | None = None) -> dict:
    """Return every local term embedded in the full sparse Hilbert space."""
    lx, ly = _geometry(ham, lx, ly)
    d = _local_dimension(ham)
    return {
        tuple(where): _embed_term(term, where, lx, ly, d)
        for where, term in ham.terms.items()
    }


def _layer_name(where) -> str:
    (i0, j0), (i1, j1) = tuple(where)
    if i0 == i1 and abs(j0 - j1) == 1:
        return "horizontal_even" if min(j0, j1) % 2 == 0 else "horizontal_odd"
    if j0 == j1 and abs(i0 - i1) == 1:
        return "vertical_even" if min(i0, i1) % 2 == 0 else "vertical_odd"
    raise ValueError(f"term {where!r} is not a nearest-neighbour square-lattice bond")


def checkerboard_layers(ham, lx: int | None = None, ly: int | None = None) -> dict:
    """Split ``H`` into horizontal/vertical even/odd commuting bond layers."""
    lx, ly = _geometry(ham, lx, ly)
    d = _local_dimension(ham)
    size = d ** (lx * ly)
    matrices = {
        name: sp.csr_matrix((size, size), dtype=complex)
        for name in LAYER_NAMES
    }
    bonds = {name: [] for name in LAYER_NAMES}
    for where, term in ham.terms.items():
        name = _layer_name(where)
        matrices[name] = matrices[name] + _embed_term(term, where, lx, ly, d)
        bonds[name].append(tuple(where))
    return {
        name: {"matrix": matrices[name].tocsr(), "bonds": tuple(bonds[name])}
        for name in LAYER_NAMES
    }


def local_term_norm_bound(ham) -> float:
    """A cheap upper bound ``sum_b ||h_b||_2 >= ||H||_2``."""
    return float(sum(la.norm(np.asarray(term), 2) for term in ham.terms.values()))


def rayleigh_residual(h, state: np.ndarray, *, h_norm_bound: float | None = None) -> dict:
    """Compute the Rayleigh quotient, residual, and variance identity.

    For normalized ``x`` and Hermitian ``H``, ``||Hx-lambda*x||^2`` equals
    ``<H^2>-lambda^2``. The direct residual is primary because the expanded
    expression loses precision close to an eigenstate.
    """
    x = np.asarray(state).reshape(-1)
    norm_sq = float(np.vdot(x, x).real)
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("state must have a finite, nonzero norm")
    hx = np.asarray(h @ x).reshape(-1)
    quotient = np.vdot(x, hx) / norm_sq
    residual = hx - quotient * x
    residual_sq = max(float(np.vdot(residual, residual).real), 0.0)
    h2_expectation = float(np.vdot(hx, hx).real / norm_sq)
    expanded = max(h2_expectation - float(abs(quotient) ** 2), 0.0)
    residual_norm = math.sqrt(residual_sq)
    x_norm = math.sqrt(norm_sq)
    scale = float(h_norm_bound) if h_norm_bound is not None else max(abs(quotient), 1.0)
    relative = residual_norm / max(scale * x_norm, np.finfo(float).tiny)
    return {
        "energy": float(quotient.real),
        "energy_imag": float(quotient.imag),
        "state_norm": x_norm,
        "residual_norm": residual_norm,
        "relative_residual": float(relative),
        "variance": float(residual_sq / norm_sq),
        "variance_expanded": float(expanded),
        "residual_identity_error": float(abs(residual_sq / norm_sq - expanded)),
    }


def rayleigh_ritz(h, states: np.ndarray, *, h_norm_bound: float | None = None) -> dict:
    """Orthonormalize a state block and solve its projected eigenproblem."""
    block = np.asarray(states)
    if block.ndim != 2:
        raise ValueError("states must have shape (hilbert_dimension, n_states)")
    q, r = np.linalg.qr(block, mode="reduced")
    if np.linalg.matrix_rank(r) != block.shape[1]:
        raise ValueError("state block is rank deficient")
    hq = np.asarray(h @ q)
    projected = q.conj().T @ hq
    projected = (projected + projected.conj().T) / 2.0
    energies, rotation = np.linalg.eigh(projected)
    vectors = q @ rotation
    metrics = [
        rayleigh_residual(h, vectors[:, i], h_norm_bound=h_norm_bound)
        for i in range(vectors.shape[1])
    ]
    return {
        "vectors": vectors,
        "energies": np.asarray(energies.real),
        "residual_norms": np.asarray([m["residual_norm"] for m in metrics]),
        "relative_residuals": np.asarray([m["relative_residual"] for m in metrics]),
        "metrics": metrics,
    }


def exact_imaginary_step(h, state: np.ndarray, tau: float) -> np.ndarray:
    """Apply ``exp(-tau H)`` and normalize each state column."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return normalize_state(spla.expm_multiply((-float(tau)) * h, np.asarray(state)))


def trotter_imaginary_step(layers: dict, state: np.ndarray, tau: float) -> np.ndarray:
    """Apply one second-order palindromic Suzuki--Trotter step."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    active = [layers[name]["matrix"] for name in LAYER_NAMES if layers[name]["bonds"]]
    if not active:
        return normalize_state(state)
    schedule = [(matrix, 0.5) for matrix in active[:-1]]
    schedule += [(active[-1], 1.0)]
    schedule += [(matrix, 0.5) for matrix in reversed(active[:-1])]
    out = np.asarray(state)
    for matrix, coefficient in schedule:
        out = spla.expm_multiply((-float(tau) * coefficient) * matrix, out)
    return normalize_state(out)


def _touches_column(where, column: int, direction: int) -> bool:
    first, second = tuple(where)
    a, b = first[1], second[1]
    return a == b == column or {a, b} == {column, column + direction}


def _sweep_bond_order(bonds: dict, ly: int, direction: int):
    columns = range(ly) if direction == 1 else range(ly - 1, -1, -1)
    items = list(bonds.items())
    if direction == -1:
        items.reverse()
    return [
        matrix
        for column in columns
        for where, matrix in items
        if _touches_column(where, column, direction)
    ]


def bond_trotter_imaginary_step(
    bonds: dict,
    state: np.ndarray,
    tau: float,
    *,
    ly: int,
    direction: int = 1,
) -> np.ndarray:
    """Dense oracle for the exact ordered gates used by ``tebd_iteration``."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    schedule = _sweep_bond_order(bonds, ly, direction)
    schedule += _sweep_bond_order(bonds, ly, -direction)
    out = np.asarray(state)
    for matrix in schedule:
        out = spla.expm_multiply((-float(tau) / 2.0) * matrix, out)
    return normalize_state(out)


def dense_state_vector(psi) -> tuple[np.ndarray, float]:
    """Return a normalized PEPS vector and its base-10 log norm safely."""
    exponent = float(getattr(psi, "exponent", 0.0))
    if not np.isfinite(exponent):
        raise ValueError(f"tensor-network exponent is non-finite: {exponent}")
    try:
        psi.exponent = 0.0
        vector = np.asarray(psi.to_dense()).reshape(-1)
    finally:
        psi.exponent = exponent
    if np.any(~np.isfinite(vector)):
        raise ValueError("dense state contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("dense state must have a finite, nonzero norm")
    return vector / norm, exponent + math.log10(norm)
