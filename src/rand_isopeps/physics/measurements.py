"""shared accuracy metrics for dense states and small peps references."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse.linalg as spla

from rand_isopeps.column.diagnostics import spectrum_diagnostics

from .dense import normalize_state, rayleigh_residual


def state_infidelity(first, second) -> float:
    """return one minus the normalized squared overlap of two states."""
    a = normalize_state(np.asarray(first).reshape(-1))
    b = normalize_state(np.asarray(second).reshape(-1))
    overlap = min(float(abs(np.vdot(a, b)) ** 2), 1.0)
    return max(1.0 - overlap, 0.0)


def subspace_metrics(reference, candidate) -> dict[str, float | list[float]]:
    """compare two equal-width state blocks without matching vector labels."""
    a, _ = np.linalg.qr(np.asarray(reference), mode="reduced")
    b, _ = np.linalg.qr(np.asarray(candidate), mode="reduced")
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        raise ValueError("state blocks must have the same shape")
    overlap = a.conj().T @ b
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values.real, 0.0, 1.0)
    angles = np.arccos(singular_values)
    projector_error = np.hypot(
        np.linalg.norm(a - b @ overlap.conj().T, ord="fro"),
        np.linalg.norm(b - a @ overlap, ord="fro"),
    )
    return {
        "max_principal_angle": float(np.max(angles, initial=0.0)),
        "projector_frobenius_error": float(projector_error),
        "principal_angles": [float(value) for value in angles],
    }


def low_energy_references(h, count: int, *, seed: int = 0) -> list[float]:
    """return the lowest exact sparse eigenvalues in deterministic order."""
    if count < 1:
        raise ValueError("count must be positive")
    dimension = int(h.shape[0])
    if dimension <= max(count + 1, 4):
        values = np.linalg.eigvalsh(h.toarray())[:count]
    else:
        if count >= dimension:
            raise ValueError("count must be smaller than the hilbert-space dimension")
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(dimension)
        values = spla.eigsh(
            h,
            k=count,
            which="SA",
            v0=v0,
            return_eigenvectors=False,
        )
    return [float(value) for value in np.sort(np.asarray(values).real)]


@lru_cache(maxsize=None)
def _z_geometry(lx: int, ly: int):
    n_sites = lx * ly
    basis = np.arange(2**n_sites, dtype=np.uint64)
    signs = np.stack([
        1 - 2 * ((basis >> np.uint64(n_sites - site - 1)) & 1).astype(np.int8)
        for site in range(n_sites)
    ])
    pairs = []
    for i in range(lx):
        for j in range(ly):
            if i + 1 < lx:
                pairs.append((i * ly + j, (i + 1) * ly + j))
            if j + 1 < ly:
                pairs.append((i * ly + j, i * ly + j + 1))
    pair_signs = np.stack([signs[a] * signs[b] for a, b in pairs]) if pairs else np.empty(
        (0, basis.size), dtype=np.int8
    )
    signs.setflags(write=False)
    pair_signs.setflags(write=False)
    return signs, pair_signs


def dense_observables(vector, lx: int, ly: int) -> dict[str, float | list[float]]:
    """measure z magnetization, nearest-neighbour zz, and global x parity."""
    state = np.asarray(vector).reshape(-1)
    expected_size = 2 ** (int(lx) * int(ly))
    if state.size != expected_size:
        raise ValueError(f"state has size {state.size}, expected {expected_size}")
    norm_sq = float(np.vdot(state, state).real)
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("state must have a finite, nonzero norm")
    probability = np.abs(state) ** 2 / norm_sq
    signs, pair_signs = _z_geometry(int(lx), int(ly))
    magnetization = signs @ probability
    correlators = pair_signs @ probability
    parity = float((np.vdot(state, state[::-1]) / norm_sq).real)
    return {
        "magnetization_z": [float(value) for value in magnetization],
        "correlator_zz": [float(value) for value in correlators],
        "parity_x": parity,
    }


def state_cut_diagnostics(vector, lx: int, ly: int) -> dict[str, list[float]]:
    """return row and column cut entropies for a small dense state."""
    tensor = normalize_state(np.asarray(vector).reshape(-1)).reshape((2,) * (lx * ly))

    def diagnose(left_axes):
        right_axes = [axis for axis in range(lx * ly) if axis not in left_axes]
        matrix = tensor.transpose(tuple(left_axes) + tuple(right_axes)).reshape(
            2 ** len(left_axes), -1
        )
        return spectrum_diagnostics(np.linalg.svd(matrix, compute_uv=False))

    row = [diagnose(list(range(cut * ly))) for cut in range(1, lx)]
    column = [
        diagnose([i * ly + j for i in range(lx) for j in range(cut)])
        for cut in range(1, ly)
    ]
    return {
        "row_renyi2": [float(entry["renyi2"]) for entry in row],
        "column_renyi2": [float(entry["renyi2"]) for entry in column],
        "row_von_neumann": [float(entry["von_neumann"]) for entry in row],
        "column_von_neumann": [float(entry["von_neumann"]) for entry in column],
    }


def energy_metrics(h, states, *, h_norm_bound: float | None = None) -> dict:
    """measure one state or an orthonormal block with the same record schema."""
    block = np.asarray(states)
    vectors = block[:, None] if block.ndim == 1 else block
    metrics = [
        rayleigh_residual(h, vectors[:, index], h_norm_bound=h_norm_bound)
        for index in range(vectors.shape[1])
    ]
    return {
        "energies": [entry["energy"] for entry in metrics],
        "residual_norms": [entry["residual_norm"] for entry in metrics],
        "relative_residuals": [entry["relative_residual"] for entry in metrics],
        "variances": [entry["variance"] for entry in metrics],
        "residual_identity_errors": [entry["residual_identity_error"] for entry in metrics],
    }
