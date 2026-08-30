"""boundary and dense diagnostics for one shared block-isopeps."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .block_state import BlockPeps, block_gram, dense_block, rotate_block


def block_slice(state: BlockPeps, index: int):
    """return one peps selected from the open block index."""
    if not (0 <= int(index) < state.size):
        raise IndexError("block index is out of range")
    peps = state.peps.copy()
    tag = peps.site_tag_id.format(*state.center)
    tensor = peps[tag]
    selected = tensor.isel({state.block_ind: int(index)})
    tensor.modify(data=selected.data, inds=selected.inds)
    return peps


def block_slices(state: BlockPeps) -> tuple:
    """return every state slice while preserving their shared construction."""
    return tuple(block_slice(state, index) for index in range(state.size))


def _boundary_overlap(ket, bra, *, max_bond: int | None, cutoff: float):
    overlap = ket.make_overlap(bra, layer_tags=("KET", "BRA"))
    return complex(
        overlap.contract_boundary(max_bond=max_bond, cutoff=float(cutoff))
    )


def boundary_block_gram(
    state: BlockPeps,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
) -> np.ndarray:
    """contract the full peps overlaps into the block gram matrix."""
    slices = block_slices(state)
    gram = np.empty((state.size, state.size), dtype=complex)
    for row, bra in enumerate(slices):
        for column, ket in enumerate(slices):
            gram[row, column] = _boundary_overlap(
                ket, bra, max_bond=max_bond, cutoff=cutoff
            )
    return gram


def boundary_projected_hamiltonian(
    state: BlockPeps,
    ham,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    hermitian_rtol: float = 1e-8,
) -> np.ndarray:
    """recover the projected hamiltonian with polarization identities."""
    basis = np.eye(state.size, dtype=complex)

    def quadratic_form(coefficients):
        transform = np.zeros((state.size, state.size), dtype=complex)
        transform[:, 0] = coefficients
        peps = block_slice(rotate_block(state, transform), 0)
        value = complex(peps.compute_local_expectation(
            ham.terms,
            max_bond=max_bond,
            cutoff=float(cutoff),
            normalized=False,
        ))
        probes.append(value)
        return value

    probes = []
    projected = np.zeros((state.size, state.size), dtype=complex)
    diagonal = [quadratic_form(vector) for vector in basis]
    for index, value in enumerate(diagonal):
        projected[index, index] = value.real
    for row in range(state.size):
        for column in range(row + 1, state.size):
            diagonal_sum = diagonal[row].real + diagonal[column].real
            real_probe = quadratic_form(basis[row] + basis[column])
            imaginary_probe = quadratic_form(basis[row] + 1.0j * basis[column])
            value = 0.5 * (
                real_probe.real - diagonal_sum
            ) + 0.5j * (
                diagonal_sum - imaginary_probe.real
            )
            projected[row, column] = value
            projected[column, row] = value.conjugate()
    scale = max((abs(value.real) for value in probes), default=0.0)
    scale = max(scale, np.finfo(float).tiny)
    imaginary_error = max(
        (abs(value.imag) for value in probes), default=0.0
    ) / scale
    if not np.isfinite(imaginary_error) or imaginary_error > hermitian_rtol:
        raise ValueError("polarized hamiltonian expectations are not real")
    return projected


def boundary_block_energies(
    state: BlockPeps,
    ham,
    *,
    max_bond: int = 64,
    cutoff: float = 1e-10,
) -> list[float]:
    """measure one normalized rayleigh quotient per block state."""
    energies = []
    for peps in block_slices(state):
        value = peps.compute_local_expectation(
            ham.terms,
            max_bond=int(max_bond),
            cutoff=float(cutoff),
            normalized=True,
        )
        energies.append(float(np.real(value)))
    return energies


def checked_gram(
    gram: np.ndarray,
    *,
    hermitian_rtol: float = 1e-8,
    eigenvalue_rtol: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """hermitize a gram matrix after checking symmetry and positive rank."""
    matrix = np.asarray(gram, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    hermitian_error = float(np.linalg.norm(matrix - matrix.conj().T) / scale)
    if not np.isfinite(hermitian_error) or hermitian_error > hermitian_rtol:
        raise ValueError("the contracted gram matrix is not hermitian")
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if largest <= 0.0 or smallest <= eigenvalue_rtol * largest:
        raise ValueError("the contracted gram matrix is not positive definite")
    return hermitian, {
        "gram_hermitian_error": hermitian_error,
        "gram_min_eigenvalue": smallest,
        "gram_max_eigenvalue": largest,
        "gram_condition": float(largest / smallest),
    }


def orthonormalize_block_from_gram(
    state: BlockPeps,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    hermitian_rtol: float = 1e-8,
    eigenvalue_rtol: float = 1e-12,
    inplace: bool = False,
) -> tuple[BlockPeps, dict]:
    """orthonormalize the block using an explicitly contracted gram matrix."""
    contracted = boundary_block_gram(state, max_bond=max_bond, cutoff=cutoff)
    gram, metrics = checked_gram(
        contracted,
        hermitian_rtol=hermitian_rtol,
        eigenvalue_rtol=eigenvalue_rtol,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    transform = (
        eigenvectors
        @ np.diag(eigenvalues ** -0.5)
        @ eigenvectors.conj().T
    )
    predicted = transform.conj().T @ gram @ transform
    metrics["gram_identity_error"] = float(
        np.linalg.norm(predicted - np.eye(state.size))
    )
    center = block_gram(state)
    metrics["center_gram_relative_error"] = float(
        np.linalg.norm(center - gram) / max(np.linalg.norm(gram), 1e-300)
    )
    return rotate_block(state, transform, inplace=inplace), metrics


def ritz_rotate_block(
    state: BlockPeps,
    ham,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    hermitian_rtol: float = 1e-8,
    eigenvalue_rtol: float = 1e-12,
    inplace: bool = False,
) -> tuple[BlockPeps, dict]:
    """solve the projected generalized eigenproblem and rotate ``alpha``."""
    contracted = boundary_block_gram(state, max_bond=max_bond, cutoff=cutoff)
    gram, metrics = checked_gram(
        contracted,
        hermitian_rtol=hermitian_rtol,
        eigenvalue_rtol=eigenvalue_rtol,
    )
    projected_raw = boundary_projected_hamiltonian(
        state,
        ham,
        max_bond=max_bond,
        cutoff=cutoff,
        hermitian_rtol=hermitian_rtol,
    )
    projected_scale = max(
        float(np.linalg.norm(projected_raw)), np.finfo(float).tiny
    )
    projected_error = float(
        np.linalg.norm(projected_raw - projected_raw.conj().T) / projected_scale
    )
    if not np.isfinite(projected_error) or projected_error > hermitian_rtol:
        raise ValueError("the projected hamiltonian is not hermitian")
    projected = (projected_raw + projected_raw.conj().T) / 2.0
    energies, rotation = la.eigh(projected, gram)
    predicted = rotation.conj().T @ gram @ rotation
    residual = projected @ rotation - (gram @ rotation) * energies[None, :]
    dual_residual = la.solve(gram, residual, assume_a="pos")
    projected_residual_norms = np.sqrt(np.maximum(
        np.real(np.sum(residual.conj() * dual_residual, axis=0)),
        0.0,
    ))
    metrics.update(
        {
            "projected_hamiltonian_hermitian_error": projected_error,
            "gram_identity_error": float(
                np.linalg.norm(predicted - np.eye(state.size))
            ),
            "energies": [float(value) for value in energies],
            "projected_residual_norms": [
                float(value) for value in projected_residual_norms
            ],
        }
    )
    return rotate_block(state, rotation, inplace=inplace), metrics


def dense_ritz_rotate_block(
    state: BlockPeps, h, *, inplace: bool = False
) -> tuple[BlockPeps, dict]:
    """solve the exact small-system ritz problem and rotate the block index."""
    from .measurements import energy_metrics

    block = dense_block(state)
    gram, _ = checked_gram(block.conj().T @ block)
    h_block = np.asarray(h @ block)
    projected = block.conj().T @ h_block
    projected = (projected + projected.conj().T) / 2.0
    energies, rotation = la.eigh(projected, gram)
    vectors = block @ rotation
    h_vectors = np.asarray(h @ vectors)
    residual = h_vectors - vectors * energies[None, :]
    applied_norm = max(float(np.linalg.norm(h_vectors)), np.finfo(float).tiny)
    metrics = energy_metrics(h, vectors)
    metrics.update({
        "energies": [float(value) for value in energies],
        "subspace_residual_norm": float(np.linalg.norm(residual)),
        "relative_subspace_residual": float(np.linalg.norm(residual) / applied_norm),
    })
    return rotate_block(state, rotation, inplace=inplace), metrics


def dense_subspace_residuals(state: BlockPeps, h) -> dict:
    """compute exact small-system ritz energies and residuals."""
    return dense_ritz_rotate_block(state, h)[1]
