"""synthetic tensors for local Moses Move experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .tn_shapes import MosesDims

EnsembleKind = Literal["ring", "noisy_ring", "gaussian"]


def rng_from_seed(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_complex(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return (scale / np.sqrt(2.0)) * z


def random_orthonormal_columns(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> np.ndarray:
    if cols > rows:
        raise ValueError("cols cannot exceed rows")
    if complex_valued:
        a = random_complex((rows, cols), rng)
    else:
        a = rng.standard_normal((rows, cols))
    q, _ = np.linalg.qr(a, mode="reduced")
    return q[:, :cols]


def random_orthonormal_rows(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> np.ndarray:
    if rows > cols:
        raise ValueError("rows cannot exceed cols")
    q = random_orthonormal_columns(cols, rows, rng, complex_valued=complex_valued)
    return q.conj().T


def make_local_ring_exact(
    dims: MosesDims,
    rng: np.random.Generator,
    decay: float = 0.92,
    complex_valued: bool = True,
) -> np.ndarray:
    """Construct a tensor exactly compatible with the two-SVD skeleton.

    The first SVD matrix is Q diag(s) W, where W has orthonormal rows with
    tensor-product row structure. The singular values are ordered in row-major
    (eta, chi) order and factor as decay ** (a * chi + c), so the second
    reshaped matrix has rank at most eta.
    """

    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must lie in (0, 1]")

    q1 = random_orthonormal_columns(dims.n1, dims.k1, rng, complex_valued=complex_valued)
    x = random_orthonormal_rows(dims.eta, dims.n2, rng, complex_valued=complex_valued)
    y = random_orthonormal_rows(dims.chi, dims.n3, rng, complex_valued=complex_valued)

    row_basis = np.einsum("au,cr->acur", x, y, optimize=True).reshape(dims.k1, dims.n2 * dims.n3)
    singular_values = decay ** np.arange(dims.k1)
    v_first = singular_values[:, None] * row_basis
    b_mat = q1 @ v_first
    return b_mat.reshape(dims.tensor_shape)


def add_relative_noise(
    tensor: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma <= 0:
        return tensor.copy()
    noise = random_complex(tensor.shape, rng)
    noise_norm = np.linalg.norm(noise)
    tensor_norm = np.linalg.norm(tensor)
    if noise_norm == 0 or tensor_norm == 0:
        return tensor + sigma * noise
    return tensor + sigma * tensor_norm * noise / noise_norm


def make_local_tensor(
    dims: MosesDims,
    rng: np.random.Generator,
    ensemble: EnsembleKind = "noisy_ring",
    noise: float = 1e-4,
    decay: float = 0.92,
) -> np.ndarray:
    if ensemble == "ring":
        return make_local_ring_exact(dims, rng, decay=decay)
    if ensemble == "noisy_ring":
        base = make_local_ring_exact(dims, rng, decay=decay)
        return add_relative_noise(base, noise, rng)
    if ensemble == "gaussian":
        tensor = random_complex(dims.tensor_shape, rng)
        norm = np.linalg.norm(tensor)
        return tensor / norm if norm else tensor
    raise ValueError(f"unknown ensemble: {ensemble}")


def make_controlled_spectrum_matrix(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    decay: Literal["exp", "power"] = "exp",
    parameter: float = 8.0,
    complex_valued: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a matrix with prescribed singular-value decay."""

    rank = min(rows, cols)
    u = random_orthonormal_columns(rows, rank, rng, complex_valued=complex_valued)
    vh = random_orthonormal_rows(rank, cols, rng, complex_valued=complex_valued)
    j = np.arange(rank, dtype=float)
    if decay == "exp":
        if parameter <= 0:
            raise ValueError("exp decay parameter must be positive")
        s = np.exp(-j / parameter)
    elif decay == "power":
        if parameter <= 0:
            raise ValueError("power decay parameter must be positive")
        s = (j + 1.0) ** (-parameter)
    else:
        raise ValueError(f"unknown decay kind: {decay}")
    return (u * s) @ vh, s
