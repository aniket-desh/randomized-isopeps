"""Random matrix product state (rMPS) sketches for tensor-network range finding.

Implements the random MPS test vectors / test matrices of Camano, Epperly, Meyer
& Tropp, *Linear algebra at exponential scale via tensor network dimension
reduction* (``docs/rmps.pdf``), Definition 1.1.

An rMPS vector ``omega`` in ``F^{d_1 * ... * d_t}`` is an MPS whose cores have iid
Gaussian entries. With per-core variances

    sigma_p^2 = 1 / chi_sk          (interior cores 1 < p < t)
    sigma_p^2 = 1 / sqrt(chi_sk)    (boundary cores p = 1, t)

the rMPS is *isotropic*: ``E[omega omega^*] = I`` (Eq. 1.4), so it can stand in
for a dense Gaussian probe in randomized linear algebra. The *sketch bond
dimension* ``chi_sk`` (the paper's ``chi``; renamed here to avoid colliding with
the isoTNS horizontal bond ``chi``) is the single reliability knob: the paper's
thesis is that rMPS probes behave like Gaussian probes **iff ``chi_sk >~ t``**
(the tensor order / number of factors). The degenerate case ``chi_sk = 1`` makes
every bond trivial, so ``omega = g_1 ⊗ g_2 ⊗ ... ⊗ g_t`` is a Gaussian-Kronecker
(a.k.a. Khatri--Rao) vector, which fails by *overwhelming orthogonality*
``|<x, omega>|^2 <= C^{-t} ||x||^2`` on tall factor counts (secs 2.1, 5.3). This
is the regime the warning is about.

These probes are the tensor-network-native replacement for a dense Gaussian test
matrix in the global column range finder (:mod:`rand_isopeps.column.global_range`):
the probe lives on the *absorbed* (input) legs of a column operator, and the
matrix--MPS product ``C @ omega`` is an MPO--MPS contraction (the paper's access
model, sec 3.1). This module builds both the dense materialization
(:func:`rmps_test_matrix`, for the tiny materialized validation) and the MPS-core
form (:func:`rmps_columns`, for the matrix-free access model).

Standalone and numpy-only, mirroring :mod:`rand_isopeps.linalg.sketches`. The
dense form is also wired into ``SketchSpec(kind="rmps")`` there so it composes
with ``rsvd_truncate`` and the cost model; this module is the canonical builder.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rmps_cores",
    "rmps_to_vector",
    "rmps_vector",
    "rmps_columns",
    "rmps_test_matrix",
    "kron_test_matrix",
]


def _is_complex(dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def _gaussian(shape: tuple[int, ...], rng: np.random.Generator, std: float, complex_valued: bool) -> np.ndarray:
    """Gaussian array with per-element standard deviation ``std`` (``E|g|^2 = std^2``)."""
    if complex_valued:
        z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return (std / np.sqrt(2.0)) * z
    return std * rng.standard_normal(shape)


def rmps_cores(
    factor_dims: tuple[int, ...],
    chi_sk: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> list[np.ndarray]:
    """One rMPS as a list of MPS cores ``[(1,d_1,chi), (chi,d_2,chi), ..., (chi,d_t,1)]``.

    Boundary cores get element std ``chi_sk**-0.25`` (variance ``1/sqrt(chi_sk)``)
    and interior cores std ``chi_sk**-0.5`` (variance ``1/chi_sk``), which makes
    ``E[omega omega^*] = I`` exactly for any ``t >= 2`` (the per-site bond chains
    lock together and the variance product ``chi_sk^{-(t-1)}`` cancels the
    ``chi_sk^{t-1}`` chain sum). ``chi_sk = 1`` gives a Gaussian-Kronecker vector
    (all bonds 1). A single factor (``t = 1``) is just a unit-variance Gaussian.
    """
    dims = tuple(int(d) for d in factor_dims)
    t = len(dims)
    if t == 0:
        raise ValueError("factor_dims must be non-empty")
    if any(d <= 0 for d in dims):
        raise ValueError(f"factor_dims must be positive, got {dims}")
    chi = max(1, int(chi_sk))
    if t == 1:
        return [_gaussian((1, dims[0], 1), rng, 1.0, complex_valued)]

    boundary_std = chi ** -0.25
    interior_std = chi ** -0.5
    cores: list[np.ndarray] = []
    for i, d in enumerate(dims):
        left = 1 if i == 0 else chi
        right = 1 if i == t - 1 else chi
        std = boundary_std if (i == 0 or i == t - 1) else interior_std
        cores.append(_gaussian((left, d, right), rng, std, complex_valued))
    return cores


def rmps_to_vector(cores: list[np.ndarray]) -> np.ndarray:
    """Contract rMPS cores ``[(l,d,r), ...]`` into a dense vector.

    Row-major over sites: index ``= i_1 (d_2..d_t) + ... + i_t``. This matches the
    flattening of :func:`rand_isopeps.column.operator.ColumnOperator.materialize`
    and ``mpo_mps_absorb.mps_to_vector`` so dense ``C @ Omega`` is consistent.
    """
    state = cores[0][0]  # (d_1, r)
    for core in cores[1:]:
        state = np.tensordot(state, core, axes=(-1, 0))
        state = state.reshape(-1, state.shape[-1])
    return state[:, 0]


def rmps_vector(
    factor_dims: tuple[int, ...],
    chi_sk: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> np.ndarray:
    """A single dense rMPS probe vector in ``F^{prod(factor_dims)}``."""
    return rmps_to_vector(rmps_cores(factor_dims, chi_sk, rng, complex_valued))


def rmps_columns(
    factor_dims: tuple[int, ...],
    ell: int,
    chi_sk: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> list[list[np.ndarray]]:
    """``ell`` independent rMPS probes, each as a list of cores (matrix-free form).

    This is the access-model form: each probe is fed to a column operator via an
    MPO--MPS product without ever materializing the ``prod(factor_dims)``-length
    dense vector. Used by the matrix-free path of the global range finder.
    """
    if ell <= 0:
        raise ValueError("ell must be positive")
    return [rmps_cores(factor_dims, chi_sk, rng, complex_valued) for _ in range(ell)]


def rmps_test_matrix(
    factor_dims: tuple[int, ...],
    ell: int,
    chi_sk: int,
    rng: np.random.Generator,
    normalize: bool = True,
    complex_valued: bool = True,
) -> np.ndarray:
    """Dense rMPS test matrix ``Omega in F^{N x ell}``, ``N = prod(factor_dims)``.

    Columns are iid rMPS vectors with sketch bond ``chi_sk``. With
    ``normalize=True`` the matrix is scaled by ``1/sqrt(ell)`` so its columns form
    an isotropic test matrix in the sense of the paper's Eq. (5.1):
    ``E[Omega Omega^*] = I`` and ``E||Omega^* x||^2 = ||x||^2``. The scaling is
    irrelevant to a range finder (it only uses ``range(C Omega)`` via a QR) but
    matters for the standalone OSI diagnostic ``sigma_min(V_r^* Omega)^2``.
    """
    if ell <= 0:
        raise ValueError("ell must be positive")
    cols = [rmps_vector(factor_dims, chi_sk, rng, complex_valued) for _ in range(ell)]
    omega = np.stack(cols, axis=1)
    if normalize:
        omega = omega / np.sqrt(ell)
    return omega


def kron_test_matrix(
    factor_dims: tuple[int, ...],
    ell: int,
    rng: np.random.Generator,
    normalize: bool = True,
    complex_valued: bool = True,
) -> np.ndarray:
    """Gaussian-Kronecker / Khatri--Rao test matrix == rMPS with ``chi_sk = 1``.

    Convenience alias for the degenerate ``chi_sk = 1`` baseline that the paper
    warns about (overwhelming orthogonality). Kept as a named entry point so
    experiments can label it explicitly rather than passing a magic ``chi_sk``.
    """
    return rmps_test_matrix(factor_dims, ell, 1, rng, normalize, complex_valued)
