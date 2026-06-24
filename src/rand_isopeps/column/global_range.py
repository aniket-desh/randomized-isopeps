"""Global randomized column range finder with rMPS probes (sketched column QR).

This is the global alternative to the sequential local Moses move. Treating the
whole active column as one matrix ``C`` (absorbed legs -> retained legs), it
approximates the column QR ``C ~ Q (Q^* C)`` with the Halko--Martinsson--Tropp
randomized range finder, using random matrix product state (rMPS) probes
(:mod:`rand_isopeps.linalg.rmps_sketch`) in place of a dense Gaussian test matrix.

Algorithm (the paper's randomized SVD, Thm 5.8):

    Omega  in F^{n x ell}     rMPS / Gaussian test matrix over the absorbed legs
    Y = C Omega               sketch (ell matrix-MPS products in the access model)
    Q = orth(Y)               orthonormal range basis (the new isometric column)
    Chat = Q (Q^* C)          rank-ell projection approximation

with optional power iterations ``Y <- C (C^* Y)`` to sharpen a flat spectrum.

What it reports (the mathematical-validation diagnostics from the briefing):

* ``rel_error``           -- ``||C - Q Q^* C||_F / ||C||_F`` (the QB / range error).
* ``rank_r_approx_error`` -- error of the rank-``r`` *truncation* of the
  randomized SVD; comparable to the Eckart--Young optimum.
* ``rank_r_floor``        -- ``||C - [C]_r||_F / ||C||_F`` (Eckart--Young optimal).
* ``excess_error``        -- ``rank_r_approx_error - rank_r_floor`` (``>= 0``).
* ``isometry_defect``     -- ``||Q^* Q - I||_F`` (is the new column isometric?).
* ``osi_sigma_min``       -- ``sigma_min(V_r^* Omega)^2``, the **oblivious subspace
  injection** diagnostic (paper sec 5.2): the injectivity of the probe on the
  top-``r`` right singular subspace. ``>~ 1/2`` means a good embedding; it
  collapses toward 0 for the ``chi_sk = 1`` Kronecker limit on tall columns
  ("overwhelming orthogonality").

The primary scientific question (briefing sec 8) is whether the randomized rMPS
range matches a dense Gaussian range and how the ``chi_sk = 1`` Kronecker baseline
fails -- a mathematical validation, **not** an end-to-end speedup claim. The
implementation-free cost comparison vs the local move lives in the experiments
(and follows the standing cost-model rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la

from rand_isopeps.linalg.randomized_svd import (
    isometry_defect_columns,
    orthonormalize,
    relative_frobenius_error,
)
from rand_isopeps.linalg.rmps_sketch import rmps_columns, rmps_test_matrix

SketchKind = "gaussian | rmps | kron"


@dataclass
class GlobalRangeResult:
    sketch: str          # "gaussian" | "rmps" | "kron"
    chi_sk: int          # sketch bond used (0 for gaussian, 1 for kron)
    ell: int             # embedding dimension (probe count)
    target_rank: int     # r, the rank the rank-r quantities reference
    rel_error: float          # ||C - Q Q^* C||_F / ||C||_F
    rank_r_approx_error: float  # ||C - Ahat_r||_F / ||C||_F
    rank_r_floor: float       # ||C - [C]_r||_F / ||C||_F (Eckart-Young)
    excess_error: float       # rank_r_approx_error - rank_r_floor (>= 0)
    isometry_defect: float    # ||Q^* Q - I||_F
    osi_sigma_min: float      # sigma_min(V_r^* Omega)^2 (subspace injection)
    runtime_s: float
    q_cols: int               # number of orthonormal range columns

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "sketch": self.sketch,
            "chi_sk": self.chi_sk,
            "ell": self.ell,
            "target_rank": self.target_rank,
            "rel_error": self.rel_error,
            "rank_r_approx_error": self.rank_r_approx_error,
            "rank_r_floor": self.rank_r_floor,
            "excess_error": self.excess_error,
            "isometry_defect": self.isometry_defect,
            "osi_sigma_min": self.osi_sigma_min,
            "runtime_s": self.runtime_s,
            "q_cols": self.q_cols,
        }


def reference_svd(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact SVD of ``C`` to share one exact reference across sketch kinds.

    Pass the result as ``ref_svd`` to :func:`global_column_range` so every sketch
    is scored against the *same* Eckart--Young floor and the same top-``r`` right
    singular subspace (and the SVD is computed once per tensor, not once per sketch).
    """
    return la.svd(c, full_matrices=False, check_finite=False, lapack_driver="gesdd")


def _gaussian_test_matrix(n: int, ell: int, rng: np.random.Generator, complex_valued: bool) -> np.ndarray:
    """Dense Gaussian test matrix ``(n, ell)``, scaled ``1/sqrt(ell)`` to match the
    rMPS normalization so the OSI diagnostic is comparable across sketch kinds."""
    if complex_valued:
        omega = (rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))) / np.sqrt(2.0)
    else:
        omega = rng.standard_normal((n, ell))
    return omega / np.sqrt(ell)


def global_column_range(
    c: np.ndarray,
    factor_dims: tuple[int, ...],
    ell: int,
    chi_sk: int = 1,
    sketch_kind: str = "rmps",
    target_rank: int | None = None,
    n_power: int = 0,
    rng: np.random.Generator | None = None,
    ref_svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> GlobalRangeResult:
    """Sketched column QR of the dense column matrix ``c`` with rMPS / Gaussian probes.

    ``factor_dims`` is the factorization of the domain (absorbed legs); its product
    must equal ``c.shape[1]``. ``ell`` is the embedding dimension; ``target_rank``
    (default ``ell``) sets the rank the Eckart--Young / OSI diagnostics reference.
    ``sketch_kind``: ``"gaussian"`` (dense gold standard), ``"rmps"`` (rMPS with
    ``chi_sk``), or ``"kron"`` (rMPS with ``chi_sk = 1`` -- the Khatri--Rao limit).
    """
    gen = np.random.default_rng() if rng is None else rng
    m, n = c.shape
    if int(np.prod(factor_dims)) != n:
        raise ValueError(f"factor_dims {factor_dims} must multiply to c.shape[1]={n}")
    ell = max(1, min(ell, n))
    r = ell if target_rank is None else target_rank
    r = max(1, min(r, ell, m, n))
    complex_valued = np.iscomplexobj(c)

    u, s, vh = reference_svd(c) if ref_svd is None else ref_svd
    cnorm = float(np.linalg.norm(s))

    if sketch_kind == "gaussian":
        omega = _gaussian_test_matrix(n, ell, gen, complex_valued)
        used_chi = 0
    elif sketch_kind in ("rmps", "kron"):
        used_chi = 1 if sketch_kind == "kron" else int(chi_sk)
        omega = rmps_test_matrix(
            factor_dims, ell, used_chi, gen, normalize=True, complex_valued=complex_valued
        )
    else:
        raise ValueError(f"unknown sketch_kind: {sketch_kind!r}")
    omega = omega.astype(c.dtype, copy=False)

    t0 = perf_counter()
    y = c @ omega
    for _ in range(max(0, n_power)):
        y = c @ (c.conj().T @ y)
    q = orthonormalize(y)
    b = q.conj().T @ c
    runtime = perf_counter() - t0

    # rank-ell projection (QB) error
    rel_error = relative_frobenius_error(c, q @ b)

    # rank-r truncation of the randomized SVD -> compare to the Eckart-Young floor
    ub, sb, vhb = la.svd(b, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    rr = min(r, sb.shape[0])
    approx_r = (q @ (ub[:, :rr] * sb[:rr])) @ vhb[:rr, :]
    rank_r_approx_error = relative_frobenius_error(c, approx_r)
    tail_r = float(np.sqrt(np.sum(s[rr:] ** 2)))
    rank_r_floor = tail_r / cnorm if cnorm else 0.0
    excess_error = rank_r_approx_error - rank_r_floor

    iso = isometry_defect_columns(q)

    # OSI: sigma_min(V_r^* Omega)^2 = injectivity of the probe on the top-r subspace.
    # V_r^* = Vh[:rr] (the conjugate-transpose of the top-r right singular vectors).
    proj_omega = vh[:rr, :] @ omega  # (rr, ell)
    sv = la.svdvals(proj_omega)
    osi = float(sv[-1] ** 2) if sv.size else 0.0

    return GlobalRangeResult(
        sketch=sketch_kind,
        chi_sk=used_chi,
        ell=ell,
        target_rank=rr,
        rel_error=rel_error,
        rank_r_approx_error=rank_r_approx_error,
        rank_r_floor=rank_r_floor,
        excess_error=excess_error,
        isometry_defect=iso,
        osi_sigma_min=osi,
        runtime_s=runtime,
        q_cols=int(q.shape[1]),
    )


def sampled_bond_growth(
    column_operator,
    ell: int,
    chi_sk: int,
    rng: np.random.Generator,
    complex_valued: bool = True,
) -> dict[str, float]:
    """Matrix-free probe of the sampled range's bond growth (Phase-3 diagnostic).

    Applies ``ell`` rMPS probes to the column operator via the matrix-free
    MPO--MPS product and reports the bond dimension of the resulting ``C omega``
    MPS columns -- the quantity that decides whether the global sampled range
    ``range(C Omega)`` admits a low-bond isometric column (the decisive question of
    briefing sec 8). The probe bond is ``chi_sk``, the operator bond ``D``; each
    product MPS has bond ``<= D * chi_sk`` before any rounding.
    """
    probes = rmps_columns(column_operator.input_dims, ell, chi_sk, rng, complex_valued)
    bonds = [column_operator.product_bond(p) for p in probes]
    return {
        "probe_bond": int(chi_sk),
        "operator_bond": int(column_operator.mpo_bond),
        "max_product_bond": int(max(bonds)),
        "mean_product_bond": float(np.mean(bonds)),
        "bound_d_times_chi": int(column_operator.mpo_bond * chi_sk),
    }
