"""Matrix-free spectrum diagnostics for a whole-column MPO."""

from __future__ import annotations

import json

import numpy as np
import scipy.linalg as la

from rand_isopeps.column.operator import ColumnOperator


def spectrum_diagnostics(singular: np.ndarray, fraction: float = 0.99) -> dict[str, float | int]:
    """Return normalized rank and entropy summaries for singular values."""
    singular = np.asarray(singular, dtype=float)
    weight = singular ** 2
    total = float(np.sum(weight))
    if total <= 0.0:
        return {"rank": 0, "r99": 0, "renyi2": 0.0, "von_neumann": 0.0}
    probability = weight / total
    positive = probability > 0.0
    threshold = float(fraction) * total
    r_fraction = int(np.searchsorted(np.cumsum(weight), threshold, side="left") + 1)
    numerical_tol = np.finfo(float).eps * max(singular.size, 1) * singular[0]
    return {
        "rank": int(np.count_nonzero(singular > numerical_tol)),
        "r99": r_fraction,
        "renyi2": float(-np.log(max(np.sum(probability ** 2), 1e-300))),
        "von_neumann": float(-np.sum(probability[positive] * np.log(probability[positive]))),
    }


def _right_canonicalize(cores: list[np.ndarray]) -> list[np.ndarray]:
    work = [np.array(core, copy=True) for core in cores]
    for site in range(len(work) - 1, 0, -1):
        left, physical, right = work[site].shape
        matrix = work[site].reshape(left, physical * right)
        u, singular, vh = la.svd(
            matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd"
        )
        rank = singular.size
        work[site] = vh.reshape(rank, physical, right)
        work[site - 1] = np.tensordot(
            work[site - 1], u * singular, axes=(2, 0)
        )
    return work


def operator_cut_spectra(column: ColumnOperator) -> list[np.ndarray]:
    """Schmidt spectra of vectorized ``C`` across every vertical MPO cut.

    The MPO cores are treated as an MPS with local dimension ``dout * din``.
    Right canonicalization followed by an exact left sweep produces
    representation-independent operator-entanglement spectra without ever
    materializing the exponentially large column matrix.
    """
    cores = [
        core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
        for core in column.cores
    ]
    work = _right_canonicalize(cores)
    spectra: list[np.ndarray] = []
    for site in range(len(work) - 1):
        left, physical, right = work[site].shape
        matrix = work[site].reshape(left * physical, right)
        u, singular, vh = la.svd(
            matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd"
        )
        spectra.append(singular)
        rank = singular.size
        work[site] = u.reshape(left, physical, rank)
        work[site + 1] = np.tensordot(
            singular[:, None] * vh, work[site + 1], axes=(1, 0)
        )
    return spectra


def _relative_tail(singular: np.ndarray, rank: int) -> float:
    weight = np.asarray(singular, dtype=float) ** 2
    return float(np.sqrt(np.sum(weight[int(rank):]) / max(np.sum(weight), 1e-300)))


def column_diagnostics(
    column: ColumnOperator,
    *,
    eta: int,
    kappa: int,
    dense_max_elements: int = 2_000_000,
    operator_spectra_cache: list[np.ndarray] | None = None,
    flat_singular_values: np.ndarray | None = None,
) -> dict[str, object]:
    """Predictors used to explain factorization difficulty.

    Operator-cut quantities are always matrix-free.  Flat matrix singular values
    are included only under the explicit dense oracle limit and otherwise set to
    ``NaN`` rather than triggering a hidden materialization.
    """
    spectra = (
        operator_cut_spectra(column)
        if operator_spectra_cache is None
        else operator_spectra_cache
    )
    if len(spectra) != max(column.lx - 1, 0):
        raise ValueError("operator spectra cache has incompatible length")
    cut_stats = [spectrum_diagnostics(s) for s in spectra]
    cut_r99 = [int(x["r99"]) for x in cut_stats]
    cut_renyi2 = [float(x["renyi2"]) for x in cut_stats]
    cut_vn = [float(x["von_neumann"]) for x in cut_stats]
    cut_tail_eta = [_relative_tail(s, eta) for s in spectra]
    cut_tail_composite = [_relative_tail(s, int(eta) * int(kappa)) for s in spectra]

    flat = {"rank": float("nan"), "r99": float("nan"),
            "renyi2": float("nan"), "von_neumann": float("nan")}
    if column.n_out * column.n_in <= int(dense_max_elements):
        singular = (
            la.svdvals(column.materialize(), check_finite=False)
            if flat_singular_values is None
            else np.asarray(flat_singular_values, dtype=float)
        )
        if singular.ndim != 1 or singular.size != min(column.n_out, column.n_in):
            raise ValueError("flat singular-value cache has incompatible shape")
        flat = spectrum_diagnostics(singular)

    return {
        "column_n_in": column.n_in,
        "column_n_out": column.n_out,
        "column_mpo_bond": column.mpo_bond,
        "column_flat_rank": flat["rank"],
        "column_flat_r99": flat["r99"],
        "column_flat_renyi2": flat["renyi2"],
        "column_flat_von_neumann": flat["von_neumann"],
        "operator_cut_max_r99": max(cut_r99, default=1),
        "operator_cut_max_renyi2": max(cut_renyi2, default=0.0),
        "operator_cut_r99": json.dumps(cut_r99),
        "operator_cut_renyi2": json.dumps(cut_renyi2),
        "operator_cut_von_neumann": json.dumps(cut_vn),
        "operator_cut_tail_eta": json.dumps(cut_tail_eta),
        "operator_cut_tail_eta_kappa": json.dumps(cut_tail_composite),
    }
