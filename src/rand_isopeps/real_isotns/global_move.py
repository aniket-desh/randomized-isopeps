"""Whole-column rMPS Moses move used by the physics loop."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.sketching.column import factorization_metrics, factorize_column

from .column_bridge import (
    compress_column,
    extract_column,
    insert_column_factorization,
    validate_peps_structure,
)


def rmps_column_move(
    psi,
    j: int,
    *,
    split: str,
    ell: int,
    eta: int,
    kappa: int,
    chi_sk: int,
    ndis: int = 0,
    absorption_max_bond: int | None = None,
    absorption_cutoff: float = 0.0,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
):
    """Sketch, factor, insert, and zip one orthogonality column.

    The result is again a standard one-tensor-per-site PEPS, so the operation is
    usable at interior columns and across repeated sweeps.  The sketch loss and
    absorption controls are reported separately.
    """
    started = perf_counter()
    column = extract_column(psi, j, split=split)
    q_cores, residual_cores, result = factorize_column(
        column,
        ell=ell,
        eta=eta,
        kappa=kappa,
        chi_sk=chi_sk,
        ndis=ndis,
        rng=rng,
    )
    moved = insert_column_factorization(
        psi,
        j,
        q_cores,
        residual_cores,
        split=split,
        inplace=inplace,
    )
    j_next = j + 1 if split == "right" else j - 1
    moved, absorption = compress_column(
        moved,
        j_next,
        max_bond=absorption_max_bond,
        cutoff=absorption_cutoff,
        inplace=True,
    )
    validate_peps_structure(moved)
    metrics = factorization_metrics(result)
    metrics.update(absorption)
    metrics.update({
        "column": int(j),
        "next_column": int(j_next),
        "split": split,
        "total_runtime_s": float(perf_counter() - started),
    })
    return moved, metrics
