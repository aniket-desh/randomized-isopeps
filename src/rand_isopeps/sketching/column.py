"""The active rMPS column-sketch interface.

This deliberately hides the large diagnostic result object from the physics
loop.  The full object is still returned as the third value for the retained
sketching experiments and for state insertion, while routine callers can log
the compact metrics dictionary.
"""

from __future__ import annotations

import numpy as np


def factorization_metrics(result) -> dict:
    """Return the small, serializable subset needed by a physics trajectory."""
    return {
        "ell": int(result.ell),
        "eta": int(result.eta),
        "kappa": int(result.kappa),
        "chi_sk": int(result.chi_sk),
        "projection_error": float(result.projection_error_dense),
        "sample_error": float(result.reconstruction_error),
        "isometry_defect": float(result.delta_global),
        "isometry_bound": float(result.delta_global_bound),
        "q_rank": int(result.q_flat_rank),
        "q_vertical_bond": int(result.max_q_vertical),
        "residual_vertical_bond": int(result.max_residual_vertical),
        "passes": int(result.passes),
        "runtime_s": float(result.runtime_s),
    }


def factorize_column(
    column,
    *,
    ell: int,
    eta: int,
    kappa: int,
    chi_sk: int,
    ndis: int = 0,
    rng: np.random.Generator | None = None,
):
    """Factor ``column`` as ``Q (Q* column)`` with the project's rMPS sketch.

    The sketch family and power iteration are intentionally fixed here.  Those
    alternatives remain available in the phase-one laboratory, but they are not
    knobs in the physics algorithm.

    Returns ``(q_cores, residual_cores, result)``.
    """
    from rand_isopeps.column.bounded_residual import bounded_residual_column_qr

    result = bounded_residual_column_qr(
        column,
        ell=ell,
        eta=eta,
        kappa=kappa,
        chi_sk=chi_sk,
        sketch_kind="rmps",
        n_power=0,
        ndis=ndis,
        rng=rng,
    )
    return result.q_cores, result.residual_cores, result
