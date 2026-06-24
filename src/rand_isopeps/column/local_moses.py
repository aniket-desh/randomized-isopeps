"""Sequential *local* column Moses move -- the baseline the global sketch replaces.

The block Moses move builds the new isometric column by sweeping it row by row:
at each site it splits off an output-isometric tensor and carries the residual
(the absorbed legs + a rank bond) to the next row, truncating the carried rank.
This module implements exactly that sweep on a :class:`ColumnOperator`, so it is
directly comparable to the global one-shot range finder
(:mod:`rand_isopeps.column.global_range`): both produce an output-isometric
column ``Q`` and a residual ``R`` with ``C ~ Q R``, and we score the same column
Frobenius error ``||C - Q R||_F / ||C||_F``.

The fork (briefing sec 5): the local sweep is greedy -- its error is a sum of
*local* truncation errors ``sum_i eps_i`` (it caps the local tensor-network rank),
whereas the global sketch targets the *flat* (whole-matrix) column rank, with
error ``~ tau_r(C)`` (the best rank-``r`` tail). Global wins when the column has a
low flat rank; local wins when it only has low local TN rank. This experiment is
where that fork is measured on tiny materialized columns.

``local_column_qr`` sweeps bottom -> top. At site ``i`` it groups the
output-isometric side ``(incoming rank, output leg)`` against the residual side
``(accumulated inputs, this input, vertical bond)``, takes a (deterministic or
randomized) truncated SVD to rank ``k``, keeps ``U`` as the output-isometric
column core, and carries ``S V^*`` -- the absorbed legs plus the rank bond -- to
the next row. The final carry is the residual column ``R`` (rank ``k`` x ``n_in``).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rand_isopeps.column.operator import ColumnOperator
from rand_isopeps.linalg.randomized_svd import (
    SketchKind,
    isometry_defect_columns,
    relative_frobenius_error,
    rsvd_truncate,
    svd_truncate,
)


@dataclass
class LocalColumnResult:
    mode: str            # "det" | "rand"
    max_rank: int        # the carried-rank cap k
    rel_error: float     # ||C - Q R||_F / ||C||_F
    isometry_defect: float  # ||Q^* Q - I||_F for the new column
    max_carried_rank: int   # largest carried rank actually reached over the sweep
    n_svd: int           # number of local SVDs (the sequential cost)
    runtime_s: float
    q_cols: int          # columns of the assembled isometric column Q

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "mode": self.mode,
            "max_rank": self.max_rank,
            "rel_error": self.rel_error,
            "isometry_defect": self.isometry_defect,
            "max_carried_rank": self.max_carried_rank,
            "n_svd": self.n_svd,
            "runtime_s": self.runtime_s,
            "q_cols": self.q_cols,
        }


def local_column_qr(
    column: ColumnOperator,
    max_rank: int,
    randomized: bool = False,
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    rng: np.random.Generator | None = None,
    reference: np.ndarray | None = None,
) -> LocalColumnResult:
    """Sequential local Moses column QR truncating the carried rank to ``max_rank``.

    ``randomized=True`` makes every local split a randomized SVD (the "randomized
    local SVD2 Moses" baseline); otherwise each split is a deterministic truncated
    SVD (the "deterministic local Moses" baseline). ``reference`` is the exact
    dense ``C`` to score against (defaults to ``column.materialize()``).
    """
    gen = np.random.default_rng() if rng is None else rng
    cores = column.cores
    lx = len(cores)
    c_dense = column.materialize() if reference is None else reference

    t0 = perf_counter()
    q_cores: list[np.ndarray] = []
    res = np.ones((1, 1, 1), dtype=c_dense.dtype)  # (carried_rank a, accumulated inputs, left bond)
    a, in_prod = 1, 1
    n_svd = 0
    max_carried = 0
    for i, w in enumerate(cores):
        ml, o, s, mr = w.shape
        # contract the carried residual's bond into this core's left MPO bond
        theta = np.tensordot(res, w, axes=(2, 0))  # (a, in_prod, o, s, mr)
        # output-isometric side (a, o) vs residual side (in_prod, s, mr)
        mat = theta.transpose(0, 2, 1, 3, 4).reshape(a * o, in_prod * s * mr)
        k = min(max_rank, mat.shape[0], mat.shape[1])
        if randomized:
            r = rsvd_truncate(mat, k, oversample=oversample, n_power=n_power, rng=gen, sketch=sketch)
        else:
            r = svd_truncate(mat, k)
        n_svd += 1
        kk = r.rank
        q_cores.append(r.u.reshape(a, o, kk))
        res = (r.s[:, None] * r.vh).reshape(kk, in_prod * s, mr)
        a, in_prod = kk, in_prod * s
        max_carried = max(max_carried, kk)
    runtime = perf_counter() - t0

    # assemble the dense isometric column Q (n_out x a) and residual R (a x n_in)
    q = q_cores[0][0]  # (o_0, k_0); incoming rank a=1 at the bottom
    for qc in q_cores[1:]:
        q = np.tensordot(q, qc, axes=(-1, 0)).reshape(-1, qc.shape[-1])
    r_mat = res[:, :, 0]  # (a_last, n_in); trailing MPO bond is 1
    approx = q @ r_mat

    return LocalColumnResult(
        mode="rand" if randomized else "det",
        max_rank=max_rank,
        rel_error=relative_frobenius_error(c_dense, approx),
        isometry_defect=isometry_defect_columns(q),
        max_carried_rank=max_carried,
        n_svd=n_svd,
        runtime_s=runtime,
        q_cols=int(q.shape[1]),
    )
