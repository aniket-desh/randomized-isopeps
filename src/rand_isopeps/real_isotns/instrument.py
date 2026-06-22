"""Per-stage randomization config + lightweight instrumentation for the real
Moses Move (roadmap Phase 0, items 0A-0C).

Kept separate from ``moses_move.py`` so the move logic stays focused: the
per-stage toggles (``MosesRandConfig``) and the per-split collector
(``MosesStats``) live here and are threaded through ``my_split`` -- the single
truncated-SVD primitive -- so every truncation site is configured and recorded
in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints; avoids a runtime import cycle with moses_move
    from .moses_move import RandSVD

# local tensor-ring SVD stages; ``absorption`` is reserved for the future 0E hook
STAGES = ("svd1", "disentangler_inner", "svd2", "absorption")


@dataclass
class MosesRandConfig:
    """Per-stage randomization for the local tensor-ring SVDs of the Moses Move.

    Each field is a ``RandSVD`` (randomize that stage) or ``None`` (deterministic):
    ``svd1`` first SVD, ``disentangler_inner`` disentangler-search SVD, ``svd2``
    final second SVD. ``absorption`` is reserved for the randomized / SRC R-column
    hook (roadmap 0E) and is not applied yet.
    """

    svd1: "RandSVD | None" = None
    disentangler_inner: "RandSVD | None" = None
    svd2: "RandSVD | None" = None
    absorption: object | None = None


@dataclass
class SplitRecord:
    """One truncated-SVD call inside a Moses Move."""

    stage: str
    m: int
    n: int
    k: int  # target rank min(chi_max, m, n)
    ell: int  # sketch width k + oversample (== k when deterministic)
    rho: float  # rank fraction ell / min(m, n)
    t_svd: float  # seconds in the SVD primitive
    tail: float  # discarded weight (an estimate when randomized)
    true_tail: float | None = None  # exact tail (debug mode, small cases only)


@dataclass
class MosesStats:
    """Per-split instrumentation collected across a Moses Move / sweep.

    Pass an instance as ``stats=`` to ``moses_move`` / ``split_3d_iso``. With
    ``debug_tail=True`` (small cases only) it also computes the exact deterministic
    tail at each randomized split, so the sketch's residual estimate is not
    over-interpreted.
    """

    debug_tail: bool = False
    keep_arrays: tuple[str, ...] = ()  # stages whose input matrix to retain (kernel benchmarks)
    records: list[SplitRecord] = field(default_factory=list)
    arrays: list[tuple[str, np.ndarray]] = field(default_factory=list)
    n_moves: int = 0

    def record(self, m, n, chi_max, rand, t_svd, tail, *, stage, array=None, actual_ell=None):
        mn = min(int(m), int(n))
        k = min(int(chi_max), int(m), int(n))
        randomized = rand is not None and getattr(rand, "sketch", None) is not None
        # Rank fraction uses the ACTUAL range-basis width when known: sparsestack rounds
        # k+oversample up to zeta*ceil(./zeta), so the nominal k+oversample under-reports rho.
        if actual_ell is not None:
            ell = min(int(actual_ell), mn)
        else:
            ell = min(k + (rand.oversample if randomized else 0), mn)
        true_tail = None
        if self.debug_tail and randomized and array is not None:
            sv = np.linalg.svd(array, compute_uv=False)
            true_tail = float(np.sqrt(float(np.sum(sv[chi_max:] ** 2))))
        if array is not None and stage in self.keep_arrays:
            self.arrays.append((stage, np.array(array, copy=True)))
        self.records.append(SplitRecord(
            stage, int(m), int(n), k, ell, (ell / mn if mn else 0.0),
            float(t_svd), float(tail), true_tail,
        ))

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage aggregates: call count, total SVD time, max rho, largest min(m,n)."""
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            s = out.setdefault(r.stage, {"n_calls": 0, "t_svd": 0.0, "max_rho": 0.0, "max_dim": 0})
            s["n_calls"] += 1
            s["t_svd"] += r.t_svd
            s["max_rho"] = max(s["max_rho"], r.rho)
            s["max_dim"] = max(s["max_dim"], min(r.m, r.n))
        return out

    def to_rows(self) -> list[dict]:
        """Flat per-split rows (for CSV / dataframe)."""
        return [asdict(r) for r in self.records]
