"""local two-SVD isometric tensor-ring decomposition skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from rand_isopeps.linalg.randomized_svd import (
    SVDResult,
    SketchKind,
    isometry_defect_columns,
    isometry_defect_rows,
    relative_frobenius_error,
    rsvd_truncate,
    svd_truncate,
)
from rand_isopeps.tn_shapes import MosesDims

LocalMode = Literal["det", "rand_first", "rand_second", "rand_both"]


@dataclass
class LocalRingResult:
    dims: MosesDims
    mode: LocalMode
    q1: np.ndarray
    u2s: np.ndarray
    vh2: np.ndarray
    singular_first: np.ndarray
    singular_second: np.ndarray
    b_hat: np.ndarray
    first_time_s: float
    second_time_s: float
    first_method: str
    second_method: str

    @property
    def total_time_s(self) -> float:
        return self.first_time_s + self.second_time_s

    def metrics(self, b: np.ndarray) -> dict[str, float | str | int]:
        return {
            "mode": self.mode,
            "first_method": self.first_method,
            "second_method": self.second_method,
            "rel_error": relative_frobenius_error(b, self.b_hat),
            "first_iso_defect": isometry_defect_columns(self.q1),
            "second_iso_defect": isometry_defect_rows(self.vh2),
            "first_time_s": self.first_time_s,
            "second_time_s": self.second_time_s,
            "total_time_s": self.total_time_s,
            "rank_first": self.singular_first.shape[0],
            "rank_second": self.singular_second.shape[0],
        }


def _truncated_svd(
    a: np.ndarray,
    k: int,
    randomized: bool,
    rng: np.random.Generator,
    oversample: int,
    n_power: int,
    sketch: SketchKind,
) -> SVDResult:
    if randomized:
        return rsvd_truncate(a, k, oversample=oversample, n_power=n_power, rng=rng, sketch=sketch)
    return svd_truncate(a, k)


def _mode_flags(mode: LocalMode) -> tuple[bool, bool]:
    if mode == "det":
        return False, False
    if mode == "rand_first":
        return True, False
    if mode == "rand_second":
        return False, True
    if mode == "rand_both":
        return True, True
    raise ValueError(f"unknown local mode: {mode}")


def local_ring_decomp(
    b: np.ndarray,
    dims: MosesDims,
    mode: LocalMode = "det",
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> LocalRingResult:
    """Run the two-SVD tensor-ring decomposition skeleton.

    This intentionally omits the optional nonlinear disentangler so that the
    randomized insertion points are isolated cleanly.
    """

    if b.shape != dims.tensor_shape:
        raise ValueError(f"expected tensor shape {dims.tensor_shape}, got {b.shape}")
    gen = np.random.default_rng(seed) if rng is None else rng
    rand_first, rand_second = _mode_flags(mode)

    first_matrix = b.reshape(dims.n1, dims.n2 * dims.n3)
    t0 = perf_counter()
    first = _truncated_svd(first_matrix, dims.k1, rand_first, gen, oversample, n_power, sketch)
    t1 = perf_counter()

    v_first = first.s[:, None] * first.vh
    v4 = v_first.reshape(dims.eta, dims.chi, dims.n2, dims.n3)
    second_matrix = v4.transpose(0, 2, 1, 3).reshape(dims.eta * dims.n2, dims.chi * dims.n3)

    t2 = perf_counter()
    second = _truncated_svd(second_matrix, dims.k2, rand_second, gen, oversample, n_power, sketch)
    t3 = perf_counter()

    u2s = second.u * second.s
    second_hat = u2s @ second.vh
    v4_hat = second_hat.reshape(dims.eta, dims.n2, dims.chi, dims.n3).transpose(0, 2, 1, 3)
    v_first_hat = v4_hat.reshape(dims.k1, dims.n2 * dims.n3)
    b_hat = (first.u @ v_first_hat).reshape(dims.tensor_shape)

    return LocalRingResult(
        dims=dims,
        mode=mode,
        q1=first.u,
        u2s=u2s,
        vh2=second.vh,
        singular_first=first.s,
        singular_second=second.s,
        b_hat=b_hat,
        first_time_s=t1 - t0,
        second_time_s=t3 - t2,
        first_method=first.method,
        second_method=second.method,
    )
