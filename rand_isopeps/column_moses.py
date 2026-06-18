"""synthetic column-level Moses Move accumulation experiments.

LEGACY / diagnostic: this is the *product-column surrogate* (independent locals,
Kronecker product), used by exp2. The faithful sequential upward carrier in
``column_carrier.py`` (exp11) supersedes it for any theoretical / accumulation
claim; keep this only as the cheap first-pass surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .local_ring_decomp import LocalMode, local_ring_decomp
from .randomized_svd import SketchKind
from .synthetic_tensors import EnsembleKind, make_local_tensor
from .tn_shapes import MosesDims


@dataclass
class ColumnRunResult:
    dims: MosesDims
    lx: int
    mode: LocalMode
    rel_error: float
    runtime_s: float
    max_local_error: float
    mean_local_error: float
    max_first_iso_defect: float
    max_second_iso_defect: float

    def as_dict(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            **self.dims.as_dict(),
            "lx": self.lx,
            "mode": self.mode,
            "rel_error": self.rel_error,
            "runtime_s": self.runtime_s,
            "max_local_error": self.max_local_error,
            "mean_local_error": self.mean_local_error,
            "max_first_iso_defect": self.max_first_iso_defect,
            "max_second_iso_defect": self.max_second_iso_defect,
        }
        return row


def product_column_relative_error(
    tensors: list[np.ndarray],
    approximations: list[np.ndarray],
) -> float:
    """Relative error for a Kronecker-product surrogate column.

    This avoids materializing the full column tensor. It is a lightweight way
    to measure how local Moses Move errors accumulate with column height.
    """

    if len(tensors) != len(approximations):
        raise ValueError("tensor and approximation lists must have equal length")

    norm_exact_sq = 1.0
    norm_approx_sq = 1.0
    overlap = 1.0 + 0.0j

    for exact, approx in zip(tensors, approximations):
        e = exact.reshape(-1)
        a = approx.reshape(-1)
        norm_exact_sq *= float(np.vdot(e, e).real)
        norm_approx_sq *= float(np.vdot(a, a).real)
        overlap *= np.vdot(e, a)

    err_sq = norm_exact_sq + norm_approx_sq - 2.0 * overlap.real
    err_sq = max(float(err_sq), 0.0)
    if norm_exact_sq == 0:
        return float(np.sqrt(err_sq))
    return float(np.sqrt(err_sq / norm_exact_sq))


def run_column_moses_surrogate(
    dims: MosesDims,
    lx: int,
    mode: LocalMode,
    ensemble: EnsembleKind = "noisy_ring",
    noise: float = 1e-4,
    decay: float = 0.92,
    spectrum_param: float = 1.0,
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    seed: int | None = None,
) -> ColumnRunResult:
    if lx <= 0:
        raise ValueError("lx must be positive")

    rng = np.random.default_rng(seed)
    tensors = [
        make_local_tensor(dims, rng, ensemble=ensemble, noise=noise, decay=decay, spectrum_param=spectrum_param)
        for _ in range(lx)
    ]
    approximations: list[np.ndarray] = []
    local_errors: list[float] = []
    first_iso: list[float] = []
    second_iso: list[float] = []

    t0 = perf_counter()
    for tensor in tensors:
        result = local_ring_decomp(
            tensor,
            dims,
            mode=mode,
            oversample=oversample,
            n_power=n_power,
            sketch=sketch,
            rng=rng,
        )
        metrics = result.metrics(tensor)
        approximations.append(result.b_hat)
        local_errors.append(float(metrics["rel_error"]))
        first_iso.append(float(metrics["first_iso_defect"]))
        second_iso.append(float(metrics["second_iso_defect"]))
    runtime = perf_counter() - t0

    return ColumnRunResult(
        dims=dims,
        lx=lx,
        mode=mode,
        rel_error=product_column_relative_error(tensors, approximations),
        runtime_s=runtime,
        max_local_error=max(local_errors),
        mean_local_error=float(np.mean(local_errors)),
        max_first_iso_defect=max(first_iso),
        max_second_iso_defect=max(second_iso),
    )
