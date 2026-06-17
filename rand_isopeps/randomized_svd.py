"""deterministic and randomized truncated SVD utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

# SketchKind / SketchSpec live in sketches.py (standalone, numpy-only). Re-export
# SketchKind here so existing callers `from .randomized_svd import SketchKind`
# keep working with the widened set (gaussian/rademacher/countsketch/sparsestack/
# khatri_rao).
from .sketches import SketchKind, SketchSpec, as_spec
from .sketches import range_sample as _structured_range_sample

@dataclass
class SVDResult:
    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray
    method: str

    @property
    def rank(self) -> int:
        return int(self.s.shape[0])

    def reconstruct(self) -> np.ndarray:
        return (self.u * self.s) @ self.vh


def _rng(seed: int | None = None, rng: np.random.Generator | None = None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _is_complex_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def _test_matrix(
    n: int,
    ell: int,
    dtype: np.dtype,
    rng: np.random.Generator,
    sketch: SketchKind,
) -> np.ndarray:
    if sketch == "gaussian":
        if _is_complex_dtype(dtype):
            omega = rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))
            return (omega / np.sqrt(2.0)).astype(dtype, copy=False)
        return rng.standard_normal((n, ell)).astype(dtype, copy=False)

    if sketch == "rademacher":
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, ell))
        if _is_complex_dtype(dtype):
            imag_signs = rng.choice(np.array([-1.0, 1.0]), size=(n, ell))
            return ((signs + 1j * imag_signs) / np.sqrt(2.0)).astype(dtype, copy=False)
        return signs.astype(dtype, copy=False)

    raise ValueError(f"unknown sketch kind: {sketch}")


def _range_sample(
    a: np.ndarray,
    ell: int,
    rng: np.random.Generator,
    sketch: "SketchKind | SketchSpec",
) -> np.ndarray:
    spec = as_spec(sketch)
    # Keep the original gaussian/rademacher/countsketch fast paths byte-for-byte
    # (exp1-11 depend on their exact RNG draw order); delegate only the new
    # structured kinds to sketches.range_sample.
    if spec.kind == "countsketch":
        buckets = rng.integers(0, ell, size=a.shape[1])
        signs = rng.choice(np.array([-1.0, 1.0]), size=a.shape[1])
        y = np.zeros((a.shape[0], ell), dtype=a.dtype)
        np.add.at(y, (slice(None), buckets), a * signs[None, :])
        return y
    if spec.kind in ("gaussian", "rademacher"):
        omega = _test_matrix(a.shape[1], ell, a.dtype, rng, spec.kind)
        return a @ omega
    return _structured_range_sample(a, ell, rng, spec)


def orthonormalize(a: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(a, mode="reduced")
    return q


def svd_truncate(a: np.ndarray, k: int) -> SVDResult:
    if k <= 0:
        raise ValueError("k must be positive")
    u, s, vh = la.svd(a, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    kk = min(k, s.shape[0])
    return SVDResult(u=u[:, :kk], s=s[:kk], vh=vh[:kk, :], method="svd")


def rsvd_truncate(
    a: np.ndarray,
    k: int,
    oversample: int = 8,
    n_power: int = 1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    sketch: "SketchKind | SketchSpec" = "gaussian",
) -> SVDResult:
    """Randomized SVD using the Halko-Martinsson-Tropp range finder.

    The implementation forms Y = A Omega, optionally applies power iterations,
    computes a small SVD of Q^* A, and maps the left singular vectors back.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if oversample < 0:
        raise ValueError("oversample must be nonnegative")
    if n_power < 0:
        raise ValueError("n_power must be nonnegative")

    m, n = a.shape
    kk = min(k, m, n)
    ell = min(max(kk + oversample, kk), m, n)
    gen = _rng(seed=seed, rng=rng)

    y = _range_sample(a, ell, gen, sketch)
    q = orthonormalize(y)

    for _ in range(n_power):
        z = orthonormalize(a.conj().T @ q)
        q = orthonormalize(a @ z)

    b = q.conj().T @ a
    u_hat, s, vh = la.svd(b, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    u = q @ u_hat
    return SVDResult(u=u[:, :kk], s=s[:kk], vh=vh[:kk, :], method=f"rsvd-{as_spec(sketch).kind}")


def relative_frobenius_error(a: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(a)
    if denom == 0:
        return float(np.linalg.norm(approx))
    return float(np.linalg.norm(a - approx) / denom)


def isometry_defect_columns(q: np.ndarray) -> float:
    gram = q.conj().T @ q
    eye = np.eye(gram.shape[0], dtype=gram.dtype)
    return float(np.linalg.norm(gram - eye))


def isometry_defect_rows(qh: np.ndarray) -> float:
    gram = qh @ qh.conj().T
    eye = np.eye(gram.shape[0], dtype=gram.dtype)
    return float(np.linalg.norm(gram - eye))
