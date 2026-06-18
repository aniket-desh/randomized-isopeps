"""structured random sketches for the randomized-SVD range finder.

Implements the sketching distributions from Camano, Epperly, Meyer, Tropp,
"Faster Linear Algebra Algorithms with Structured Random Matrices"
(arXiv:2508.21189), behind a small ``SketchSpec`` / ``range_sample`` interface.

``range_sample(a, ell, rng, spec)`` returns ``Y`` of shape ``(a.shape[0], w)``
with ``w >= ell`` -- the action of a test matrix ``Omega in F^{a.shape[1] x w}``
on ``a`` (i.e. ``a @ Omega`` or its structured equivalent). The width is exactly
``ell`` for every kind except ``sparsestack``, which rounds up to
``zeta*ceil(ell/zeta)`` so its ``zeta`` blocks are equal-width (the extra columns
only widen the range basis; the range finder is unaffected). Each sketch is
isotropically
normalized so ``E||Omega^* x||^2 = ||x||^2``. The randomized range finder only
uses ``range(Y)`` (via a QR), so the scaling is irrelevant there, but we keep it
correct for any standalone OSI diagnostic.

Kinds:

* ``gaussian`` / ``rademacher`` -- dense baselines.
* ``countsketch`` -- a single CountSketch block (one signed nonzero per input
  column, hashed into ``ell`` buckets).
* ``sparsestack`` -- ``zeta`` stacked CountSketch blocks concatenated and scaled
  ``1/sqrt(zeta)`` (Def. 1.7; recommended ``zeta=4``). The averaging over blocks
  is what makes it a reliable ``(r, 1/2)``-OSI where a single CountSketch is not.
* ``khatri_rao`` -- product-structured sketch whose columns are Kronecker
  products of isotropic base vectors over ``factor_dims`` (Def. 1.12). Targets a
  grouped product index such as the second-SVD right index ``chi*n3 =
  chi*(chi*eta)``. Use a **spherical** base: a Rademacher base has injectivity 0
  on the worst Kronecker subspace ("overwhelming orthogonality", sec. 5.3).
  NOTE: ``Omega`` is materialized and applied as ``a @ Omega``, so this tests the
  *distribution's* accuracy/safety, NOT a matvec speedup -- the structure-
  exploiting matrix-free form (applying ``A Omega`` via tensor contractions
  without forming ``Omega``) is future work. Do not claim a Khatri-Rao speedup
  from this implementation.

Only ``sparsestack`` and ``khatri_rao`` are routed here from ``rsvd_truncate``;
the dense / countsketch fast paths stay native to ``randomized_svd`` so the
existing experiments are byte-for-byte unchanged. The full set is implemented
here so ``range_sample`` is a complete standalone sketch library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SketchKind = Literal["gaussian", "rademacher", "countsketch", "sparsestack", "khatri_rao"]
BaseKind = Literal["gaussian", "rademacher", "spherical"]


@dataclass(frozen=True)
class SketchSpec:
    """Parameters of a structured sketch. ``kind`` alone reproduces the simple
    sketches; ``zeta`` tunes sparsestack; ``factor_dims``/``base`` tune khatri_rao."""

    kind: SketchKind = "gaussian"
    zeta: int = 4  # sparsestack: number of stacked CountSketch blocks (= nnz per row)
    factor_dims: tuple[int, ...] | None = None  # khatri_rao: per-factor sizes, outer..inner
    base: BaseKind = "spherical"  # khatri_rao base distribution


def as_spec(sketch: "SketchKind | SketchSpec") -> SketchSpec:
    """Normalize a string kind or a SketchSpec into a SketchSpec."""
    return sketch if isinstance(sketch, SketchSpec) else SketchSpec(kind=sketch)


def _is_complex(dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def _signs(rng: np.random.Generator, size, dtype) -> np.ndarray:
    """Rademacher signs; complex Rademacher (1/sqrt2 normalized) for complex dtype."""
    s = rng.choice(np.array([-1.0, 1.0]), size=size)
    if _is_complex(dtype):
        t = rng.choice(np.array([-1.0, 1.0]), size=size)
        return ((s + 1j * t) / np.sqrt(2.0)).astype(dtype, copy=False)
    return s.astype(dtype, copy=False)


def _dense(a: np.ndarray, ell: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    n = a.shape[1]
    if kind == "gaussian":
        if _is_complex(a.dtype):
            omega = (rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))) / np.sqrt(2.0)
        else:
            omega = rng.standard_normal((n, ell))
    else:  # rademacher
        omega = _signs(rng, (n, ell), a.dtype)
    return (a @ omega.astype(a.dtype, copy=False)) / np.sqrt(ell)


def _countsketch_block(a: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """One CountSketch block: hash each column of ``a`` into one of ``b`` signed buckets."""
    n = a.shape[1]
    buckets = rng.integers(0, b, size=n)
    signs = _signs(rng, n, a.dtype)
    y = np.zeros((a.shape[0], b), dtype=a.dtype)
    np.add.at(y, (slice(None), buckets), a * signs[None, :])
    return y


def _sparsestack(a: np.ndarray, ell: int, rng: np.random.Generator, zeta: int) -> np.ndarray:
    zeta = max(1, int(zeta))
    b = max(1, -(-ell // zeta))  # ceil so total width zeta*b >= ell >= target rank
    blocks = [_countsketch_block(a, b, rng) for _ in range(zeta)]
    return np.concatenate(blocks, axis=1) / np.sqrt(zeta)


def _base_vectors(m: int, ell: int, rng: np.random.Generator, base: str, complex_dtype: bool) -> np.ndarray:
    """``ell`` isotropic base vectors in F^m as an (m, ell) array (E[v v^*] = I)."""
    if base == "rademacher":
        return _signs(rng, (m, ell), np.complex128 if complex_dtype else np.float64)
    if complex_dtype:
        g = (rng.standard_normal((m, ell)) + 1j * rng.standard_normal((m, ell))) / np.sqrt(2.0)
    else:
        g = rng.standard_normal((m, ell))
    if base == "gaussian":
        return g
    # spherical: uniform on the sphere of radius sqrt(m) -- isotropic, avoids the
    # discrete-base "overwhelming orthogonality" failure of a Rademacher base.
    norms = np.linalg.norm(g, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return np.sqrt(m) * g / norms


def _khatri_rao(a: np.ndarray, ell: int, rng: np.random.Generator, spec: SketchSpec) -> np.ndarray:
    factor_dims = spec.factor_dims
    if factor_dims is None or int(np.prod(factor_dims)) != a.shape[1]:
        raise ValueError(
            f"khatri_rao factor_dims {factor_dims} must multiply to a.shape[1]={a.shape[1]}"
        )
    complex_dtype = _is_complex(a.dtype)
    factors = [_base_vectors(m, ell, rng, spec.base, complex_dtype) for m in factor_dims]
    # column-wise Kronecker over the factors (outer .. inner), preserving ell columns
    cols = factors[0]
    for f in factors[1:]:
        cols = (cols[:, None, :] * f[None, :, :]).reshape(-1, ell)
    omega = cols.astype(a.dtype, copy=False) / np.sqrt(ell)
    return a @ omega


def range_sample(a: np.ndarray, ell: int, rng: np.random.Generator, sketch: "SketchKind | SketchSpec") -> np.ndarray:
    """Apply a sketch to ``a``: return ``Y`` of shape ``(a.shape[0], w)``, ``w >= ell``
    (``w == ell`` except for sparsestack, which rounds up to ``zeta*ceil(ell/zeta)``)."""
    spec = as_spec(sketch)
    if spec.kind in ("gaussian", "rademacher"):
        return _dense(a, ell, rng, spec.kind)
    if spec.kind == "countsketch":
        return _countsketch_block(a, ell, rng)  # b = ell buckets; isotropic as-is
    if spec.kind == "sparsestack":
        return _sparsestack(a, ell, rng, spec.zeta)
    if spec.kind == "khatri_rao":
        return _khatri_rao(a, ell, rng, spec)
    raise ValueError(f"unknown sketch kind: {spec.kind}")
