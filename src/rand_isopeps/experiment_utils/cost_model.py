"""Implementation-independent cost model for the SVD methods.

Counts ALGORITHM work -- FLOPs (multiply-adds), passes over the input matrix, and
peak intermediate memory -- analytically from matrix shapes, with NO library / BLAS /
hardware dependence. This is the fair way to compare deterministic vs randomized SVD,
and to compare sketch families, because **wall-clock conflates the algorithm with
implementation maturity and hardware**: a dense Gaussian sketch is a BLAS-3 GEMM that
runs ~100x faster than an equivalent CountSketch scatter *despite doing FEWER FLOPs*.
Use these counts for cross-method / cross-sketch claims; treat wall-clock as a labeled,
machine-dependent secondary metric.

Conventions
-----------
* FLOP = one scalar multiply-add counted as 2 flops; a real ``(m x p) @ (p x n)`` GEMM
  is ``2 m p n``.
* SVD flop counts use the standard Golub-Reinsch figure (Trefethen & Bau, *Numerical
  Linear Algebra*, Lecture 31): a thin SVD of an ``a x b`` matrix (``a >= b``) computing
  ``U, Sigma, V`` costs ``~ 2 a b^2 + 11 b^3``. QR (thin, ``a x b``, ``a >= b``) costs
  ``~ 2 a b^2 - (2/3) b^3``.
* These are MODEL counts used consistently for *ratios* -- the exact LAPACK constants
  vary by driver, but they cancel in the det-vs-rand and sketch-vs-sketch comparisons.
* "Passes over A" = how many times the m x n input is streamed (architecture-free; the
  classic randomized-SVD selling point). "Peak intermediate floats" = the working set
  beyond the input A (the memory-footprint proxy).
"""

from __future__ import annotations

from dataclasses import dataclass


def gemm_flops(m: int, p: int, n: int) -> float:
    """FLOPs for a real (m x p) @ (p x n) matrix multiply."""
    return 2.0 * m * p * n


def qr_flops(a: int, b: int) -> float:
    """FLOPs for a thin QR of an a x b matrix (Householder, a >= b)."""
    a, b = max(a, b), min(a, b)
    return 2.0 * a * b * b - (2.0 / 3.0) * b ** 3


def svd_flops(m: int, n: int) -> float:
    """FLOPs for a thin SVD of an m x n matrix (Golub-Reinsch, U,Sigma,V)."""
    a, b = max(m, n), min(m, n)
    return 2.0 * a * b * b + 11.0 * b ** 3


def sketch_flops(m: int, n: int, ell: int, sketch: str = "gaussian", zeta: int = 4) -> float:
    """FLOPs to apply a width-``ell`` sketch Omega to A (m x n) forming Y = A Omega.

    gaussian/rademacher: a dense GEMM, ``2 m n ell``. CountSketch: each of the m*n
    entries is signed and scattered into one bucket, ``~2 m n`` (independent of ell).
    SparseStack: ``zeta`` stacked CountSketch blocks, ``~2 zeta m n``. Khatri-Rao here
    is materialized (Omega formed then dense-applied), so ``2 m n ell`` like a dense
    sketch -- it does NOT yet exploit its structure for a cheaper matvec.
    """
    if sketch in ("gaussian", "rademacher", "khatri_rao"):
        return gemm_flops(m, n, ell)
    if sketch == "countsketch":
        return 2.0 * m * n
    if sketch == "sparsestack":
        return 2.0 * zeta * m * n
    raise ValueError(f"unknown sketch kind: {sketch}")


def svd_truncate_flops(m: int, n: int) -> float:
    """Deterministic truncated SVD = a full thin SVD then slice."""
    return svd_flops(m, n)


def rsvd_flops(m: int, n: int, k: int, ell: int, n_power: int = 1,
               sketch: str = "gaussian", zeta: int = 4) -> float:
    """FLOPs for the HMT randomized SVD: sketch, QR, power iters, B = Q^H A, small SVD, map back."""
    total = sketch_flops(m, n, ell, sketch, zeta)      # Y = A Omega
    total += qr_flops(m, ell)                           # Q = orth(Y)
    for _ in range(n_power):                            # (A A^H)^q A Omega, reorthonormalized
        total += gemm_flops(n, m, ell) + qr_flops(n, ell)   # Z = orth(A^H Q)
        total += gemm_flops(m, n, ell) + qr_flops(m, ell)   # Q = orth(A Z)
    total += gemm_flops(ell, m, n)                      # B = Q^H A
    total += svd_flops(ell, n)                          # small SVD of B
    total += gemm_flops(m, ell, ell)                    # U = Q Uhat
    return total


def passes_over_a(method: str, n_power: int = 1) -> int:
    """How many times the input matrix A is streamed."""
    if method == "det":
        return 1
    return 2 + 2 * n_power  # Y = A Omega (1), B = Q^H A (1), each power iter reads A twice


def peak_intermediate_floats(method: str, m: int, n: int, k: int, ell: int,
                             sketch: str = "gaussian", zeta: int = 4) -> float:
    """Working-set floats beyond the input A (memory-footprint proxy)."""
    mn = min(m, n)
    if method == "det":
        return float(m * mn + mn * n + 2 * mn * mn)     # U, Vh, + LAPACK workspace ~ 2 min^2
    omega = (n * ell if sketch in ("gaussian", "rademacher", "khatri_rao")
             else zeta * n)                              # dense Omega vs sparse (nnz)
    return float(omega + m * ell + ell * n + m * ell)    # Omega + Y + B + Q


@dataclass
class CostComparison:
    """Det vs randomized algorithm-work, for one matrix and one randomized config."""
    m: int
    n: int
    k: int
    ell: int
    det_flops: float
    rand_flops: float
    flop_speedup: float          # det_flops / rand_flops (implementation-free)
    det_passes: int
    rand_passes: int
    det_peak_floats: float
    rand_peak_floats: float
    mem_ratio: float             # det_peak / rand_peak

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


def compare(m: int, n: int, k: int, ell: int, n_power: int = 1,
            sketch: str = "gaussian", zeta: int = 4) -> CostComparison:
    """Implementation-free det-vs-rand comparison for a truncated SVD of an m x n matrix."""
    df = svd_truncate_flops(m, n)
    rf = rsvd_flops(m, n, k, ell, n_power, sketch, zeta)
    dp = peak_intermediate_floats("det", m, n, k, ell, sketch, zeta)
    rp = peak_intermediate_floats("rand", m, n, k, ell, sketch, zeta)
    return CostComparison(
        m=m, n=n, k=k, ell=ell,
        det_flops=df, rand_flops=rf, flop_speedup=df / rf if rf else float("nan"),
        det_passes=passes_over_a("det", n_power), rand_passes=passes_over_a("rand", n_power),
        det_peak_floats=dp, rand_peak_floats=rp, mem_ratio=dp / rp if rp else float("nan"),
    )


def local_ring_compare(dims, randomize_first: bool, randomize_second: bool,
                       oversample: int = 8, n_power: int = 1, sketch: str = "gaussian",
                       zeta: int = 4) -> dict:
    """Implementation-free cost for the synthetic local two-SVD ring decomposition.

    ``dims`` is a ``MosesDims`` (exposes ``matrix_shape``/``target_rank`` for "first"/"second").
    The decomp does a first SVD (``n1 x n2n3 -> k1``) and a second SVD
    (``(eta n2) x (chi n3) -> k2``); a stage is randomized per its flag. Returns the
    whole-decomp FLOP-speedup (det / rand), peak-memory ratio, and pass counts -- the fair,
    implementation-free analogue of the wall-clock speedup the rand modes are timed by.
    """
    det_flops = rand_flops = 0.0
    det_peak = rand_peak = 0.0
    det_passes = rand_passes = 0
    for target, randomize in (("first", randomize_first), ("second", randomize_second)):
        m, n = dims.matrix_shape(target)
        k = dims.target_rank(target)
        df = svd_truncate_flops(m, n)
        det_flops += df
        det_peak = max(det_peak, peak_intermediate_floats("det", m, n, k, k, sketch, zeta))
        det_passes += passes_over_a("det")
        if randomize:
            ell = min(k + oversample, m, n)
            rand_flops += rsvd_flops(m, n, k, ell, n_power, sketch, zeta)
            rand_peak = max(rand_peak, peak_intermediate_floats("rand", m, n, k, ell, sketch, zeta))
            rand_passes += passes_over_a("rand", n_power)
        else:
            rand_flops += df
            rand_peak = max(rand_peak, peak_intermediate_floats("det", m, n, k, k, sketch, zeta))
            rand_passes += passes_over_a("det")
    return {
        "flop_speedup": det_flops / rand_flops if rand_flops else float("nan"),
        "det_flops": det_flops, "rand_flops": rand_flops,
        "mem_ratio": det_peak / rand_peak if rand_peak else float("nan"),
        "det_passes": det_passes, "rand_passes": rand_passes,
    }
