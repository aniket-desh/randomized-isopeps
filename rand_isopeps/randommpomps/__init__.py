"""Vendored Successive Randomized Compression (SRC) code.

This subpackage is a vendored, lightly-patched copy of the ``tensornetwork``
modules from Chris Camaño's RandomMPOMPS repository, which accompanies

    C. Camaño, E. N. Epperly, J. A. Tropp,
    "Successive randomized compression: A randomized algorithm for the
    compressed MPO-MPS product", arXiv:2504.06475.

Source: https://github.com/chriscamano/RandomMPOMPS (cloned to /tmp during setup).
All algorithmic credit belongs to the original authors; the code is reproduced
here so this repo can call SRC directly.

Patches applied while vendoring (no algorithmic changes):
- ``torch`` / ``matplotlib`` / ``seaborn`` imports are guarded so the SRC path
  imports without those heavy/optional dependencies (the numpy SRC routines
  ``random_contraction`` / ``random_contraction_inc`` do not use torch).
- ``incrementalqr`` no longer mutates the global OMP/OPENBLAS thread counts or
  appends a C++ build path on import; it imports the optional C++ extension as a
  package-relative submodule (``libincrementalqr``) and otherwise falls back to a
  pure-Python incremental QR. Build the C++ extension with
  ``bash rand_isopeps/randommpomps/build_incrementalqr.sh`` (needs pybind11 +
  OpenBLAS); ``incrementalqr.cpp`` is the vendored source.

Note: the upstream repository did not include a license file at the time of
vendoring. This copy is kept for local research use only.

Public entry points re-exported for convenience:
- ``random_contraction``       -- SRC (standard / einsum-friendly QR variant)
- ``random_contraction_inc``   -- SRC with incremental QR
- ``mps_mpo_blas``             -- deterministic dense-bond product (+ optional rounding)
- ``MPS``, ``MPO``             -- the tensor-train classes
- ``Cutoff``, ``FixedDimension``, ``no_truncation`` -- stopping rules
"""

from .stopping import Cutoff, FixedDimension, no_truncation
from .MPO import MPO
from .MPS import MPS
from .contraction import mps_mpo_blas, random_contraction, random_contraction_inc

__all__ = [
    "Cutoff",
    "FixedDimension",
    "no_truncation",
    "MPO",
    "MPS",
    "mps_mpo_blas",
    "random_contraction",
    "random_contraction_inc",
]
