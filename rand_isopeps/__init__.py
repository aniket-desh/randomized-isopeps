"""randomized isoPEPS / Moses Move kernels.

Shared library for the ``randomized-isopeps`` repo. It holds the reusable
randomized-linear-algebra kernels, tensor-ring decomposition, MPO-MPS
absorption, synthetic ensembles, and plotting/IO/parallel helpers used by the
experiment suites (e.g. ``synthetic/``).
"""

from .tn_shapes import MosesDims

__all__ = ["MosesDims"]
