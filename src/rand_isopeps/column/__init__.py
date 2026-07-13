"""Column-level (whole-column) Moses move: global rMPS-sketched column QR.

This subpackage holds the *global column* alternative to the sequential local
Moses move. Where the local move (``rand_isopeps.synthetic.column_carrier``)
sweeps a column row by row -- splitting each site with two local SVDs and
carrying a residual -- the global move treats the entire active column as one
linear map ``C_j : X_j -> Y_j`` (absorbed legs -> retained legs) and approximates
its column QR ``C_j ~ Q_j R_j`` in one shot with a randomized range finder using
random matrix product state (rMPS) probes.

* :mod:`rand_isopeps.column.operator` -- the ``ColumnOperator`` access seam: a
  column as an MPO that can be materialized (tiny validation) or applied
  matrix-free via MPO--MPS products (the access model the algorithm really uses).
* :mod:`rand_isopeps.column.global_range` -- the global randomized range finder
  and its diagnostics (approximation error, excess over the best rank-r SVD,
  isometry defect, the OSI subspace-injection diagnostic).

The seam is deliberately synthetic-now / real-PEPS-column-later: a real isoTNS
column, once its top/bottom environment is contracted, is exactly an MPO from
absorbed to retained legs, so the same range finder and experiments run on real
columns by swapping the ``ColumnOperator`` constructor.
"""

from .operator import ColumnOperator, controlled_spectrum_column_matrix, random_column_operator
from .global_range import GlobalRangeResult, global_column_range, reference_svd, sampled_bond_growth
from .bounded_residual import (
    BoundedResidualResult,
    ProjectionScore,
    apply_boundary_factorization,
    bounded_residual_column_qr,
    score_projection_error,
)
from .diagnostics import column_diagnostics, operator_cut_spectra, spectrum_diagnostics

__all__ = [
    "ColumnOperator",
    "random_column_operator",
    "controlled_spectrum_column_matrix",
    "GlobalRangeResult",
    "global_column_range",
    "reference_svd",
    "sampled_bond_growth",
    "BoundedResidualResult",
    "ProjectionScore",
    "bounded_residual_column_qr",
    "score_projection_error",
    "apply_boundary_factorization",
    "column_diagnostics",
    "operator_cut_spectra",
    "spectrum_diagnostics",
]
