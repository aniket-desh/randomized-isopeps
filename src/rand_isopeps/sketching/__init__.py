"""Small public surface for the project's randomized column sketch.

The detailed phase-one implementations remain in :mod:`rand_isopeps.column`
and :mod:`rand_isopeps.linalg` so the historical experiments stay reproducible.
New physics code should import this package instead of depending on those
experiment-facing modules directly.
"""

from .column import factorize_column, factorization_metrics

__all__ = ["factorize_column", "factorization_metrics"]
