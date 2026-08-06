"""Physics-first imaginary-time isoPEPS with rMPS-sketched column moves.

The active APIs are :mod:`rand_isopeps.physics`,
:mod:`rand_isopeps.real_isotns.physics_loop`, and
:mod:`rand_isopeps.sketching`. Phase-one method-development modules remain
available so their experiments stay reproducible.
"""

from .tn_shapes import MosesDims

__all__ = ["MosesDims"]
