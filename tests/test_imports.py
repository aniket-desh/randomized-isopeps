"""Smoke tests for the src/ layout: package imports and back-compat shims.

After the src/ restructure, several modules were relocated under subpackages
(linalg/, moses/, synthetic/, experiment_utils/, real_isotns/) and thin
back-compat shims were left at the old top-level paths. These tests assert that
the package imports and that each shim re-exports the *same* callable object as
its canonical module.
"""

import importlib

import pytest


def test_package_imports():
    import rand_isopeps  # noqa: F401


def test_randomized_svd_backcompat_is_canonical():
    from rand_isopeps.randomized_svd import rsvd_truncate as compat
    from rand_isopeps.linalg.randomized_svd import rsvd_truncate as canonical

    assert compat is canonical


def test_disentangler_backcompat_is_canonical():
    from rand_isopeps.disentangler import disentangle_altmin as compat
    from rand_isopeps.moses.disentangler import disentangle_altmin as canonical

    assert compat is canonical


def test_io_utils_backcompat_is_canonical():
    # io_utils -> experiment_utils.io. Compare a public name shared by both.
    compat_mod = importlib.import_module("rand_isopeps.io_utils")
    canonical_mod = importlib.import_module("rand_isopeps.experiment_utils.io")
    shared = [
        name
        for name in vars(canonical_mod)
        if not name.startswith("_") and callable(getattr(canonical_mod, name))
    ]
    assert shared, "expected at least one public callable in experiment_utils.io"
    for name in shared:
        assert getattr(compat_mod, name) is getattr(canonical_mod, name)


def test_synthetic_tensors_backcompat_is_canonical():
    from rand_isopeps.synthetic_tensors import make_disentanglable_local as compat
    from rand_isopeps.synthetic.tensors import make_disentanglable_local as canonical

    assert compat is canonical


def test_sketching_facade_reuses_validated_kernels():
    from rand_isopeps.column.operator import ColumnOperator as implementation_operator
    from rand_isopeps.linalg.rmps_sketch import rmps_cores as implementation_rmps
    from rand_isopeps.linalg.sketches import range_sample as implementation_distribution
    from rand_isopeps.sketching.distributions import range_sample
    from rand_isopeps.sketching.operator import ColumnOperator
    from rand_isopeps.sketching.rmps import rmps_cores

    assert ColumnOperator is implementation_operator
    assert rmps_cores is implementation_rmps
    assert range_sample is implementation_distribution


def test_quimb_dependent_imports_skip_when_missing():
    """isotns/tebd pull in quimb; this should SKIP (not fail) without quimb."""
    pytest.importorskip("quimb")
    from rand_isopeps.isotns import RandSVD  # noqa: F401
    from rand_isopeps.tebd import tfi_ham  # noqa: F401
