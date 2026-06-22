"""Smoke tests for the Phase-0 statistical aggregation helpers (no optional deps)."""

import numpy as np

from rand_isopeps.aggregate import bootstrap_ci, failure_rate, median_band


def test_bootstrap_ci_brackets_point():
    vals = list(range(1, 100))
    point, lo, hi = bootstrap_ci(vals, ci=90.0, seed=1)
    assert abs(point - np.median(vals)) < 1e-9
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_ci_nan_and_empty():
    point, lo, hi = bootstrap_ci([1.0, np.nan, 3.0], seed=0)
    assert not np.isnan(point)
    assert all(np.isnan(x) for x in bootstrap_ci([np.nan, np.nan]))


def test_failure_rate():
    vals = [1.0, 1.2, 0.9, 2.0]
    assert failure_rate(vals, 1.1, above=True) == 0.5   # {1.2, 2.0}
    assert failure_rate(vals, 1.0, above=False) == 0.5  # {1.0, 0.9}
    assert np.isnan(failure_rate([np.nan], 1.0))


def test_median_band_ci_option():
    rows = [{"m": "a", "x": 1, "v": v} for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
    iqr = median_band(rows, "m", "x", "v", ["a"])
    cib = median_band(rows, "m", "x", "v", ["a"], ci=90.0, seed=0)
    xs, med, lo, hi = cib["a"]
    assert xs == [1.0]
    assert abs(med[0] - 3.0) < 1e-9
    assert lo[0] <= med[0] <= hi[0]
    # default (IQR) band is unchanged / still valid
    assert iqr["a"][1][0] == med[0]
