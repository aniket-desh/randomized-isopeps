"""Median + percentile aggregation for experiment sweeps.

Experiment rows are dicts with a grouping key (e.g. the mode/method), an x key
(e.g. ``eta`` or ``lx``), and a value key (a metric). For robust plots over many
trials we summarize with the median and an inter-percentile band rather than the
mean, which is sensitive to BLAS warmup spikes and outliers.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

Band = tuple[list[float], list[float], list[float], list[float]]  # x, median, low, high


def median_band(
    rows: list[dict[str, object]],
    group_key: str,
    x_key: str,
    value_key: str,
    group_order: tuple[str, ...] | list[str],
    low: float = 25.0,
    high: float = 75.0,
    drop_nan: bool = True,
    ci: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Band]:
    """Group ``rows`` by ``(group, x)`` and summarize ``value_key`` per x.

    Returns ``{group: (xs, medians, lows, highs)}`` where ``lows``/``highs`` are
    the ``low``/``high`` percentiles across trials at each x (the default IQR
    band). If ``ci`` is set (e.g. ``ci=90``) the band is instead a bootstrap
    confidence interval for the median. Groups are returned in ``group_order``
    (skipping any with no data).
    """
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if drop_nan and np.isnan(value):
            continue
        buckets[(str(row[group_key]), float(row[x_key]))].append(value)

    out: dict[str, Band] = {}
    for group in group_order:
        xs = sorted({x for (g, x) in buckets if g == group})
        if not xs:
            continue
        medians, lows, highs = [], [], []
        for x in xs:
            vals = np.asarray(buckets[(group, x)], dtype=float)
            medians.append(float(np.median(vals)))
            if ci is None:
                lows.append(float(np.percentile(vals, low)))
                highs.append(float(np.percentile(vals, high)))
            else:
                _, lo, hi = bootstrap_ci(vals, n_boot=n_boot, ci=ci, seed=seed)
                lows.append(lo)
                highs.append(hi)
        out[group] = (xs, medians, lows, highs)
    return out


def bootstrap_ci(values, stat=np.median, n_boot: int = 2000, ci: float = 90.0, seed: int = 0):
    """Bootstrap confidence interval for ``stat`` of ``values`` (NaNs dropped).

    Returns ``(point, lo, hi)`` at the central ``ci`` percent level -- use for
    paper-quality bands on a median speedup / excess error (resample the paired
    per-instance values and report the interval, not just a single estimate).
    """
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(a))
    rng = np.random.default_rng(seed)
    boot = stat(rng.choice(a, size=(n_boot, a.size), replace=True), axis=1)
    half = (100.0 - ci) / 2.0
    return point, float(np.percentile(boot, half)), float(np.percentile(boot, 100.0 - half))


def failure_rate(values, threshold: float, *, above: bool = True) -> float:
    """Fraction of (non-NaN) ``values`` past ``threshold`` (``above`` -> ``>=``).

    For the failure-probability stress test, e.g. ``Pr[eps_rand/eps_det > 1+delta]``
    or ``Pr[T_rand > T_det]``.
    """
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a >= threshold)) if above else float(np.mean(a <= threshold))
