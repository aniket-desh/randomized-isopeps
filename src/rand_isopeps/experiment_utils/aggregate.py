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
) -> dict[str, Band]:
    """Group ``rows`` by ``(group, x)`` and summarize ``value_key`` per x.

    Returns ``{group: (xs, medians, lows, highs)}`` where ``lows``/``highs`` are
    the ``low``/``high`` percentiles across trials at each x. Groups are returned
    in ``group_order`` (skipping any with no data).
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
            lows.append(float(np.percentile(vals, low)))
            highs.append(float(np.percentile(vals, high)))
        out[group] = (xs, medians, lows, highs)
    return out
