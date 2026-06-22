"""Wall-clock + peak-memory measurement for experiment drivers (roadmap 0C)."""

from __future__ import annotations

import resource
import sys
from contextlib import contextmanager
from time import perf_counter


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB (high-water mark since start).

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux; normalize to MiB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024


@contextmanager
def timed():
    """Time a block and capture peak RSS.

    ``with timed() as rec: ...`` then read ``rec['wall_s']`` and
    ``rec['peak_rss_mb']`` afterwards.
    """
    rec: dict[str, float] = {}
    t0 = perf_counter()
    try:
        yield rec
    finally:
        rec["wall_s"] = perf_counter() - t0
        rec["peak_rss_mb"] = peak_rss_mb()
