"""coarse-grained parallel helpers for experiment sweeps."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

from threadpoolctl import threadpool_limits

T = TypeVar("T")
R = TypeVar("R")


def auto_worker_count(requested: int | None) -> int:
    """Resolve a user worker request.

    `0` or `None` means a conservative local default: leave one CPU free and
    cap at four workers. These experiments use BLAS-heavy kernels, so more
    processes are not always faster on a laptop.
    """

    if requested is not None and requested > 0:
        return requested
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu - 1))


def flatten(list_of_lists: Iterable[Sequence[R]]) -> list[R]:
    out: list[R] = []
    for rows in list_of_lists:
        out.extend(rows)
    return out


def run_parallel(
    func: Callable[[T], R],
    tasks: Sequence[T],
    workers: int,
) -> list[R]:
    if workers <= 1 or len(tasks) <= 1:
        return [func(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(func, tasks))
    except PermissionError:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(func, tasks))


def with_blas_threads(num_threads: int | None):
    """Return a threadpoolctl context manager for worker kernels."""

    if num_threads is None or num_threads <= 0:
        return threadpool_limits(limits=None)
    return threadpool_limits(limits=num_threads)
