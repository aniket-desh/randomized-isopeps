"""coarse-grained parallel helpers for experiment sweeps."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypeVar

from threadpoolctl import threadpool_limits

T = TypeVar("T")
R = TypeVar("R")


def total_ram_bytes() -> int | None:
    """Physical RAM in bytes (POSIX ``sysconf``), or ``None`` if unavailable."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, AttributeError, OSError):
        return None


def auto_worker_count(
    requested: int | None,
    est_bytes_per_task: int | None = None,
    mem_fraction: float = 0.6,
) -> int:
    """Resolve a worker request, optionally capping by memory.

    `0` or `None` means a conservative local default: leave one CPU free and cap
    at four workers (these kernels are BLAS-heavy, so more processes are not always
    faster on a laptop). When ``est_bytes_per_task`` is given, the count is *also*
    capped so ``workers * est_bytes_per_task <= mem_fraction * RAM`` -- defense in
    depth against N workers each holding a large dense column (the 108 GB OOM was
    4 workers x a ~27 GB column; see reports/incidents/2026-06-29).
    """

    if requested is not None and requested > 0:
        base = requested
    else:
        cpu = os.cpu_count() or 1
        base = max(1, min(4, cpu - 1))
    if est_bytes_per_task and est_bytes_per_task > 0:
        ram = total_ram_bytes()
        if ram:
            mem_cap = max(1, int((mem_fraction * ram) // est_bytes_per_task))
            base = min(base, mem_cap)
    return base


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


def run_parallel_stream(
    func: Callable[[T], R],
    tasks: Sequence[T],
    workers: int,
) -> Iterator[R]:
    """Like :func:`run_parallel` but *yields* results in completion order.

    Lets the caller persist each trial as it finishes (incremental CSV) so a crash
    mid-sweep keeps the rows already computed instead of losing the whole run.
    """
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield func(task)
        return
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(func, task) for task in tasks]
            for fut in as_completed(futures):
                yield fut.result()
    except PermissionError:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(func, task) for task in tasks]
            for fut in as_completed(futures):
                yield fut.result()


def with_blas_threads(num_threads: int | None):
    """Return a threadpoolctl context manager for worker kernels."""

    if num_threads is None or num_threads <= 0:
        return threadpool_limits(limits=None)
    return threadpool_limits(limits=num_threads)
