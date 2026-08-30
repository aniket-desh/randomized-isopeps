"""small numpy/cupy dispatch helpers for maintained numerical kernels."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

import numpy as np


def _values(items: Iterable[Any]):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _values(item)
        else:
            yield item


def _is_cupy_value(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "cupy"


def _cupy():
    return importlib.import_module("cupy")


def array_namespace(*values: Any):
    """return cupy when any value is a cupy array, otherwise numpy."""
    return _cupy() if any(_is_cupy_value(value) for value in _values(values)) else np


def asarray(value: Any, *, like: Any | None = None, dtype=None):
    """convert with the backend selected by ``like`` or ``value``."""
    xp = array_namespace(value if like is None else like)
    return xp.asarray(value, dtype=dtype)


def to_numpy(value: Any) -> np.ndarray:
    """copy a device array to numpy and leave numpy inputs on the host."""
    xp = array_namespace(value)
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def synchronize(value: Any | None = None) -> None:
    """synchronize the current device stream selected by ``value``."""
    xp = array_namespace(value)
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def svd(value: Any, *, full_matrices: bool = False):
    """compute an svd with the input array's backend."""
    return array_namespace(value).linalg.svd(value, full_matrices=full_matrices)


def svdvals(value: Any):
    """compute singular values with the input array's backend."""
    return array_namespace(value).linalg.svd(value, compute_uv=False)
