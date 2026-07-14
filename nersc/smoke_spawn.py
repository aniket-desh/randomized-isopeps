#!/usr/bin/env python3
"""Exercise a spawned worker pool after parent-side threaded BLAS setup."""

import numpy as np
from threadpoolctl import threadpool_limits

from rand_isopeps.experiment_utils.parallel import run_parallel


def square(value):
    return value * value


def main():
    rng = np.random.default_rng(7)
    with threadpool_limits(limits=2):
        np.linalg.svd(rng.standard_normal((128, 128)), compute_uv=False)
    result = run_parallel(square, [1, 2, 3, 4], workers=2)
    expected = [1, 4, 9, 16]
    if result != expected:
        raise RuntimeError(f"spawn smoke mismatch: {result} != {expected}")
    print("spawn-after-BLAS smoke passed")


if __name__ == "__main__":
    main()
