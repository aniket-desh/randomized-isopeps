import numpy as np
from threadpoolctl import threadpool_limits

from rand_isopeps.experiment_utils.parallel import run_parallel, run_parallel_stream


def _square(value):
    return value * value


def test_run_parallel_spawn_pool():
    assert run_parallel(_square, [1, 2, 3, 4], workers=2) == [1, 4, 9, 16]


def test_run_parallel_stream_spawn_pool():
    assert sorted(run_parallel_stream(_square, [1, 2, 3, 4], workers=2)) == [1, 4, 9, 16]


def test_run_parallel_after_parent_blas_initialization():
    rng = np.random.default_rng(7)
    with threadpool_limits(limits=2):
        np.linalg.svd(rng.standard_normal((128, 128)), compute_uv=False)
    assert run_parallel(_square, [2, 3], workers=2) == [4, 9]
