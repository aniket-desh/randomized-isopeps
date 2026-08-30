from __future__ import annotations

import numpy as np
import pytest

import rand_isopeps.backend as backend


def _cupy_or_skip():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
        cupy.zeros(1)
    except Exception as exc:
        pytest.skip(f"CuPy cannot use a CUDA device: {exc}")
    return cupy


def test_numpy_backend_does_not_import_cupy(monkeypatch):
    def fail_import(_name):
        raise AssertionError("CPU dispatch imported CuPy")

    monkeypatch.setattr(backend.importlib, "import_module", fail_import)
    value = np.arange(6.0).reshape(2, 3)
    assert backend.array_namespace(value) is np
    assert backend.asarray(value) is value
    assert backend.to_numpy(value) is value
    backend.synchronize(value)


def test_numpy_svd_helpers_match_numpy():
    rng = np.random.default_rng(4)
    value = rng.standard_normal((7, 4))
    u, singular, vh = backend.svd(value)
    assert np.allclose((u * singular) @ vh, value)
    assert np.allclose(backend.svdvals(value), np.linalg.svd(value, compute_uv=False))


def test_cupy_backend_round_trip_and_svd():
    cupy = _cupy_or_skip()
    value = np.arange(12.0).reshape(4, 3)
    device = backend.asarray(value, like=cupy.zeros(1))
    assert backend.array_namespace(device) is cupy
    assert np.array_equal(backend.to_numpy(device), value)
    u, singular, vh = backend.svd(device)
    backend.synchronize(device)
    assert np.allclose(backend.to_numpy((u * singular) @ vh), value)
    assert np.allclose(
        backend.to_numpy(backend.svdvals(device)),
        np.linalg.svd(value, compute_uv=False),
    )
