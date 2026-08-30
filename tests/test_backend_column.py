from __future__ import annotations

import numpy as np
import pytest

from rand_isopeps.backend import array_namespace, asarray, to_numpy
from rand_isopeps.column.bounded_residual import (
    _combined_sampled_cores,
    _range_block_mps,
    block_mps_to_matrix,
    bounded_residual_column_qr,
    factor_sampled_block_mps,
    materialize_q,
)
from rand_isopeps.column.operator import ColumnOperator, random_column_operator
from rand_isopeps.linalg.rmps_sketch import rmps_cores


def _cupy_or_skip():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
        cupy.zeros(1)
    except Exception as exc:
        pytest.skip(f"CuPy cannot use a CUDA device: {exc}")
    return cupy


def _operator_and_probe():
    operator = random_column_operator(
        lx=3,
        in_dim=2,
        out_dim=3,
        mpo_bond=2,
        rng=np.random.default_rng(11),
        ensemble="decay",
    )
    probe = rmps_cores(
        operator.input_dims,
        chi_sk=2,
        rng=np.random.default_rng(17),
        complex_valued=True,
    )
    return operator, probe


def test_numpy_column_products_match_dense_operator():
    operator, probe = _operator_and_probe()
    vector = block_mps_to_matrix(probe)[:, 0]
    found = block_mps_to_matrix(operator.matvec_mps(probe))[:, 0]
    expected = operator.materialize() @ vector
    assert np.allclose(found, expected)

    output_probe = rmps_cores(
        operator.output_dims,
        chi_sk=2,
        rng=np.random.default_rng(23),
        complex_valued=True,
    )
    output_vector = block_mps_to_matrix(output_probe)[:, 0]
    found_adjoint = block_mps_to_matrix(operator.rmatvec_mps(output_probe))[:, 0]
    expected_adjoint = operator.materialize().conj().T @ output_vector
    assert np.allclose(found_adjoint, expected_adjoint)


def test_cupy_column_and_range_products_match_numpy():
    cupy = _cupy_or_skip()
    operator, probe = _operator_and_probe()
    device_operator = ColumnOperator([cupy.asarray(core) for core in operator.cores])

    cpu_product = operator.matvec_mps(probe)
    device_product = device_operator.matvec_mps(probe)
    for cpu_core, device_core in zip(cpu_product, device_product):
        assert np.allclose(to_numpy(device_core), cpu_core, rtol=2e-12, atol=2e-12)

    output_probe = rmps_cores(
        operator.output_dims,
        chi_sk=2,
        rng=np.random.default_rng(29),
        complex_valued=True,
    )
    cpu_adjoint = operator.rmatvec_mps(output_probe)
    device_adjoint = device_operator.rmatvec_mps(output_probe)
    for cpu_core, device_core in zip(cpu_adjoint, device_adjoint):
        assert np.allclose(to_numpy(device_core), cpu_core, rtol=2e-12, atol=2e-12)

    cpu_sampled, *_ = _range_block_mps(
        operator, 3, 2, 1, np.random.default_rng(31), True
    )
    device_sampled, *_ = _range_block_mps(
        device_operator, 3, 2, 1, np.random.default_rng(31), True
    )
    for cpu_core, device_core in zip(cpu_sampled, device_sampled):
        assert np.allclose(to_numpy(device_core), cpu_core, rtol=3e-12, atol=3e-12)

    assert np.allclose(to_numpy(device_operator.materialize()), operator.materialize())
    assert device_operator.frobenius_norm() == pytest.approx(
        operator.frobenius_norm(), rel=2e-12
    )


@pytest.mark.parametrize("sketch_kind", ["gaussian", "rademacher", "sparsestack"])
def test_dense_validation_sketches_are_repeatable(sketch_kind):
    operator, _ = _operator_and_probe()
    options = dict(
        ell=3,
        eta=2,
        kappa=1,
        chi_sk=2,
        sketch_kind=sketch_kind,
        n_power=0,
        ndis=0,
        dense_oracle_max_elements=100_000,
    )
    first = bounded_residual_column_qr(
        operator, rng=np.random.default_rng(37), **options
    )
    second = bounded_residual_column_qr(
        operator, rng=np.random.default_rng(37), **options
    )
    expected_width = 4 if sketch_kind == "sparsestack" else 3
    assert first.ell == expected_width
    assert first.reconstruction_error == pytest.approx(second.reconstruction_error)
    assert first.projection_error_dense == pytest.approx(second.projection_error_dense)
    assert np.isfinite(first.delta_global)


def test_cupy_sampling_and_factorization_preserve_bounded_result():
    cupy = _cupy_or_skip()
    operator, _ = _operator_and_probe()
    like = cupy.zeros(1)
    device_operator = ColumnOperator([
        asarray(core, like=like) for core in operator.cores
    ])
    options = dict(
        ell=3,
        eta=2,
        kappa=1,
        chi_sk=2,
        n_power=0,
        ndis=0,
        dense_oracle_max_elements=100_000,
    )
    cpu = bounded_residual_column_qr(
        operator, rng=np.random.default_rng(37), **options
    )
    device = bounded_residual_column_qr(
        device_operator, rng=np.random.default_rng(37), **options
    )
    assert device.reconstruction_error == pytest.approx(
        cpu.reconstruction_error, rel=5e-10, abs=5e-12
    )
    assert device.projection_error_dense == pytest.approx(
        cpu.projection_error_dense, rel=5e-10, abs=5e-12
    )
    assert array_namespace(device.q_cores) is cupy
    assert array_namespace(device.residual_cores) is cupy
    cpu_projected = materialize_q(cpu.q_cores) @ cpu.residual_operator.materialize()
    device_projected = (
        materialize_q(device.q_cores) @ device.residual_operator.materialize()
    )
    assert np.allclose(
        to_numpy(device_projected), cpu_projected, rtol=5e-10, atol=5e-11
    )


def test_cupy_factorization_matches_cpu_reconstruction():
    cupy = _cupy_or_skip()
    operator, _ = _operator_and_probe()
    sampled, *_ = _range_block_mps(
        operator, 3, 2, 0, np.random.default_rng(41), True
    )
    device_sampled = [cupy.asarray(core) for core in sampled]
    cpu_q, cpu_r, *_ = factor_sampled_block_mps(
        sampled, eta=2, kappa=2, ndis=0, rng=np.random.default_rng(43)
    )
    device_q, device_r, *_ = factor_sampled_block_mps(
        device_sampled, eta=2, kappa=2, ndis=0, rng=np.random.default_rng(43)
    )
    cpu_reconstruction = block_mps_to_matrix(
        _combined_sampled_cores(cpu_q, cpu_r)
    )
    device_reconstruction = block_mps_to_matrix(
        _combined_sampled_cores(device_q, device_r)
    )
    assert array_namespace(device_q, device_r) is cupy
    assert np.allclose(
        to_numpy(device_reconstruction),
        cpu_reconstruction,
        rtol=5e-10,
        atol=5e-11,
    )


def test_cupy_factorization_rejects_disentangling():
    cupy = _cupy_or_skip()
    operator, _ = _operator_and_probe()
    sampled, *_ = _range_block_mps(
        operator, 2, 2, 0, np.random.default_rng(47), True
    )
    with pytest.raises(ValueError, match="requires ndis=0"):
        factor_sampled_block_mps(
            [cupy.asarray(core) for core in sampled],
            eta=2,
            kappa=2,
            ndis=1,
        )


def test_cupy_quimb_column_bridge_preserves_device_arrays():
    cupy = _cupy_or_skip()
    qtn = pytest.importorskip("quimb.tensor")
    from rand_isopeps.real_isotns.column_bridge import (
        compress_column,
        extract_column,
        insert_column_factorization,
    )

    psi = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=53)
    psi.apply_to_arrays(cupy.asarray)
    operator = extract_column(psi, 0, split="right")
    assert array_namespace(operator.cores) is cupy
    result = bounded_residual_column_qr(
        operator,
        ell=operator.n_in,
        eta=8,
        kappa=1,
        chi_sk=2,
        ndis=0,
        rng=np.random.default_rng(59),
    )
    moved = insert_column_factorization(
        psi,
        0,
        result.q_cores,
        result.residual_cores,
        split="right",
    )
    moved, _ = compress_column(
        moved, 1, max_bond=None, cutoff=0.0, inplace=True
    )
    assert all(array_namespace(tensor.data) is cupy for tensor in moved.tensors)


def test_cupy_physics_iteration_keeps_state_and_gates_on_device():
    cupy = _cupy_or_skip()
    qtn = pytest.importorskip("quimb.tensor")
    from rand_isopeps.real_isotns.physics_loop import tebd_iteration
    from rand_isopeps.real_isotns.tebd2 import tfi_ham

    psi = qtn.PEPS.product_state(
        {
            (0, 0): np.array([1.0, 0.0]),
            (0, 1): np.array([0.0, 1.0]),
        }
    )
    psi.apply_to_arrays(cupy.asarray)
    moved, metrics = tebd_iteration(
        psi,
        tfi_ham(1, 2),
        0.01,
        column_backend="none",
        gate_options={"max_bond": None, "cutoff": 0.0},
        trotter_order=1,
    )
    assert all(array_namespace(tensor.data) is cupy for tensor in moved.tensors)
    assert metrics["update_runtime_s"] >= 0.0
