"""accuracy-gated cpu/gpu crossover measurements."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.backend import synchronize, to_numpy
from rand_isopeps.column.operator import ColumnOperator, random_column_operator
from rand_isopeps.compression.mpo_mps_absorb import (
    canonicalize_mps_exact,
    max_mps_bond,
    mps_to_vector,
)
from rand_isopeps.linalg.rmps_sketch import rmps_cores


def _device_arrays(arrays, backend: str):
    if backend == "numpy":
        return [np.asarray(value) for value in arrays]
    if backend != "cupy":
        raise ValueError(f"unsupported backend: {backend!r}")
    import cupy as cp

    return [cp.asarray(value) for value in arrays]


def _power_output(operator, probe, n_power: int):
    output = canonicalize_mps_exact(operator.matvec_mps(probe))
    for _ in range(int(n_power)):
        output = canonicalize_mps_exact(operator.rmatvec_mps(output))
        output = canonicalize_mps_exact(operator.matvec_mps(output))
    return output


def _power_parity_error(cpu_operator, operator, probe_cpu, probe, n_power: int) -> float:
    """compare the complete matrix-power probe on cpu and the selected backend."""
    reference = mps_to_vector(_power_output(cpu_operator, probe_cpu, n_power))
    found = to_numpy(mps_to_vector(_power_output(operator, probe, n_power)))
    return float(
        np.linalg.norm(reference - found) / max(np.linalg.norm(reference), 1e-300)
    )


def run_gpu_pilot(task: dict) -> list[dict]:
    """time synchronized matrix-free products after a cpu parity check."""
    problem, method = task["problem"], task["method"]
    backend = str(task.get("backend", "numpy"))
    lx = int(problem["lx"])
    rng = np.random.default_rng(int(task["seeds"]["problem"]))
    cpu_operator = random_column_operator(
        lx,
        int(problem.get("in_dim", 2)),
        int(problem.get("out_dim", 2)),
        int(problem.get("mpo_bond", 4)),
        rng,
        ensemble=str(problem.get("family", "gaussian")),
    )
    operator = ColumnOperator(_device_arrays(cpu_operator.cores, backend))
    ell = int(method.get("ell", 8))
    probes_cpu = [
        canonicalize_mps_exact(
            rmps_cores(
                cpu_operator.input_dims,
                int(method.get("chi_sk", 8)),
                np.random.default_rng(int(task["seeds"]["sketch"]) + index),
                complex_valued=True,
            )
        )
        for index in range(ell)
    ]
    probes = [_device_arrays(probe, backend) for probe in probes_cpu]

    n_power = int(method.get("n_power", 0))
    parity_error = _power_parity_error(
        cpu_operator,
        operator,
        probes_cpu[0],
        probes[0],
        n_power,
    )
    tolerance = float(task["measurement"].get("parity_tolerance", 1e-11))
    if parity_error > tolerance:
        raise RuntimeError(
            f"backend parity error {parity_error:.3e} exceeds {tolerance:.3e}"
        )

    repeats = int(task["measurement"].get("repeats", 5))
    timings = []
    for _ in range(repeats + 1):
        synchronize(operator.cores)
        started = perf_counter()
        outputs = [_power_output(operator, probe, n_power) for probe in probes]
        synchronize(outputs)
        timings.append(float(perf_counter() - started))
    measured = timings[1:]
    row = {
        "backend": backend,
        "lx": lx,
        "mpo_bond": int(problem.get("mpo_bond", 4)),
        "ell": ell,
        "chi_sk": int(method.get("chi_sk", 8)),
        "n_power": n_power,
        "parity_error": parity_error,
        "parity_tolerance": tolerance,
        "parity_passed": True,
        "parity_scope": "full_power_chain",
        "parity_matrix_products": 1 + 2 * n_power,
        "power_rounding": "exact_two_sided_qr",
        "maximum_output_bond": max(max_mps_bond(output) for output in outputs),
        "timings_s": measured,
        "median_runtime_s": float(np.median(measured)),
        "minimum_runtime_s": float(np.min(measured)),
        "matrix_products": ell * (1 + 2 * n_power),
    }
    if backend == "cupy":
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        row["gpu_memory_bytes"] = int(cp.get_default_memory_pool().used_bytes())
        row["device"] = int(cp.cuda.runtime.getDevice())
        row["device_name"] = str(name)
        row["compute_capability"] = [
            int(properties.get("major", -1)),
            int(properties.get("minor", -1)),
        ]
        row["cupy_version"] = str(cp.__version__)
        row["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        row["cuda_driver_version"] = int(cp.cuda.runtime.driverGetVersion())
    return [row]
