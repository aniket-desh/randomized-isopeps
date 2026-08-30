import json

import numpy as np
import pytest

from rand_isopeps.campaign.aggregate import MissingDataError
from rand_isopeps.campaign import gpu_gate, gpu_pilot
from rand_isopeps.campaign.gpu_gate import require_gpu_gate, validate_gpu_pilot


def _kernel(backend, error=0.0):
    return {
        "experiment": "gpu_pilot",
        "backend": backend,
        "status": "ok",
        "parity_error": error,
        "parity_passed": True,
    }


def _physics(backend, energy):
    return {
        "experiment": "physics",
        "backend": backend,
        "status": "ok",
        "problem": {"study": "gpu_pilot", "lx": 2},
        "task_method": {"name": "peps_sketch"},
        "seeds": {"problem": 7},
        "iteration": 1,
        "energies": [energy],
        "ground_energy_errors": [energy + 1.0],
    }


def test_gpu_gate_requires_both_kernel_and_physics_parity():
    rows = [_kernel("cupy", 1e-13), _physics("numpy", -1.0), _physics("cupy", -1.0)]
    gate = validate_gpu_pilot(rows, tolerance=1e-12)
    assert gate["passed"] is True
    assert gate["physics_pairs"] == 1


def test_power_parity_checks_the_adjoint_chain(monkeypatch):
    class Operator:
        def __init__(self, adjoint_scale):
            self.adjoint_scale = adjoint_scale

        def matvec_mps(self, value):
            return 2.0 * np.asarray(value)

        def rmatvec_mps(self, value):
            return self.adjoint_scale * np.asarray(value)

    monkeypatch.setattr(gpu_pilot, "mps_to_vector", np.asarray)
    cpu = Operator(3.0)
    device = Operator(4.0)
    probe = np.asarray([1.0, 2.0])

    assert gpu_pilot._power_parity_error(cpu, device, probe, probe, 0) == 0.0
    assert gpu_pilot._power_parity_error(cpu, device, probe, probe, 1) > 0.3


def test_gpu_gate_fails_closed_on_missing_or_bad_pair():
    with pytest.raises(MissingDataError, match="paired numpy"):
        validate_gpu_pilot([_kernel("cupy"), _physics("cupy", -1.0)])
    with pytest.raises(RuntimeError, match="exceeded"):
        validate_gpu_pilot(
            [_kernel("cupy"), _physics("numpy", -1.0), _physics("cupy", -0.9)],
            tolerance=1e-3,
        )


def test_gpu_gate_rejects_powered_rows_without_full_chain_evidence():
    kernel = _kernel("cupy")
    kernel["n_power"] = 1
    with pytest.raises(RuntimeError, match="full power-chain"):
        validate_gpu_pilot([
            kernel,
            _physics("numpy", -1.0),
            _physics("cupy", -1.0),
        ])


def test_gpu_gate_rejects_failed_and_missing_expected_tasks():
    rows = [_kernel("cupy"), _physics("numpy", -1.0), _physics("cupy", -1.0)]
    rows[0].update({"task_id": "kernel"})
    rows[1].update({"task_id": "cpu"})
    rows[2].update({"task_id": "gpu"})
    expected = [{"task_id": name} for name in ("kernel", "cpu", "gpu", "missing")]
    with pytest.raises(MissingDataError, match="1 gpu pilot tasks"):
        validate_gpu_pilot(rows, expected_tasks=expected)
    rows[0]["status"] = "failed"
    with pytest.raises(RuntimeError, match="pilot records failed"):
        validate_gpu_pilot(rows)


def test_gpu_gate_ignores_results_from_an_old_manifest_revision():
    rows = [_kernel("cupy"), _physics("numpy", -1.0), _physics("cupy", -1.0)]
    for row, task_id in zip(rows, ("kernel", "cpu", "gpu")):
        row["task_id"] = task_id
    old = _kernel("cupy")
    old["task_id"] = "old-kernel"

    gate = validate_gpu_pilot(
        [*rows, old],
        expected_tasks=[{"task_id": task_id} for task_id in ("kernel", "cpu", "gpu")],
    )

    assert gate["expected_task_ids"] == ["cpu", "gpu", "kernel"]
    assert gate["kernel_rows"] == 1


def test_gpu_gate_binds_exact_cupy_and_cuda_runtime(monkeypatch, tmp_path):
    current = {
        "name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": [8, 0],
        "cupy_version": "14.0.0",
        "cuda_runtime_version": 13000,
    }
    monkeypatch.setattr(gpu_gate, "_current_device", lambda: dict(current))
    payload = {
        "schema_version": "gpu_gate_v1",
        "passed": True,
        "campaign_code_revision": gpu_gate.campaign_code_revision(),
        "campaign_schema_versions": ["paper_campaign_v1"],
        "devices": [dict(current)],
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    task = {
        "schema_version": "paper_campaign_v1",
        "experiment": "physics",
        "backend": "cupy",
        "problem": {"study": "bond_sweep"},
    }

    require_gpu_gate(task, path)
    payload["devices"][0]["cupy_version"] = "13.6.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CuPy"):
        require_gpu_gate(task, path)
