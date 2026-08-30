"""executed local and global column methods with shared accuracy metrics."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from rand_isopeps.column.local_moses import local_column_qr
from rand_isopeps.physics.measurements import state_infidelity

from .sketches import sketch_matrix


def is_local_method(name: str) -> bool:
    return name.startswith("local_") or name in {
        "sequential_moses",
        "sequential_moses_riemannian",
    }


def _spectral_metrics(matrix, approximation, rank: int) -> dict[str, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    norm = max(float(np.linalg.norm(singular_values)), np.finfo(float).tiny)
    error = float(np.linalg.norm(matrix - approximation) / norm)
    floor = float(np.linalg.norm(singular_values[int(rank):]) / norm)
    return {
        "projection_error": error,
        "spectral_floor": floor,
        "projection_excess": float(np.sqrt(max(error**2 - floor**2, 0.0))),
    }


def run_synthetic_method(task: dict) -> dict:
    """run one local or global method on a paired synthetic column."""
    from .column_cases import synthetic_column

    problem, method = task["problem"], task["method"]
    operator, supplied_matrix, factor_dims = synthetic_column(
        problem, int(task["seeds"]["problem"])
    )
    matrix = supplied_matrix
    if matrix is None:
        matrix = operator.materialize()
    name = str(method["name"])
    rank = min(int(method.get("eta", method.get("target_rank", 4))), min(matrix.shape))
    started = perf_counter()
    if name == "sequential_moses_riemannian":
        raise ValueError("riemannian moses requires a physical peps column")
    if is_local_method(name):
        if operator is None:
            raise ValueError("sequential column methods require an mpo column")
        randomized = name in {"local_gaussian", "local_sparsestack"}
        sketch = name.removeprefix("local_") if randomized else "gaussian"
        result = local_column_qr(
            operator,
            rank,
            randomized=randomized,
            oversample=int(method.get("oversample", 4)),
            n_power=int(method.get("n_power", 0)),
            sketch=sketch,
            rng=np.random.default_rng(int(task["seeds"]["sketch"])),
            reference=matrix,
        )
        metrics = result.as_dict()
        metrics.update({
            "method": name,
            "projection_error": float(result.rel_error),
            "spectral_floor": float(np.linalg.norm(np.linalg.svd(matrix, compute_uv=False)[rank:])
                                      / np.linalg.norm(matrix)),
            "q_rank": int(result.q_cols),
        })
        metrics["projection_excess"] = float(np.sqrt(max(
            metrics["projection_error"] ** 2 - metrics["spectral_floor"] ** 2,
            0.0,
        )))
    else:
        ell = min(int(method.get("ell", rank + 4)), matrix.shape[1])
        omega = sketch_matrix(
            matrix.shape[1],
            ell,
            method,
            np.random.default_rng(int(task["seeds"]["sketch"])),
            factor_dims=factor_dims,
            dtype=matrix.dtype,
        )
        sampled = matrix @ omega
        for _ in range(int(method.get("n_power", 0))):
            sampled = matrix @ (matrix.conj().T @ sampled)
        q, _ = np.linalg.qr(sampled, mode="reduced")
        projected = q.conj().T @ matrix
        u_small, singular_small, vh_small = np.linalg.svd(projected, full_matrices=False)
        used_rank = min(rank, singular_small.size)
        q_rank = q @ u_small[:, :used_rank]
        approximation = (q_rank * singular_small[:used_rank]) @ vh_small[:used_rank]
        range_error = float(
            np.linalg.norm(matrix - q @ projected) / max(np.linalg.norm(matrix), 1e-300)
        )
        embedded = np.linalg.svd(
            np.linalg.svd(matrix, full_matrices=False)[2][:rank] @ omega,
            compute_uv=False,
        )
        metrics = {
            "method": name,
            "ell": ell,
            "effective_ell": int(omega.shape[1]),
            "chi_sk": int(method.get("chi_sk", 0)),
            "range_rank": int(q.shape[1]),
            "q_rank": int(q_rank.shape[1]),
            "range_error": range_error,
            "isometry_defect": float(
                np.linalg.norm(q_rank.conj().T @ q_rank - np.eye(q_rank.shape[1]))
            ),
            "osi_sigma_min": float(embedded[-1] ** 2) if embedded.size else 0.0,
            **_spectral_metrics(matrix, approximation, rank),
        }
    return {
        "lx": int(problem["lx"]),
        "family": str(problem.get("family", "gaussian")),
        "state": str(problem.get("state", problem.get("family", "gaussian"))),
        "study": str(task["measurement"].get("study", "synthetic")),
        "decay": float(problem.get("decay", 0.0)),
        "mpo_bond": int(problem.get("mpo_bond", 0)),
        "n_out": int(matrix.shape[0]),
        "n_in": int(matrix.shape[1]),
        "target_rank": rank,
        "runtime_s": float(perf_counter() - started),
        **metrics,
    }


def _local_move(psi, j: int, split: str, method: dict, seed: int):
    from rand_isopeps.real_isotns.instrument import MosesRandConfig, MosesStats
    from rand_isopeps.real_isotns.moses_move import RandSVD, moses_move

    name = str(method["name"])
    config = None
    if name in {"local_gaussian", "local_sparsestack"}:
        random_svd = RandSVD(
            sketch=name.removeprefix("local_"),
            oversample=int(method.get("oversample", 4)),
            n_power=int(method.get("n_power", 0)),
            rng=np.random.default_rng(int(seed)),
        )
        config = MosesRandConfig(svd2=random_svd)
    stats = MosesStats()
    out = psi.copy()
    started = perf_counter()
    errors = moses_move(
        out,
        int(j),
        int(method.get("chi", 8)),
        int(method.get("eta", 4)),
        float(method.get("cutoff", 1e-10)),
        int(method.get("ndis", 0)),
        orientation="col",
        sweep="up",
        split=split,
        renorm=False,
        rand=config,
        stats=stats,
        absorb_max_bond=method.get("absorption_bond"),
        absorb_cutoff=float(method.get("absorption_cutoff", 1e-10)),
        disentangler=str(method.get("disentangler", "altmin")),
    )
    return out, {
        "method": name,
        "local_error_squared": float(np.sum(np.asarray(errors, dtype=float) ** 2)),
        "factor_runtime_s": float(perf_counter() - started),
        "local_svd_count": len(stats.records),
        "largest_svd_dimension": max((min(row.m, row.n) for row in stats.records), default=0),
    }


def _global_move(psi, column, j: int, split: str, method: dict, seeds: dict, measurement: dict):
    from rand_isopeps.column.bounded_residual import (
        bounded_residual_column_qr,
        score_projection_error,
    )
    from rand_isopeps.real_isotns.column_bridge import (
        compress_column,
        insert_column_factorization,
    )

    name = str(method["name"])
    sketch_kind = name.removeprefix("global_")
    valid_sketches = {"rmps", "kron", "gaussian", "rademacher", "sparsestack"}
    if sketch_kind not in valid_sketches:
        raise ValueError(f"unsupported global method: {name!r}")
    factor = bounded_residual_column_qr(
        column,
        ell=min(int(method.get("ell", 8)), column.n_in),
        eta=int(method.get("eta", 4)),
        kappa=int(method.get("kappa", 2)),
        chi_sk=int(method.get("chi_sk", 8)),
        sketch_kind=sketch_kind,
        n_power=int(method.get("n_power", 0)),
        ndis=int(method.get("ndis", 0)),
        rng=np.random.default_rng(int(seeds["sketch"])),
        dense_oracle_max_elements=int(measurement.get("dense_oracle_max_elements", 2_000_000)),
    )
    score = score_projection_error(
        column,
        factor,
        n_probes=int(measurement.get("score_probes", 32)),
        chi_score=int(measurement.get("score_chi", 8)),
        rng=np.random.default_rng(int(seeds["score"])),
        confidence=float(measurement.get("confidence", 90.0)),
        n_bootstrap=int(measurement.get("bootstrap", 500)),
    )
    started = perf_counter()
    out = insert_column_factorization(
        psi,
        int(j),
        factor.q_cores,
        factor.residual_cores,
        split=split,
        inplace=False,
    )
    next_column = int(j) + (1 if split == "right" else -1)
    out, absorption = compress_column(
        out,
        next_column,
        max_bond=method.get("absorption_bond"),
        cutoff=float(method.get("absorption_cutoff", 1e-10)),
        inplace=True,
    )
    return out, {
        "method": name,
        "projection_error": float(score.estimate),
        "projection_standard_error": float(score.standard_error),
        "projection_ci_low": float(score.ci_low),
        "projection_ci_high": float(score.ci_high),
        "projection_error_dense": float(factor.projection_error_dense),
        "spectral_floor": float(factor.spectral_tail_dense),
        "projection_excess": float(factor.projection_excess_dense),
        "sampled_reconstruction_error": float(factor.reconstruction_error),
        "isometry_defect": float(factor.delta_global),
        "isometry_bound": float(factor.delta_global_bound),
        "q_rank": int(factor.q_flat_rank),
        "q_vertical_bond": int(factor.max_q_vertical),
        "residual_vertical_bond": int(factor.max_residual_vertical),
        "factor_runtime_s": float(factor.runtime_s),
        "absorption_runtime_s": float(perf_counter() - started),
        "matrix_mps_products": int(factor.matrix_mps_products),
        **absorption,
    }


def move_physical_column(
    psi,
    column,
    j: int,
    split: str,
    method: dict,
    seeds: dict,
    measurement: dict,
):
    """execute one requested move and return the standard peps plus metrics."""
    name = str(method["name"])
    if is_local_method(name):
        return _local_move(psi, j, split, method, int(seeds["sketch"]))
    return _global_move(psi, column, j, split, method, seeds, measurement)


def _boundary_metrics(reference, moved, hamiltonian, bond: int) -> dict:
    from rand_isopeps.real_isotns.tebd2 import energy

    overlap = complex((reference.H | moved).contract_boundary(max_bond=int(bond)))
    norm_before = float(np.real(
        (reference.H | reference).contract_boundary(max_bond=int(bond))
    ))
    norm_after = float(np.real(
        (moved.H | moved).contract_boundary(max_bond=int(bond))
    ))
    result = {
        "state_infidelity": max(
            1.0 - abs(overlap) ** 2 / max(norm_before * norm_after, 1e-300),
            0.0,
        ),
        "norm_drift": (
            math.sqrt(max(norm_after, 0.0) / max(norm_before, 1e-300)) - 1.0
        ),
    }
    if hamiltonian is not None:
        result["energy"] = energy(moved, hamiltonian, max_bond=int(bond))
    return result


def physical_state_metrics(
    reference,
    moved,
    hamiltonian,
    problem: dict,
    measurement: dict | None = None,
) -> dict:
    """score a moved peps exactly when small and by contraction otherwise."""
    from rand_isopeps.physics import dense_state_vector, rayleigh_residual, sparse_hamiltonian

    lx, ly = int(problem["lx"]), int(problem["ly"])
    if lx * ly <= int(problem.get("dense_oracle_max_sites", 16)):
        before, before_log = dense_state_vector(reference)
        after, after_log = dense_state_vector(moved)
        metrics = {
            "state_infidelity": state_infidelity(before, after),
            "norm_drift": float(math.expm1(math.log(10.0) * (after_log - before_log))),
        }
        if hamiltonian is not None:
            h = sparse_hamiltonian(hamiltonian, lx, ly)
            before_energy = rayleigh_residual(h, before)["energy"]
            after_result = rayleigh_residual(h, after)
            metrics.update({
                "energy": float(after_result["energy"]),
                "energy_change": float(after_result["energy"] - before_energy),
                "residual_norm": float(after_result["residual_norm"]),
                "variance": float(after_result["variance"]),
            })
        return metrics
    settings = measurement or {}
    default_high = int(problem.get("measurement_bond", 64))
    bonds = tuple(int(value) for value in settings.get(
        "measurement_bonds", (max(default_high // 2, 1), default_high)
    ))
    if len(bonds) != 2 or bonds[0] >= bonds[1]:
        raise ValueError("large-column measurements require increasing b,2b bonds")
    estimates = [
        _boundary_metrics(reference, moved, hamiltonian, bond) for bond in bonds
    ]
    keys = ("state_infidelity", "norm_drift")
    if hamiltonian is not None:
        keys += ("energy",)
    differences = {
        key: abs(float(estimates[1][key]) - float(estimates[0][key]))
        for key in keys
    }
    tolerance = float(settings.get("measurement_convergence_tolerance", 1e-5))
    return {
        **estimates[1],
        "measurement_bonds": list(bonds),
        "measurement_estimates": estimates,
        "measurement_differences": differences,
        "measurement_max_difference": max(differences.values(), default=0.0),
        "measurement_convergence_tolerance": tolerance,
        "measurement_converged": bool(
            max(differences.values(), default=0.0) <= tolerance
        ),
    }
