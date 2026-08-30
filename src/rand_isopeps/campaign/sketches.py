"""small-system sketch comparisons used by the paper campaign."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from rand_isopeps.linalg.rmps_sketch import rmps_test_matrix
from rand_isopeps.linalg.sketches import SketchSpec, range_sample

from .seeds import derive_seed


def sketch_matrix(
    dimension: int,
    ell: int,
    method: dict,
    rng: np.random.Generator,
    *,
    factor_dims: tuple[int, ...],
    dtype=np.complex128,
):
    """materialize a validation sketch through the common sketch interface."""
    name = str(method["name"])
    if name in {"global_rmps", "rmps"}:
        spec = SketchSpec(
            kind="rmps",
            factor_dims=factor_dims,
            chi_sk=int(method.get("chi_sk", 1)),
        )
    elif name in {"global_kron", "kron"}:
        spec = SketchSpec(kind="rmps", factor_dims=factor_dims, chi_sk=1)
    elif name in {"global_sparsestack", "local_sparsestack", "sparsestack"}:
        spec = SketchSpec(kind="sparsestack", zeta=int(method.get("zeta", 4)))
    else:
        kind = name.removeprefix("global_").removeprefix("local_")
        if kind not in {"gaussian", "rademacher", "countsketch"}:
            raise ValueError(f"unsupported materialized sketch: {name!r}")
        spec = SketchSpec(kind=kind)
    identity = np.eye(int(dimension), dtype=dtype)
    return range_sample(identity, int(ell), rng, spec)


def _orthonormal_complement(vectors, rng: np.random.Generator):
    dimension, rank = vectors.shape
    draw = rng.standard_normal((dimension, dimension - rank))
    draw = draw + 1j * rng.standard_normal(draw.shape)
    draw -= vectors @ (vectors.conj().T @ draw)
    complement, _ = np.linalg.qr(draw, mode="reduced")
    return np.concatenate([vectors, complement], axis=1)


def test_subspace(
    factor_dims: tuple[int, ...],
    rank: int,
    family: str,
    rng: np.random.Generator,
):
    """construct a paired generic or product-aligned right subspace."""
    dimension = int(np.prod(factor_dims))
    rank = min(int(rank), dimension)
    if family == "haar":
        draw = rng.standard_normal((dimension, rank))
        draw = draw + 1j * rng.standard_normal(draw.shape)
        vectors, _ = np.linalg.qr(draw, mode="reduced")
        return vectors
    if family == "product_aligned":
        indices = np.linspace(0, dimension - 1, rank, dtype=int)
        return np.eye(dimension, dtype=complex)[:, indices]
    raise ValueError(f"unknown subspace family: {family!r}")


def gaussian_limit_metrics(task: dict) -> dict:
    """measure probe moments, osi, and downstream range quality."""
    problem = task["problem"]
    method = task["method"]
    factor_dims = tuple(int(value) for value in problem.get(
        "factor_dims", [int(problem.get("factor_dim", 2))] * int(problem["lx"])
    ))
    rank = int(problem.get("rank", 2))
    ell = min(int(method.get("ell", problem.get("ell", 8))), int(np.prod(factor_dims)))
    problem_rng = np.random.default_rng(int(task["seeds"]["problem"]))
    sketch_rng = np.random.default_rng(int(task["seeds"]["sketch"]))
    vectors = test_subspace(
        factor_dims,
        rank,
        str(problem.get("subspace", "haar")),
        problem_rng,
    )
    basis = _orthonormal_complement(vectors, problem_rng)
    dimension = basis.shape[0]
    singular_values = np.exp(-np.arange(dimension) / float(problem.get("decay", 4.0)))
    left_draw = problem_rng.standard_normal((dimension, dimension))
    left_draw = left_draw + 1j * problem_rng.standard_normal(left_draw.shape)
    left, _ = np.linalg.qr(left_draw)
    column = (left * singular_values) @ basis.conj().T
    omega = sketch_matrix(
        dimension,
        ell,
        method,
        sketch_rng,
        factor_dims=factor_dims,
        dtype=column.dtype,
    )
    embedded = vectors.conj().T @ omega
    injection = np.linalg.svd(embedded, compute_uv=False)
    sample = np.sqrt(omega.shape[1]) * (vectors[:, 0].conj() @ omega)
    second = float(np.mean(np.abs(sample) ** 2))
    fourth = float(np.mean(np.abs(sample) ** 4))
    moment_ratio = fourth / max(second**2, np.finfo(float).tiny)
    sampled = column @ omega
    q, _ = np.linalg.qr(sampled, mode="reduced")
    projection_error = float(
        np.linalg.norm(column - q @ (q.conj().T @ column)) / np.linalg.norm(column)
    )
    tail = float(
        np.sqrt(np.sum(singular_values[min(q.shape[1], dimension):] ** 2))
        / np.linalg.norm(singular_values)
    )
    return {
        "benchmark": "column_embedding",
        "lx": len(factor_dims),
        "factor_dims": list(factor_dims),
        "subspace": str(problem.get("subspace", "haar")),
        "rank": rank,
        "method": str(method["name"]),
        "chi_sk": int(method.get("chi_sk", 0)),
        "ell": ell,
        "effective_ell": int(omega.shape[1]),
        "chi_sk_over_lx": float(method.get("chi_sk", 0)) / len(factor_dims),
        "probe_second_moment": second,
        "probe_fourth_moment": fourth,
        "normalized_fourth_moment_error": abs(moment_ratio - 2.0) / 2.0,
        "osi_sigma_min": float(injection[-1] ** 2) if injection.size else 0.0,
        "projection_error": projection_error,
        "spectral_floor": tail,
        "projection_excess": max(projection_error - tail, 0.0),
        "embedding_failed": bool(not injection.size or injection[-1] ** 2 < 0.5),
    }


def walsh_subspace(tensor_order: int, rank: int) -> np.ndarray:
    """return the leading normalized walsh--hadamard columns."""
    dimension = 2 ** int(tensor_order)
    if not (1 <= int(rank) <= dimension):
        raise ValueError("rank must lie between one and the ambient dimension")
    return la.hadamard(dimension, dtype=float)[:, : int(rank)] / np.sqrt(dimension)


def _rmps_batch(
    tensor_order: int,
    chi_sk: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw a batch of real isotropic rmps vectors without python sample loops."""
    order = int(tensor_order)
    chi = int(chi_sk)
    if order < 2 or chi < 1 or count < 1:
        raise ValueError("tensor_order >= 2, chi_sk >= 1, and count >= 1 are required")
    boundary_std = chi ** -0.25
    interior_std = chi ** -0.5
    state = boundary_std * rng.standard_normal((count, 2, chi))
    for _ in range(1, order - 1):
        core = interior_std * rng.standard_normal((count, chi, 2, chi))
        state = np.einsum("bpl,bldr->bpdr", state, core, optimize=True)
        state = state.reshape(count, -1, chi)
    final = boundary_std * rng.standard_normal((count, chi, 2))
    return np.einsum("bpl,bld->bpd", state, final, optimize=True).reshape(count, -1)


def _quadratic_samples(
    problem: dict,
    method: dict,
    seed: int,
) -> np.ndarray:
    order = int(problem["tensor_order"])
    rank = int(problem["rank"])
    samples = int(problem["samples"])
    batch_size = int(problem.get("batch_size", 32))
    epsilon = float(problem["epsilon"])
    subspace = walsh_subspace(order, rank)
    rng = np.random.default_rng(int(seed))
    values = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        if method["name"] == "rmps":
            vectors = _rmps_batch(order, int(method["chi_sk"]), count, rng)
        elif method["name"] == "global_gaussian":
            vectors = rng.standard_normal((count, 2**order))
        else:
            raise ValueError(f"unsupported quadratic-form method: {method['name']!r}")
        overlap = vectors @ subspace
        values.append(
            np.sum(overlap * overlap, axis=1)
            + epsilon * np.sum(vectors * vectors, axis=1)
        )
    return np.concatenate(values)


def walsh_variance_metrics(task: dict, replicate: int) -> dict:
    """reproduce the quadratic-form panel of the rmps paper's figure 2."""
    problem, method = task["problem"], task["method"]
    seed = derive_seed(int(task["seeds"]["sketch"]), "trial", replicate)
    values = _quadratic_samples(problem, method, seed)
    dimension = 2 ** int(problem["tensor_order"])
    rank = int(problem["rank"])
    epsilon = float(problem["epsilon"])
    trace = rank + epsilon * dimension
    frobenius_squared = rank * (1.0 + epsilon) ** 2 + (dimension - rank) * epsilon**2
    sample_mean = float(np.mean(values))
    return {
        "benchmark": "rmps_figure2_variance",
        "method": str(method["name"]),
        "chi_sk": int(method.get("chi_sk", 0)),
        "tensor_order": int(problem["tensor_order"]),
        "rank": rank,
        "samples": int(problem["samples"]),
        "normalized_quadratic_variance": float(np.var(values, ddof=1) / trace**2),
        "quadratic_sample_mean": sample_mean,
        "quadratic_sample_m2": float(np.sum((values - sample_mean) ** 2)),
        "trace_value": float(trace),
        "gaussian_theory_variance": float(2.0 * frobenius_squared / trace**2),
        "replicate": int(replicate),
        "trial_seed": int(seed),
    }


def _apply_walsh_matrix(
    omega: np.ndarray,
    subspace: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    return subspace @ (subspace.T @ omega) + float(epsilon) * omega


def _psd_nystrom(omega: np.ndarray, y: np.ndarray, ridge: float):
    shifted = y + float(ridge) * omega
    gram = (omega.T @ shifted + shifted.T @ omega) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * max(float(eigenvalues[-1]), 1.0)
    keep = eigenvalues > tolerance
    factor = shifted @ (
        eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    )
    vectors, singular, _ = np.linalg.svd(factor, full_matrices=False)
    return vectors, np.maximum(singular**2 - float(ridge), 0.0)


def _gram_nystrom(omega: np.ndarray, y: np.ndarray):
    """dense gram-access form of the mps gram-nystrom algorithm."""
    gram = (omega.T @ omega + (omega.T @ omega).T) / 2.0
    cross = (omega.T @ y + y.T @ omega) / 2.0
    output_gram = (y.T @ y + (y.T @ y).T) / 2.0
    identity = np.eye(gram.shape[0])
    chol_gram = la.cholesky(gram, lower=False, check_finite=False)
    inverse = la.solve_triangular(
        chol_gram, identity, lower=False, check_finite=False
    )
    whitened = (inverse.T @ cross @ inverse)
    whitened = (whitened + whitened.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(whitened)
    root_epsilon = np.sqrt(np.finfo(float).eps)
    shift = max(0.0, root_epsilon * eigenvalues[-1] - eigenvalues[0])
    chol_cross = la.cholesky(
        (cross + shift * gram + (cross + shift * gram).T) / 2.0,
        lower=False,
        check_finite=False,
    )
    shifted_output = output_gram + 2.0 * shift * cross + shift**2 * gram
    inverse_cross = la.solve_triangular(
        chol_cross, identity, lower=False, check_finite=False
    )
    reduced = inverse_cross.T @ shifted_output @ inverse_cross
    reduced = (reduced + reduced.T) / 2.0
    shifted_values, rotations = np.linalg.eigh(reduced)
    order = np.argsort(shifted_values)[::-1]
    shifted_values = shifted_values[order]
    rotations = rotations[:, order]
    keep = shifted_values > root_epsilon * max(float(shifted_values[0]), 1.0)
    shifted_values = shifted_values[keep]
    rotations = rotations[:, keep]
    weights = la.solve_triangular(
        chol_cross, rotations, lower=False, check_finite=False
    ) / np.sqrt(shifted_values)[None, :]
    vectors = (y + shift * omega) @ weights
    return vectors, np.maximum(shifted_values - shift, 0.0)


def _relative_nuclear_error(
    subspace: np.ndarray,
    epsilon: float,
    vectors: np.ndarray,
    values: np.ndarray,
) -> float:
    """evaluate the nuclear residual through a small low-rank eigensystem."""
    positive = np.asarray(values) > np.finfo(float).eps
    vectors = np.asarray(vectors)[:, positive]
    values = np.asarray(values)[positive]
    joined = np.concatenate([subspace, vectors], axis=1)
    basis, _ = np.linalg.qr(joined, mode="reduced")
    projected_subspace = basis.T @ subspace
    projected_vectors = basis.T @ vectors
    update = projected_subspace @ projected_subspace.T
    update -= (projected_vectors * values[None, :]) @ projected_vectors.T
    residual_values = np.linalg.eigvalsh(
        (update + update.T) / 2.0 + float(epsilon) * np.eye(basis.shape[1])
    )
    dimension = subspace.shape[0]
    numerator = np.sum(np.abs(residual_values))
    numerator += (dimension - basis.shape[1]) * float(epsilon)
    denominator = subspace.shape[1] + float(epsilon) * dimension
    return float(numerator / denominator)


def walsh_nystrom_metrics(task: dict, replicate: int) -> dict:
    """reproduce one gaussian or mps gram-nystrom point from figure 2."""
    problem, method = task["problem"], task["method"]
    order = int(problem["tensor_order"])
    embedding_dim = int(problem["embedding_dim"])
    subspace = walsh_subspace(order, int(problem["rank"]))
    seed = derive_seed(int(task["seeds"]["sketch"]), "trial", replicate)
    rng = np.random.default_rng(seed)
    if method["name"] == "gaussian_nystrom":
        omega = rng.standard_normal((2**order, embedding_dim))
    elif method["name"] == "mps_gram_nystrom":
        omega = rmps_test_matrix(
            (2,) * order,
            embedding_dim,
            int(method["chi_sk"]),
            rng,
            normalize=False,
            complex_valued=False,
        )
    else:
        raise ValueError(f"unsupported Nystrom method: {method['name']!r}")
    epsilon = float(problem["epsilon"])
    y = _apply_walsh_matrix(omega, subspace, epsilon)
    if method["name"] == "gaussian_nystrom":
        vectors, values = _psd_nystrom(omega, y, float(problem["ridge"]))
    else:
        vectors, values = _gram_nystrom(omega, y)
    return {
        "benchmark": "rmps_figure2_nystrom",
        "method": str(method["name"]),
        "chi_sk": int(method.get("chi_sk", 0)),
        "tensor_order": order,
        "rank": int(problem["rank"]),
        "embedding_dim": embedding_dim,
        "relative_nuclear_error": _relative_nuclear_error(
            subspace, epsilon, vectors, values
        ),
        "replicate": int(replicate),
        "trial_seed": int(seed),
    }


def run_gaussian_limit(task: dict) -> list[dict]:
    """run a small batch of paired draws to avoid scheduler-sized millisecond jobs."""
    kind = str(task["problem"].get("kind", "sketch_embedding"))
    if kind == "walsh_variance":
        return [
            walsh_variance_metrics(task, replicate)
            for replicate in range(int(task["measurement"]["replicates"]))
        ]
    if kind == "walsh_nystrom":
        return [
            walsh_nystrom_metrics(task, replicate)
            for replicate in range(int(task["measurement"]["replicates"]))
        ]
    if kind != "sketch_embedding":
        raise ValueError(f"unsupported gaussian-limit problem: {kind!r}")
    rows = []
    for replicate in range(int(task["measurement"].get("replicates", 1))):
        configured = dict(task)
        seeds = dict(task["seeds"])
        seeds["problem"] = derive_seed(int(task["seeds"]["problem"]), "replicate", replicate)
        seeds["sketch"] = derive_seed(int(task["seeds"]["sketch"]), "replicate", replicate)
        seeds["score"] = derive_seed(int(task["seeds"]["score"]), "replicate", replicate)
        configured["seeds"] = seeds
        rows.append({
            **gaussian_limit_metrics(configured),
            "replicate": replicate,
            "seeds": seeds,
        })
    return rows
