"""rmps-to-gaussian benchmark manifests."""

from __future__ import annotations

import itertools

import numpy as np

from .manifest_common import method_seeds, task


def build_gaussian_limit() -> list[dict]:
    """preserve the rmps Figure 2 grids and add column embeddings."""
    tasks = []
    for problem_index, (column_size, subspace, ell) in enumerate(itertools.product(
        range(2, 8), ("haar", "product_aligned"), (4, 8, 16)
    )):
        ambient_dim = 2**column_size
        if ell > ambient_dim:
            continue
        problem = {
            "kind": "sketch_embedding",
            "lx": column_size,
            "column_size": column_size,
            "factor_dim": 2,
            "factor_dims": [2] * column_size,
            "ambient_dim": ambient_dim,
            "subspace": subspace,
            "rank": min(4, ell),
            "subspace_rank": min(4, ell),
        }
        methods = [{"name": "global_gaussian", "ell": ell}]
        methods.extend(
            {
                "name": "rmps",
                "ell": ell,
                "chi_sk": chi_sk,
                "normalization": "isotropic",
            }
            for chi_sk in (1, 2, 4, 8, 16, 32, 64)
        )
        for method_index, method in enumerate(methods):
            tasks.append(task(
                "gaussian_limit",
                problem,
                method,
                method_seeds(problem_index, method_index, 0),
                {
                    "replicates": 60,
                    "metrics": [
                        "fourth_moment_ratio",
                        "osi_min",
                        "projection_error",
                        "error_to_gaussian",
                    ],
                    "primary_metric": "normalized_fourth_moment_error",
                },
                resources={"hardware": "cpu", "cpus": 1, "gpus": 0},
            ))

    variance_problem = {
        "kind": "walsh_variance",
        "tensor_order": 10,
        "ambient_dim": 2**10,
        "rank": 20,
        "epsilon": 1e-5,
        "samples": 500,
        "batch_size": 16,
    }
    variance_methods = [{"name": "global_gaussian"}]
    variance_methods.extend(
        {"name": "rmps", "chi_sk": chi_sk}
        for chi_sk in (*range(1, 101, 2), 100)
    )
    for method_index, method in enumerate(variance_methods):
        tasks.append(task(
            "gaussian_limit",
            variance_problem,
            method,
            method_seeds(10_000, method_index, 0),
            {
                "replicates": 20,
                "metrics": ["normalized_quadratic_variance"],
                "primary_metric": "normalized_quadratic_variance",
                "quantiles": [0.1, 0.9],
            },
            dtype="float64",
            resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
        ))

    embedding_dims = tuple(dict.fromkeys(
        int(round(value)) for value in np.linspace(2, 100, 16)
    ))
    nystrom_methods = [{"name": "gaussian_nystrom"}]
    nystrom_methods.extend(
        {"name": "mps_gram_nystrom", "chi_sk": chi_sk}
        for chi_sk in (1, 2, 3, 4, 5, 10, 20)
    )
    for embedding_index, embedding_dim in enumerate(embedding_dims):
        problem = {
            "kind": "walsh_nystrom",
            "tensor_order": 10,
            "ambient_dim": 2**10,
            "rank": 20,
            "epsilon": 1e-5,
            "ridge": 1e-12,
            "embedding_dim": embedding_dim,
        }
        for method_index, method in enumerate(nystrom_methods):
            tasks.append(task(
                "gaussian_limit",
                problem,
                method,
                method_seeds(11_000 + embedding_index, method_index, 0),
                {
                    "replicates": 20,
                    "metrics": ["relative_nuclear_error"],
                    "primary_metric": "relative_nuclear_error",
                    "quantiles": [0.1, 0.9],
                },
                dtype="float64",
                resources={"hardware": "cpu", "cpus": 4, "gpus": 0},
            ))
    return tasks
