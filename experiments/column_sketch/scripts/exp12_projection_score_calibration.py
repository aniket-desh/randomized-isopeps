#!/usr/bin/env python3
"""Calibrate fresh-rMPS projection scoring against exact dense Frobenius errors."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from rand_isopeps.experiment_utils.run_store import (
    VersionedCsvStore,
    content_hash,
    make_identity,
    write_manifest,
)
from rand_isopeps.parallel import auto_worker_count, run_parallel_stream, with_blas_threads

SCHEMA_VERSION = "projection-score-calibration-v5"
SUITE = "projection_score_calibration"

IDENTITY_FIELDS = [
    "run_uuid", "run_key", "config_hash", "git_commit", "schema_version",
    "timestamp_utc", "hostname", "platform", "pid", "seed_hierarchy",
]
RESULT_FIELDS = [
    "lx", "ensemble", "problem_index", "problem_seed", "construction_seed",
    "score_index", "score_seed", "dtype", "in_dim", "out_dim", "mpo_bond",
    "eta", "kappa", "chi_sk", "ell", "n_power", "ndis", "score_chi",
    "score_probes", "confidence", "exact_projection_error",
    "estimated_projection_error", "absolute_bias", "relative_bias",
    "standard_error", "ci_low", "ci_high", "interval_covered",
    "factor_runtime_s", "dense_oracle_runtime_s", "score_runtime_s", "matrix_mps_products",
    "score_matrix_mps_products", "score_contraction_count", "score_peak_mps_bond",
    "factor_contraction_flops_estimate", "score_contraction_flops_estimate",
    "max_q_vertical", "max_residual_vertical", "peak_allocated_bytes",
    "status", "failure_reason",
]
FIELDS = IDENTITY_FIELDS + RESULT_FIELDS


def _row_specs(args, lx, ensemble, problem_index, completed_keys):
    problem_seed = args.seed + 100003 * problem_index + 1009 * lx
    construction_seed = problem_seed + 17
    specs = []
    effective_chis = dict.fromkeys(
        max(lx, args.chi_sk) if score_chi == 0 else score_chi
        for score_chi in args.score_chis
    )
    for effective_chi in effective_chis:
        for score_probes in args.score_probes_grid:
            for score_index in range(args.score_seeds):
                score_seed = problem_seed + 7919 + 104729 * score_index
                config = {
                    "problem": {
                        "lx": lx, "ensemble": ensemble, "in_dim": args.in_dim,
                        "out_dim": args.out_dim, "mpo_bond": args.mpo_bond,
                        "real": args.real,
                    },
                    "factorization": {
                        "eta": args.eta, "kappa": args.kappa,
                        "chi_sk": args.chi_sk, "ell_requested": args.ell,
                        "n_power": args.n_power, "ndis": args.ndis,
                        "dense_max_elements": args.dense_max_elements,
                    },
                    "scoring": {
                        "score_chi": effective_chi, "score_probes": score_probes,
                        "confidence": args.confidence, "bootstrap": args.bootstrap,
                    },
                    "execution": {
                        "workers": args.workers, "blas_threads": args.blas_threads,
                    },
                    "problem_index": problem_index, "score_index": score_index,
                }
                seeds = {
                    "problem": problem_seed,
                    "construction": construction_seed,
                    "score": score_seed,
                }
                identity = make_identity(
                    config, seeds, method="fresh_rmps_score",
                    schema_version=SCHEMA_VERSION,
                    root=Path(__file__).resolve().parents[3],
                )
                if identity["run_key"] not in completed_keys:
                    specs.append((identity, effective_chi, score_probes, score_index,
                                  score_seed, problem_seed, construction_seed))
    return specs


def _run_problem(task):
    args, lx, ensemble, problem_index, completed_keys = task
    specs = _row_specs(args, lx, ensemble, problem_index, completed_keys)
    if not specs:
        return []

    with with_blas_threads(args.blas_threads):
        from rand_isopeps.column.bounded_residual import (
            bounded_residual_column_qr,
            score_projection_error,
        )
        from rand_isopeps.column.operator import random_column_operator

        problem_seed, construction_seed = specs[0][-2:]
        op = random_column_operator(
            lx, args.in_dim, args.out_dim, args.mpo_bond,
            np.random.default_rng(problem_seed), ensemble=ensemble,
            complex_valued=not args.real,
        )
        # Keep the calibration target genuinely lossy even at the smallest Lx;
        # a full-domain sketch makes the exact residual roundoff-sized and turns
        # relative bias into a meaningless division by numerical noise.
        effective_ell = min(args.ell, max(1, op.n_in // 2))
        result = bounded_residual_column_qr(
            op, ell=effective_ell, eta=args.eta, kappa=args.kappa,
            chi_sk=args.chi_sk, sketch_kind="rmps", n_power=args.n_power,
            ndis=args.ndis, rng=np.random.default_rng(construction_seed),
            dense_oracle_max_elements=args.dense_max_elements,
        )
        exact = result.projection_error_dense
        if not np.isfinite(exact):
            raise RuntimeError(
                f"dense oracle disabled at lx={lx}; increase --dense-max-elements "
                "or reduce the calibration ceiling"
            )

        rows = []
        for identity, score_chi, probes, score_index, score_seed, _, _ in specs:
            base = {
                **identity, "lx": lx, "ensemble": ensemble,
                "problem_index": problem_index, "problem_seed": problem_seed,
                "construction_seed": construction_seed, "score_index": score_index,
                "score_seed": score_seed, "dtype": str(op.cores[0].dtype),
                "in_dim": args.in_dim, "out_dim": args.out_dim,
                "mpo_bond": args.mpo_bond, "eta": args.eta, "kappa": args.kappa,
                "chi_sk": args.chi_sk, "ell": effective_ell,
                "n_power": args.n_power, "ndis": args.ndis,
                "score_chi": score_chi, "score_probes": probes,
                "confidence": args.confidence, "exact_projection_error": exact,
                "factor_runtime_s": result.runtime_s,
                "dense_oracle_runtime_s": result.dense_oracle_runtime_s,
                "matrix_mps_products": result.matrix_mps_products,
                "factor_contraction_flops_estimate": result.contraction_flops_estimate,
                "max_q_vertical": result.max_q_vertical,
                "max_residual_vertical": result.max_residual_vertical,
                "peak_allocated_bytes": result.peak_allocated_bytes,
            }
            try:
                score = score_projection_error(
                    op, result, n_probes=probes, chi_score=score_chi,
                    rng=np.random.default_rng(score_seed), confidence=args.confidence,
                    n_bootstrap=args.bootstrap,
                )
                absolute_bias = score.estimate - exact
                row = {
                    **base, "estimated_projection_error": score.estimate,
                    "absolute_bias": absolute_bias,
                    "relative_bias": absolute_bias / max(abs(exact), 1e-15),
                    "standard_error": score.standard_error,
                    "ci_low": score.ci_low, "ci_high": score.ci_high,
                    "interval_covered": int(score.ci_low <= exact <= score.ci_high),
                    "score_runtime_s": score.runtime_s,
                    "score_matrix_mps_products": score.matrix_mps_products,
                    "score_contraction_count": score.contraction_count,
                    "score_contraction_flops_estimate": score.contraction_flops_estimate,
                    "score_peak_mps_bond": score.peak_mps_bond,
                    "status": "ok", "failure_reason": "",
                }
            except (MemoryError, ValueError, RuntimeError) as exc:
                row = {
                    **base,
                    **{field: math.nan for field in (
                        "estimated_projection_error", "absolute_bias", "relative_bias",
                        "standard_error", "ci_low", "ci_high", "interval_covered",
                        "score_runtime_s", "score_matrix_mps_products",
                        "score_contraction_count", "score_peak_mps_bond",
                        "score_contraction_flops_estimate",
                    )},
                    "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
        return rows


def run(args):
    root = Path(__file__).resolve().parents[3]
    out = root / "outputs" / SUITE / SCHEMA_VERSION
    store = VersionedCsvStore(out / "results.csv", FIELDS)
    write_manifest(out / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "description": __doc__,
        "fields": FIELDS,
        "exact_metric": "||(I-QQ*)C||_F / ||C||_F",
        "estimator": "fresh isotropic rMPS paired numerator-denominator ratio",
        "coverage_target_percent": args.confidence,
        "invocation_manifests": "manifests/*.json",
    })
    invocation_hash = content_hash(vars(args))
    write_manifest(out / "manifests" / f"{invocation_hash}.json", {
        "schema_version": SCHEMA_VERSION,
        "invocation_hash": invocation_hash,
        "arguments": vars(args),
    })
    completed = frozenset(store.rows)
    tasks = [
        (args, lx, ensemble, problem, completed)
        for lx in args.lxs for ensemble in args.ensembles
        for problem in range(args.problem_seeds)
    ]
    for rows in run_parallel_stream(
        _run_problem, tasks, auto_worker_count(args.workers)
    ):
        for row in rows:
            store.append(row)
    return out / "results.csv", out / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lxs", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--ensembles", nargs="+", default=["gaussian", "decay"])
    parser.add_argument("--in-dim", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=4)
    parser.add_argument("--mpo-bond", type=int, default=2)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--kappa", type=int, default=2)
    parser.add_argument("--chi-sk", type=int, default=4)
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--n-power", type=int, default=0)
    parser.add_argument("--ndis", type=int, default=5)
    parser.add_argument("--score-chis", type=int, nargs="+", default=[1, 2, 4, 8, 0],
                        help="0 means max(lx, chi_sk)")
    parser.add_argument("--score-probes-grid", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--problem-seeds", type=int, default=4)
    parser.add_argument("--score-seeds", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=90.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--dense-max-elements", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lxs = [2, 3]
        args.ensembles = ["gaussian"]
        args.problem_seeds = 1
        args.score_seeds = 2
        args.score_chis = [1, 4]
        args.score_probes_grid = [8, 32]
        args.bootstrap = 100
        args.ndis = 1
        args.workers = 1
    return args


if __name__ == "__main__":
    os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
    csv_path, manifest_path = run(parse_args())
    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
