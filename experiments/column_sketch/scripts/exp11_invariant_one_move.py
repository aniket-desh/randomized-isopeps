#!/usr/bin/env python3
"""Invariant one-move pilot for the executed bounded-residual factorization.

This replaces the invalid ``local_eps / global_eps`` comparison.  Every method
starts from the same prepared PEPS and performs exactly one boundary-column move.
The primary error is normalized state infidelity; energy and observables are
confirmation metrics.  Whole-column projection errors remain diagnostics for
global methods only.

Rows are versioned and resume by a hash of the complete method configuration,
seed hierarchy, schema, and git commit.  Conflicting duplicate payloads abort.
The first paper-scale run should be launched through the repository's NERSC
deploy -> sbatch -> publish -> local-analysis loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from rand_isopeps.experiment_utils.run_store import (
    VersionedCsvStore,
    content_hash,
    make_identity,
    write_manifest,
)
from rand_isopeps.parallel import (
    auto_worker_count,
    run_parallel_stream,
    with_blas_threads,
)

SCHEMA_VERSION = "invariant-one-move-v10"
SUITE = "invariant_one_move"

IDENTITY_FIELDS = [
    "run_uuid", "run_key", "config_hash", "git_commit", "schema_version",
    "timestamp_utc", "hostname", "platform", "pid", "seed_hierarchy",
]
RESULT_FIELDS = [
    "method", "state", "lx", "ly", "prep_key", "prep_seed", "sketch_index", "sketch_seed",
    "score_seed", "dtype",
    "center_j", "split", "prep_converged", "prep_final_energy",
    "prep_final_rel_change", "prep_sweeps",
    "prep_energy_variance", "prep_exact_ground_energy", "prep_ground_energy_error",
    "prep_parity_x", "prep_mean_magnetization_z", "prep_abs_magnetization_z",
    "prep_max_abs_magnetization_z", "prep_row_cut_renyi2",
    "prep_column_cut_renyi2", "prep_row_cut_von_neumann",
    "prep_column_cut_von_neumann", "prep_correlation_length_z",
    "column_n_in", "column_n_out", "column_mpo_bond", "column_flat_rank",
    "column_flat_r99", "column_flat_renyi2", "column_flat_von_neumann",
    "operator_cut_max_r99", "operator_cut_max_renyi2", "operator_cut_r99",
    "operator_cut_renyi2", "operator_cut_von_neumann", "operator_cut_tail_eta",
    "operator_cut_tail_eta_kappa",
    "prep_eta", "eta", "kappa", "chi_sk", "ell", "n_power", "ndis",
    "state_infidelity", "norm_drift", "energy", "relative_energy_error",
    "relative_energy_change", "energy_variance", "parity_x",
    "max_magnetization_z_error", "max_correlator_zz_error",
    "projection_error", "projection_error_source", "projection_error_dense", "spectral_tail_dense",
    "projection_excess_dense", "q_flat_rank", "residual_consistency_error",
    "score_standard_error",
    "score_ci_low", "score_ci_high", "score_interval_covered", "score_relative_bias",
    "score_chi", "score_probes", "score_runtime_s", "score_matrix_mps_products",
    "score_contraction_count", "score_contraction_flops_estimate", "score_peak_mps_bond",
    "sampled_reconstruction_error", "delta_local", "delta_global",
    "delta_global_bound", "residual_dims", "max_q_vertical",
    "max_residual_vertical", "max_sample_residual_vertical",
    "cut_tails_first", "cut_tails_second", "factor_runtime_s", "dense_oracle_runtime_s",
    "absorption_runtime_s", "metric_runtime_s", "total_runtime_s",
    "matrix_mps_products", "passes", "contraction_count",
    "contraction_flops_estimate", "peak_allocated_bytes",
    "largest_svd_rows", "largest_svd_cols", "status", "failure_reason",
]
FIELDS = IDENTITY_FIELDS + RESULT_FIELDS
PREPARATION_RESULT_FIELDS = [
    "state", "lx", "ly", "prep_seed", "prep_eta",
    "prep_converged", "prep_final_energy", "prep_final_rel_change",
    "prep_sweeps", "prep_energy_trace", "prep_energy_variance",
    "prep_exact_ground_energy", "prep_ground_energy_error", "prep_parity_x",
    "prep_mean_magnetization_z", "prep_abs_magnetization_z",
    "prep_max_abs_magnetization_z", "prep_row_cut_renyi2",
    "prep_column_cut_renyi2", "prep_row_cut_von_neumann",
    "prep_column_cut_von_neumann", "prep_correlation_length_z",
    "prep_column_cache_runtime_s", "prep_column_cache_bytes",
]
PREPARATION_FIELDS = IDENTITY_FIELDS + PREPARATION_RESULT_FIELDS


def _dense_hamiltonian(ham, lx, ly):
    """Sparse exact Hamiltonian in PEPS site-index order (small systems only)."""
    import quimb as qu
    import scipy.sparse as sp

    dims = [2] * (lx * ly)
    h = sp.csr_matrix((2 ** (lx * ly), 2 ** (lx * ly)), dtype=complex)
    for where, term in ham.terms.items():
        sites = tuple(where) if isinstance(where[0], tuple) else (where,)
        inds = tuple(int(i * ly + j) for i, j in sites)
        # ``LocalHam2D`` stores a fused two-site operator.  ``pkron`` is needed
        # for vertical bonds whose row-major site indices are non-adjacent;
        # ``ikron`` only pads a fused operator correctly on contiguous sites.
        h = h + qu.pkron(term, dims, inds, sparse=True)
    return h.tocsr()


@lru_cache(maxsize=None)
def _z_geometry(lx, ly):
    n = lx * ly
    basis = np.arange(2 ** n, dtype=np.uint64)
    signs = np.stack([
        1 - 2 * ((basis >> np.uint64(n - 1 - site)) & np.uint64(1)).astype(np.int8)
        for site in range(n)
    ])
    pairs = []
    for i in range(lx):
        for j in range(ly):
            if i + 1 < lx:
                pairs.append((i * ly + j, (i + 1) * ly + j))
            if j + 1 < ly:
                pairs.append((i * ly + j, i * ly + j + 1))
    pair_signs = np.stack([signs[a] * signs[b] for a, b in pairs]) if pairs else np.empty(
        (0, basis.size), dtype=np.int8
    )
    signs.setflags(write=False)
    pair_signs.setflags(write=False)
    return signs, pair_signs


def _dense_observables(vector, lx, ly):
    norm = float(np.vdot(vector, vector).real)
    probability = np.abs(np.asarray(vector).reshape(-1)) ** 2
    signs, pair_signs = _z_geometry(lx, ly)
    mags = signs @ probability / max(norm, 1e-300)
    corrs = pair_signs @ probability / max(norm, 1e-300)
    # X on every site maps each computational basis index to its bitwise
    # complement, which is simply reversed row-major ordering for qubits.
    parity = float((np.vdot(vector, np.asarray(vector)[::-1]) / max(norm, 1e-300)).real)
    return np.asarray(mags), np.asarray(corrs), parity


_GROUND_ENERGY_CACHE = {}
_DENSE_HAMILTONIAN_CACHE = OrderedDict()


def _cached_dense_hamiltonian(ham, state, lx, ly):
    """Reuse the sparse exact Hamiltonian across preparation seeds in a worker."""
    key = (state, int(lx), int(ly))
    if key not in _DENSE_HAMILTONIAN_CACHE:
        _DENSE_HAMILTONIAN_CACHE[key] = _dense_hamiltonian(ham, lx, ly)
        while len(_DENSE_HAMILTONIAN_CACHE) > 2:
            _DENSE_HAMILTONIAN_CACHE.popitem(last=False)
    else:
        _DENSE_HAMILTONIAN_CACHE.move_to_end(key)
    return _DENSE_HAMILTONIAN_CACHE[key]


def _state_cut_diagnostics(vector, lx, ly):
    from rand_isopeps.column.diagnostics import spectrum_diagnostics

    tensor = np.asarray(vector).reshape((2,) * (lx * ly))

    def stats_for_axes(left_axes):
        right_axes = [axis for axis in range(lx * ly) if axis not in left_axes]
        matrix = tensor.transpose(tuple(left_axes) + tuple(right_axes)).reshape(
            2 ** len(left_axes), -1
        )
        return spectrum_diagnostics(np.linalg.svd(matrix, compute_uv=False))

    row = [
        stats_for_axes(list(range(cut * ly)))
        for cut in range(1, lx)
    ]
    col = [
        stats_for_axes([i * ly + j for i in range(lx) for j in range(cut)])
        for cut in range(1, ly)
    ]
    return {
        "prep_row_cut_renyi2": json.dumps([s["renyi2"] for s in row]),
        "prep_column_cut_renyi2": json.dumps([s["renyi2"] for s in col]),
        "prep_row_cut_von_neumann": json.dumps([s["von_neumann"] for s in row]),
        "prep_column_cut_von_neumann": json.dumps([s["von_neumann"] for s in col]),
    }


def _correlation_length_z(vector, lx, ly, mags):
    norm = float(np.vdot(vector, vector).real)
    probability = np.abs(np.asarray(vector).reshape(-1)) ** 2
    signs, _ = _z_geometry(lx, ly)
    by_distance = {}
    for i in range(lx):
        for j in range(ly):
            a = i * ly + j
            for di, dj in ((1, 0), (0, 1)):
                distance = 1
                while i + distance * di < lx and j + distance * dj < ly:
                    b = (i + distance * di) * ly + j + distance * dj
                    zz = np.dot(signs[a] * signs[b], probability) / max(norm, 1e-300)
                    by_distance.setdefault(distance, []).append(abs(float(zz) - mags[a] * mags[b]))
                    distance += 1
    x, y = [], []
    for distance, values in sorted(by_distance.items()):
        value = float(np.mean(values))
        if value > 1e-14:
            x.append(float(distance))
            y.append(float(np.log(value)))
    if len(x) < 2:
        return float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    return float(-1.0 / slope) if slope < 0.0 else float("nan")


def _exact_ground_energy(state, lx, ly, ed_max_sites):
    if lx * ly > ed_max_sites or state == "random":
        return float("nan")
    from rand_isopeps.real_isotns.tebd2 import ham_from_spec
    import scipy.sparse.linalg as spla

    h = _dense_hamiltonian(ham_from_spec(state, lx, ly), lx, ly)
    if h.shape[0] <= 4:
        return float(np.linalg.eigvalsh(h.toarray())[0].real)
    v0 = np.random.default_rng(1907).standard_normal(h.shape[0])
    v0 /= np.linalg.norm(v0)
    return float(spla.eigsh(
        h, k=1, which="SA", v0=v0, return_eigenvectors=False
    )[0].real)


def _exact_context(
    psi, ham, lx, ly, max_sites, state, ed_max_sites,
    ground_energy_override=float("nan"),
):
    if lx * ly > max_sites:
        return None
    vector = np.asarray(psi.to_dense()).reshape(-1)
    norm = float(np.linalg.norm(vector))
    vector = vector / max(norm, 1e-300)
    h = _cached_dense_hamiltonian(ham, state, lx, ly) if ham is not None else None
    if h is None:
        energy = variance = float("nan")
    else:
        hv = h @ vector
        energy = float(np.vdot(vector, hv).real)
        variance = float(max(np.vdot(hv, hv).real - energy ** 2, 0.0))
    mags, corrs, parity = _dense_observables(vector, lx, ly)
    if np.isfinite(ground_energy_override):
        ground_energy = float(ground_energy_override)
    elif h is not None and lx * ly <= ed_max_sites:
        key = (state, lx, ly)
        if key not in _GROUND_ENERGY_CACHE:
            import scipy.sparse.linalg as spla

            if h.shape[0] <= 4:
                ground = float(np.linalg.eigvalsh(h.toarray())[0].real)
            else:
                ground = float(spla.eigsh(h, k=1, which="SA", return_eigenvectors=False)[0].real)
            _GROUND_ENERGY_CACHE[key] = ground
        ground_energy = _GROUND_ENERGY_CACHE[key]
    else:
        ground_energy = float("nan")
    cut_diagnostics = _state_cut_diagnostics(vector, lx, ly)
    return {
        "vector": vector, "raw_norm": norm, "hamiltonian": h,
        "energy": energy, "variance": variance,
        "mags": mags, "corrs": corrs, "parity": parity, "ground_energy": ground_energy,
        "correlation_length_z": _correlation_length_z(vector, lx, ly, mags),
        **cut_diagnostics,
    }


def _boundary_overlap(a, b, max_bond):
    aa = (a.H | a).contract_boundary(max_bond=max_bond)
    bb = (b.H | b).contract_boundary(max_bond=max_bond)
    ab = (a.H | b).contract_boundary(max_bond=max_bond)
    return complex(ab), float(np.real(aa)), float(np.real(bb))


def _boundary_observables(psi, lx, ly, max_bond):
    import quimb as qu

    z = qu.pauli("Z")
    zz = z & z
    mags = []
    corrs = []
    for i in range(lx):
        for j in range(ly):
            mags.append(float(psi.compute_local_expectation(
                {(i, j): z}, normalized=True, max_bond=max_bond
            ).real))
            if i + 1 < lx:
                corrs.append(float(psi.compute_local_expectation(
                    {((i, j), (i + 1, j)): zz}, normalized=True, max_bond=max_bond
                ).real))
            if j + 1 < ly:
                corrs.append(float(psi.compute_local_expectation(
                    {((i, j), (i, j + 1)): zz}, normalized=True, max_bond=max_bond
                ).real))
    return np.asarray(mags), np.asarray(corrs)


def _state_metrics(reference, method, ham, exact, args):
    t0 = perf_counter()
    if exact is not None:
        vector = np.asarray(method.to_dense()).reshape(-1)
        norm = float(np.linalg.norm(vector))
        vector_n = vector / max(norm, 1e-300)
        overlap = np.vdot(exact["vector"], vector_n)
        infidelity = max(1.0 - abs(overlap) ** 2, 0.0)
        norm_ref = float(exact["raw_norm"])
        norm_drift = norm / max(norm_ref, 1e-300) - 1.0
        if exact["hamiltonian"] is not None:
            hv = exact["hamiltonian"] @ vector_n
            energy = float(np.vdot(vector_n, hv).real)
            variance = float(max(np.vdot(hv, hv).real - energy ** 2, 0.0))
        else:
            energy = variance = float("nan")
        mags, corrs, parity = _dense_observables(vector_n, args.lx, args.ly)
    else:
        overlap, norm_ref2, norm2 = _boundary_overlap(reference, method, args.boundary_bond)
        infidelity = max(1.0 - abs(overlap) ** 2 / max(norm_ref2 * norm2, 1e-300), 0.0)
        norm_drift = math.sqrt(max(norm2, 0.0) / max(norm_ref2, 1e-300)) - 1.0
        from rand_isopeps.real_isotns.tebd2 import energy as tn_energy
        energy = tn_energy(method, ham, args.boundary_bond) if ham is not None else float("nan")
        variance = parity = float("nan")
        mags, corrs = _boundary_observables(method, args.lx, args.ly, args.boundary_bond)

    ref_energy = exact["energy"] if exact is not None else (
        __import__("rand_isopeps.real_isotns.tebd2", fromlist=["energy"]).energy(
            reference, ham, args.boundary_bond
        ) if ham is not None else float("nan")
    )
    ref_mags = exact["mags"] if exact is not None else _boundary_observables(
        reference, args.lx, args.ly, args.boundary_bond
    )[0]
    ref_corrs = exact["corrs"] if exact is not None else _boundary_observables(
        reference, args.lx, args.ly, args.boundary_bond
    )[1]
    rel_energy = abs(energy - ref_energy) / max(abs(ref_energy), 1e-15) if ham is not None else float("nan")
    return {
        "state_infidelity": float(infidelity),
        "norm_drift": float(norm_drift),
        "energy": float(energy),
        "relative_energy_error": float(rel_energy),
        "relative_energy_change": float((energy - ref_energy) / max(abs(ref_energy), 1e-15))
        if ham is not None else float("nan"),
        "energy_variance": float(variance),
        "parity_x": float(parity),
        "max_magnetization_z_error": float(np.max(np.abs(mags - ref_mags))),
        "max_correlator_zz_error": float(np.max(np.abs(corrs - ref_corrs))) if corrs.size else 0.0,
        "metric_runtime_s": float(perf_counter() - t0),
    }


def _prepare(args, state, prep_seed):
    from rand_isopeps.real_isotns.moses_move import random_isotns
    from rand_isopeps.real_isotns.tebd2 import ham_from_spec, imaginary_time_converged

    psi = random_isotns(
        args.lx, args.ly, bond=args.bond, phys=args.phys, chi=args.chi,
        eta=args.prep_eta, cutoff=args.cutoff, Ndis=args.ndis, seed=prep_seed,
    )
    ham = ham_from_spec(state, args.lx, args.ly)
    if ham is None:
        return psi, ham, {
            "converged": True, "final_energy": float("nan"),
            "final_relative_energy_change": float("nan"), "steps": [],
        }
    psi, prep = imaginary_time_converged(
        psi, ham, taus=args.taus, chi=args.chi, eta=args.prep_eta,
        cutoff=args.cutoff, Ndis=args.ndis, energy_rtol=args.energy_rtol,
        stable_sweeps=args.stable_sweeps, min_sweeps_per_tau=args.min_sweeps_per_tau,
        max_sweeps_per_tau=args.max_sweeps_per_tau, e_max_bond=args.boundary_bond,
    )
    return psi, ham, {
        "converged": prep.converged, "final_energy": prep.final_energy,
        "final_relative_energy_change": prep.final_relative_energy_change,
        "steps": prep.steps,
    }


def _prep_fields(prep):
    trace = [{"tau": s.tau, "stage": s.stage, "sweep": s.sweep, "energy": s.energy,
              "rel": s.relative_energy_change, "stable": s.stable_count}
             for s in prep["steps"]]
    return {
        "prep_converged": int(prep["converged"]),
        "prep_final_energy": prep["final_energy"],
        "prep_final_rel_change": prep["final_relative_energy_change"],
        "prep_sweeps": len(trace),
        "prep_energy_trace": json.dumps(trace, separators=(",", ":")),
    }


def _prepared_state_fields(exact):
    if exact is None:
        return {
            "prep_energy_variance": float("nan"),
            "prep_exact_ground_energy": float("nan"),
            "prep_ground_energy_error": float("nan"),
            "prep_parity_x": float("nan"),
            "prep_mean_magnetization_z": float("nan"),
            "prep_abs_magnetization_z": float("nan"),
            "prep_max_abs_magnetization_z": float("nan"),
            "prep_row_cut_renyi2": "",
            "prep_column_cut_renyi2": "",
            "prep_row_cut_von_neumann": "",
            "prep_column_cut_von_neumann": "",
            "prep_correlation_length_z": float("nan"),
        }
    ground = exact["ground_energy"]
    ground_error = (
        abs(exact["energy"] - ground) / max(abs(ground), 1e-15)
        if np.isfinite(ground) else float("nan")
    )
    return {
        "prep_energy_variance": exact["variance"],
        "prep_exact_ground_energy": ground,
        "prep_ground_energy_error": ground_error,
        "prep_parity_x": exact["parity"],
        "prep_mean_magnetization_z": float(np.mean(exact["mags"])),
        "prep_abs_magnetization_z": float(np.mean(np.abs(exact["mags"]))),
        "prep_max_abs_magnetization_z": float(np.max(np.abs(exact["mags"]))),
        "prep_row_cut_renyi2": exact["prep_row_cut_renyi2"],
        "prep_column_cut_renyi2": exact["prep_column_cut_renyi2"],
        "prep_row_cut_von_neumann": exact["prep_row_cut_von_neumann"],
        "prep_column_cut_von_neumann": exact["prep_column_cut_von_neumann"],
        "prep_correlation_length_z": exact["correlation_length_z"],
    }


def _empty_method_fields():
    return {field: float("nan") for field in (
        "projection_error", "projection_error_dense", "spectral_tail_dense",
        "projection_excess_dense", "q_flat_rank", "residual_consistency_error",
        "score_standard_error",
        "score_ci_low", "score_ci_high", "score_interval_covered", "score_relative_bias",
        "score_chi", "score_probes", "score_runtime_s", "score_matrix_mps_products",
        "score_contraction_count", "score_contraction_flops_estimate", "score_peak_mps_bond",
        "sampled_reconstruction_error", "delta_local", "delta_global",
        "delta_global_bound", "max_q_vertical", "max_residual_vertical",
        "max_sample_residual_vertical", "factor_runtime_s", "dense_oracle_runtime_s",
        "absorption_runtime_s",
        "matrix_mps_products", "passes", "contraction_count", "peak_allocated_bytes",
        "contraction_flops_estimate", "largest_svd_rows", "largest_svd_cols",
    )} | {
        "projection_error_source": "", "residual_dims": "",
        "cut_tails_first": "", "cut_tails_second": "",
    }


def _global_method(
    psi, op, method, config, seed, score_seed, args,
    dense_reference=None, reference_singular_values=None,
):
    from rand_isopeps.column.bounded_residual import (
        apply_boundary_factorization,
        bounded_residual_column_qr,
        score_projection_error,
    )

    factor = bounded_residual_column_qr(
        op, ell=config["ell"], eta=config["eta"], kappa=config["kappa"],
        chi_sk=config["chi_sk"], sketch_kind=config["sketch_kind"],
        n_power=config["n_power"], ndis=config["ndis"],
        rng=np.random.default_rng(seed), reference=dense_reference,
        reference_singular_values=reference_singular_values,
    )
    score = None
    if args.score_probes > 0:
        score = score_projection_error(
            op, factor, n_probes=args.score_probes, chi_score=args.score_chi,
            rng=np.random.default_rng(score_seed), confidence=args.score_confidence,
            n_bootstrap=args.score_bootstrap,
        )
    t_abs = perf_counter()
    moved = apply_boundary_factorization(psi, config["center_j"], factor, split=config["split"])
    absorption_runtime = perf_counter() - t_abs
    shapes = [shape for cut in factor.cuts for shape in (cut.first_shape, cut.second_shape) if shape]
    fields = {
        "projection_error": (
            score.estimate if score is not None else factor.projection_error_dense
        ),
        "projection_error_source": "fresh_rmps" if score is not None else "dense_oracle",
        "projection_error_dense": factor.projection_error_dense,
        "spectral_tail_dense": factor.spectral_tail_dense,
        "projection_excess_dense": factor.projection_excess_dense,
        "q_flat_rank": factor.q_flat_rank,
        "residual_consistency_error": factor.residual_consistency_error,
        "score_standard_error": score.standard_error if score is not None else float("nan"),
        "score_ci_low": score.ci_low if score is not None else float("nan"),
        "score_ci_high": score.ci_high if score is not None else float("nan"),
        "score_interval_covered": int(
            score.ci_low <= factor.projection_error_dense <= score.ci_high
        ) if score is not None and np.isfinite(factor.projection_error_dense) else float("nan"),
        "score_relative_bias": (
            (score.estimate - factor.projection_error_dense)
            / max(abs(factor.projection_error_dense), 1e-15)
        ) if score is not None and np.isfinite(factor.projection_error_dense) else float("nan"),
        "score_chi": score.chi_score if score is not None else float("nan"),
        "score_probes": score.n_probes if score is not None else 0,
        "score_runtime_s": score.runtime_s if score is not None else 0.0,
        "score_matrix_mps_products": score.matrix_mps_products if score is not None else 0,
        "score_contraction_count": score.contraction_count if score is not None else 0,
        "score_contraction_flops_estimate": (
            score.contraction_flops_estimate if score is not None else 0
        ),
        "score_peak_mps_bond": score.peak_mps_bond if score is not None else 0,
        "sampled_reconstruction_error": factor.reconstruction_error,
        "delta_local": factor.delta_local,
        "delta_global": factor.delta_global,
        "delta_global_bound": factor.delta_global_bound,
        "residual_dims": json.dumps(factor.residual_dims),
        "max_q_vertical": factor.max_q_vertical,
        "max_residual_vertical": factor.max_residual_vertical,
        "max_sample_residual_vertical": factor.max_sample_residual_vertical,
        "cut_tails_first": json.dumps([c.discarded_first for c in factor.cuts]),
        "cut_tails_second": json.dumps([c.discarded_second for c in factor.cuts]),
        "factor_runtime_s": factor.runtime_s,
        "dense_oracle_runtime_s": factor.dense_oracle_runtime_s,
        "absorption_runtime_s": absorption_runtime,
        "matrix_mps_products": factor.matrix_mps_products,
        "passes": factor.passes,
        "contraction_count": factor.contraction_count,
        "contraction_flops_estimate": factor.contraction_flops_estimate,
        "peak_allocated_bytes": factor.peak_allocated_bytes,
        "largest_svd_rows": max((s[0] for s in shapes), default=0),
        "largest_svd_cols": max((s[1] for s in shapes), default=0),
    }
    return moved, fields


def _local_method(psi, method, config, sketch_seed, args, center_j, split):
    from rand_isopeps.real_isotns.instrument import MosesRandConfig, MosesStats
    from rand_isopeps.real_isotns.moses_move import RandSVD, moses_move

    moved = psi.copy()
    stats = MosesStats()
    cfg = None
    if method == "local_rsvd2":
        rand = RandSVD(
            sketch="gaussian", oversample=args.oversample, n_power=config["n_power"],
            rng=np.random.default_rng(sketch_seed),
        )
        cfg = MosesRandConfig(svd2=rand)
    t0 = perf_counter()
    moses_move(
        moved, center_j, args.chi, config["eta"], args.cutoff, args.ndis,
        orientation="col", sweep="up", split=split, renorm=False,
        rand=cfg, stats=stats,
    )
    elapsed = perf_counter() - t0
    fields = _empty_method_fields()
    fields.update({
        "factor_runtime_s": elapsed,
        "dense_oracle_runtime_s": 0.0,
        "absorption_runtime_s": 0.0,  # included in the monolithic local runtime
        "matrix_mps_products": 0,
        "passes": 0,
        "contraction_count": len(stats.records),
        "contraction_flops_estimate": float("nan"),
        "largest_svd_rows": max((r.m for r in stats.records), default=0),
        "largest_svd_cols": max((r.n for r in stats.records), default=0),
        "cut_tails_first": json.dumps([r.tail for r in stats.records if r.stage == "svd1"]),
        "cut_tails_second": json.dumps([r.tail for r in stats.records if r.stage == "svd2"]),
    })
    return moved, fields


def _method_grid(args, op):
    """Unique executed configurations sharing one prepared input state."""
    methods = []
    enabled = set(args.methods)
    for eta in args.eta_grid:
        ell_values = tuple(dict.fromkeys(
            tuple(args.ell_grid)
            + tuple(int(eta) + int(p) for p in args.ell_oversampling_grid)
        ))
        if "local_det" in enabled:
            methods.append((
                "local_det", -1,
                {"eta": eta, "kappa": 1, "chi_sk": 0, "ell": 0,
                 "n_power": 0, "sketch_kind": None},
            ))
        for sketch_index in range(args.sketch_seeds):
            for n_power in args.n_power_grid:
                if "local_rsvd2" in enabled:
                    methods.append((
                        "local_rsvd2", sketch_index,
                        {"eta": eta, "kappa": 1, "chi_sk": 0, "ell": 0,
                         "n_power": n_power, "sketch_kind": None},
                    ))
            for ell_requested in ell_values:
                ell = min(int(ell_requested), op.n_in)
                for n_power in args.n_power_grid:
                    if "global_gaussian" in enabled:
                        methods.append((
                            "global_gaussian", sketch_index,
                            {"eta": eta, "kappa": 1, "chi_sk": 0, "ell": ell,
                             "n_power": n_power, "sketch_kind": "gaussian"},
                        ))
                    for chi_sk in args.chi_sk_grid:
                        if "global_rmps_plain" in enabled:
                            methods.append((
                                "global_rmps_plain", sketch_index,
                                {"eta": eta, "kappa": 1, "chi_sk": chi_sk,
                                 "ell": ell, "n_power": n_power,
                                 "sketch_kind": "rmps"},
                            ))
                        for kappa in args.kappa_grid:
                            if kappa == 1 or "global_rmps_bounded" not in enabled:
                                continue  # exactly the plain method above
                            methods.append((
                                "global_rmps_bounded", sketch_index,
                                {"eta": eta, "kappa": kappa, "chi_sk": chi_sk,
                                 "ell": ell, "n_power": n_power,
                                 "sketch_kind": "rmps"},
                            ))
                    for kappa in args.kappa_grid:
                        if "global_kron" in enabled:
                            methods.append((
                                "global_kron", sketch_index,
                                {"eta": eta, "kappa": kappa, "chi_sk": 1,
                                 "ell": ell, "n_power": n_power,
                                 "sketch_kind": "kron"},
                            ))
    unique = []
    seen = set()
    for method, sketch_index, config in methods:
        key = (method, sketch_index, tuple(sorted(config.items())))
        if key not in seen:
            seen.add(key)
            unique.append((method, sketch_index, config))
    return unique


def _run_problem(task):
    (args, state, prep_index, completed_keys, completed_prep_keys,
     ground_energies) = task
    prep_seed = args.seed + 100003 * prep_index
    with with_blas_threads(args.blas_threads):
        from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column

        psi, ham, prep = _prepare(args, state, prep_seed)
        center_j, split = find_center_column(psi)
        op = from_quimb_column(psi, center_j, split=split, normalize=False)
        cache_t0 = perf_counter()
        dense_reference = reference_singular_values = None
        if op.n_out * op.n_in <= 2_000_000:
            import scipy.linalg as la

            dense_reference = op.materialize()
            reference_singular_values = la.svdvals(
                dense_reference, check_finite=False
            )
        from rand_isopeps.column.diagnostics import operator_cut_spectra

        operator_spectra_cache = operator_cut_spectra(op)
        column_cache_runtime = perf_counter() - cache_t0
        column_cache_bytes = sum(s.nbytes for s in operator_spectra_cache)
        if dense_reference is not None:
            column_cache_bytes += dense_reference.nbytes + reference_singular_values.nbytes
        exact = _exact_context(
            psi, ham, args.lx, args.ly, args.exact_max_sites, state, args.ed_max_sites,
            ground_energies.get(state, float("nan")),
        )
        prep_all_fields = {
            **_prep_fields(prep), **_prepared_state_fields(exact),
            "prep_column_cache_runtime_s": column_cache_runtime,
            "prep_column_cache_bytes": column_cache_bytes,
        }
        prep_config = {
            "state": state, "lx": args.lx, "ly": args.ly,
            "bond": args.bond, "phys": args.phys, "chi": args.chi,
            "prep_eta": args.prep_eta, "cutoff": args.cutoff,
            "ndis": args.ndis, "taus": args.taus,
            "energy_rtol": args.energy_rtol,
            "stable_sweeps": args.stable_sweeps,
            "min_sweeps_per_tau": args.min_sweeps_per_tau,
            "max_sweeps_per_tau": args.max_sweeps_per_tau,
            "boundary_bond": args.boundary_bond,
            "exact_max_sites": args.exact_max_sites,
            "ed_max_sites": args.ed_max_sites,
        }
        prep_identity = make_identity(
            prep_config, {"preparation": prep_seed}, method="preparation",
            schema_version=SCHEMA_VERSION,
            root=Path(__file__).resolve().parents[3],
        )
        prep_key = prep_identity["run_key"]
        prep_row = {
            **prep_identity, "state": state, "lx": args.lx, "ly": args.ly,
            "prep_seed": prep_seed, "prep_eta": args.prep_eta,
            **prep_all_fields,
        }
        prep_fields = dict(prep_all_fields)
        prep_fields.pop("prep_energy_trace")
        prep_fields.pop("prep_column_cache_runtime_s")
        prep_fields.pop("prep_column_cache_bytes")
        rows = []
        predictor_cache = {}

        for method, sketch_index, method_cfg in _method_grid(args, op):
            is_global = method_cfg["sketch_kind"] is not None
            sketch_seed = args.seed + 100003 * prep_index + 1009 * max(sketch_index, 0) + 17
            score_seed = args.seed + 100003 * prep_index + 1009 * max(sketch_index, 0) + 7919
            config = {
                "problem": {
                    "state": state, "lx": args.lx, "ly": args.ly,
                    "bond": args.bond, "phys": args.phys, "chi": args.chi,
                },
                "preparation": {
                    "prep_eta": args.prep_eta, "cutoff": args.cutoff,
                    "taus": args.taus, "energy_rtol": args.energy_rtol,
                    "stable_sweeps": args.stable_sweeps,
                    "min_sweeps_per_tau": args.min_sweeps_per_tau,
                    "max_sweeps_per_tau": args.max_sweeps_per_tau,
                    "boundary_bond": args.boundary_bond,
                },
                "method": method,
                "prep_index": prep_index, "sketch_index": sketch_index,
                "effective": {
                    **method_cfg, "center_j": center_j, "split": split,
                    "ndis": args.ndis, "oversample": args.oversample,
                },
                "measurement": {
                    "score_chi": args.score_chi,
                    "score_probes": args.score_probes,
                    "score_confidence": args.score_confidence,
                    "score_bootstrap": args.score_bootstrap,
                    "exact_max_sites": args.exact_max_sites,
                    "ed_max_sites": args.ed_max_sites,
                    "diagnostic_dense_max_elements": args.diagnostic_dense_max_elements,
                },
                "execution": {
                    "workers": args.workers, "blas_threads": args.blas_threads,
                },
            }
            seeds = {
                "preparation": prep_seed,
                "sketch": sketch_seed if sketch_index >= 0 else -1,
                "score": score_seed if is_global else -1,
            }
            identity = make_identity(
                config, seeds, method=method, schema_version=SCHEMA_VERSION,
                root=Path(__file__).resolve().parents[3],
            )
            if identity["run_key"] in completed_keys:
                continue
            from rand_isopeps.column.diagnostics import column_diagnostics
            predictor_key = (method_cfg["eta"], method_cfg["kappa"])
            if predictor_key not in predictor_cache:
                predictor_cache[predictor_key] = column_diagnostics(
                    op, eta=method_cfg["eta"], kappa=method_cfg["kappa"],
                    dense_max_elements=args.diagnostic_dense_max_elements,
                    operator_spectra_cache=operator_spectra_cache,
                    flat_singular_values=(
                        reference_singular_values
                        if op.n_out * op.n_in <= args.diagnostic_dense_max_elements
                        else None
                    ),
                )
            predictors = predictor_cache[predictor_key]
            base = {
                **identity, "method": method, "state": state, "lx": args.lx, "ly": args.ly,
                "prep_key": prep_key,
                "prep_seed": prep_seed, "sketch_index": sketch_index,
                "sketch_seed": sketch_seed if sketch_index >= 0 else -1,
                "score_seed": score_seed if is_global else -1,
                "dtype": str(op.cores[0].dtype), "center_j": center_j, "split": split,
                **prep_fields, "prep_eta": args.prep_eta, "eta": method_cfg["eta"],
                **predictors,
                "kappa": method_cfg["kappa"], "chi_sk": method_cfg["chi_sk"],
                "ell": method_cfg["ell"], "n_power": method_cfg["n_power"],
                "ndis": args.ndis,
            }
            t_method = perf_counter()
            try:
                if not is_global:
                    moved, fields = _local_method(
                        psi, method, method_cfg, sketch_seed, args, center_j, split
                    )
                else:
                    cfg = dict(method_cfg, ndis=args.ndis, center_j=center_j, split=split)
                    moved, fields = _global_method(
                        psi, op, method, cfg, sketch_seed, score_seed, args,
                        dense_reference, reference_singular_values,
                    )
                metrics = _state_metrics(psi, moved, ham, exact, args)
                total = perf_counter() - t_method
                row = {**base, **metrics, **fields, "total_runtime_s": total,
                       "status": "ok", "failure_reason": ""}
            except (MemoryError, ValueError, RuntimeError) as exc:
                row = {**base, **{field: float("nan") for field in RESULT_FIELDS if field not in base},
                       "method": method, "state": state, "lx": args.lx, "ly": args.ly,
                       "prep_seed": prep_seed, "sketch_index": sketch_index,
                       "sketch_seed": sketch_seed if sketch_index >= 0 else -1,
                       "score_seed": score_seed if is_global else -1,
                       "dtype": str(op.cores[0].dtype), "center_j": center_j, "split": split,
                       **prep_fields, "prep_eta": args.prep_eta,
                       "eta": method_cfg["eta"], "kappa": method_cfg["kappa"],
                       "chi_sk": method_cfg["chi_sk"], "ell": method_cfg["ell"],
                       "n_power": method_cfg["n_power"], "ndis": args.ndis,
                       "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
        return {
            "preparation": None if prep_key in completed_prep_keys else prep_row,
            "rows": rows,
        }


def run(args):
    root = Path(__file__).resolve().parents[3]
    out = root / "outputs" / SUITE / SCHEMA_VERSION
    store = VersionedCsvStore(out / "results.csv", FIELDS)
    prep_store = VersionedCsvStore(out / "preparations.csv", PREPARATION_FIELDS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": __doc__,
        "fields": FIELDS,
        "preparation_fields": PREPARATION_FIELDS,
        "preparation_table": "preparations.csv",
        "primary_metric": "state_infidelity",
        "pairing": "same prepared state; sketch seeds nested within preparation seed",
        "invocation_manifests": "manifests/*.json",
    }
    write_manifest(out / "manifest.json", manifest)
    invocation_hash = content_hash(vars(args))
    write_manifest(out / "manifests" / f"{invocation_hash}.json", {
        "schema_version": SCHEMA_VERSION,
        "invocation_hash": invocation_hash,
        "arguments": vars(args),
    })

    completed_keys = frozenset(store.rows)
    completed_prep_keys = frozenset(prep_store.rows)
    with with_blas_threads(args.blas_threads):
        ground_energies = {
            state: _exact_ground_energy(state, args.lx, args.ly, args.ed_max_sites)
            for state in args.states
        }
    tasks = [
        (args, state, prep, completed_keys, completed_prep_keys, ground_energies)
        for state in args.states for prep in range(args.prep_seeds)
    ]
    workers = auto_worker_count(args.workers)
    for result in run_parallel_stream(_run_problem, tasks, workers):
        if result["preparation"] is not None:
            prep_store.append(result["preparation"])
        for row in result["rows"]:
            store.append(row)
    return out / "results.csv", out / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=int, default=3)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--states", nargs="+", default=["tfim@3.5"])
    parser.add_argument(
        "--methods", nargs="+",
        choices=["local_det", "local_rsvd2", "global_gaussian",
                 "global_rmps_plain", "global_rmps_bounded", "global_kron"],
        default=["local_det", "local_rsvd2", "global_gaussian",
                 "global_rmps_plain", "global_rmps_bounded", "global_kron"],
    )
    parser.add_argument("--bond", type=int, default=2)
    parser.add_argument("--phys", type=int, default=2)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--eta-grid", type=int, nargs="+")
    parser.add_argument("--prep-eta", type=int, default=0,
                        help="fixed preparation bond; 0 means max target eta")
    parser.add_argument("--kappa", type=int, default=2)
    parser.add_argument("--kappa-grid", type=int, nargs="+")
    parser.add_argument("--chi-sk", type=int, default=4)
    parser.add_argument("--chi-sk-grid", type=int, nargs="+")
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--ell-grid", type=int, nargs="+")
    parser.add_argument("--ell-oversampling-grid", type=int, nargs="+", default=[],
                        help="also use ell=eta+p for each listed oversampling p")
    parser.add_argument("--n-power", type=int, default=0)
    parser.add_argument("--n-power-grid", type=int, nargs="+")
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--ndis", type=int, default=10)
    parser.add_argument("--score-chi", type=int, default=8)
    parser.add_argument("--score-probes", type=int, default=64)
    parser.add_argument("--score-confidence", type=float, default=90.0)
    parser.add_argument("--score-bootstrap", type=int, default=1000)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.1, 0.03, 0.01])
    parser.add_argument("--energy-rtol", type=float, default=1e-6)
    parser.add_argument("--stable-sweeps", type=int, default=3)
    parser.add_argument("--min-sweeps-per-tau", type=int, default=3)
    parser.add_argument("--max-sweeps-per-tau", type=int, default=40)
    parser.add_argument("--prep-seeds", type=int, default=2)
    parser.add_argument("--sketch-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29000)
    parser.add_argument("--exact-max-sites", type=int, default=16)
    parser.add_argument("--ed-max-sites", type=int, default=16)
    parser.add_argument("--diagnostic-dense-max-elements", type=int, default=2_000_000)
    parser.add_argument("--boundary-bond", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.lx = 2
        args.ly = 2
        args.states = ["random"]
        args.chi = 2
        args.eta = 2
        args.kappa = 2
        args.chi_sk = 2
        args.ell = 4
        args.ndis = 1
        args.score_chi = 4
        args.score_probes = 16
        args.score_bootstrap = 100
        args.prep_seeds = 1
        args.sketch_seeds = 1
        args.workers = 1
    args.eta_grid = tuple(dict.fromkeys(args.eta_grid or [args.eta]))
    args.kappa_grid = tuple(dict.fromkeys(args.kappa_grid or [args.kappa]))
    args.chi_sk_grid = tuple(dict.fromkeys(args.chi_sk_grid or [args.chi_sk]))
    if args.ell_grid is None and not args.ell_oversampling_grid:
        args.ell_grid = (args.ell,)
    else:
        args.ell_grid = tuple(dict.fromkeys(args.ell_grid or []))
    args.ell_oversampling_grid = tuple(dict.fromkeys(args.ell_oversampling_grid))
    args.n_power_grid = tuple(dict.fromkeys(args.n_power_grid or [args.n_power]))
    args.prep_eta = args.prep_eta or max(args.eta_grid)
    return args


if __name__ == "__main__":
    os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")
    csv_path, manifest_path = run(parse_args())
    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
