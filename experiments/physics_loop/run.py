#!/usr/bin/env python3
"""Run one fixed-schedule dense or isoPEPS imaginary-time trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QUIMB_NUMBA_CACHE", "False")

import numpy as np
import scipy.sparse.linalg as spla
from threadpoolctl import threadpool_limits

from rand_isopeps.physics import (
    bond_hamiltonians,
    bond_trotter_imaginary_step,
    checkerboard_layers,
    dense_state_vector,
    exact_imaginary_step,
    local_term_norm_bound,
    normalize_state,
    rayleigh_residual,
    rayleigh_ritz,
    run_iterations,
    sparse_hamiltonian,
    trotter_imaginary_step,
)
from rand_isopeps.real_isotns.physics_loop import tebd_iteration
from rand_isopeps.real_isotns.tebd2 import energy, ham_from_spec


MODES = ("dense_exact", "dense_trotter", "peps_full", "peps_local", "peps_sketch")


def _schedule(args) -> list[float]:
    if not args.stage:
        return [float(args.tau)] * int(args.iterations)
    schedule = []
    for value in args.stage:
        tau_text, count_text = value.split(":", 1)
        tau, count = float(tau_text), int(count_text)
        if tau <= 0.0 or count < 1:
            raise ValueError("each --stage must be TAU:ITERATIONS with positive values")
        schedule.extend([tau] * count)
    return schedule


def _site_vectors(lx: int, ly: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vectors = {}
    for i in range(lx):
        for j in range(ly):
            value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            vectors[(i, j)] = value / np.linalg.norm(value)
    return vectors


def _dense_product(vectors: dict, lx: int, ly: int) -> np.ndarray:
    state = np.asarray([1.0 + 0.0j])
    for i in range(lx):
        for j in range(ly):
            state = np.kron(state, vectors[(i, j)])
    return normalize_state(state)


def _reference_energies(h, count: int) -> list[float]:
    dimension = h.shape[0]
    if dimension <= max(count + 1, 4):
        values = np.linalg.eigvalsh(h.toarray())[:count]
    else:
        values = spla.eigsh(h, k=count, which="SA", return_eigenvectors=False)
        values = np.sort(values.real)
    return [float(value) for value in values]


def _dense_run(
    args, h, layers, norm_bound, schedule, vectors, references, on_record=None
):
    dimension = h.shape[0]
    if args.states == 1:
        state = _dense_product(vectors, args.lx, args.ly)
    else:
        rng = np.random.default_rng(args.seed)
        state = rng.standard_normal((dimension, args.states))
        state = state + 1j * rng.standard_normal(state.shape)
        state, _ = np.linalg.qr(state, mode="reduced")

    def update(current, iteration):
        tau = schedule[iteration - 1]
        started = perf_counter()
        if args.mode == "dense_exact":
            current = exact_imaginary_step(h, current, tau)
        else:
            current = trotter_imaginary_step(layers, current, tau)
        if args.states > 1:
            current = rayleigh_ritz(h, current, h_norm_bound=norm_bound)["vectors"]
        return current, {
            "tau": tau,
            "update_runtime_s": float(perf_counter() - started),
            "max_bond": None,
        }

    def measure(current, _iteration):
        started = perf_counter()
        if args.states == 1:
            metrics = [rayleigh_residual(h, current, h_norm_bound=norm_bound)]
        else:
            metrics = rayleigh_ritz(h, current, h_norm_bound=norm_bound)["metrics"]
        result = {
            "energies": [entry["energy"] for entry in metrics],
            "ground_energy_errors": [
                entry["energy"] - references[index]
                for index, entry in enumerate(metrics)
            ],
            "residual_norms": [entry["residual_norm"] for entry in metrics],
            "relative_residuals": [entry["relative_residual"] for entry in metrics],
            "variances": [entry["variance"] for entry in metrics],
            "residual_identity_errors": [entry["residual_identity_error"] for entry in metrics],
            "residual_method": "dense_exact",
        }
        result["measurement_runtime_s"] = float(perf_counter() - started)
        return result

    return run_iterations(
        state,
        iterations=len(schedule),
        update=update,
        measure=measure,
        on_record=on_record,
    )


def _infidelity(first, second) -> float:
    overlap = abs(np.vdot(normalize_state(first), normalize_state(second))) ** 2
    return float(max(0.0, 1.0 - min(float(overlap), 1.0)))


def _peps_run(
    args, ham, h, bonds, norm_bound, schedule, vectors, references, on_record=None
):
    if args.states != 1:
        raise ValueError("PEPS modes currently support one state; use a dense mode for block Ritz")
    import quimb.tensor as qtn

    state = qtn.PEPS.product_state(vectors)
    sketch_rng = np.random.default_rng(args.sketch_seed)
    exact_states = full_trotter_states = None
    if h is not None:
        initial = _dense_product(vectors, args.lx, args.ly)
        exact_states = [initial]
        full_trotter_states = [initial]
        for tau in schedule:
            exact_states.append(exact_imaginary_step(h, exact_states[-1], tau))
            full_trotter_states.append(bond_trotter_imaginary_step(
                bonds,
                full_trotter_states[-1],
                tau,
                ly=args.ly,
                direction=1,
            ))

    if args.mode == "peps_full":
        backend = "none"
        gate_options = {"max_bond": None, "cutoff": 0.0}
        column_options = {}
    elif args.mode == "peps_local":
        backend = "local"
        gate_options = {"max_bond": args.gate_bond, "cutoff": args.cutoff}
        column_options = {
            "chi": args.chi,
            "eta": args.eta,
            "cutoff": args.cutoff,
            "ndis": args.ndis,
            "absorption_max_bond": args.absorption_bond,
            "absorption_cutoff": args.absorption_cutoff,
        }
    else:
        backend = "rmps"
        gate_options = {"max_bond": args.gate_bond, "cutoff": args.cutoff}
        column_options = {
            "ell": args.ell,
            "eta": args.eta,
            "kappa": args.kappa,
            "chi_sk": args.chi_sk,
            "ndis": args.ndis,
            "absorption_max_bond": args.absorption_bond,
            "absorption_cutoff": args.absorption_cutoff,
        }

    def update(current, iteration):
        updated, metrics = tebd_iteration(
            current,
            ham,
            schedule[iteration - 1],
            direction=1,
            column_backend=backend,
            gate_options=gate_options,
            column_options=column_options,
            rng=sketch_rng,
            inplace=False,
        )
        if args.abort_bond is not None and updated.max_bond() > args.abort_bond:
            raise RuntimeError(
                f"bond dimension {updated.max_bond()} exceeded --abort-bond={args.abort_bond}; "
                "the state was not silently truncated"
            )
        return updated, metrics

    def measure(current, iteration):
        started = perf_counter()
        if args.lx * args.ly <= args.exact_max_sites:
            vector, log10_norm = dense_state_vector(current)
            metrics = rayleigh_residual(h, vector, h_norm_bound=norm_bound)
            exact_metrics = rayleigh_residual(
                h, exact_states[iteration], h_norm_bound=norm_bound
            )
            result = {
                "energies": [metrics["energy"]],
                "ground_energy_errors": [metrics["energy"] - references[0]],
                "residual_norms": [metrics["residual_norm"]],
                "relative_residuals": [metrics["relative_residual"]],
                "variances": [metrics["variance"]],
                "residual_identity_errors": [metrics["residual_identity_error"]],
                "state_log10_norm": log10_norm,
                "state_infidelity_to_exact_evolution": _infidelity(
                    vector, exact_states[iteration]
                ),
                "state_infidelity_to_full_trotter": _infidelity(
                    vector, full_trotter_states[iteration]
                ),
                "trotter_infidelity": _infidelity(
                    full_trotter_states[iteration], exact_states[iteration]
                ),
                "energy_error_to_exact_evolution": (
                    metrics["energy"] - exact_metrics["energy"]
                ),
                "residual_error_to_exact_evolution": (
                    metrics["residual_norm"] - exact_metrics["residual_norm"]
                ),
                "residual_method": "dense_peps_oracle",
            }
            result["measurement_runtime_s"] = float(perf_counter() - started)
            return result
        measured_energy = energy(current, ham, max_bond=args.energy_bond)
        result = {
            "energies": [measured_energy],
            "ground_energy_errors": [float("nan")],
            "residual_norms": [float("nan")],
            "relative_residuals": [float("nan")],
            "variances": [float("nan")],
            "residual_identity_errors": [float("nan")],
            "state_log10_norm": float("nan"),
            "state_infidelity_to_exact_evolution": float("nan"),
            "state_infidelity_to_full_trotter": float("nan"),
            "trotter_infidelity": float("nan"),
            "energy_error_to_exact_evolution": float("nan"),
            "residual_error_to_exact_evolution": float("nan"),
            "residual_method": "unavailable_large_peps",
        }
        result["measurement_runtime_s"] = float(perf_counter() - started)
        return result

    return run_iterations(
        state,
        iterations=len(schedule),
        update=update,
        measure=measure,
        on_record=on_record,
    )


def run(args, on_record=None):
    schedule = _schedule(args)
    n_sites = args.lx * args.ly
    if args.mode.startswith("dense") and n_sites > args.exact_max_sites:
        raise ValueError("dense modes exceed --exact-max-sites")
    ham = ham_from_spec(args.hamiltonian, args.lx, args.ly)
    if ham is None:
        raise ValueError("the physics loop requires a Hamiltonian")
    h = sparse_hamiltonian(ham, args.lx, args.ly) if n_sites <= args.exact_max_sites else None
    layers = checkerboard_layers(ham, args.lx, args.ly) if h is not None else None
    bonds = bond_hamiltonians(ham, args.lx, args.ly) if h is not None else None
    norm_bound = local_term_norm_bound(ham)
    vectors = _site_vectors(args.lx, args.ly, args.seed)
    references = _reference_energies(h, args.states) if h is not None else []

    def decorate(record):
        record.update({
            "mode": args.mode,
            "lx": args.lx,
            "ly": args.ly,
            "hamiltonian": args.hamiltonian,
            "seed": args.seed,
            "reference_energies": references,
        })
        if on_record is not None:
            on_record(record)

    with threadpool_limits(limits=args.blas_threads):
        if args.mode.startswith("dense"):
            final_state, records = _dense_run(
                args, h, layers, norm_bound, schedule, vectors, references, decorate
            )
        else:
            final_state, records = _peps_run(
                args,
                ham,
                h,
                bonds,
                norm_bound,
                schedule,
                vectors,
                references,
                decorate,
            )
    return final_state, records


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--lx", type=int, default=2)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--hamiltonian", default="tfim@3.04")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--stage", action="append", help="repeatable TAU:ITERATIONS schedule")
    parser.add_argument("--states", type=int, default=1, help="dense block-subspace size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sketch-seed", type=int, default=1)
    parser.add_argument("--chi", type=int, default=8)
    parser.add_argument("--eta", type=int, default=4)
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--kappa", type=int, default=2)
    parser.add_argument("--chi-sk", type=int, default=4)
    parser.add_argument("--ndis", type=int, default=0)
    parser.add_argument("--gate-bond", type=int, default=8)
    parser.add_argument("--absorption-bond", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--absorption-cutoff", type=float, default=1e-10)
    parser.add_argument("--energy-bond", type=int, default=32)
    parser.add_argument("--exact-max-sites", type=int, default=12)
    parser.add_argument("--abort-bond", type=int)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional JSONL output; stdout is always emitted")
    return parser


def main():
    args = _parser().parse_args()
    if args.lx < 1 or args.ly < 1 or args.states < 1:
        raise ValueError("lattice dimensions and --states must be positive")
    handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", encoding="utf-8")

    def emit(record):
        line = json.dumps(record, allow_nan=True, sort_keys=True)
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    try:
        run(args, on_record=emit)
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
