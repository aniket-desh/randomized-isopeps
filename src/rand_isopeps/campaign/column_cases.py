"""paired synthetic and physical columns for campaign tasks."""

from __future__ import annotations

import numpy as np

from rand_isopeps.column.operator import (
    ColumnOperator,
    controlled_spectrum_column_matrix,
    random_column_operator,
)


def synthetic_column(problem: dict, seed: int):
    """build a deterministic column and optional dense reference."""
    rng = np.random.default_rng(int(seed))
    lx = int(problem["lx"])
    in_dims = tuple(int(value) for value in problem.get("in_dims", [2] * lx))
    out_dims = tuple(int(value) for value in problem.get("out_dims", [2] * lx))
    family = str(problem.get("family", "gaussian"))
    if family in {"controlled_exp", "controlled_power"}:
        matrix = controlled_spectrum_column_matrix(
            out_dims,
            in_dims,
            rng,
            decay_kind=family.removeprefix("controlled_"),
            parameter=float(problem.get("decay", 4.0)),
        )
        return None, matrix, in_dims
    operator = random_column_operator(
        lx,
        in_dims,
        out_dims,
        int(problem.get("mpo_bond", 2)),
        rng,
        ensemble=family,
        decay=float(problem.get("decay", 0.6)),
        noise=float(problem.get("noise", 0.15)),
        complex_valued=str(problem.get("dtype", "complex128")).startswith("complex"),
    )
    return operator, None, operator.input_dims


def _product_peps(lx: int, ly: int, seed: int):
    import quimb.tensor as qtn

    rng = np.random.default_rng(int(seed))
    vectors = {}
    for site in np.ndindex(lx, ly):
        value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        vectors[site] = value / np.linalg.norm(value)
    return qtn.PEPS.product_state(vectors)


def physical_state(problem: dict, seed: int):
    """prepare one paired peps and return its center column metadata."""
    from rand_isopeps.column.from_quimb import find_center_column, from_quimb_column
    from rand_isopeps.real_isotns.moses_move import random_isotns
    from rand_isopeps.real_isotns.tebd2 import ham_from_spec, imaginary_time_converged

    lx, ly = int(problem["lx"]), int(problem["ly"])
    state_name = str(problem.get("state", "random"))
    if state_name == "random_raw":
        import quimb.tensor as qtn

        state = qtn.PEPS.rand(
            lx,
            ly,
            bond_dim=int(problem.get("bond", 2)),
            phys_dim=2,
            seed=int(seed),
        )
        center, split = 0, "right"
        preparation = {"converged": True, "sweeps": 0, "kind": "raw_random_peps"}
        hamiltonian = None
    elif state_name == "random_product":
        state = _product_peps(lx, ly, seed)
        center, split = 0, "right"
        preparation = {"converged": True, "sweeps": 0, "kind": "random_product"}
        hamiltonian = None
    else:
        state = random_isotns(
            lx,
            ly,
            bond=int(problem.get("bond", 2)),
            phys=2,
            chi=int(problem.get("chi", 8)),
            eta=int(problem.get("prep_eta", problem.get("eta", 8))),
            cutoff=float(problem.get("cutoff", 1e-10)),
            Ndis=int(problem.get("ndis", 0)),
            seed=int(seed),
        )
        hamiltonian = ham_from_spec(state_name, lx, ly)
        if hamiltonian is None:
            preparation = {"converged": True, "sweeps": 0}
        else:
            state, result = imaginary_time_converged(
                state,
                hamiltonian,
                taus=tuple(problem.get("prep_taus", (0.3, 0.1, 0.03, 0.01))),
                chi=int(problem.get("chi", 8)),
                eta=int(problem.get("prep_eta", problem.get("eta", 8))),
                cutoff=float(problem.get("cutoff", 1e-10)),
                Ndis=int(problem.get("ndis", 0)),
                energy_rtol=float(problem.get("energy_rtol", 1e-6)),
                stable_sweeps=int(problem.get("stable_sweeps", 3)),
                min_sweeps_per_tau=int(problem.get("min_sweeps_per_tau", 3)),
                max_sweeps_per_tau=int(problem.get("max_sweeps_per_tau", 40)),
                e_max_bond=int(problem.get("measurement_bond", 64)),
            )
            preparation = {
                "converged": bool(result.converged),
                "sweeps": len(result.steps),
                "energy": float(result.final_energy),
                "relative_energy_change": float(result.final_relative_energy_change),
            }
        center, split = find_center_column(state)
    return {
        "state": state,
        "hamiltonian": hamiltonian,
        "center": int(center),
        "split": split,
        "column": from_quimb_column(state, center, split=split, normalize=False),
        "preparation": preparation,
    }
