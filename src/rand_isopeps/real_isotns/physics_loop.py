"""One small, reusable isoPEPS imaginary-time iteration."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .global_move import rmps_column_move
from .moses_move import moses_move
from .tebd2 import _gates, _touches_center


def _gate_items(gates, direction: int):
    items = list(gates.items())
    return items if direction == 1 else list(reversed(items))


def _directional_sweep(
    psi,
    gates,
    *,
    direction: int,
    column_backend: str,
    gate_options: dict,
    column_options: dict,
    rng,
):
    columns = range(psi.Ly) if direction == 1 else range(psi.Ly - 1, -1, -1)
    split = "right" if direction == 1 else "left"
    sweep = "up"
    records = []
    for j in columns:
        for where, gate in _gate_items(gates, direction):
            if _touches_center(where, j, direction):
                psi.gate_(
                    gate,
                    where=where,
                    contract="reduce-split",
                    max_bond=gate_options.get("max_bond"),
                    cutoff=float(gate_options.get("cutoff", 0.0)),
                )
        if not (0 <= j + direction < psi.Ly):
            continue
        before = int(psi.max_bond())
        if column_backend == "local":
            started = perf_counter()
            errors = moses_move(
                psi,
                j,
                int(column_options["chi"]),
                int(column_options["eta"]),
                float(column_options.get("cutoff", 0.0)),
                int(column_options.get("ndis", 0)),
                orientation="col",
                sweep=sweep,
                split=split,
                renorm=False,
                rand=None,
                absorb_max_bond=column_options.get("absorption_max_bond"),
                absorb_cutoff=float(column_options.get("absorption_cutoff", 0.0)),
            )
            records.append({
                "column": int(j),
                "next_column": int(j + direction),
                "split": split,
                "bond_before": before,
                "bond_after": int(psi.max_bond()),
                "local_error_squared": float(np.sum(np.asarray(errors, dtype=float) ** 2)),
                "total_runtime_s": float(perf_counter() - started),
            })
        elif column_backend == "rmps":
            psi, record = rmps_column_move(
                psi,
                j,
                split=split,
                ell=int(column_options["ell"]),
                eta=int(column_options["eta"]),
                kappa=int(column_options["kappa"]),
                chi_sk=int(column_options["chi_sk"]),
                ndis=int(column_options.get("ndis", 0)),
                absorption_max_bond=column_options.get("absorption_max_bond"),
                absorption_cutoff=float(column_options.get("absorption_cutoff", 0.0)),
                rng=rng,
                inplace=False,
            )
            record.update({"bond_before": before, "bond_after": int(psi.max_bond())})
            records.append(record)
        else:
            raise ValueError("column_backend must be 'local' or 'rmps'")
    return psi, records


def _full_peps_iteration(psi, ham, tau: float, direction: int, gate_options: dict):
    half_gates = _gates(ham, tau / 2.0)
    for sweep_direction in (direction, -direction):
        columns = (
            range(psi.Ly)
            if sweep_direction == 1
            else range(psi.Ly - 1, -1, -1)
        )
        for j in columns:
            for where, gate in _gate_items(half_gates, sweep_direction):
                if _touches_center(where, j, sweep_direction):
                    psi.gate_(
                        gate,
                        where=where,
                        contract="reduce-split",
                        max_bond=gate_options.get("max_bond"),
                        cutoff=float(gate_options.get("cutoff", 0.0)),
                    )
    return psi


def tebd_iteration(
    psi,
    ham,
    tau: float,
    *,
    direction: int = 1,
    column_backend: str = "local",
    gate_options: dict | None = None,
    column_options: dict | None = None,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
):
    """Apply one symmetric second-order imaginary-time PEPS iteration.

    ``column_backend='none'`` is the dynamically growing, untruncated PEPS
    baseline.  ``'local'`` uses the sequential Moses move and ``'rmps'`` uses
    the whole-column sketch.  A physics iteration is a half sweep followed by
    its exact reverse, so each bond Hamiltonian appears twice with ``tau/2``.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    gate_options = {} if gate_options is None else dict(gate_options)
    column_options = {} if column_options is None else dict(column_options)
    out = psi if inplace else psi.copy()
    started = perf_counter()

    if column_backend == "none":
        out = _full_peps_iteration(out, ham, float(tau), direction, gate_options)
        column_records = []
    else:
        half_gates = _gates(ham, float(tau) / 2.0)
        out, forward = _directional_sweep(
            out,
            half_gates,
            direction=direction,
            column_backend=column_backend,
            gate_options=gate_options,
            column_options=column_options,
            rng=rng,
        )
        out, backward = _directional_sweep(
            out,
            half_gates,
            direction=-direction,
            column_backend=column_backend,
            gate_options=gate_options,
            column_options=column_options,
            rng=rng,
        )
        column_records = forward + backward

    out.equalize_norms_(1.0)
    return out, {
        "backend": column_backend,
        "tau": float(tau),
        "trotter_order": 2,
        "column_moves": len(column_records),
        "max_bond": int(out.max_bond()),
        "update_runtime_s": float(perf_counter() - started),
        "columns": column_records,
    }
