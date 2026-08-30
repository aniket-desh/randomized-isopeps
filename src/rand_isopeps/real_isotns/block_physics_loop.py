"""first-order center-local imaginary-time sweeps for block-isopeps."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.physics.block_state import (
    BlockPeps,
    block_gram,
    move_block_center,
    orthonormalize_block,
    rotate_block_lattice,
    validate_block_state,
)

from .block_column import block_column_move
from .block_gate import apply_block_gate
from .block_local_move import block_local_column_move
from .tebd2 import _gates


def _vertical(where) -> bool:
    return int(where[0][1]) == int(where[1][1])


def _rotate_where(where, ly: int):
    return tuple((ly - 1 - int(y), int(x)) for x, y in where)


def _relocate_center(state: BlockPeps, row: int):
    """move alpha along its column using exact qr factorizations."""
    moves = 0
    while state.center[0] != int(row):
        step = 1 if state.center[0] < int(row) else -1
        target = (state.center[0] + step, state.center[1])
        state = move_block_center(state, target, inplace=True)
        moves += 1
    return state, moves


def _move_column(
    state: BlockPeps,
    column: int,
    direction: int,
    backend: str,
    column_options: dict,
    rng,
):
    split = "right" if direction == 1 else "left"
    if backend == "rmps_shared_q":
        return block_column_move(
            state,
            column,
            split=split,
            ell=column_options.get("ell"),
            eta=int(column_options["eta"]),
            kappa=int(column_options["kappa"]),
            chi_sk=int(column_options["chi_sk"]),
            ndis=int(column_options.get("ndis", 0)),
            absorption_max_bond=column_options.get("absorption_max_bond"),
            absorption_cutoff=float(column_options.get("absorption_cutoff", 0.0)),
            rng=rng,
            inplace=True,
        )
    return block_local_column_move(
        state,
        column,
        split=split,
        chi=int(column_options["chi"]),
        eta=int(column_options["eta"]),
        cutoff=float(column_options.get("cutoff", 0.0)),
        ndis=int(column_options.get("ndis", 0)),
        disentangler=str(column_options.get("disentangler", "altmin")),
        absorption_max_bond=column_options.get("absorption_max_bond"),
        absorption_cutoff=float(column_options.get("absorption_cutoff", 0.0)),
        inplace=True,
    )


def _vertical_pass(
    state: BlockPeps,
    gates: dict,
    *,
    direction: int,
    backend: str,
    gate_options: dict,
    column_options: dict,
    rng,
    orientation: str,
    resume: dict | None = None,
    progress_callback=None,
):
    progress = {} if resume is None else dict(resume)
    completed_columns = int(progress.get("completed_columns", 0))
    boundary = 0 if direction == 1 else state.peps.Ly - 1
    if completed_columns == 0 and state.center[1] != boundary:
        raise ValueError("the block center is not on the incoming column boundary")
    incoming_row = int(progress.get("incoming_row", state.center[0]))
    if incoming_row not in (0, state.peps.Lx - 1):
        raise ValueError("the block center must enter a sweep at a row endpoint")
    bonds = {frozenset(where): (where, gate) for where, gate in gates.items()}
    columns = tuple(
        range(state.peps.Ly)
        if direction == 1
        else range(state.peps.Ly - 1, -1, -1)
    )
    if not 0 <= completed_columns <= len(columns):
        raise ValueError("vertical-pass checkpoint has an invalid column count")
    gate_records = list(progress.get("gate_records", ()))
    column_records = list(progress.get("column_records", ()))
    center_qr_moves = int(progress.get("center_qr_moves", 0))
    out = state
    if completed_columns:
        expected_column = (
            columns[completed_columns]
            if completed_columns < len(columns)
            else columns[-1]
        )
        if out.center != (incoming_row, expected_column):
            raise ValueError("vertical-pass checkpoint has an inconsistent center")
    for column_index in range(completed_columns, len(columns)):
        column = columns[column_index]
        if out.center[1] != column:
            raise ValueError("the block center did not follow the column sweep")
        row_direction = 1 if out.center[0] == 0 else -1
        if out.center[0] not in (0, out.peps.Lx - 1):
            raise ValueError("the block center must enter a column at an endpoint")
        rows = (
            range(out.peps.Lx - 1)
            if row_direction == 1
            else range(out.peps.Lx - 2, -1, -1)
        )
        for lower in rows:
            pair = ((lower, column), (lower + 1, column))
            original_where, gate = bonds[frozenset(pair)]
            target = (lower + 1 if row_direction == 1 else lower, column)
            info = {}
            out = apply_block_gate(
                out,
                gate,
                original_where,
                move_to=target,
                max_bond=gate_options.get("max_bond"),
                cutoff=float(gate_options.get("cutoff", 0.0)),
                info=info,
                inplace=True,
            )
            gate_records.append({
                "orientation": orientation,
                "where": [list(site) for site in original_where],
                **info,
            })
        if 0 <= column + direction < out.peps.Ly:
            out, record = _move_column(
                out, column, direction, backend, column_options, rng
            )
            out, moves = _relocate_center(out, incoming_row)
            center_qr_moves += moves
            record["center_qr_moves"] = int(moves)
            column_records.append({"orientation": orientation, **record})
        else:
            out, moves = _relocate_center(out, incoming_row)
            center_qr_moves += moves
        if progress_callback is not None:
            progress_callback(out, {
                "completed_columns": column_index + 1,
                "incoming_row": incoming_row,
                "gate_records": list(gate_records),
                "column_records": list(column_records),
                "center_qr_moves": center_qr_moves,
            })
    return out, gate_records, column_records, center_qr_moves


def _orthonormalize_center(state: BlockPeps):
    gram = block_gram(state)
    eigenvalues = np.linalg.eigvalsh((gram + gram.conj().T) / 2.0)
    if eigenvalues[0] <= np.finfo(float).eps * max(float(eigenvalues[-1]), 1.0):
        raise ValueError("the block center is rank deficient")
    out, _ = orthonormalize_block(state, inplace=True)
    return out, {
        "gram_min_eigenvalue": float(eigenvalues[0]),
        "gram_max_eigenvalue": float(eigenvalues[-1]),
        "gram_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "gram_identity_error": float(
            np.linalg.norm(block_gram(out) - np.eye(out.size))
        ),
        "method": "orthogonality_center_qr",
    }


def _finalize_iteration(state: BlockPeps, gram_options: dict):
    """restore the lattice orientation and validate the outgoing block."""
    state = rotate_block_lattice(state, turns=3, inplace=True)
    state, gram_record = _orthonormalize_center(state)
    if gram_options.get("validate_boundary"):
        from rand_isopeps.physics.block_measurements import boundary_block_gram

        boundary = boundary_block_gram(
            state,
            max_bond=gram_options.get("max_bond"),
            cutoff=float(gram_options.get("cutoff", 0.0)),
        )
        gram_record["boundary_gram_error"] = float(
            np.linalg.norm(boundary - np.eye(state.size))
        )
    validate_block_state(state)
    return state, gram_record


def _validate_iteration_inputs(
    state: BlockPeps,
    tau: float,
    direction: int,
    backend: str,
    gate_options: dict | None,
    column_options: dict | None,
    gram_options: dict | None,
    resume: dict | None,
):
    """validate one block sweep and copy its mutable option mappings."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if backend not in ("rmps_shared_q", "local_shared_q"):
        raise ValueError("unsupported shared-q backend")
    validate_block_state(state)
    gate_options = {} if gate_options is None else dict(gate_options)
    column_options = {} if column_options is None else dict(column_options)
    gram_options = {} if gram_options is None else dict(gram_options)
    progress = {} if resume is None else dict(resume)
    phase = progress.get("phase")
    if phase not in (None, "vertical", "horizontal"):
        raise ValueError("block checkpoint has an invalid phase")
    for key, expected in (("direction", int(direction)), ("backend", backend)):
        if key in progress and progress[key] != expected:
            raise ValueError(f"block checkpoint has a stale {key}")
    if "tau" in progress and float(progress["tau"]) != float(tau):
        raise ValueError("block checkpoint has a stale tau")
    return gate_options, column_options, gram_options, progress, phase


def _elapsed_clock(started: float, elapsed_before: float):
    def elapsed():
        return elapsed_before + float(perf_counter() - started)

    return elapsed


def _pass_callback(
    callback,
    *,
    phase: str,
    direction: int,
    backend: str,
    tau: float,
    elapsed,
    first_pass: tuple[list, list, int, int] | None = None,
):
    """adapt a column checkpoint to the public iteration checkpoint schema."""
    def report(current, pass_progress):
        if callback is None:
            return
        checkpoint = {
            "phase": phase,
            "direction": int(direction),
            "backend": backend,
            "tau": float(tau),
            "elapsed_runtime_s": elapsed(),
        }
        if first_pass is not None:
            first_gates, first_columns, first_qrs, old_ly = first_pass
            checkpoint.update({
                "old_ly": old_ly,
                "first_gates": list(first_gates),
                "first_columns": list(first_columns),
                "first_qrs": first_qrs,
            })
        checkpoint["pass"] = pass_progress
        callback(current, checkpoint)

    return report


def _first_orientation(
    state: BlockPeps,
    vertical_gates: dict,
    *,
    phase: str | None,
    progress: dict,
    direction: int,
    backend: str,
    tau: float,
    gate_options: dict,
    column_options: dict,
    rng,
    elapsed,
    progress_callback,
):
    """run or restore the vertical half of the oriented sweep."""
    if phase == "horizontal":
        return (
            state,
            list(progress["first_gates"]),
            list(progress["first_columns"]),
            int(progress["first_qrs"]),
            int(progress["old_ly"]),
        )
    callback = _pass_callback(
        progress_callback,
        phase="vertical",
        direction=direction,
        backend=backend,
        tau=tau,
        elapsed=elapsed,
    )
    state, gates, columns, qrs = _vertical_pass(
        state,
        vertical_gates,
        direction=direction,
        backend=backend,
        gate_options=gate_options,
        column_options=column_options,
        rng=rng,
        orientation="vertical",
        resume=progress.get("pass") if phase == "vertical" else None,
        progress_callback=callback,
    )
    old_ly = int(state.peps.Ly)
    return rotate_block_lattice(state, inplace=True), gates, columns, qrs, old_ly


def _second_orientation(
    state: BlockPeps,
    horizontal_gates: dict,
    first_pass: tuple[list, list, int, int],
    *,
    phase: str | None,
    progress: dict,
    direction: int,
    backend: str,
    tau: float,
    gate_options: dict,
    column_options: dict,
    rng,
    elapsed,
    progress_callback,
):
    """run the horizontal bonds after rotating them into column order."""
    first_gates, first_columns, first_qrs, old_ly = first_pass
    rotated_gates = {
        _rotate_where(where, old_ly): gate
        for where, gate in horizontal_gates.items()
    }
    callback = _pass_callback(
        progress_callback,
        phase="horizontal",
        direction=direction,
        backend=backend,
        tau=tau,
        elapsed=elapsed,
        first_pass=first_pass,
    )
    state, gates, columns, qrs = _vertical_pass(
        state,
        rotated_gates,
        direction=direction,
        backend=backend,
        gate_options=gate_options,
        column_options=column_options,
        rng=rng,
        orientation="horizontal",
        resume=progress.get("pass") if phase == "horizontal" else None,
        progress_callback=callback,
    )
    return state, first_gates + gates, first_columns + columns, first_qrs + qrs


def _iteration_record(
    state: BlockPeps,
    gate_records: list,
    column_records: list,
    center_qr_moves: int,
    gram_record: dict,
    *,
    backend: str,
    tau: float,
    direction: int,
    elapsed,
):
    return {
        "backend": backend,
        "tau": float(tau),
        "trotter_order": 1,
        "direction": int(direction),
        "next_direction": -int(direction),
        "gate_order": "forward" if direction == 1 else "reverse",
        "sweep_schedule": "alternating_forward_reverse",
        "gate_count": len(gate_records),
        "gate_discarded_weight": float(
            sum(row["discarded_weight"] for row in gate_records)
        ),
        "center_qr_moves": int(center_qr_moves),
        "column_moves": len(column_records),
        "max_bond": int(state.peps.max_bond()),
        "update_runtime_s": elapsed(),
        "gram": gram_record,
        "gates": gate_records,
        "columns": column_records,
    }


def block_tebd_iteration(
    state: BlockPeps,
    ham,
    tau: float,
    *,
    direction: int,
    backend: str = "rmps_shared_q",
    gate_options: dict | None = None,
    column_options: dict | None = None,
    gram_options: dict | None = None,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
    resume: dict | None = None,
    progress_callback=None,
):
    """apply one oriented split-exponential sweep to a shared state block.

    the center finishes at the opposite corner, so repeated fixed-bond updates
    alternate forward and reverse gate order. returning it horizontally would
    require another approximate column move rather than an exact qr. progress
    callbacks expose completed columns; an individual gate or column
    factorization remains atomic because it has no stable partial state.
    """
    gate_options, column_options, gram_options, progress, phase = (
        _validate_iteration_inputs(
            state,
            tau,
            direction,
            backend,
            gate_options,
            column_options,
            gram_options,
            resume,
        )
    )
    out = state if inplace else state.copy()
    gates = _gates(ham, float(tau))
    vertical = {where: gate for where, gate in gates.items() if _vertical(where)}
    horizontal = {where: gate for where, gate in gates.items() if not _vertical(where)}
    started = perf_counter()
    elapsed_before = float(progress.get("elapsed_runtime_s", 0.0))
    elapsed = _elapsed_clock(started, elapsed_before)
    first_pass = _first_orientation(
        out,
        vertical,
        phase=phase,
        progress=progress,
        direction=direction,
        backend=backend,
        tau=tau,
        gate_options=gate_options,
        column_options=column_options,
        rng=rng,
        elapsed=elapsed,
        progress_callback=progress_callback,
    )
    out, gate_records, column_records, center_qr_moves = _second_orientation(
        first_pass[0],
        horizontal,
        first_pass[1:],
        phase=phase,
        progress=progress,
        direction=direction,
        backend=backend,
        tau=tau,
        gate_options=gate_options,
        column_options=column_options,
        rng=rng,
        elapsed=elapsed,
        progress_callback=progress_callback,
    )
    out, gram_record = _finalize_iteration(out, gram_options)
    record = _iteration_record(
        out,
        gate_records,
        column_records,
        center_qr_moves,
        gram_record,
        backend=backend,
        tau=tau,
        direction=direction,
        elapsed=elapsed,
    )
    return out, record
