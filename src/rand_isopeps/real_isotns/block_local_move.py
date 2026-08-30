"""block-aware sequential moses column moves."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from rand_isopeps.physics.block_state import BlockPeps, validate_block_state

from .column_bridge import validate_peps_structure
from .moses_move import moses_move


def _sweep_for_center(state: BlockPeps) -> tuple[str, int]:
    row = state.center[0]
    if row == state.peps.Lx - 1:
        return "up", 0
    if row == 0:
        return "down", state.peps.Lx - 1
    raise ValueError("a local block move requires alpha at a column endpoint")


def block_local_column_move(
    state: BlockPeps,
    column: int,
    *,
    split: str,
    chi: int,
    eta: int,
    cutoff: float,
    ndis: int,
    disentangler: str = "altmin",
    absorption_max_bond: int | None = None,
    absorption_cutoff: float = 0.0,
    inplace: bool = False,
):
    """move alpha through the carrier and top residual into the next column."""
    validate_block_state(state)
    if state.center[1] != int(column):
        raise ValueError("the block center must lie on the moved column")
    direction = 1 if split == "right" else -1 if split == "left" else 0
    if direction == 0:
        raise ValueError("split must be 'right' or 'left'")
    next_column = int(column) + direction
    if not (0 <= next_column < state.peps.Ly):
        raise ValueError("the residual would be pushed off the lattice")
    sweep, next_row = _sweep_for_center(state)
    out = state if inplace else state.copy()
    before = int(out.peps.max_bond())
    started = perf_counter()
    errors = moses_move(
        out.peps,
        int(column),
        int(chi),
        int(eta),
        float(cutoff),
        int(ndis),
        orientation="col",
        sweep=sweep,
        split=split,
        renorm=False,
        rand=None,
        absorb_max_bond=absorption_max_bond,
        absorb_cutoff=float(absorption_cutoff),
        block_ind=out.block_ind,
        disentangler=disentangler,
    )
    moved = BlockPeps(out.peps, out.block_ind, (next_row, next_column))
    validate_peps_structure(moved.peps)
    return moved, {
        "backend": "local_shared_q",
        "column": int(column),
        "next_column": int(next_column),
        "split": split,
        "vertical_sweep": sweep,
        "block_size": int(moved.size),
        "bond_before": before,
        "bond_after": int(moved.peps.max_bond()),
        "local_error_squared": float(
            np.sum(np.asarray(errors, dtype=float) ** 2)
        ),
        "total_runtime_s": float(perf_counter() - started),
    }
