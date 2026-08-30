"""whole-column shared-q moves for a block-isopeps."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rand_isopeps.column.operator import ColumnOperator, mpo_frobenius_norm
from rand_isopeps.physics.block_state import BlockPeps, validate_block_state
from rand_isopeps.sketching.column import factorization_metrics, factorize_column

from .column_bridge import compress_column, validate_peps_structure


@dataclass(frozen=True)
class BlockColumnLayout:
    """index layout needed to return a factored block column to a peps."""

    column: int
    split: str
    block_row: int
    block_ind: str
    block_size: int
    toward_dims: tuple[int, ...]

    @property
    def next_column(self) -> int:
        return self.column + (1 if self.split == "right" else -1)

    @property
    def next_center(self) -> tuple[int, int]:
        return self.block_row, self.next_column


def _tag(peps, x: int, y: int) -> str:
    return peps.site_tag_id.format(x, y)


def _bond(peps, first: str, second: str):
    shared = tuple(index for index in peps[first].inds if index in peps[second].inds)
    return shared[0] if shared else None


def _physical_index(peps, x: int, y: int) -> str:
    return peps.site_ind((x, y))


def _direction(split: str) -> int:
    if split not in ("right", "left"):
        raise ValueError("split must be 'right' or 'left'")
    return 1 if split == "right" else -1


def extract_block_column(
    state: BlockPeps,
    column: int,
    *,
    split: str,
    normalize: bool = False,
) -> tuple[ColumnOperator, BlockColumnLayout]:
    """extract one column with the block leg fused into its local input."""
    validate_block_state(state)
    peps = state.peps
    direction = _direction(split)
    next_column = int(column) + direction
    away_column = int(column) - direction
    if state.center[1] != int(column):
        raise ValueError("the block center must lie on the extracted column")
    if not (0 <= next_column < peps.Ly):
        raise ValueError("the residual would be pushed off the lattice")

    cores = []
    toward_dims = []
    for x in range(peps.Lx):
        tensor = peps[_tag(peps, x, int(column))]
        physical = _physical_index(peps, x, int(column))
        toward = _bond(
            peps,
            _tag(peps, x, int(column)),
            _tag(peps, x, next_column),
        )
        if toward is None:
            raise ValueError("an active site has no bond toward the next column")
        away = (
            _bond(
                peps,
                _tag(peps, x, int(column)),
                _tag(peps, x, away_column),
            )
            if 0 <= away_column < peps.Ly
            else None
        )
        up = (
            _bond(peps, _tag(peps, x, int(column)), _tag(peps, x - 1, int(column)))
            if x > 0
            else None
        )
        down = (
            _bond(peps, _tag(peps, x, int(column)), _tag(peps, x + 1, int(column)))
            if x + 1 < peps.Lx
            else None
        )
        is_block_row = x == state.center[0]
        order = [index for index in (up, physical, away, toward) if index is not None]
        if is_block_row:
            order.append(state.block_ind)
        if down is not None:
            order.append(down)
        array = np.asarray(tensor.transpose(*order).data)

        up_dim = tensor.ind_size(up) if up is not None else 1
        physical_dim = tensor.ind_size(physical)
        away_dim = tensor.ind_size(away) if away is not None else 1
        toward_dim = tensor.ind_size(toward)
        down_dim = tensor.ind_size(down) if down is not None else 1
        block_dim = state.size if is_block_row else 1
        core = array.reshape(
            up_dim,
            physical_dim * away_dim,
            toward_dim * block_dim,
            down_dim,
        )
        cores.append(core)
        toward_dims.append(toward_dim)

    if normalize:
        norm = mpo_frobenius_norm(cores)
        if norm > 0.0:
            cores[0] = cores[0] / norm
    layout = BlockColumnLayout(
        column=int(column),
        split=split,
        block_row=int(state.center[0]),
        block_ind=state.block_ind,
        block_size=state.size,
        toward_dims=tuple(toward_dims),
    )
    return ColumnOperator(cores), layout


def _squeeze_named_axes(data, names):
    for axis in reversed(range(len(names))):
        if names[axis] is None:
            data = np.squeeze(data, axis=axis)
    return data, [name for name in names if name is not None]


def insert_block_factorization(
    state: BlockPeps,
    q_cores,
    residual_cores,
    layout: BlockColumnLayout,
    *,
    inplace: bool = False,
) -> BlockPeps:
    """insert a shared q and move the unfused block leg through the residual."""
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("block-column insertion requires quimb") from exc

    validate_block_state(state)
    if state.block_ind != layout.block_ind or state.center != (
        layout.block_row,
        layout.column,
    ):
        raise ValueError("the layout does not describe the supplied block state")
    peps = state.peps
    if len(q_cores) != peps.Lx or len(residual_cores) != peps.Lx:
        raise ValueError("the factorization height does not match the PEPS")
    if len(layout.toward_dims) != peps.Lx:
        raise ValueError("the layout height does not match the PEPS")

    direction = _direction(layout.split)
    next_column = layout.next_column
    away_column = layout.column - direction
    if not (0 <= next_column < peps.Ly):
        raise ValueError("the residual would be pushed off the lattice")

    source = peps.copy()
    out_peps = peps if inplace else peps.copy()
    q_vertical = [qtn.rand_uuid() for _ in range(max(peps.Lx - 1, 0))]
    r_vertical = [qtn.rand_uuid() for _ in range(max(peps.Lx - 1, 0))]
    horizontal = [qtn.rand_uuid() for _ in range(peps.Lx)]

    for x, (q_core, residual_core) in enumerate(zip(q_cores, residual_cores)):
        active = source[_tag(source, x, layout.column)]
        neighbour = source[_tag(source, x, next_column)]
        physical = _physical_index(source, x, layout.column)
        toward = _bond(
            source,
            _tag(source, x, layout.column),
            _tag(source, x, next_column),
        )
        away = (
            _bond(
                source,
                _tag(source, x, layout.column),
                _tag(source, x, away_column),
            )
            if 0 <= away_column < source.Ly
            else None
        )
        if toward is None:
            raise ValueError("an active site has no bond toward the next column")
        toward_dim = int(layout.toward_dims[x])
        if toward_dim != active.ind_size(toward):
            raise ValueError("the stored toward-bond dimension is stale")
        block_dim = layout.block_size if x == layout.block_row else 1
        if residual_core.shape[2] != toward_dim * block_dim:
            raise ValueError("the residual input does not match the fused block layout")

        physical_dim = active.ind_size(physical)
        away_dim = active.ind_size(away) if away is not None else 1
        if q_core.shape[1] != physical_dim * away_dim:
            raise ValueError("q output does not match the retained peps legs")

        q_data = np.asarray(q_core).reshape(
            q_core.shape[0],
            physical_dim,
            away_dim,
            q_core.shape[2],
            q_core.shape[3],
        )
        q_names = [
            q_vertical[x - 1] if x > 0 else None,
            physical,
            away,
            horizontal[x],
            q_vertical[x] if x + 1 < peps.Lx else None,
        ]
        q_data, q_inds = _squeeze_named_axes(q_data, q_names)

        residual_data = np.asarray(residual_core).reshape(
            residual_core.shape[0],
            residual_core.shape[1],
            toward_dim,
            block_dim,
            residual_core.shape[3],
        )
        residual_names = [
            r_vertical[x - 1] if x > 0 else None,
            horizontal[x],
            toward,
            layout.block_ind if x == layout.block_row else None,
            r_vertical[x] if x + 1 < peps.Lx else None,
        ]
        residual_data, residual_inds = _squeeze_named_axes(
            residual_data, residual_names
        )

        q_tensor = qtn.Tensor(q_data, inds=q_inds)
        residual_tensor = qtn.Tensor(residual_data, inds=residual_inds)
        active_out = out_peps[_tag(out_peps, x, layout.column)]
        neighbour_out = out_peps[_tag(out_peps, x, next_column)]
        active_out.modify(data=q_tensor.data, inds=q_tensor.inds)
        absorbed = residual_tensor @ neighbour
        neighbour_out.modify(data=absorbed.data, inds=absorbed.inds)

    moved = BlockPeps(out_peps, layout.block_ind, layout.next_center)
    return moved


def block_column_move(
    state: BlockPeps,
    column: int,
    *,
    split: str,
    ell: int | None,
    eta: int,
    kappa: int,
    chi_sk: int,
    ndis: int = 0,
    absorption_max_bond: int | None = None,
    absorption_cutoff: float = 0.0,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
):
    """factor one whole block column once, then insert and compress it."""
    started = perf_counter()
    column_operator, layout = extract_block_column(
        state, column, split=split, normalize=False
    )
    sketch_width = column_operator.n_in if ell is None else int(ell)
    q_cores, residual_cores, result = factorize_column(
        column_operator,
        ell=sketch_width,
        eta=eta,
        kappa=kappa,
        chi_sk=chi_sk,
        ndis=ndis,
        rng=rng,
    )
    moved = insert_block_factorization(
        state,
        q_cores,
        residual_cores,
        layout,
        inplace=inplace,
    )
    compressed, absorption = compress_column(
        moved.peps,
        layout.next_column,
        max_bond=absorption_max_bond,
        cutoff=absorption_cutoff,
        inplace=True,
    )
    moved = BlockPeps(compressed, layout.block_ind, layout.next_center)
    validate_peps_structure(moved.peps)
    metrics = factorization_metrics(result)
    metrics.update(absorption)
    metrics.update(
        {
            "block_size": int(layout.block_size),
            "column": int(layout.column),
            "next_column": int(layout.next_column),
            "split": layout.split,
            "total_runtime_s": float(perf_counter() - started),
        }
    )
    return moved, metrics
