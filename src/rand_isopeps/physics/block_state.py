"""small block-isopeps state operations at the orthogonality center."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Site = tuple[int, int]


@dataclass
class BlockPeps:
    """a peps with one open block index at its orthogonality center."""

    peps: object
    block_ind: str
    center: Site

    def __post_init__(self) -> None:
        self.center = tuple(map(int, self.center))
        validate_block_state(self)

    @property
    def size(self) -> int:
        """return the number of states represented by the open block index."""
        return int(_center_tensor(self).ind_size(self.block_ind))

    def copy(self) -> "BlockPeps":
        """copy the tensor network and retain the block metadata."""
        return BlockPeps(self.peps.copy(), self.block_ind, self.center)


def _site_tag(peps, site: Site) -> str:
    return peps.site_tag_id.format(*site)


def _center_tensor(state: BlockPeps):
    tensors = state.peps.select_tensors(
        tags=[_site_tag(state.peps, state.center)], which="all"
    )
    if len(tensors) != 1:
        raise ValueError("the block center must contain exactly one tensor")
    return tensors[0]


def validate_block_state(state: BlockPeps) -> None:
    """raise unless exactly one tensor owns the block index at ``center``."""
    peps = state.peps
    x, y = state.center
    if not (0 <= x < peps.Lx and 0 <= y < peps.Ly):
        raise ValueError("the block center lies outside the PEPS")
    owners = [tensor for tensor in peps.tensors if state.block_ind in tensor.inds]
    if len(owners) != 1:
        raise ValueError("the block index must belong to exactly one tensor")
    center = _center_tensor(state)
    if owners[0] is not center:
        raise ValueError("the block index owner does not match center")
    if center.ind_size(state.block_ind) < 1:
        raise ValueError("the block index must have positive size")


def _ordered_center(state: BlockPeps):
    tensor = _center_tensor(state)
    other_inds = tuple(index for index in tensor.inds if index != state.block_ind)
    ordered = tensor.transpose(*other_inds, state.block_ind)
    return tensor, other_inds, np.asarray(ordered.data)


def dense_block(state: BlockPeps) -> np.ndarray:
    """contract the peps into a matrix with one state per column."""
    validate_block_state(state)
    physical = tuple(
        state.peps.site_ind((x, y))
        for x in range(state.peps.Lx)
        for y in range(state.peps.Ly)
    )
    block = np.asarray(state.peps.to_dense(physical, (state.block_ind,)))
    scale = 10.0 ** float(getattr(state.peps, "exponent", 0.0))
    return block.reshape(-1, state.size) * scale


def block_gram(state: BlockPeps) -> np.ndarray:
    """compute the exact block gram matrix from the center tensor."""
    validate_block_state(state)
    _, _, data = _ordered_center(state)
    matrix = data.reshape(-1, state.size)
    gram = matrix.conj().T @ matrix
    scale = 10.0 ** (2.0 * float(getattr(state.peps, "exponent", 0.0)))
    return gram * scale


def _replace_center_data(state: BlockPeps, ordered_data: np.ndarray) -> None:
    tensor, other_inds, _ = _ordered_center(state)
    ordered_inds = (*other_inds, state.block_ind)
    ordered_shape = tuple(tensor.ind_size(index) for index in ordered_inds)
    data = ordered_data.reshape(ordered_shape)
    permutation = tuple(ordered_inds.index(index) for index in tensor.inds)
    tensor.modify(data=np.transpose(data, permutation))


def orthonormalize_block(
    state: BlockPeps, *, inplace: bool = False
) -> tuple[BlockPeps, np.ndarray]:
    """orthonormalize the center columns and return the removed factor."""
    out = state if inplace else state.copy()
    _, _, data = _ordered_center(out)
    matrix = data.reshape(-1, out.size)
    q, r = np.linalg.qr(matrix, mode="reduced")
    if q.shape[1] != out.size or np.linalg.matrix_rank(r) != out.size:
        raise ValueError("the block center is rank deficient")
    exponent = float(getattr(out.peps, "exponent", 0.0))
    _replace_center_data(out, q)
    out.peps.exponent = 0.0
    validate_block_state(out)
    return out, r * (10.0 ** exponent)


def rotate_block(
    state: BlockPeps, rotation: np.ndarray, *, inplace: bool = False
) -> BlockPeps:
    """apply a square change of basis to the block-state columns."""
    transform = np.asarray(rotation)
    if transform.shape != (state.size, state.size):
        raise ValueError("rotation must have shape (block_size, block_size)")
    out = state if inplace else state.copy()
    _, _, data = _ordered_center(out)
    matrix = data.reshape(-1, out.size) @ transform
    _replace_center_data(out, matrix)
    validate_block_state(out)
    return out


def move_block_center(
    state: BlockPeps, target: Site, *, inplace: bool = False
) -> BlockPeps:
    """move the block index to a neighboring site with an exact qr."""
    target = tuple(map(int, target))
    source = state.center
    if abs(source[0] - target[0]) + abs(source[1] - target[1]) != 1:
        raise ValueError("the target must neighbor the block center")
    if not (0 <= target[0] < state.peps.Lx and 0 <= target[1] < state.peps.Ly):
        raise ValueError("the target lies outside the PEPS")

    out = state if inplace else state.copy()
    peps = out.peps
    source_tensor = peps[_site_tag(peps, source)]
    target_tensor = peps[_site_tag(peps, target)]
    bonds = tuple(source_tensor.bonds(target_tensor))
    if len(bonds) != 1:
        raise ValueError("the center and target must share exactly one bond")
    bond = bonds[0]
    source_inds = tuple(
        index
        for index in source_tensor.inds
        if index not in (bond, out.block_ind)
    )
    combined = source_tensor @ target_tensor
    left, right = combined.split(
        left_inds=source_inds,
        method="qr",
        absorb="right",
        bond_ind=bond,
    )
    source_tensor.modify(data=left.data, inds=left.inds)
    target_tensor.modify(data=right.data, inds=right.inds)
    out.center = target
    validate_block_state(out)
    return out


def _rotate_site(site: Site, lx: int, ly: int) -> Site:
    x, y = site
    return ly - 1 - y, x


def rotate_block_lattice(
    state: BlockPeps,
    turns: int = 1,
    *,
    inplace: bool = False,
) -> BlockPeps:
    """rotate the lattice counterclockwise while preserving the represented block."""
    out = state if inplace else state.copy()
    for _ in range(int(turns) % 4):
        peps = out.peps
        lx, ly = int(peps.Lx), int(peps.Ly)
        tag_map = {}
        index_map = {}
        for x in range(lx):
            for y in range(ly):
                new_x, new_y = _rotate_site((x, y), lx, ly)
                tag_map[peps.site_tag_id.format(x, y)] = peps.site_tag_id.format(
                    new_x, new_y
                )
                index_map[peps.site_ind_id.format(x, y)] = peps.site_ind_id.format(
                    new_x, new_y
                )
        for x in range(lx):
            tag_map[peps.x_tag_id.format(x)] = peps.y_tag_id.format(x)
        for y in range(ly):
            tag_map[peps.y_tag_id.format(y)] = peps.x_tag_id.format(ly - 1 - y)
        peps.retag_(tag_map)
        peps.reindex_(index_map)
        peps._Lx, peps._Ly = ly, lx
        out.center = _rotate_site(out.center, lx, ly)
    validate_block_state(out)
    return out
