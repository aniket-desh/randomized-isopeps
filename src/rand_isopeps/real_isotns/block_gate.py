"""full-rank two-site gates for a block-isopeps center."""

from __future__ import annotations

from rand_isopeps.physics.block_state import BlockPeps, Site, validate_block_state


def _site_tag(peps, site: Site) -> str:
    return peps.site_tag_id.format(*site)


def _reduce_site(tensor, active_inds):
    environment = tuple(index for index in tensor.inds if index not in active_inds)
    if not environment:
        return None, tensor
    return tensor.split(
        left_inds=environment,
        method="qr",
        absorb="right",
    )


def apply_block_gate(
    state: BlockPeps,
    gate,
    where: tuple[Site, Site],
    *,
    move_to: Site,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    info: dict | None = None,
    reduced: bool = True,
    inplace: bool = False,
) -> BlockPeps:
    """apply a two-site gate at the block center with a reduced qr update."""
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("block gates require quimb") from exc

    first, second = (tuple(map(int, site)) for site in where)
    target = tuple(map(int, move_to))
    if target not in (first, second):
        raise ValueError("move_to must be one of the gated sites")
    if state.center not in (first, second):
        raise ValueError("the gate must touch the block center")
    if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
        raise ValueError("the gated sites must be nearest neighbours")

    out = state if inplace else state.copy()
    peps = out.peps
    first_tensor = peps[_site_tag(peps, first)]
    second_tensor = peps[_site_tag(peps, second)]
    bonds = tuple(first_tensor.bonds(second_tensor))
    if len(bonds) != 1:
        raise ValueError("the gated sites must share exactly one bond")
    bond = bonds[0]
    first_phys = peps.site_ind(first)
    second_phys = peps.site_ind(second)
    if first_tensor.ind_size(first_phys) != second_tensor.ind_size(second_phys):
        raise ValueError("the two physical dimensions must match")
    dimension = first_tensor.ind_size(first_phys)

    if reduced:
        first_frame, first_core = _reduce_site(
            first_tensor,
            {bond, first_phys, out.block_ind},
        )
        second_frame, second_core = _reduce_site(
            second_tensor,
            {bond, second_phys, out.block_ind},
        )
    else:
        first_frame, first_core = None, first_tensor
        second_frame, second_core = None, second_tensor

    gate_data = gate.reshape(dimension, dimension, dimension, dimension)
    first_out, second_out = qtn.rand_uuid(), qtn.rand_uuid()
    gate_tensor = qtn.Tensor(
        gate_data,
        inds=(first_out, second_out, first_phys, second_phys),
    )
    combined = first_core @ second_core @ gate_tensor
    combined.reindex_({first_out: first_phys, second_out: second_phys})

    first_inds = [
        index
        for index in first_core.inds
        if index not in (bond, out.block_ind)
    ]
    if target == first:
        first_inds.append(out.block_ind)
    absorb = -1 if target == first else 1
    split_info = {"error": None} if info is not None else None
    left, right = combined.split(
        left_inds=first_inds,
        method="svd",
        absorb=absorb,
        cutoff=float(cutoff),
        max_bond=max_bond,
        bond_ind=bond,
        info=split_info,
    )
    first_updated = left if first_frame is None else first_frame @ left
    second_updated = right if second_frame is None else second_frame @ right
    first_tensor.modify(data=first_updated.data, inds=first_updated.inds)
    second_tensor.modify(data=second_updated.data, inds=second_updated.inds)
    out.center = target
    if info is not None:
        error = split_info.get("error", 0.0)
        if error is None:
            error = 0.0
        info["discarded_weight"] = float(abs(error) ** 2)
        info["bond_dimension"] = int(left.ind_size(bond))
        info["update"] = "reduced" if reduced else "full"
    validate_block_state(out)
    return out
