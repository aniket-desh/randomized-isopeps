"""Convert a sketched column factorization back into a standard quimb PEPS."""

from __future__ import annotations

import math

import numpy as np


def _tag(psi, x: int, y: int) -> str:
    return psi.site_tag_id.format(x, y)


def _bond(psi, a_tag: str, b_tag: str):
    a, b = psi[a_tag], psi[b_tag]
    shared = tuple(index for index in a.inds if index in b.inds)
    return shared[0] if shared else None


def _physical_index(psi, x: int, y: int) -> str:
    try:
        return psi.site_ind((x, y))
    except Exception:
        return f"k{x},{y}"


def extract_column(psi, j: int, *, split: str):
    """Extract the current orthogonality column without rescaling it."""
    from rand_isopeps.column.from_quimb import from_quimb_column

    return from_quimb_column(psi, j, split=split, normalize=False)


def insert_column_factorization(
    psi,
    j: int,
    q_cores,
    residual_cores,
    *,
    split: str,
    inplace: bool = False,
):
    """Insert ``Q`` and absorb ``Q* C`` sitewise into the next column.

    For an interior column, ``Q``'s fused output is reshaped back into the
    physical leg and the existing bond away from the move.  The target column
    temporarily has parallel vertical bonds; :func:`compress_column` converts
    those into one controlled PEPS bond.
    """
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("column insertion requires quimb") from exc

    if split not in ("right", "left"):
        raise ValueError("split must be 'right' or 'left'")
    if len(q_cores) != psi.Lx or len(residual_cores) != psi.Lx:
        raise ValueError("factorization height does not match the PEPS")
    direction = 1 if split == "right" else -1
    j_next, j_away = j + direction, j - direction
    if not (0 <= j_next < psi.Ly):
        raise ValueError("residual would be pushed off the lattice")

    source = psi.copy()
    out = psi if inplace else psi.copy()
    q_vertical = [qtn.rand_uuid() for _ in range(max(psi.Lx - 1, 0))]
    r_vertical = [qtn.rand_uuid() for _ in range(max(psi.Lx - 1, 0))]
    horizontal = [qtn.rand_uuid() for _ in range(psi.Lx)]

    active_old = [source[_tag(source, x, j)].copy() for x in range(source.Lx)]
    neighbour_old = [source[_tag(source, x, j_next)].copy() for x in range(source.Lx)]
    for x, (q_core, r_core, active, neighbour) in enumerate(
        zip(q_cores, residual_cores, active_old, neighbour_old)
    ):
        phys = _physical_index(source, x, j)
        away = (
            _bond(source, _tag(source, x, j), _tag(source, x, j_away))
            if 0 <= j_away < source.Ly else None
        )
        toward = _bond(source, _tag(source, x, j), _tag(source, x, j_next))
        if toward is None or r_core.shape[2] != active.ind_size(toward):
            raise ValueError("residual input leg does not match the original horizontal bond")

        phys_dim = active.ind_size(phys)
        away_dim = active.ind_size(away) if away is not None else 1
        if q_core.shape[1] != phys_dim * away_dim:
            raise ValueError("Q output cannot be unfused into physical and away-bond legs")

        q_data = np.asarray(q_core).reshape(
            q_core.shape[0], phys_dim, away_dim, q_core.shape[2], q_core.shape[3]
        )
        q_inds: list[str] = []
        if x == 0:
            q_data = np.squeeze(q_data, axis=0)
        else:
            q_inds.append(q_vertical[x - 1])
        q_inds.append(phys)
        if away is None:
            away_axis = 1 if x == 0 else 2
            q_data = np.squeeze(q_data, axis=away_axis)
        else:
            q_inds.append(away)
        q_inds.append(horizontal[x])
        if x == source.Lx - 1:
            q_data = np.squeeze(q_data, axis=-1)
        else:
            q_inds.append(q_vertical[x])
        q_tensor = qtn.Tensor(q_data, inds=q_inds, tags=active.tags)

        r_data = np.asarray(r_core)
        r_inds: list[str] = []
        if x == 0:
            r_data = np.squeeze(r_data, axis=0)
        else:
            r_inds.append(r_vertical[x - 1])
        r_inds.extend([horizontal[x], toward])
        if x == source.Lx - 1:
            r_data = np.squeeze(r_data, axis=-1)
        else:
            r_inds.append(r_vertical[x])
        r_tensor = qtn.Tensor(r_data, inds=r_inds)

        out.delete(active.tags)
        out.delete(neighbour.tags)
        out |= q_tensor
        out |= r_tensor @ neighbour
    return out


def _effective_vertical_bond(psi, j: int) -> int:
    largest = 1
    for x in range(psi.Lx - 1):
        lower, upper = psi[_tag(psi, x, j)], psi[_tag(psi, x + 1, j)]
        shared = tuple(index for index in lower.inds if index in upper.inds)
        largest = max(largest, math.prod(lower.ind_size(index) for index in shared))
    return int(largest)


def compress_column(
    psi,
    j: int,
    *,
    max_bond: int | None,
    cutoff: float,
    inplace: bool = False,
):
    """Zip parallel vertical bonds into one bond per row pair.

    ``max_bond=None, cutoff=0`` is the exact no-truncation baseline.  The return
    value is ``(peps, metrics)``; quimb's zip-up routine does not expose a
    discarded-weight estimate, so this function records the controlled ranks
    and never fabricates one.
    """
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("column compression requires quimb") from exc
    if max_bond is not None and int(max_bond) < 1:
        raise ValueError("max_bond must be positive or None")
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    out = psi if inplace else psi.copy()
    before = _effective_vertical_bond(out, j)
    column = out.select(tags=[out.y_tag(j)], which="all")
    site_tags = [_tag(out, x, j) for x in range(out.Lx)]
    compressed = qtn.tensor_network_1d_compress(
        column,
        site_tags=site_tags,
        sweep_reverse=True,
        method="zipup",
        max_bond=max_bond,
        cutoff=float(cutoff),
    )
    out.delete([out.y_tag(j)])
    out |= compressed
    after = _effective_vertical_bond(out, j)
    return out, {
        "absorption_bond_before": int(before),
        "absorption_bond_after": int(after),
        "absorption_max_bond": None if max_bond is None else int(max_bond),
        "absorption_cutoff": float(cutoff),
    }


def validate_peps_structure(psi) -> None:
    """Raise if a move did not return one tensor and one bond per lattice edge."""
    if len(psi.tensors) != psi.Lx * psi.Ly:
        raise ValueError("PEPS does not contain exactly one tensor per lattice site")
    tensors = {}
    for i in range(psi.Lx):
        for j in range(psi.Ly):
            selected = psi.select_tensors(tags=[_tag(psi, i, j)], which="all")
            if len(selected) != 1:
                raise ValueError(f"site {(i, j)} has {len(selected)} tensors, expected one")
            tensor = selected[0]
            if _physical_index(psi, i, j) not in tensor.inds:
                raise ValueError(f"site {(i, j)} lost its physical index")
            tensors[(i, j)] = tensor

    sites = list(tensors)
    for index, site in enumerate(sites):
        for other in sites[index + 1:]:
            bonds = tensors[site].bonds(tensors[other])
            distance = abs(site[0] - other[0]) + abs(site[1] - other[1])
            if distance == 1 and len(bonds) != 1:
                raise ValueError(f"edge {site}--{other} does not have one bond")
            if distance != 1 and bonds:
                raise ValueError(f"non-neighbouring sites {site} and {other} share a bond")
