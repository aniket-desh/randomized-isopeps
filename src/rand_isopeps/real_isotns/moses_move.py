"""real isoTNS Moses Move on quimb PEPS, with randomized SVD / disentangler hooks.

Adapted (cleaned + rewired) from Alec Dektor's reference implementation
``iso_methods.py`` in github.com/adektor/isoTNS_sampling. The tensor-network
bookkeeping (the column sweep, the upward carrier, the sideways R-column
absorption via ``tensor_network_1d_compress``) is kept on quimb; the single
truncated-SVD primitive ``my_split`` and the alternating-minimization
``disentangle`` are rewritten so the SVD step runs through *our* numpy
primitives (``rand_isopeps.randomized_svd``). Passing ``rand=RandSVD(...)``
swaps the **local tensor-ring SVDs** -- the first SVD, the disentangler-search
SVD, and the second SVD -- for a randomized / structured sketch, with the gauge
kept exact. NOTE: the sideways **R-column zip-up absorption** is still quimb's
*deterministic* ``tensor_network_1d_compress`` (it is not threaded through
``RandSVD``); it is a known randomization insertion point (the block-isoPEPS
paper lists zip-up / variational / randomized-SVD options) studied synthetically
in exp3, not yet wired into the real move.

This replaces Dektor's ``my_split`` use of the private quimb internals
``tensor_core._SPLIT_FNS`` / ``_parse_split_opts`` (removed in quimb >= 1.14)
with a direct numpy SVD; the only quimb private symbol kept is
``decomp._trim_and_renorm_svd_result`` (cutoff/max-bond truncation + absorb),
isolated to ``my_split`` so future quimb churn touches one place.

The ``disentangle`` alternating minimisation here is the same algorithm we
verified bit-for-bit against Dektor's ``disentangle-python`` reference (see
``rand_isopeps.disentangler`` and log.md 2026-06-16): rank-k truncation then an
orthogonal Procrustes update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import quimb.utils as uts
from quimb.tensor.decomp import _trim_and_renorm_svd_result
from quimb.tensor.tensor_core import tags_to_oset

from rand_isopeps.linalg.randomized_svd import SketchSpec, rsvd_truncate


@dataclass
class RandSVD:
    """Randomization config threaded through the Moses Move's *local tensor-ring*
    SVDs (first SVD, disentangler search, second SVD). ``sketch=None`` is the
    deterministic path (numpy SVD); otherwise those SVDs use our randomized range
    finder with the given sketch (a kind string or SketchSpec). The R-column
    zip-up absorption is NOT affected (it stays quimb's deterministic compress)."""

    sketch: "str | SketchSpec | None" = None
    oversample: int = 8
    n_power: int = 1
    rng: np.random.Generator | None = None


# ----------------------------- the SVD primitive ----------------------------- #

def _raw_svd(array: np.ndarray, chi_max: int, rand: RandSVD | None):
    """Untruncated-ish factorisation U, s, Vh feeding the truncation step.

    Deterministic: full numpy SVD. Randomized: our range finder to rank chi_max
    (the truncation/absorb is then applied identically by quimb's helper).
    """
    if rand is None or rand.sketch is None:
        return np.linalg.svd(array, full_matrices=False)
    k = min(chi_max, array.shape[0], array.shape[1])
    res = rsvd_truncate(array, k, oversample=rand.oversample, n_power=rand.n_power,
                        rng=rand.rng, sketch=rand.sketch)
    return res.u, res.s, res.vh


def my_split(T, left_inds, absorb, chi_max, cutoff=1e-10, get="tensors",
             right_inds=None, ltags=None, rtags=None, stags=None, rand=None):
    """Truncated SVD of a quimb Tensor across (left_inds | right_inds).

    Returns the factor tensors (and singular-value tensor if ``absorb is None``)
    plus the truncation error. ``absorb`` follows quimb's convention (1 -> right,
    -1 -> left, None -> separate). The raw SVD is ours (numpy / randomized); the
    cutoff/max-bond truncation + absorb reuse quimb's ``_trim_and_renorm``.
    """
    if left_inds is None:
        left_inds = uts.oset(T.inds) - uts.oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)
    if right_inds is None:
        right_inds = uts.oset(T.inds) - uts.oset(left_inds)
    else:
        right_inds = tags_to_oset(right_inds)

    nleft, nright = len(left_inds), len(right_inds)
    TT = T.transpose(*left_inds, *right_inds)
    left_dims, right_dims = TT.shape[:nleft], TT.shape[nleft:]
    array = np.asarray(TT.data).reshape(int(np.prod(left_dims)), int(np.prod(right_dims)))

    left, s, right = _raw_svd(array, chi_max, rand)
    # discarded Frobenius weight at rank chi_max. In deterministic mode ``s`` is
    # the full spectrum so this is the exact rank-chi_max tail; in randomized mode
    # ``s`` are the rsvd's approximate top-chi_max values, so this is an *estimate*
    # of the residual (it under-counts what the sketch failed to capture).
    arr_norm2 = float(np.vdot(array, array).real)
    err = float(np.sqrt(max(arr_norm2 - float(np.sum(np.abs(s[:chi_max]) ** 2)), 0.0)))

    left, s, right = _trim_and_renorm_svd_result(left, s, right, cutoff, 3, chi_max, absorb, False)

    if nleft != 1:
        left = left.reshape(*left_dims, left.shape[-1])
    if nright != 1:
        right = right.reshape(right.shape[0], *right_dims)

    bond = qtn.rand_uuid()
    Tl = qtn.Tensor(left, inds=(*left_inds, bond), tags=tags_to_oset(ltags))
    Tr = qtn.Tensor(right, inds=(bond, *right_inds), tags=tags_to_oset(rtags))
    if absorb is None:
        Ts = qtn.Tensor(s, inds=(bond,), tags=tags_to_oset(stags))
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    if get == "tensors":
        return *tensors, err
    return qtn.TensorNetwork(tensors, virtual=True), err


# ----------------------------- the disentangler ------------------------------ #

def disentangle(X, dis_bonds, svd_bonds, eta, cutoff=1e-6, Ndis=30, rand=None):
    """Alternating-minimization disentangler (Dektor / Wei et al.).

    Optimizes a unitary Q on ``dis_bonds`` that minimizes the rank-``eta``
    truncation tail across ``svd_bonds``: rank-eta truncation of A(QX), then an
    orthogonal Procrustes update Q = U V. ``rand`` randomizes the inner rank-eta
    SVD (the *sketched search*); the Procrustes gauge update stays an exact unit-
    ary. A fresh sketch is drawn each iteration (the reliable mode).
    """
    dims = [X.ind_size(b) for b in dis_bonds]
    dim = qu.prod(dims)
    # match the state dtype so the gauge is a complex unitary for complex states
    # (the Procrustes update Q = U V below stays an exact unitary either way).
    eye = qu.eye(dim, dtype=X.data.dtype).reshape(list(dims) + list(dims))
    bonds_out = [ix + "*" for ix in dis_bonds]
    Q = qtn.Tensor(data=eye, inds=list(dis_bonds) + bonds_out, tags=())

    err = 0.0
    for _ in range(Ndis):
        QX = Q @ X
        QX = QX.reindex({ix: ix[:-1] if ix.endswith("*") else ix for ix in QX.inds})
        U, R, err = my_split(QX, svd_bonds, -1, eta, cutoff=cutoff,
                             ltags=QX.tags, rtags=QX.tags, rand=rand)
        M = (U @ R).reindex({ix: ix + "*" for ix in dis_bonds})
        MX = M @ X.H
        Uq, _, Vq = MX.split(left_inds=dis_bonds, absorb=None)  # exact gauge update
        Q = Uq @ Vq
        if abs(err) < cutoff:
            break

    QX = Q @ X
    QX = QX.reindex({ix: ix[:-1] if ix.endswith("*") else ix for ix in QX.inds})
    U, R, err = my_split(QX, svd_bonds, -1, eta, cutoff=cutoff, ltags=QX.tags, rtags=QX.tags, rand=rand)
    return Q, U, R, err


def split_dim(d: int) -> tuple[int, int]:
    """Factor a bond dimension d = d1*d2 for the intermediate ring bond split."""
    for d1 in range(int(np.sqrt(d)), 0, -1):
        if d % d1 == 0:
            return d1, d // d1
    return 1, d


# --------------------------- the local ring split ---------------------------- #

def split_3d_iso(T, left_inds, right_inds, up_inds, chi, eta, Ndis, cutoff, rand=None):
    """Split an active tensor into the isometric 3-tensor ring (Moses local step).

    First SVD to bond chi*eta (the isometry L + residual UR), then disentangle +
    second SVD to bond eta, absorbing the gauge inverse into L. Returns
    (L isometric column tensor, U upward carrier, R sideways residual, errors).
    """
    L, UR, err0 = my_split(T, left_inds, 1, chi * eta, cutoff=cutoff,
                           right_inds=right_inds.union(up_inds), ltags=T.tags, rtags=T.tags, rand=rand)
    bond = L.bonds(UR).pop()
    ds = L.ind_size(bond)
    d1, d2 = split_dim(ds)

    new_inds = (qtn.rand_uuid(), qtn.rand_uuid())
    L.unfuse({bond: new_inds}, {bond: (d1, d2)}, inplace=True)
    UR.unfuse({bond: new_inds}, {bond: (d1, d2)}, inplace=True)

    dis_bonds = L.bonds(UR)
    svd_bonds = up_inds.copy()
    svd_bonds.add(new_inds[1])

    Q, U, R, err1 = disentangle(UR, dis_bonds, svd_bonds, eta, Ndis=Ndis, rand=rand)
    L = L @ Q.H
    L = L.reindex({ix: ix[:-1] if ix.endswith("*") else ix for ix in L.inds})

    errs = np.reshape(np.stack([np.asarray(err1), np.asarray(err0)]), (2, 1))
    return L, U, R, errs


# ------------------------------- the Moses Move ------------------------------ #

def moses_move(psi, j, chi, eta, cutoff, Ndis,
               orientation="col", sweep="up", split="right", renorm=True, rand=None):
    """Moses move on column/row ``j`` of a quimb PEPS (modifies ``psi`` in place).

    Sweeps the column splitting each site into an isometric ring (split_3d_iso),
    contracting the upward carrier U into the site above, peeling the residual R
    to the neighbor column, then absorbing the R-column with a zip-up compression.
    ``rand=RandSVD(...)`` randomizes every SVD in the move. Returns the per-site
    (disentangler, first-SVD) truncation errors.
    """
    if orientation == "col":
        from_which = "xmax" if sweep == "up" else "xmin"
    else:
        from_which = "ymin" if sweep == "right" else "ymax"
    j_next = j + 1 if split in ("right", "down") else j - 1

    rot = qtn.tensor_2d.Rotator2D(tn=psi, xrange=None, yrange=None, from_which=from_which)
    R = qtn.Tensor(data=0)
    sweep_inds = list(rot.sweep)
    mm_errs = None

    for idx in range(len(sweep_inds) - 1):
        i, i_next = sweep_inds[idx], sweep_inds[idx + 1]
        site = rot.site_tag(i, j)
        site_up = rot.site_tag(i_next, j)
        site_right = rot.site_tag(i, j_next)

        up_bonds = psi[site].bonds(psi[site_up])
        right_bonds = psi[site].bonds(psi[site_right]).union(psi[site].bonds(R))

        L, U, R, errs = split_3d_iso(psi[site], left_inds=None, right_inds=right_bonds,
                                     up_inds=up_bonds, chi=chi, eta=eta, Ndis=Ndis,
                                     cutoff=cutoff, rand=rand)
        mm_errs = errs if mm_errs is None else np.concatenate([mm_errs, errs], axis=1)

        U.drop_tags()
        psi[site_up] = U @ psi[site_up]
        psi.delete(psi[site].tags)
        psi |= L
        psi |= R.retag({site: rot.site_tag(i, j_next), rot.y_tag(j): rot.y_tag(j_next)}, inplace=True)

    # top tensor: a plain split (no ring), peel R to the neighbor
    i = sweep_inds[-1]
    site = rot.site_tag(i, j)
    right_bonds = psi[site].bonds(psi[rot.site_tag(i, j_next)]).union(psi[site].bonds(R))
    L, R, err0 = my_split(psi[site], None, 1, chi * eta, cutoff=cutoff,
                          right_inds=right_bonds, ltags=psi[site].tags, rtags=psi[site].tags, rand=rand)
    errs = np.reshape(np.stack([np.asarray(0.0), np.asarray(err0)]), (2, 1))
    mm_errs = np.concatenate([mm_errs, errs], axis=1)

    psi.delete(psi[site].tags)
    psi |= L
    psi |= R.retag({site: rot.site_tag(i, j_next), rot.y_tag(j): rot.y_tag(j_next)}, inplace=True)

    # absorb the R-column into the neighbor column (sideways MPO-MPS compression)
    RX = psi.select(tags=[rot.y_tag(j_next)], which="all")
    new_col_tags = [rot.site_tag(i, j_next) for i in sweep_inds]
    new_col = qtn.tensor_network_1d_compress(RX, site_tags=new_col_tags, sweep_reverse=True, method="zipup")
    psi.delete([rot.y_tag(j_next)])
    psi |= new_col

    if renorm:
        site = rot.site_tag(i, j_next)
        psi[site] = psi[site] / psi[site].norm()
    return mm_errs


def random_isotns(Lx, Ly, bond=2, phys=2, chi=64, eta=64, cutoff=1e-10, Ndis=20,
                  seed=None, rand=None):
    """Build a random isoTNS by Moses-sweeping a random quimb PEPS to canonical
    form (mirrors Dektor's ``rand_isoTNS``). ``rand`` randomizes the sweep's SVDs.
    Returns the canonicalized PEPS."""
    psi = qtn.PEPS.rand(Lx, Ly, bond_dim=bond, phys_dim=phys, seed=seed)
    psi.normalize()
    for j in range(Ly - 1, 0, -1):
        moses_move(psi, j, chi, eta, cutoff, Ndis, orientation="col",
                   sweep=("down" if j % 2 == 0 else "up"), split="left", renorm=False, rand=rand)
    psi.canonize_column(0, "down")
    return psi
