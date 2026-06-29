"""Extract a real isoTNS column as a `ColumnOperator` (the PEPS bridge).

Builds the whole-column map ``C_j : X_j -> Y_j`` that the block Moses move
factors, directly from a canonical quimb isoTNS, so the global rMPS column QR
(:mod:`rand_isopeps.column.global_range`) and the local baseline run unchanged on
a real physical state.

Leg map (corrected against a live Gram-test of the real move,
``moses_move.py:224,276-280``). For the active column ``j`` with the orthogonality
centre on it and the residual pushed toward ``j_next`` (``split="right" ->
j_next=j+1``):

* ``din`` (absorbed -> domain ``X_j``, pushed into ``j_next``) = the **horizontal
  bond toward ``j_next``** of each row (one per row; an accumulated rank, not a bare
  ``chi`` leg).
* ``dout`` (retained -> codomain ``Y_j``, the new isometric column) = the
  **physical leg** times the **away-from-``j_next`` horizontal bond** (the bond
  toward ``j_other = 2j - j_next``), if that neighbour exists. At a lattice
  boundary there is no away bond, so ``dout`` is the physical leg alone.
* ``ml`` / ``mr`` (MPO bonds) = the **vertical** up / down bonds (intra-column
  entanglement); boundary rows give size-1 bonds.

The cores are emitted in row order ``x = 0..Lx-1`` so ``materialize()`` flattens
row-major (matching ``rmps_to_vector``) and ``factor_dims = op.input_dims``.

Only the orthogonality-centre column has a non-trivial ``C_j``: an off-centre
column is an exact isometry (flat spectrum) -- it is already canonicalised toward
the centre. ``random_isotns`` leaves the centre at column 0, so the meaningful
extraction is ``from_quimb_column(psi, 0, split="right")``.
"""

from __future__ import annotations

import numpy as np

from rand_isopeps.column.operator import (
    ColumnOperator,
    dense_column_nbytes,
    mpo_frobenius_norm,
)
from rand_isopeps.compression.mpo_mps_absorb import mps_norm_sq, mps_to_vector
from rand_isopeps.linalg.rmps_sketch import rmps_cores


def _bond(psi, a_tag, b_tag):
    """The single shared index between two sites (by tag), or None if not adjacent."""
    ta, tb = psi[a_tag], psi[b_tag]
    shared = tuple(ix for ix in ta.inds if ix in tb.inds)
    return shared[0] if shared else None


def _phys_ind(psi, x, j):
    try:
        return psi.site_ind((x, j))
    except Exception:
        return f"k{x},{j}"


def column_tensors_to_core(arr, up, phys, away, din, down):
    """Reshape a row tensor (already transposed to [up?, phys, away?, din, down?])
    into a ``(ml, dout, din, mr)`` MPO core with size-1 boundary bonds."""
    full = arr.reshape(up, phys, away, din, down)          # missing legs are size 1
    return full.reshape(up, phys * away, din, down)         # dout = phys x away


def from_quimb_column(psi, j, split="right", normalize=True):
    """Build ``C_j`` as a vertical MPO ``ColumnOperator`` from a canonical isoTNS.

    ``psi`` must be in isoTNS canonical form with the orthogonality centre on
    column ``j`` (e.g. column 0 right after :func:`random_isotns`). ``split`` is
    the side the residual is pushed to (``"right" -> j_next=j+1``). Returns a
    :class:`ColumnOperator`; call ``.materialize()`` for the dense ``C_j`` or pass
    ``op.input_dims`` as ``factor_dims`` to the range finder.
    """
    Lx, Ly = psi.Lx, psi.Ly
    j_next = j + 1 if split == "right" else j - 1
    j_away = j - 1 if split == "right" else j + 1
    if not (0 <= j_next < Ly):
        raise ValueError(f"j_next={j_next} out of range; cannot push the residual off the lattice")

    def tag(x, y):
        return psi.site_tag_id.format(x, y)

    cores = []
    for x in range(Lx):
        t = psi[tag(x, j)]
        phys_ix = _phys_ind(psi, x, j)
        din_ix = _bond(psi, tag(x, j), tag(x, j_next))
        if din_ix is None:
            raise ValueError(f"row {x}: no horizontal bond toward j_next={j_next}")
        away_ix = _bond(psi, tag(x, j), tag(x, j_away)) if 0 <= j_away < Ly else None
        up_ix = _bond(psi, tag(x, j), tag(x - 1, j)) if x - 1 >= 0 else None
        down_ix = _bond(psi, tag(x, j), tag(x + 1, j)) if x + 1 < Lx else None

        order = [ix for ix in (up_ix, phys_ix, away_ix, din_ix, down_ix) if ix is not None]
        arr = np.asarray(t.transpose(*order).data)
        dims = dict(
            up=t.ind_size(up_ix) if up_ix else 1,
            phys=t.ind_size(phys_ix),
            away=t.ind_size(away_ix) if away_ix else 1,
            din=t.ind_size(din_ix),
            down=t.ind_size(down_ix) if down_ix else 1,
        )
        cores.append(column_tensors_to_core(arr, dims["up"], dims["phys"], dims["away"],
                                            dims["din"], dims["down"]))

    op = ColumnOperator(cores)
    if normalize:
        # ``||C||_F`` via transfer-matrix contraction -- never materialize the
        # ``(d*chi)^Lx`` column just to take a norm (a hidden large-Lx blow-up).
        nrm = mpo_frobenius_norm(cores)
        if nrm > 0:
            cores[0] = cores[0] / nrm
            op = ColumnOperator(cores)
    return op


def _isometry_defect_mf(op, *, n_probes=8, chi_score=4, seed=0):
    """Matrix-free isometry defect ``median_omega |||C omega||^2 / ||omega||^2 - 1|``.

    For an exact isometry ``C^*C = I`` so the ratio is ``1`` for *every* probe and the
    defect is ``0`` to roundoff; the entangled centre column has ``C^*C != I`` so the
    ratio departs from 1. Uses isotropic rMPS probes over the input legs: ``||omega||^2``
    via the transfer-matrix MPS norm and ``||C omega||^2`` from the small ``n_out``-length
    output vector -- the ``prod(r_i)`` dense vector is never formed."""
    rng = np.random.default_rng(seed)
    complex_valued = np.iscomplexobj(op.cores[0])
    ratios = []
    for _ in range(n_probes):
        w = rmps_cores(op.input_dims, chi_score, rng, complex_valued=complex_valued)
        wn = mps_norm_sq(w)
        if wn <= 0:
            continue
        cw = mps_to_vector(op.matvec_mps(w))
        ratios.append(abs(float(np.vdot(cw, cw).real) / wn - 1.0))
    return float(np.median(ratios)) if ratios else 0.0


def find_center_column(psi, tol=1e-04):
    """Return ``(j, split)`` of the orthogonality-centre **boundary** column.

    After a Moses-sweep canonicalization (``random_isotns``) or a TEBD2 sweep
    (``imaginary_time``) the centre sits at a *boundary* column -- 0 or ``Ly-1``,
    depending on the last sweep direction (``canonize_column`` does NOT relocate the
    2D centre, so do not assume column 0). The off-centre boundary is an exact
    isometry (flat spectrum, ``sigma_min/sigma_max == 1``, spread ``0``); the centre
    carries entanglement (a spread spectrum). This returns the boundary whose column
    map is non-isometric.

    Small ``Lx`` (dense column within budget): the exact SVD spread, as before.
    Large ``Lx`` (dense column over budget): a matrix-free isometry-defect estimate
    (:func:`_isometry_defect_mf`) so center-finding never materializes the
    ``(d*chi)^Lx`` column -- the regime that OOM-crashed the host.

    Scope: this handles centres left at a boundary, which is every state these code
    paths produce. If BOTH boundaries are isometric the centre is *interior* (e.g.
    from some other gauge), and we **raise** rather than silently extract a flat
    isometric column -- the failure mode an adversarial check flagged. (Generalising
    to interior centres needs a full per-column scan and is deliberately out of scope.)
    """
    from rand_isopeps.column.operator import DEFAULT_DENSE_MAX_GB

    scores = {}
    for j, split in ((0, "right"), (psi.Ly - 1, "left")):
        op = from_quimb_column(psi, j, split=split, normalize=False)
        # exact spread when the column is cheap to materialize; matrix-free defect
        # (a different scale, but 0 for an isometry and >0 for the centre, same as
        # the spread) once the dense column would blow the memory budget.
        peak = 3.0 * dense_column_nbytes(op.n_out, op.n_in)
        if peak <= DEFAULT_DENSE_MAX_GB * (1024 ** 3):
            s = np.linalg.svd(op.materialize(), compute_uv=False)
            s = s[s > 0]
            scores[(j, split)] = float(1.0 - s[-1] / s[0]) if s.size else 0.0
        else:
            scores[(j, split)] = _isometry_defect_mf(op, seed=hash((j, split)) & 0xFFFF)
    (jc, split), score = max(scores.items(), key=lambda kv: kv[1])
    if score < tol:
        raise ValueError(
            "no boundary column is the orthogonality centre (both are ~isometric); the "
            "centre is likely interior -- canonicalize it to a boundary first, or extend "
            "find_center_column to a full per-column scan."
        )
    return jc, split
