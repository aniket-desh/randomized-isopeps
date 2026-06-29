"""Arrow-compatible structured column QR -- the actual global Moses move.

The reframe (theory review): we do NOT sketch to enforce the isoPEPS isometry. We
sketch to make the otherwise-intractable *whole-column* QR tractable, and then an
**arrow-compatible TT/MPS QR/SVD sweep** on the sampled range turns that range into
a genuinely isometric isoPEPS column. Dense ``orth(C Omega)`` gives only a matrix
isometry ``Q^* Q = I``; the structured sweep gives the **local** isometry
``q_i^* q_i = I`` at every column tensor (the arrows in the tensor-network diagrams).

Pipeline:

    C  --rMPS sketch-->  Y = C Omega          (thin: ell = r + s probe columns)
       --TT-SVD left-canonical sweep-->        Q_iso  (each core left-isometric)
       -->  R = Q_iso^* C                       (residual, from the ORIGINAL column)
       -->  C ~ Q_iso R

The sweep reshapes ``Y`` over the retained (output) legs and left-canonicalizes it,
truncating each vertical bond to ``eta_q`` with an SVD. Keeping ``q_i = U`` (the
left singular vectors) makes every core *exactly* isometric -- truncation changes the
represented subspace, not the arrows.

Feasibility gate (Q4 of the theory review): range capture does NOT by itself give a
valid low-bond isoPEPS column. The correct statement is **range-capture + small
TT-rounding tail => valid low-bond isometric column with controlled error**. This
module measures the four diagnostics that decide it:

* ``delta_local``  = ``max_i ||q_i^* q_i - I||_F``   -- per-site arrow isometry (should be ~eps).
* ``delta_global`` = ``||Q_iso^* Q_iso - I||_F``      -- global matrix isometry (should be ~eps).
* ``eps_proj``     = ``||(I - Q_iso Q_iso^*) C||_F / ||C||_F`` -- did the rounded column capture C?
* ``tau_round``    = ``(sum of discarded TT tails)^{1/2} / ||Y||`` -- the rounding error.

The whole project rides on whether a *small* ``eta_q`` keeps ``eps_proj`` small (the
sampled range of a physical column rounds cheaply into an isometric column), vs a
random column needing a large ``eta_q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from rand_isopeps.column.operator import ColumnOperator
from rand_isopeps.linalg.randomized_svd import orthonormalize, relative_frobenius_error
from rand_isopeps.linalg.rmps_sketch import rmps_test_matrix


@dataclass
class StructuredQRResult:
    sketch: str
    chi_sk: int
    ell: int
    eta_q: int
    final_bond: int          # the carried R-column bond (= columns of Q_iso)
    eps_proj: float          # ||(I - Q Q*) C|| / ||C||  (range capture after rounding)
    tau_round: float         # TT-rounding tail of Y, relative to ||Y||
    delta_local: float       # max_i ||q_i* q_i - I||  (per-site arrow isometry)
    delta_global: float      # ||Q* Q - I||
    r_bond: int              # rank of the residual R = Q* C (its effective interface dim)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "sketch": self.sketch, "chi_sk": self.chi_sk, "ell": self.ell, "eta_q": self.eta_q,
            "final_bond": self.final_bond, "eps_proj": self.eps_proj, "tau_round": self.tau_round,
            "delta_local": self.delta_local, "delta_global": self.delta_global, "r_bond": self.r_bond,
        }


def _gaussian(n, ell, rng, complex_valued):
    if complex_valued:
        om = (rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))) / np.sqrt(2.0)
    else:
        om = rng.standard_normal((n, ell))
    return om / np.sqrt(ell)


def structured_column_qr(
    column: ColumnOperator,
    ell: int,
    eta_q: int,
    chi_sk: int = 8,
    sketch_kind: str = "rmps",
    n_power: int = 0,
    rng: np.random.Generator | None = None,
    reference: np.ndarray | None = None,
) -> StructuredQRResult:
    """rMPS-sketched, arrow-compatible structured column QR of a ``ColumnOperator``.

    ``ell`` probe columns, vertical bonds truncated to ``eta_q``. ``sketch_kind`` in
    ``{"gaussian", "rmps", "kron"}`` (``kron`` = ``chi_sk=1``). Returns the four
    feasibility diagnostics. Materialized (dense) path for the tiny validation; the
    matrix-free MPO--MPS form is future work.
    """
    gen = np.random.default_rng() if rng is None else rng
    c = column.materialize() if reference is None else reference
    n_out, n_in = c.shape
    out_dims = column.output_dims
    complex_valued = np.iscomplexobj(c)
    ell = max(1, min(ell, n_in))

    if sketch_kind == "gaussian":
        omega = _gaussian(n_in, ell, gen, complex_valued)
        used_chi = 0
    elif sketch_kind in ("rmps", "kron"):
        used_chi = 1 if sketch_kind == "kron" else int(chi_sk)
        omega = rmps_test_matrix(column.input_dims, ell, used_chi, gen,
                                 normalize=True, complex_valued=complex_valued)
    else:
        raise ValueError(f"unknown sketch_kind: {sketch_kind!r}")
    omega = omega.astype(c.dtype, copy=False)

    y = c @ omega
    for _ in range(max(0, n_power)):
        y = c @ (c.conj().T @ y)
    y_norm = float(np.linalg.norm(y))

    # TT-SVD left-canonical sweep of Y reshaped over the retained (output) legs.
    work = y.reshape(*out_dims, ell)
    cores: list[np.ndarray] = []
    discarded = 0.0
    delta_local = 0.0
    left = 1
    lx = len(out_dims)
    for i, o in enumerate(out_dims):
        mat = work.reshape(left * o, -1)
        u, sv, vh = la.svd(mat, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        k = max(1, min(eta_q, sv.shape[0]))
        q = u[:, :k]
        # per-site arrow isometry (U has orthonormal columns -> ~machine precision)
        gram = q.conj().T @ q
        delta_local = max(delta_local, float(np.linalg.norm(gram - np.eye(k))))
        cores.append(q.reshape(left, o, k))
        discarded += float(np.sum(sv[k:] ** 2))
        work = (sv[:k, None] * vh[:k, :])
        left = k
    final_bond = left

    # assemble Q_iso (n_out x final_bond) from the left-isometric cores
    q_iso = cores[0].reshape(out_dims[0], -1)  # left=1 at the bottom
    for core in cores[1:]:
        q_iso = np.tensordot(q_iso, core, axes=(-1, 0)).reshape(-1, core.shape[-1])

    eye = np.eye(final_bond, dtype=q_iso.dtype)
    delta_global = float(np.linalg.norm(q_iso.conj().T @ q_iso - eye))
    proj = q_iso @ (q_iso.conj().T @ c)
    eps_proj = relative_frobenius_error(c, proj)
    tau_round = float(np.sqrt(max(discarded, 0.0))) / y_norm if y_norm else 0.0
    r = q_iso.conj().T @ c  # the residual column R = Q_iso^* C
    r_bond = int(np.linalg.matrix_rank(r, tol=1e-9 * max(1.0, float(np.linalg.norm(r)))))

    return StructuredQRResult(
        sketch=sketch_kind, chi_sk=used_chi, ell=ell, eta_q=eta_q, final_bond=final_bond,
        eps_proj=eps_proj, tau_round=tau_round, delta_local=delta_local,
        delta_global=delta_global, r_bond=r_bond,
    )
