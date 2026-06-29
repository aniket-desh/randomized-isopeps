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

from rand_isopeps.compression.mpo_mps_absorb import mps_norm_sq, mps_to_vector
from rand_isopeps.column.operator import ColumnOperator
from rand_isopeps.linalg.randomized_svd import orthonormalize, relative_frobenius_error
from rand_isopeps.linalg.rmps_sketch import rmps_cores, rmps_test_matrix


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


def _tt_left_canonical_sweep(y, out_dims, eta_q):
    """TT-SVD left-canonical sweep of the sampled range ``Y`` (``n_out x ell``).

    Identical for the dense and matrix-free paths -- ``Y`` is always small
    (``n_out = prod(out_dims)`` rows, ``ell`` columns), it is only the *building* of
    ``Y`` and the *scoring* that differ. Returns ``(q_iso, final_bond, tau_round,
    delta_local)`` where each TT core ``q_i`` is left-isometric (the arrow)."""
    ell = y.shape[1]
    y_norm = float(np.linalg.norm(y))
    work = y.reshape(*out_dims, ell)
    cores: list[np.ndarray] = []
    discarded = 0.0
    delta_local = 0.0
    left = 1
    for o in out_dims:
        mat = work.reshape(left * o, -1)
        u, sv, vh = la.svd(mat, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        k = max(1, min(eta_q, sv.shape[0]))
        q = u[:, :k]
        gram = q.conj().T @ q
        delta_local = max(delta_local, float(np.linalg.norm(gram - np.eye(k))))
        cores.append(q.reshape(left, o, k))
        discarded += float(np.sum(sv[k:] ** 2))
        work = (sv[:k, None] * vh[:k, :])
        left = k
    final_bond = left

    q_iso = cores[0].reshape(out_dims[0], -1)
    for core in cores[1:]:
        q_iso = np.tensordot(q_iso, core, axes=(-1, 0)).reshape(-1, core.shape[-1])
    tau_round = float(np.sqrt(max(discarded, 0.0))) / y_norm if y_norm else 0.0
    return q_iso, final_bond, tau_round, delta_local


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

    q_iso, final_bond, tau_round, delta_local = _tt_left_canonical_sweep(y, out_dims, eta_q)

    eye = np.eye(final_bond, dtype=q_iso.dtype)
    delta_global = float(np.linalg.norm(q_iso.conj().T @ q_iso - eye))
    proj = q_iso @ (q_iso.conj().T @ c)
    eps_proj = relative_frobenius_error(c, proj)
    r = q_iso.conj().T @ c  # the residual column R = Q_iso^* C
    r_bond = int(np.linalg.matrix_rank(r, tol=1e-9 * max(1.0, float(np.linalg.norm(r)))))

    return StructuredQRResult(
        sketch=sketch_kind, chi_sk=used_chi, ell=ell, eta_q=eta_q, final_bond=final_bond,
        eps_proj=eps_proj, tau_round=tau_round, delta_local=delta_local,
        delta_global=delta_global, r_bond=r_bond,
    )


def make_column_qr(column: ColumnOperator, dense_max_gb: float | None = None):
    """Return ``(qr_fn, c_or_none)`` picking the dense or matrix-free QR by budget.

    The accuracy-matching experiments call the QR many times per column (a config
    search over ``ell, eta_q, chi_sk, q``). When the dense column fits the budget we
    materialize ONCE and reuse it as the ``reference`` (exact ``eps_proj``, fast);
    above the budget we switch to the matrix-free QR (estimated ``eps_proj``, no
    ``(d*chi)^Lx`` allocation). ``qr_fn(ell, eta_q, chi_sk, q, rng)`` returns a
    :class:`StructuredQRResult` either way, so the experiment is agnostic to which
    path ran. ``c_or_none`` is the dense column (small-Lx) or ``None`` (large-Lx)."""
    from rand_isopeps.column.operator import DEFAULT_DENSE_MAX_GB, dense_column_nbytes

    budget = DEFAULT_DENSE_MAX_GB if dense_max_gb is None else dense_max_gb
    peak = 3.0 * dense_column_nbytes(column.n_out, column.n_in)
    if peak <= budget * (1024 ** 3):
        c = column.materialize()

        def qr_fn(ell, eta_q, chi_sk, q, rng):
            return structured_column_qr(column, ell=ell, eta_q=eta_q, chi_sk=chi_sk,
                                        sketch_kind="rmps", n_power=q, rng=rng, reference=c)
        return qr_fn, c

    def qr_fn(ell, eta_q, chi_sk, q, rng):
        return structured_column_qr_matrix_free(column, ell=ell, eta_q=eta_q, chi_sk=chi_sk,
                                                sketch_kind="rmps", n_power=q, rng=rng)
    return qr_fn, None


def _sketch_column_mf(column: ColumnOperator, omega_cores, n_power, complex_valued):
    """One sampled-range column ``(C (C^* C)^q omega)`` as a dense ``n_out`` vector.

    Forward ``C`` and adjoint ``C^*`` stay in MPS form (bond ``<= mpo_bond^(2q+1) *
    chi_sk``); only the final output MPS -- over the ``n_out = prod(o_i)`` retained
    legs, which is small for a boundary column -- is densified. The ``prod(r_i)``
    input-leg vector is NEVER formed (that is the ``8^Lx`` allocation this avoids)."""
    a = column.matvec_mps(omega_cores)          # C omega   (MPS over output legs)
    for _ in range(max(0, n_power)):
        b = column.rmatvec_mps(a)                # C^* a     (MPS over input legs)
        a = column.matvec_mps(b)                 # C b       (MPS over output legs)
    return mps_to_vector(a).astype(np.complex128 if complex_valued else np.float64, copy=False)


def _estimate_eps_proj_mf(column, q_iso, n_probes, chi_score, rng, complex_valued):
    """Matrix-free Hutchinson estimate of ``||(I-QQ*)C||_F / ||C||_F``.

    With isotropic rMPS probes (``E[omega omega^*] = I``, exact for the rmps_cores
    construction at any ``chi_sk``) the estimator is UNBIASED: ``E||(I-QQ*)C omega||^2
    = ||(I-QQ*)C||_F^2`` and ``E||C omega||^2 = ||C||_F^2``. ``chi_score`` controls the
    *variance* (the Kron ``chi_sk=1`` limit is unbiased but high-variance -- use
    ``chi_score >~ Lx``). Probes here are fresh and independent of the sketch probes,
    else ``Q`` (fit to them) makes the estimate optimistic. Each ``C omega`` is an
    ``n_out``-vector; no dense ``C`` is touched."""
    num = den = 0.0
    for _ in range(n_probes):
        w = rmps_cores(column.input_dims, chi_score, rng, complex_valued=complex_valued)
        z = mps_to_vector(column.matvec_mps(w))
        resid = z - q_iso @ (q_iso.conj().T @ z)
        num += float(np.vdot(resid, resid).real)
        den += float(np.vdot(z, z).real)
    return float(np.sqrt(num / max(den, 1e-300)))


def structured_column_qr_matrix_free(
    column: ColumnOperator,
    ell: int,
    eta_q: int,
    chi_sk: int = 8,
    sketch_kind: str = "rmps",
    n_power: int = 0,
    rng: np.random.Generator | None = None,
    score_probes: int = 32,
    score_chi: int | None = None,
    complex_valued: bool = True,
) -> StructuredQRResult:
    """Matrix-free structured column QR -- never forms the ``(d*chi)^Lx`` dense ``C``.

    Same algorithm and diagnostics as :func:`structured_column_qr`, but every dense
    allocation is replaced by the access model:

    * **sketch** -- ``ell`` rMPS probes applied via :func:`_sketch_column_mf`
      (forward + adjoint MPO--MPS), densifying only the small ``n_out`` outputs into
      ``Y`` (``n_out x ell``);
    * **sweep** -- the identical small TT-SVD sweep (:func:`_tt_left_canonical_sweep`);
    * **scoring** -- ``eps_proj`` is a probe estimate (:func:`_estimate_eps_proj_mf`),
      NOT the exact Frobenius ratio. Calibrate it against the dense path at small
      ``Lx`` (see ``tests/test_structured_qr_matrix_free``), then trust it at large
      ``Lx``. ``r_bond`` becomes the rank of the small sketch residual ``Q^* Y``.

    ``score_chi`` defaults to ``max(chi_sk, Lx)`` to keep the scoring estimator's
    variance low (isotropy is exact at any ``chi``; variance is the knob)."""
    gen = np.random.default_rng() if rng is None else rng
    out_dims = column.output_dims
    lx = column.lx
    ell = max(1, min(int(ell), column.n_in))
    used_chi = 1 if sketch_kind == "kron" else int(chi_sk)
    if sketch_kind not in ("rmps", "kron"):
        raise ValueError(f"matrix-free path supports rmps/kron probes, got {sketch_kind!r}")

    # sketch Y = C (C^* C)^q Omega, column by column, matrix-free
    cols = []
    for _ in range(ell):
        omega_cores = rmps_cores(column.input_dims, used_chi, gen, complex_valued=complex_valued)
        cols.append(_sketch_column_mf(column, omega_cores, n_power, complex_valued))
    y = np.stack(cols, axis=1)

    q_iso, final_bond, tau_round, delta_local = _tt_left_canonical_sweep(y, out_dims, eta_q)
    eye = np.eye(final_bond, dtype=q_iso.dtype)
    delta_global = float(np.linalg.norm(q_iso.conj().T @ q_iso - eye))

    score_chi = max(used_chi, lx) if score_chi is None else int(score_chi)
    eps_proj = _estimate_eps_proj_mf(column, q_iso, score_probes, score_chi, gen, complex_valued)
    # residual interface dim from the small sketch residual Q^* Y (<= eta_q rows)
    r_sketch = q_iso.conj().T @ y
    r_bond = int(np.linalg.matrix_rank(r_sketch, tol=1e-9 * max(1.0, float(np.linalg.norm(r_sketch)))))

    return StructuredQRResult(
        sketch=sketch_kind, chi_sk=used_chi, ell=ell, eta_q=eta_q, final_bond=final_bond,
        eps_proj=eps_proj, tau_round=tau_round, delta_local=delta_local,
        delta_global=delta_global, r_bond=r_bond,
    )
