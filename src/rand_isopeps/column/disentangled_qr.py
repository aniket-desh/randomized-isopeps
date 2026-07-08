"""Disentangled structured column QR -- a Moses disentangler inside the sketch sweep.

The global structured QR (:mod:`rand_isopeps.column.structured_qr`) sketches the column
``Y = C Omega`` and left-canonicalizes it with a plain TT-SVD sweep, truncating each
vertical bond to ``eta_q``. exp08/exp09 found the catch: to MATCH the local Moses move's
accuracy the sweep needs a *larger* vertical bond (``eta_q = 6-8`` vs local ``eta = 4``),
because it has no disentangler to shrink the bond, and that fatter carry bond inflates
every downstream column.

This module inserts the **verified Moses disentangler** (:func:`disentangle_altmin`,
bit-for-bit vs Dektor's reference) into the sketch sweep, per the theory-review figure
``tikz/rmps_disentangled_column_method``:

    C --rMPS sketch--> Y = C Omega
      --disentangled TT-SVD sweep-->  Q_iso        (each vertical cut disentangled)
      -->  R = Q_iso^* C

The subtlety the review stresses: a unitary placed *naively* on an existing vertical bond
cannot lower its Schmidt rank (``sigma(MD) = sigma(M)``). The Moses disentangler defeats
this by acting on a **composite** bond ``eta*kappa`` (first SVD to that larger bond) and
**reshuffling** before the rank-``eta`` truncation: at vertical cut ``i`` the ``eta`` half
is regrouped with the *next* output leg ``o_{i+1}`` and the ``kappa`` half with the rest,
so the disentangler ``D`` and the reshuffle ``A`` do not commute and ``D`` genuinely
changes the spectrum the truncation sees. That is exactly ``cut_forward`` +
``disentangle_altmin`` with the free-dimension shim :class:`_CutDims` below.

The mechanism question (exp10a, the go/no-go): does disentangling the thin sampled range
lower the rank-``eta`` tail at each cut enough to beat the no-disentangler method's larger
``eta_q`` tail? :func:`mechanism_profile` measures the three tails per cut that decide it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from rand_isopeps.column.operator import ColumnOperator
from rand_isopeps.experiment_utils.cost_model import svd_flops
from rand_isopeps.linalg.rmps_sketch import rmps_test_matrix
from rand_isopeps.moses.disentangler import disentangle_altmin, tail_energy


@dataclass
class _CutDims:
    """Free-dimension shim exposing the six attributes ``cut_forward`` /
    ``cut_inverse`` / ``disentangle_altmin`` read off a ``MosesDims`` (``eta``,
    ``chi``, ``n2``, ``n3``, ``k1``, ``k2``), but with ``n2``/``n3`` set to the
    sketch-sweep environment rather than the synthetic block-Moses geometry.

    At vertical cut ``i`` of the sampled range: ``eta`` = the kept vertical bond,
    ``chi = kappa`` = the disentangler's extra freedom, ``n2 = o_{i+1}`` = the next
    output leg the vertical bond is regrouped with, ``n3`` = the remaining
    environment. Then ``cut_forward`` reshuffles the composite carrier
    ``(eta*kappa, n2*n3)`` to ``(eta*n2, kappa*n3)`` -- the non-commuting regroup
    that makes the disentangler bite."""

    eta: int
    chi: int
    n2: int
    n3: int

    @property
    def k1(self) -> int:
        return self.eta * self.chi

    @property
    def k2(self) -> int:
        return self.eta


def sketch_range(
    column: ColumnOperator,
    ell: int,
    chi_sk: int,
    n_power: int,
    rng: np.random.Generator,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled range ``Y = C (C^* C)^q Omega`` (dense, ``n_out x ell``) and the column ``C``.

    Same rMPS sketch as :func:`structured_column_qr`, factored out so the mechanism
    scan and the plain sweep share one construction."""
    c = column.materialize() if reference is None else reference
    n_in = c.shape[1]
    ell = max(1, min(int(ell), n_in))
    omega = rmps_test_matrix(column.input_dims, ell, int(chi_sk), rng,
                             normalize=True, complex_valued=np.iscomplexobj(c))
    omega = omega.astype(c.dtype, copy=False)
    y = c @ omega
    for _ in range(max(0, int(n_power))):
        y = c @ (c.conj().T @ y)
    return y, c


def _null_gauge_tail_std(vtilde: np.ndarray, eta: int, n_draws: int,
                         rng: np.random.Generator) -> tuple[float, float]:
    """NULL test: a unitary on the *existing* composite bond (no reshuffle) cannot
    change the rank-``eta`` tail. Draw ``n_draws`` Haar-ish unitaries ``D``, left-
    multiply ``vtilde`` (rotating the bond WITHOUT regrouping), and return the mean
    and std of ``tail_energy(sigma(D vtilde), eta)``. Std ~ machine precision is the
    point: ``sigma(D vtilde) = sigma(vtilde)``, so naive gauging does nothing."""
    rho = vtilde.shape[0]
    tails = []
    for _ in range(max(1, n_draws)):
        g = rng.standard_normal((rho, rho))
        if np.iscomplexobj(vtilde):
            g = g + 1j * rng.standard_normal((rho, rho))
        d, _ = la.qr(g)
        tails.append(tail_energy(la.svdvals(d @ vtilde), eta))
    return float(np.mean(tails)), float(np.std(tails))


@dataclass
class CutMechanism:
    """Per-vertical-cut tails deciding whether disentangling the sketch helps."""

    cut: int
    rows: int                 # a = (left bond so far) * o_i
    cols: int                 # b = remaining output legs * ell
    kappa: int
    rho: int                  # composite bond eta*kappa actually available (<= min(a,b))
    disentangled: bool        # was there room (rho == eta*kappa and a next leg to regroup)?
    tail_eta_I: float         # rank-eta tail of M_i, NO disentangler (the naive fixed-eta cost)
    tail_etaq_I: dict[int, float]   # rank-eta_q tails of M_i (the no-D larger-bond method)
    comp_tail: float          # tail(M_i, rho): first-SVD-to-composite truncation loss
    dis_tail: float           # tail_final: rank-eta tail after the disentangler (note's tau(D*))
    dis_total: float          # comp_tail + dis_tail: honest total lost at vertical bond eta
    dis_iters: int
    null_mean: float          # naive-gauge rank-eta tail (== tail_eta_I to roundoff)
    null_std: float           # spread over random naive gauges (~ machine precision)


def mechanism_profile(
    y: np.ndarray,
    out_dims: tuple[int, ...],
    eta: int,
    kappa: int,
    eta_qs: tuple[int, ...],
    ndis: int,
    rng: np.random.Generator,
    null_draws: int = 8,
) -> list[CutMechanism]:
    """Walk the exact left-canonical cuts of the sampled range ``Y`` and, at each
    internal vertical cut, measure the three tails of the mechanism test.

    The sweep is *exact* (no truncation of the propagated carrier) so every cut sees
    the intrinsic spectrum of ``Y``; the disentangler is then applied to the top-
    ``rho = eta*kappa`` carrier of that cut. Returns one :class:`CutMechanism` per
    internal cut ``i = 0 .. Lx-2`` (the last output leg has no next leg to regroup)."""
    lx = len(out_dims)
    ell = y.shape[1]
    work = y.reshape(*out_dims, ell)
    left = 1
    profile: list[CutMechanism] = []
    for i in range(lx - 1):
        o = out_dims[i]
        mat = work.reshape(left * o, -1)          # M_i : (a, b)
        a, b = mat.shape
        u, sv, vh = la.svd(mat, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        tail_eta = tail_energy(sv, eta)
        tail_qs = {int(q): tail_energy(sv, int(q)) for q in eta_qs}

        rho = min(eta * kappa, sv.shape[0])
        o_next = int(out_dims[i + 1])
        vtilde = sv[:rho, None] * vh[:rho, :]     # top-rho carrier: (rho, b)
        can = (rho == eta * kappa) and (b % o_next == 0) and (b // o_next >= 1) and (a >= rho)

        engaged = bool(can and kappa > 1 and ndis >= 1)
        if engaged:
            n3 = b // o_next
            dims = _CutDims(eta=eta, chi=kappa, n2=o_next, n3=n3)
            res = disentangle_altmin(vtilde, dims, k=eta, maxiter=int(ndis), rng=rng)
            dis_tail, dis_iters = res.tail_final, res.iters
            null_mean, null_std = _null_gauge_tail_std(vtilde, eta, null_draws, rng)
        else:
            dis_tail, dis_iters = tail_eta, 0     # ndis=0 or no room: identity disentangler
            null_mean, null_std = tail_energy(sv[:rho], eta), 0.0

        comp_tail = tail_energy(sv, rho)
        profile.append(CutMechanism(
            cut=i, rows=a, cols=b, kappa=kappa, rho=rho, disentangled=engaged,
            tail_eta_I=tail_eta, tail_etaq_I=tail_qs, comp_tail=comp_tail,
            dis_tail=dis_tail, dis_total=comp_tail + dis_tail, dis_iters=dis_iters,
            null_mean=null_mean, null_std=null_std,
        ))
        # exact left-canonical carry upward (no truncation): full spectrum propagates
        work = (sv[:, None] * vh)
        left = sv.shape[0]
    return profile


@dataclass
class MechanismSummary:
    """Column-level roll-up of the per-cut mechanism tails (relative to ``||Y||``)."""

    y_norm: float
    tau_eta_I: float          # sqrt(sum_i tail_eta_I) / ||Y||   (fixed-eta, no disentangler)
    tau_etaq_I: dict[int, float]   # per eta_q: sqrt(sum_i tail_etaq_I) / ||Y||
    tau_dis: float            # sqrt(sum_i dis_tail) / ||Y||      (note's disentangled rank-eta)
    tau_dis_total: float      # sqrt(sum_i dis_total) / ||Y||     (honest: composite + second SVD)
    max_null_std: float       # worst per-cut naive-gauge tail spread (~ machine precision)
    n_disentangled: int       # how many cuts had room to disentangle


def summarize(profile: list[CutMechanism], y_norm: float) -> MechanismSummary:
    """Roll the per-cut tails into column-level relative rounding errors."""
    scale = y_norm if y_norm else 1.0

    def rel(vals: list[float]) -> float:
        return float(np.sqrt(max(sum(vals), 0.0))) / scale

    eta_qs = sorted(profile[0].tail_etaq_I) if profile else []
    return MechanismSummary(
        y_norm=float(y_norm),
        tau_eta_I=rel([c.tail_eta_I for c in profile]),
        tau_etaq_I={q: rel([c.tail_etaq_I[q] for c in profile]) for q in eta_qs},
        tau_dis=rel([c.dis_tail for c in profile]),
        tau_dis_total=rel([c.dis_total for c in profile]),
        max_null_std=max((c.null_std for c in profile), default=0.0),
        n_disentangled=sum(1 for c in profile if c.disentangled),
    )


# --------------------------- disentangler FLOP model --------------------------- #
def disentangler_flops(out_dims: tuple[int, ...], ell: int, eta: int, kappa: int,
                       ndis: int) -> float:
    """FLOPs the disentangler ADDS to the sketch sweep (beyond the plain TT-SVD).

    At each internal cut the plain sweep does one SVD; the disentangled sweep adds,
    per cut, a first SVD to the composite bond ``rho = eta*kappa`` and ``ndis``
    alternating-minimization iterations, each an inner rank-``eta`` SVD of the
    reshuffled ``(eta*o_{i+1}) x (kappa*n3)`` matrix plus a ``rho x rho`` Procrustes
    SVD. Only cuts with composite headroom (``rho <= rows`` and a next leg) pay it;
    the rest fall back to the plain single SVD (charged elsewhere). ``o_{i+1}`` and
    the environment come from the same left-canonical shapes the sweep truncates."""
    if ndis < 1:
        return 0.0                                    # no disentangler -> nothing added
    rho = eta * kappa
    tot = 0.0
    left = 1
    n_out_dims = len(out_dims)
    for i in range(n_out_dims - 1):
        o = int(out_dims[i])
        rows = left * o
        env = int(np.prod(out_dims[i + 1:])) * ell
        o_next = int(out_dims[i + 1])
        # exact left-canonical carry cap: the propagated vertical bond
        left = min(rows, env)
        if rho > rows or rho > env or o_next < 1:
            continue                                  # no room -> plain single SVD only
        n3 = env // o_next
        tot += svd_flops(rows, rho)                   # first SVD to composite rho
        per_iter = svd_flops(eta * o_next, kappa * n3) + svd_flops(rho, rho)
        tot += int(ndis) * per_iter
    return tot


@dataclass
class DisentangledCost:
    """Accuracy proxy + validity + cost knobs for the disentangled-global column.

    The disentangled column at fixed vertical bond ``eta`` with disentangler freedom
    ``kappa`` retains, per cut, the composite ``eta*kappa`` subspace reorganized into a
    vertical ``eta`` bond and a horizontal residual bond ``kappa``. Its captured RANGE
    is therefore that of the plain sweep at ``eta_q = eta*kappa`` MINUS the residual-
    truncation loss (the disentangler's rank-``eta`` second-SVD tails, ``sqrt(sum
    dis_tail)/||Y||``). The proxy is only trustworthy when that loss is far below the
    column error it stands in for -- ``residual_loss`` is reported so the caller can
    gate on it (``residual_loss << eps_proxy``)."""

    eta: int
    kappa: int
    eta_q_equiv: int          # eta*kappa: the plain vertical bond whose RANGE this matches
    residual_loss: float      # sqrt(sum_i dis_tail)/||Y||: energy the kappa residual drops
    n_disentangled: int
