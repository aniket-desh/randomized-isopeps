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
from rand_isopeps.column.structured_qr import structured_column_qr
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
    cut_workers: int = 1,
) -> list[CutMechanism]:
    """Walk the exact left-canonical cuts of the sampled range ``Y`` and, at each
    internal vertical cut, measure the three tails of the mechanism test.

    The sweep is *exact* (no truncation of the propagated carrier) so every cut sees
    the intrinsic spectrum of ``Y``; the disentangler is then applied to the top-
    ``rho = eta*kappa`` carrier of that cut. Returns one :class:`CutMechanism` per
    internal cut ``i = 0 .. Lx-2`` (the last output leg has no next leg to regroup).

    ``cut_workers > 1`` runs the per-cut disentangler optimizations in a thread pool:
    each ``disentangle_altmin`` acts on a *copy* of its cut's carrier and nothing feeds
    forward (the carry is the exact untruncated spectrum), so the cuts are independent
    given the sequential SVD chain. Per-cut RNG streams are spawned up front, so the
    result is bit-identical at any ``cut_workers`` (BLAS releases the GIL; pair with
    ``--blas-threads 1``)."""
    lx = len(out_dims)
    ell = y.shape[1]
    work = y.reshape(*out_dims, ell)
    left = 1
    child_rngs = rng.spawn(max(lx - 1, 1))

    # phase 1 -- the sequential exact SVD chain: collect every cut's carrier + tails.
    jobs: list[dict] = []
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

        jobs.append(dict(cut=i, a=a, b=b, rho=rho, o_next=o_next, engaged=engaged,
                         vtilde=vtilde, tail_eta=tail_eta, tail_qs=tail_qs,
                         comp_tail=tail_energy(sv, rho),
                         fallback_null=tail_energy(sv[:rho], eta)))
        # exact left-canonical carry upward (no truncation): full spectrum propagates
        work = (sv[:, None] * vh)
        left = sv.shape[0]

    # phase 2 -- the independent per-cut disentangler optimizations (parallelizable).
    def _run_cut(job: dict) -> CutMechanism:
        gen = child_rngs[job["cut"]]
        if job["engaged"]:
            dims = _CutDims(eta=eta, chi=kappa, n2=job["o_next"], n3=job["b"] // job["o_next"])
            res = disentangle_altmin(job["vtilde"], dims, k=eta, maxiter=int(ndis), rng=gen)
            dis_tail, dis_iters = res.tail_final, res.iters
            null_mean, null_std = _null_gauge_tail_std(job["vtilde"], eta, null_draws, gen)
        else:
            dis_tail, dis_iters = job["tail_eta"], 0   # ndis=0 or no room: identity disentangler
            null_mean, null_std = job["fallback_null"], 0.0
        return CutMechanism(
            cut=job["cut"], rows=job["a"], cols=job["b"], kappa=kappa, rho=job["rho"],
            disentangled=job["engaged"], tail_eta_I=job["tail_eta"], tail_etaq_I=job["tail_qs"],
            comp_tail=job["comp_tail"], dis_tail=dis_tail,
            dis_total=job["comp_tail"] + dis_tail, dis_iters=dis_iters,
            null_mean=null_mean, null_std=null_std,
        )

    if cut_workers > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(cut_workers)) as pool:
            return list(pool.map(_run_cut, jobs))
    return [_run_cut(job) for job in jobs]


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


@dataclass
class DisentangledColumnResult:
    """The disentangled-global column's accuracy, bracketed rigorously (verified builds)."""

    eta: int
    kappa: int
    eps_best: float           # ||(I-QQ*)C||/||C|| of the composite (eta*kappa) isometry -- BEST case
    eps_worst: float          # same at the plain vertical bond eta -- WORST case (no residual)
    best_iso_defect: float    # ||Q*Q - I|| of the best-case column (should be ~eps)
    residual_loss: float      # disentangler's rank-eta tails, sqrt(sum)/||Y|| (<< eps => near best)
    n_out: int
    final_bond_best: int      # columns of the best-case isometry (its captured range dim)


def disentangled_column_qr(
    column: ColumnOperator,
    eta: int,
    kappa: int,
    ell: int,
    chi_sk: int = 8,
    n_power: int = 1,
    ndis: int = 10,
    rng: np.random.Generator | None = None,
    reference: np.ndarray | None = None,
    cut_workers: int = 1,
) -> DisentangledColumnResult:
    """Disentangled-global column accuracy, rigorously **bracketed** (no fragile build).

    The disentangled column holds vertical bond ``eta`` and spills the excess into a
    bounded horizontal residual bond ``kappa``. Its true projection error
    ``||(I - Q Q^*) C||_F / ||C||_F`` is bracketed between two builds we CAN do exactly
    with the verified structured QR:

    * ``eps_best`` -- the composite ``eta*kappa`` isometry (residual fully kept); a genuine
      output-isometric column (``best_iso_defect ~ eps``). No disentangled column can beat
      this (it cannot capture more than the composite it retains).
    * ``eps_worst`` -- the plain vertical bond ``eta`` (no residual). The disentangled
      column's range contains this, so it is at least this accurate.

    The disentangler's own rank-``eta`` second-SVD tails (``residual_loss``, from
    :func:`mechanism_profile`) measure how much of the composite the bounded-``kappa``
    residual must drop; ``residual_loss << eps_best`` means the true error hugs the best
    edge. This is the honest, verifiable characterization -- the exact bounded-residual
    isoPEPS build (with the zip-up residual MPS) is the one remaining refinement.
    """
    gen = np.random.default_rng() if rng is None else rng
    c = column.materialize() if reference is None else reference
    n_out = int(c.shape[0])
    eta_eq = min(int(eta) * int(kappa), n_out)

    # Both bracket endpoints use the SAME sketch (a clean bracket differs only by eta_q, not
    # by sketch noise) -- so at kappa=1 (eta_eq == eta) the two builds are bit-identical.
    ell_shared = max(ell, eta_eq + 4)
    seed_shared = int(gen.integers(1 << 30))
    best = structured_column_qr(column, ell=ell_shared, eta_q=eta_eq, chi_sk=chi_sk,
                                n_power=n_power, rng=np.random.default_rng(seed_shared), reference=c)
    worst = structured_column_qr(column, ell=ell_shared, eta_q=int(eta), chi_sk=chi_sk,
                                 n_power=n_power, rng=np.random.default_rng(seed_shared), reference=c)
    y, _ = sketch_range(column, ell=max(ell, eta_eq + 2), chi_sk=chi_sk, n_power=n_power,
                        rng=gen, reference=c)
    prof = mechanism_profile(y, column.output_dims, int(eta), int(kappa), (int(eta),),
                             ndis=int(ndis), rng=np.random.default_rng(int(gen.integers(1 << 30))),
                             cut_workers=int(cut_workers))
    resid_loss = summarize(prof, float(np.linalg.norm(y))).tau_dis

    return DisentangledColumnResult(
        eta=int(eta), kappa=int(kappa), eps_best=best.eps_proj, eps_worst=worst.eps_proj,
        best_iso_defect=best.delta_global, residual_loss=resid_loss, n_out=n_out,
        final_bond_best=best.final_bond,
    )
