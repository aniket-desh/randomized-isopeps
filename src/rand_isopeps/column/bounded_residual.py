"""Executed whole-column factorization with a sequential bounded residual.

The older :mod:`rand_isopeps.column.disentangled_qr` experiment optimizes each
vertical cut of ``Y = C Omega`` independently.  Those cuts are useful mechanism
diagnostics, but they do not define one tensor network: the carrier chosen at one
cut is not used at the next cut.  This module implements the missing direct
object.

The sampled columns are kept as a *block MPS*.  A bottom-to-top Moses sweep then
factorizes each active tensor into

``Q_i[vertical_down, output, residual, vertical_up]`` and
``R_i[residual_down, residual, residual_up]``.

At an internal site the first SVD keeps a composite space of dimension at most
``eta * kappa``.  A full unitary gauge reorganizes that same space, a second SVD
keeps the propagated residual rank at most ``eta``, and its left factor is
contracted into the next sampled-range core.  Consequently every optimization
acts on the carrier produced by the preceding site.  ``kappa=1`` removes the
sideways residual legs and reduces to the ordinary left TT-SVD sweep.

The returned residual of the *original* column is explicit and matrix-free:
contracting the output leg of each ``Q_i.conj()`` with the corresponding MPO core
of ``C`` produces an MPO for ``R_C = Q^* C``.  It can be composed with or absorbed
into the neighbouring column without materializing either exponentially large
matrix.  Dense materialization is used only by the small-system oracle helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
import tracemalloc

import numpy as np

from rand_isopeps.backend import (
    array_namespace,
    asarray as backend_asarray,
    svd as backend_svd,
    svdvals as backend_svdvals,
    synchronize,
    to_numpy,
)
from rand_isopeps.column.operator import ColumnOperator
from rand_isopeps.compression.mpo_mps_absorb import max_mps_bond
from rand_isopeps.linalg.rmps_sketch import rmps_cores
from rand_isopeps.linalg.sketches import range_sample
from rand_isopeps.moses.disentangler import disentangle_altmin, unitary_defect


@dataclass(frozen=True)
class _SweepDims:
    """Dimensions read by the verified Moses disentangler reshuffle."""

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


@dataclass
class BoundedCutRecord:
    """Executed dimensions, spectra, and losses at one sequential sweep site."""

    site: int
    kind: str
    first_shape: tuple[int, int]
    second_shape: tuple[int, int] | None
    q_down: int
    q_up: int
    residual_down: int
    residual_up: int
    kappa_actual: int
    singular_first: np.ndarray = field(repr=False)
    singular_second: np.ndarray | None = field(default=None, repr=False)
    discarded_first: float = 0.0
    discarded_second: float = 0.0
    gauge_iters: int = 0
    gauge_unitary_defect: float = 0.0

    @property
    def composite_dim(self) -> int:
        return int(self.q_up * self.kappa_actual)


@dataclass
class BoundedResidualResult:
    """An executed arrow-isometric column and its contractible residual.

    ``q_cores`` use ``(vertical_down, output, residual, vertical_up)``.
    ``sample_residual_cores`` use ``(residual_down, residual, residual_up)``;
    their final right boundary is the sample dimension ``ell``.  The combination
    reconstructs the sampled block-MPS.  ``residual_cores`` are MPO cores for
    ``Q^* C`` in the standard ``(ml, dout, din, mr)`` convention.
    """

    eta: int
    kappa: int
    chi_sk: int
    ell: int
    n_power: int
    sketch_kind: str
    q_cores: list[np.ndarray] = field(repr=False)
    sample_residual_cores: list[np.ndarray] = field(repr=False)
    residual_cores: list[np.ndarray] = field(repr=False)
    sampled_cores: list[np.ndarray] = field(repr=False)
    cuts: list[BoundedCutRecord]
    reconstruction_error: float
    residual_consistency_error: float
    delta_local: float
    delta_global: float
    delta_global_bound: float
    projection_error_dense: float
    spectral_tail_dense: float
    projection_excess_dense: float
    q_flat_rank: int
    max_q_vertical: int
    max_residual_vertical: int
    max_sample_residual_vertical: int
    residual_dims: tuple[int, ...]
    matrix_mps_products: int
    passes: int
    contraction_count: int
    contraction_flops_estimate: int
    peak_allocated_bytes: int
    runtime_s: float
    dense_oracle_runtime_s: float

    @property
    def residual_operator(self) -> ColumnOperator:
        """The explicit matrix-free residual ``R_C = Q^* C``."""
        return ColumnOperator(self.residual_cores)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "eta": self.eta,
            "kappa": self.kappa,
            "chi_sk": self.chi_sk,
            "ell": self.ell,
            "n_power": self.n_power,
            "sketch_kind": self.sketch_kind,
            "reconstruction_error": self.reconstruction_error,
            "residual_consistency_error": self.residual_consistency_error,
            "delta_local": self.delta_local,
            "delta_global": self.delta_global,
            "delta_global_bound": self.delta_global_bound,
            "projection_error_dense": self.projection_error_dense,
            "spectral_tail_dense": self.spectral_tail_dense,
            "projection_excess_dense": self.projection_excess_dense,
            "q_flat_rank": self.q_flat_rank,
            "max_q_vertical": self.max_q_vertical,
            "max_residual_vertical": self.max_residual_vertical,
            "max_sample_residual_vertical": self.max_sample_residual_vertical,
            "matrix_mps_products": self.matrix_mps_products,
            "passes": self.passes,
            "contraction_count": self.contraction_count,
            "contraction_flops_estimate": self.contraction_flops_estimate,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "runtime_s": self.runtime_s,
            "dense_oracle_runtime_s": self.dense_oracle_runtime_s,
        }


@dataclass(frozen=True)
class ProjectionScore:
    """Fresh-probe estimate of the normalized whole-column residual.

    The confidence interval is a paired nonparametric bootstrap over the
    numerator/denominator contribution from each probe.  It quantifies scoring
    noise only; construction randomness is handled by the experiment's nested
    sketch seeds.
    """

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    numerator_mean: float
    denominator_mean: float
    n_probes: int
    chi_score: int
    matrix_mps_products: int
    contraction_count: int
    contraction_flops_estimate: int
    peak_mps_bond: int
    runtime_s: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "numerator_mean": self.numerator_mean,
            "denominator_mean": self.denominator_mean,
            "n_probes": self.n_probes,
            "chi_score": self.chi_score,
            "matrix_mps_products": self.matrix_mps_products,
            "contraction_count": self.contraction_count,
            "contraction_flops_estimate": self.contraction_flops_estimate,
            "peak_mps_bond": self.peak_mps_bond,
            "runtime_s": self.runtime_s,
        }


def block_mps_to_matrix(cores: list[np.ndarray]) -> np.ndarray:
    """Materialize a block MPS whose final right boundary labels its columns."""
    if not cores or cores[0].shape[0] != 1:
        raise ValueError("block MPS must have a unit left boundary")
    xp = array_namespace(cores)
    state = xp.asarray(cores[0])[0]
    for core in cores[1:]:
        if state.shape[-1] != core.shape[0]:
            raise ValueError("block-MPS bond mismatch")
        state = xp.tensordot(state, xp.asarray(core), axes=(-1, 0))
        state = state.reshape(-1, state.shape[-1])
    return state


def _is_complex(value) -> bool:
    return np.issubdtype(np.dtype(value.dtype), np.complexfloating)


def _contract_flops(output_entries: int, summed_entries: int, complex_valued: bool) -> int:
    """Shape-derived real-FLOP estimate for one multiply-sum contraction."""
    per_output = (8 * summed_entries - 2) if complex_valued else (2 * summed_entries - 1)
    return int(output_entries) * max(int(per_output), 1)


def _mpo_mps_flops(mpo: list[np.ndarray], mps: list[np.ndarray]) -> int:
    total = 0
    for w, a in zip(mpo, mps):
        ml, dout, din, mr = map(int, w.shape)
        dl, _, dr = map(int, a.shape)
        total += _contract_flops(
            dl * ml * dout * dr * mr, din,
            _is_complex(w) or _is_complex(a),
        )
    return total


def _mps_inner_flops(a: list[np.ndarray], b: list[np.ndarray]) -> int:
    total = 0
    env_a = env_b = 1
    for aa, bb in zip(a, b):
        right_a, right_b = int(aa.shape[2]), int(bb.shape[2])
        physical = int(aa.shape[1])
        complex_valued = _is_complex(aa) or _is_complex(bb)
        # Match the cheaper of the two pairwise paths used by optimized einsum,
        # rather than charging the exponentially worse simultaneous contraction.
        via_a = _contract_flops(
            env_b * physical * right_a, env_a, complex_valued
        ) + _contract_flops(
            right_a * right_b, env_b * physical, complex_valued
        )
        via_b = _contract_flops(
            env_a * physical * right_b, env_b, complex_valued
        ) + _contract_flops(
            right_a * right_b, env_a * physical, complex_valued
        )
        total += min(via_a, via_b)
        env_a, env_b = right_a, right_b
    return total


def _measure_peak_allocations(function):
    """Trace Python/NumPy allocations while always stopping cleanly."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        owns_tracer = not tracemalloc.is_tracing()
        if owns_tracer:
            tracemalloc.start()
        try:
            return function(*args, **kwargs)
        finally:
            if owns_tracer:
                tracemalloc.stop()

    return wrapped


def block_mps_inner(a: list[np.ndarray], b: list[np.ndarray]) -> complex:
    """Frobenius inner product of two block MPSs with the same column boundary."""
    if len(a) != len(b):
        raise ValueError("block MPS lengths must match")
    xp = array_namespace(a, b)
    dtype = xp.result_type(*[core.dtype for core in (*a, *b)])
    env = xp.ones((1, 1), dtype=dtype)
    for aa, bb in zip(a, b):
        if aa.shape[1] != bb.shape[1]:
            raise ValueError("block MPS physical dimensions must match")
        aa = xp.asarray(aa)
        bb = xp.asarray(bb)
        env = xp.einsum("ab,aic,bid->cd", env, aa.conj(), bb, optimize=True)
    if env.shape[0] != env.shape[1]:
        raise ValueError("block MPS column boundaries must match")
    return complex(to_numpy(xp.trace(env)))


def block_mps_relative_error(exact: list[np.ndarray], approx: list[np.ndarray]) -> float:
    """Relative Frobenius error without materializing the sampled matrix."""
    # On oracle-sized problems, direct subtraction avoids catastrophic cancellation
    # when the reconstruction is at machine precision.  Scaling runs stay on the
    # transfer-matrix path below.
    elements = int(np.prod([core.shape[1] for core in exact])) * int(exact[-1].shape[2])
    if elements <= 2_000_000:
        a = block_mps_to_matrix(exact)
        b = block_mps_to_matrix(approx)
        xp = array_namespace(a, b)
        numerator = xp.linalg.norm(a - b)
        denominator = float(to_numpy(xp.linalg.norm(a)))
        return float(to_numpy(numerator)) / max(denominator, 1e-300)
    ne = float(block_mps_inner(exact, exact).real)
    na = float(block_mps_inner(approx, approx).real)
    ov = block_mps_inner(exact, approx)
    err2 = max(ne + na - 2.0 * ov.real, 0.0)
    return float(np.sqrt(err2 / max(ne, 1e-300)))


def dense_to_block_mps(y: np.ndarray, out_dims: tuple[int, ...]) -> list[np.ndarray]:
    """Exact TT-SVD of a small dense ``(prod(out_dims), ell)`` oracle matrix."""
    if y.ndim != 2 or y.shape[0] != int(np.prod(out_dims)):
        raise ValueError("dense sampled range has incompatible output dimensions")
    ell = int(y.shape[1])
    work = y.reshape(*out_dims, ell)
    cores: list[np.ndarray] = []
    left = 1
    for o in out_dims[:-1]:
        mat = work.reshape(left * int(o), -1)
        u, s, vh = backend_svd(mat)
        rank = int(s.shape[0])
        cores.append(u.reshape(left, int(o), rank))
        work = s[:, None] * vh
        left = rank
    cores.append(work.reshape(left, int(out_dims[-1]), ell))
    return cores


def _right_canonicalize_block_mps(cores: list[np.ndarray]) -> list[np.ndarray]:
    """Return an exactly equivalent right-canonical block MPS.

    The final sample boundary is included in the right side of the top-core
    matricization.  After this sweep, the unprocessed upper environment is an
    isometry at every site, so a local first SVD has the same singular values as
    the corresponding cut of the full sampled matrix.  This is what makes the
    ``kappa=1`` path reduce to the representation-independent TT-SVD sweep.
    """
    xp = array_namespace(cores)
    work = [xp.asarray(core).copy() for core in cores]
    for i in range(len(work) - 1, 0, -1):
        left, phys, right = map(int, work[i].shape)
        mat = work[i].reshape(left, phys * right)
        u, singular, vh = backend_svd(mat)
        rank = int(singular.shape[0])
        work[i] = vh.reshape(rank, phys, right)
        transfer = u * singular
        work[i - 1] = xp.tensordot(work[i - 1], transfer, axes=(2, 0))
    return work


def _stack_mps_columns(columns: list[list[np.ndarray]], scale: float = 1.0) -> list[np.ndarray]:
    """Direct-sum independent MPSs into one block MPS with a final ``ell`` boundary."""
    if not columns:
        raise ValueError("at least one sampled column is required")
    ell = len(columns)
    lx = len(columns[0])
    if any(len(col) != lx for col in columns):
        raise ValueError("all sampled MPS columns must have the same length")
    xp = array_namespace(columns)
    dtype = xp.result_type(*[core.dtype for col in columns for core in col])
    out: list[np.ndarray] = []
    for i in range(lx):
        phys = int(columns[0][i].shape[1])
        if any(int(col[i].shape[1]) != phys for col in columns):
            raise ValueError("sampled columns have inconsistent physical dimensions")
        left_sizes = [int(col[i].shape[0]) for col in columns]
        right_sizes = [int(col[i].shape[2]) for col in columns]
        if i == 0:
            core = xp.zeros((1, phys, sum(right_sizes)), dtype=dtype)
            off = 0
            for col, width in zip(columns, right_sizes):
                core[0, :, off:off + width] = col[i][0]
                off += width
            core *= scale
        elif i == lx - 1:
            core = xp.zeros((sum(left_sizes), phys, ell), dtype=dtype)
            off = 0
            for sample, (col, width) in enumerate(zip(columns, left_sizes)):
                if col[i].shape[2] != 1:
                    raise ValueError("each sampled MPS must have a unit right boundary")
                core[off:off + width, :, sample] = col[i][:, :, 0]
                off += width
        else:
            core = xp.zeros((sum(left_sizes), phys, sum(right_sizes)), dtype=dtype)
            lo = ro = 0
            for col, lw, rw in zip(columns, left_sizes, right_sizes):
                core[lo:lo + lw, :, ro:ro + rw] = col[i]
                lo += lw
                ro += rw
        out.append(core)
    return out


def _choose_composite_dims(limit: int, eta: int, kappa: int) -> tuple[int, int]:
    """Largest factorable ``vertical * residual <= limit`` within both caps."""
    best = (1, 1)
    best_prod = 1
    for vertical in range(1, min(int(eta), int(limit)) + 1):
        residual = min(int(kappa), int(limit) // vertical)
        prod = vertical * residual
        if prod > best_prod or (prod == best_prod and vertical > best[0]):
            best = (vertical, residual)
            best_prod = prod
    return best


def _combined_sampled_cores(q_cores: list[np.ndarray], r_cores: list[np.ndarray]) -> list[np.ndarray]:
    """Contract each Q/R horizontal leg into a block-MPS reconstruction."""
    out: list[np.ndarray] = []
    for q, r in zip(q_cores, r_cores):
        if q.shape[2] != r.shape[1]:
            raise ValueError("Q/residual horizontal dimensions do not match")
        # (q_down, r_down, output, q_up, r_up)
        xp = array_namespace(q, r)
        q = xp.asarray(q)
        r = xp.asarray(r)
        site = xp.einsum("aohb,chd->acobd", q, r, optimize=True)
        out.append(site.reshape(q.shape[0] * r.shape[0], q.shape[1], q.shape[3] * r.shape[2]))
    return out


def materialize_q(q_cores: list[np.ndarray]) -> np.ndarray:
    """Dense small-system oracle for the arrow-isometric column Q."""
    return ColumnOperator(q_cores).materialize()


def _residual_from_column(q_cores: list[np.ndarray], column: ColumnOperator) -> list[np.ndarray]:
    """Contract ``Q^* C`` corewise into a residual MPO, without dense matrices."""
    if len(q_cores) != column.lx:
        raise ValueError("Q and column lengths must match")
    out: list[np.ndarray] = []
    for q, c in zip(q_cores, column.cores):
        if q.shape[1] != c.shape[1]:
            raise ValueError("Q output dimension does not match the column")
        # q=(a,o,h,b), c=(x,o,d,y) -> (a*x,h,d,b*y)
        xp = array_namespace(q, c)
        q = xp.asarray(q)
        c = xp.asarray(c)
        core = xp.einsum("aohb,xody->axhdby", q.conj(), c, optimize=True)
        out.append(core.reshape(q.shape[0] * c.shape[0], q.shape[2], c.shape[2],
                                q.shape[3] * c.shape[3]))
    return out


def _range_block_mps(
    column: ColumnOperator,
    ell: int,
    chi_sk: int,
    n_power: int,
    rng: np.random.Generator,
    complex_valued: bool,
) -> tuple[list[np.ndarray], int, int, int]:
    """Form ``C(C^*C)^q Omega`` through MPO-MPS products and retain MPS columns."""
    columns: list[list[np.ndarray]] = []
    contractions = 0
    flops = 0
    for _ in range(int(ell)):
        omega = rmps_cores(column.input_dims, int(chi_sk), rng,
                           complex_valued=complex_valued)
        omega = [backend_asarray(core, like=column.cores[0]) for core in omega]
        flops += _mpo_mps_flops(column.cores, omega)
        sampled = column.matvec_mps(omega)
        contractions += column.lx
        for _ in range(max(0, int(n_power))):
            xp = array_namespace(column.cores)
            adjoint = [xp.asarray(core).conj().transpose(0, 2, 1, 3)
                       for core in column.cores]
            flops += _mpo_mps_flops(adjoint, sampled)
            sampled = column.rmatvec_mps(sampled)
            flops += _mpo_mps_flops(column.cores, sampled)
            sampled = column.matvec_mps(sampled)
            contractions += 2 * column.lx
        columns.append(sampled)
    return (
        _stack_mps_columns(columns, scale=1.0 / np.sqrt(ell)),
        len(columns), contractions, flops,
    )


def factor_sampled_block_mps(
    sampled_cores: list[np.ndarray],
    eta: int,
    kappa: int,
    ndis: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[BoundedCutRecord],
    float, float, int, int, int,
]:
    """Sequentially factor one sampled block MPS into consistent Q/R columns.

    Returns ``(q_cores, r_cores, cuts, delta_local, global_bound,
    contraction_count, peak_bytes)``.  Every internal second-SVD carrier is
    contracted into the next *actual* sampled core before its first SVD.
    """
    if eta < 1 or kappa < 1:
        raise ValueError("eta and kappa must be positive")
    if not sampled_cores or sampled_cores[0].shape[0] != 1:
        raise ValueError("sampled block MPS must have a unit left boundary")
    for left, right in zip(sampled_cores, sampled_cores[1:]):
        if left.shape[2] != right.shape[0]:
            raise ValueError("sampled block-MPS bond mismatch")

    xp = array_namespace(sampled_cores)
    if xp is not np and int(ndis) > 0:
        raise ValueError("GPU bounded-residual factorization currently requires ndis=0")
    gen = np.random.default_rng() if rng is None else rng
    sampled_cores = _right_canonicalize_block_mps(sampled_cores)
    dtype = xp.result_type(*[core.dtype for core in sampled_cores])
    # (Q vertical down, sampled-MPS bond down, residual vertical down)
    carrier = xp.ones((1, 1, 1), dtype=dtype)
    q_cores: list[np.ndarray] = []
    r_cores: list[np.ndarray] = []
    records: list[BoundedCutRecord] = []
    delta_local = 0.0
    contraction_count = 0
    contraction_flops = 0
    peak_bytes = int(carrier.nbytes + sum(c.nbytes for c in sampled_cores))

    for site, core in enumerate(sampled_cores):
        m_down, output, m_up = map(int, core.shape)
        if carrier.shape[1] != m_down:
            raise ValueError(f"carrier/sample bond mismatch at site {site}")
        q_down, _, residual_down = map(int, carrier.shape)
        # theta=(q_down, output, sampled_up, residual_down)
        theta = xp.einsum("amc,mob->aobc", carrier, core, optimize=True)
        contraction_count += 1
        contraction_flops += _contract_flops(
            theta.size, m_down, _is_complex(carrier) or _is_complex(core)
        )
        mat = theta.reshape(q_down * output, m_up * residual_down)

        if site == len(sampled_cores) - 1:
            rank = max(1, min(int(eta), min(mat.shape)))
            u, singular, vh = backend_svd(mat)
            qmat = u[:, :rank]
            q_cores.append(qmat.reshape(q_down, output, rank, 1))
            rtop = singular[:rank, None] * vh[:rank]
            # right boundary m_up is the sample label ell
            r_cores.append(rtop.reshape(rank, m_up, residual_down).transpose(2, 0, 1))
            discarded = float(to_numpy(xp.sum(singular[rank:] ** 2)))
            defect = float(to_numpy(
                xp.linalg.norm(qmat.conj().T @ qmat - xp.eye(rank, dtype=qmat.dtype))
            ))
            delta_local = max(delta_local, defect)
            records.append(BoundedCutRecord(
                site=site, kind="top", first_shape=mat.shape, second_shape=None,
                q_down=q_down, q_up=1, residual_down=residual_down,
                residual_up=m_up, kappa_actual=rank, singular_first=singular.copy(),
                discarded_first=discarded,
            ))
            peak_bytes = max(peak_bytes, sum(x.nbytes for x in (theta, mat, u, singular, vh, rtop)))
            break

        rank_limit = min(mat.shape)
        q_up, residual_dim = _choose_composite_dims(
            min(rank_limit, int(eta) * int(kappa)), int(eta), int(kappa)
        )
        composite = q_up * residual_dim
        u1, singular1, vh1 = backend_svd(mat)
        vtilde = singular1[:composite, None] * vh1[:composite]
        second_rows = q_up * m_up
        second_cols = residual_dim * residual_down
        residual_up = max(1, min(int(eta), second_rows, second_cols))

        gauge_iters = 0
        gauge_defect = 0.0
        gauge = xp.eye(composite, dtype=dtype)
        # Product gauges on q_up x residual_dim are spectrally inert after the
        # reshuffle.  Only a full gauge with both factors nontrivial can help.
        if int(ndis) > 0 and q_up > 1 and residual_dim > 1:
            dims = _SweepDims(q_up, residual_dim, m_up, residual_down)
            dis = disentangle_altmin(vtilde, dims, k=residual_up, maxiter=int(ndis), rng=gen)
            gauge = dis.q
            gauge_iters = int(dis.iters)
            gauge_defect = unitary_defect(gauge)

        qmat = u1[:, :composite] @ gauge.conj().T
        q_core = qmat.reshape(q_down, output, q_up, residual_dim).transpose(0, 1, 3, 2)
        gauged = (gauge @ vtilde).reshape(q_up, residual_dim, m_up, residual_down)
        second = gauged.transpose(0, 2, 1, 3).reshape(second_rows, second_cols)
        u2, singular2, vh2 = backend_svd(second)
        # Mirror the real Moses move's absorb=-1 convention: singular values ride
        # with the upward carrier; the peeled residual core is row-isometric.
        carrier = (u2[:, :residual_up] * singular2[:residual_up]).reshape(
            q_up, m_up, residual_up
        )
        r_core = vh2[:residual_up].reshape(residual_up, residual_dim, residual_down).transpose(2, 1, 0)

        q_cores.append(q_core)
        r_cores.append(r_core)
        defect = float(to_numpy(
            xp.linalg.norm(
                qmat.conj().T @ qmat - xp.eye(composite, dtype=qmat.dtype)
            )
        ))
        delta_local = max(delta_local, defect)
        records.append(BoundedCutRecord(
            site=site, kind="internal", first_shape=mat.shape, second_shape=second.shape,
            q_down=q_down, q_up=q_up, residual_down=residual_down,
            residual_up=residual_up, kappa_actual=residual_dim,
            singular_first=singular1.copy(), singular_second=singular2.copy(),
            discarded_first=float(to_numpy(xp.sum(singular1[composite:] ** 2))),
            discarded_second=float(to_numpy(xp.sum(singular2[residual_up:] ** 2))),
            gauge_iters=gauge_iters, gauge_unitary_defect=gauge_defect,
        ))
        peak_bytes = max(
            peak_bytes,
            sum(x.nbytes for x in (theta, mat, u1, singular1, vh1, vtilde, gauge,
                                    qmat, gauged, second, u2, singular2, vh2,
                                    carrier, q_core, r_core)),
        )

    # If E_i = Q_i^*Q_i-I, recursive contraction gives a conservative spectral
    # perturbation bound prod_i(1+||E_i||_2)-1.  Dense oracle runs also report the
    # requested exact Frobenius global defect below.
    local_defects = []
    for q in q_cores:
        matrix = q.reshape(q.shape[0] * q.shape[1], -1)
        identity = xp.eye(matrix.shape[1], dtype=matrix.dtype)
        local_defects.append(float(to_numpy(
            xp.linalg.norm(matrix.conj().T @ matrix - identity)
        )))
    global_bound = float(np.expm1(sum(np.log1p(max(0.0, value))
                                      for value in local_defects)))
    return (
        q_cores, r_cores, records, delta_local, global_bound,
        contraction_count, peak_bytes, contraction_flops,
    )


@_measure_peak_allocations
def bounded_residual_column_qr(
    column: ColumnOperator,
    ell: int,
    eta: int,
    kappa: int,
    chi_sk: int = 8,
    sketch_kind: str = "rmps",
    n_power: int = 0,
    ndis: int = 10,
    rng: np.random.Generator | None = None,
    reference: np.ndarray | None = None,
    reference_singular_values: np.ndarray | None = None,
    dense_oracle_max_elements: int = 2_000_000,
) -> BoundedResidualResult:
    """Execute the matrix-free sampled-range factorization and build ``Q^* C``.

    ``sketch_kind='rmps'`` is the proposed method and ``'kron'`` is its
    ``chi_sk=1`` negative control.  The Gaussian, Rademacher, and SparseStack
    methods are dense small-system controls and require a materializable column.
    """
    if ell < 1 or eta < 1 or kappa < 1:
        raise ValueError("ell, eta, and kappa must be positive")
    if n_power < 0:
        raise ValueError("n_power must be nonnegative")
    if sketch_kind not in ("rmps", "kron", "gaussian", "rademacher", "sparsestack"):
        raise ValueError(f"unknown sketch kind: {sketch_kind!r}")

    gen = np.random.default_rng() if rng is None else rng
    xp = array_namespace(column.cores)
    if xp is not np and int(ndis) > 0:
        raise ValueError("GPU bounded-residual factorization currently requires ndis=0")
    synchronize(column.cores)
    t0 = perf_counter()
    used_chi = 1 if sketch_kind == "kron" else int(chi_sk)
    complex_valued = any(_is_complex(core) for core in column.cores)
    ell = max(1, min(int(ell), column.n_in))

    if sketch_kind in ("gaussian", "rademacher", "sparsestack"):
        if reference is None:
            c = column.materialize()
            c_host = to_numpy(c)
        else:
            c_host = to_numpy(reference)
            c = backend_asarray(c_host, like=column.cores[0])
        sampled_host = range_sample(c_host, ell, gen, sketch_kind)
        y = backend_asarray(sampled_host, like=c)
        ell = int(y.shape[1])
        for _ in range(int(n_power)):
            y = c @ (c.conj().T @ y)
        sampled_cores = dense_to_block_mps(y, column.output_dims)
        matrix_products = ell * (1 + 2 * int(n_power))
        sketch_contractions = matrix_products
        sketch_flops = _contract_flops(
            column.n_out * ell, column.n_in, complex_valued
        ) + int(n_power) * (
            _contract_flops(column.n_in * ell, column.n_out, complex_valued)
            + _contract_flops(column.n_out * ell, column.n_in, complex_valued)
        )
    else:
        sampled_cores, _, sketch_contractions, sketch_flops = _range_block_mps(
            column, ell, used_chi, int(n_power), gen, complex_valued
        )
        matrix_products = ell * (1 + 2 * int(n_power))

    (q_cores, sample_r, cuts, delta_local, delta_bound, sweep_contractions,
     peak, sweep_flops) = (
        factor_sampled_block_mps(sampled_cores, int(eta), int(kappa), int(ndis), gen)
    )
    reconstructed = _combined_sampled_cores(q_cores, sample_r)
    reconstruction_flops = sum(
        _contract_flops(
            q.shape[0] * r.shape[0] * q.shape[1] * q.shape[3] * r.shape[2],
            q.shape[2], _is_complex(q) or _is_complex(r),
        )
        for q, r in zip(q_cores, sample_r)
    )
    reconstruction_error = block_mps_relative_error(sampled_cores, reconstructed)
    residual_cores = _residual_from_column(q_cores, column)
    residual_flops = sum(
        _contract_flops(
            q.shape[0] * c.shape[0] * q.shape[2] * c.shape[2]
            * q.shape[3] * c.shape[3],
            q.shape[1], _is_complex(q) or _is_complex(c),
        )
        for q, c in zip(q_cores, column.cores)
    )
    contraction_count = sketch_contractions + sweep_contractions + 2 * column.lx
    contraction_flops = sketch_flops + sweep_flops + reconstruction_flops + residual_flops
    peak = max(peak, sum(x.nbytes for x in q_cores + sample_r + residual_cores))
    if tracemalloc.is_tracing():
        _, traced_factor_peak = tracemalloc.get_traced_memory()
        peak = max(peak, int(traced_factor_peak))
    synchronize(residual_cores)
    factor_runtime = perf_counter() - t0

    residual_dims = tuple(int(q.shape[2]) for q in q_cores)
    q_columns = int(np.prod(residual_dims))
    dense_possible = (
        column.n_out * q_columns <= int(dense_oracle_max_elements)
        and column.n_out * column.n_in <= int(dense_oracle_max_elements)
    )
    delta_global = float("nan")
    residual_consistency = float("nan")
    projection_error = float("nan")
    spectral_tail = float("nan")
    projection_excess = float("nan")
    oracle_t0 = perf_counter()
    if dense_possible:
        q_dense = materialize_q(q_cores)
        dense_xp = array_namespace(q_dense)
        delta_global = float(to_numpy(dense_xp.linalg.norm(
            q_dense.conj().T @ q_dense
            - dense_xp.eye(q_dense.shape[1], dtype=q_dense.dtype)
        )))
        y_dense = block_mps_to_matrix(sampled_cores)
        rs_dense = block_mps_to_matrix(sample_r)
        y_norm = float(to_numpy(dense_xp.linalg.norm(y_dense)))
        residual_consistency = float(to_numpy(
            dense_xp.linalg.norm(rs_dense - q_dense.conj().T @ y_dense)
        )) / max(y_norm, 1e-300)
        c_dense = (
            column.materialize()
            if reference is None
            else backend_asarray(reference, like=q_dense)
        )
        if reference_singular_values is None:
            singular_c = backend_svdvals(c_dense)
        else:
            singular_c = backend_asarray(
                reference_singular_values, like=q_dense, dtype=float
            )
            if singular_c.ndim != 1 or singular_c.size != min(c_dense.shape):
                raise ValueError("reference singular values have incompatible shape")
        c_norm_sq = float(to_numpy(dense_xp.sum(singular_c ** 2)))
        r_dense = ColumnOperator(residual_cores).materialize()
        projection_error = float(to_numpy(
            dense_xp.linalg.norm(c_dense - q_dense @ r_dense)
        )) / max(np.sqrt(c_norm_sq), 1e-300)
        tail_sq = float(to_numpy(dense_xp.sum(singular_c[q_columns:] ** 2)))
        spectral_tail = float(np.sqrt(tail_sq / max(c_norm_sq, 1e-300)))
        projection_excess = float(np.sqrt(max(
            projection_error ** 2 - spectral_tail ** 2, 0.0
        )))
        peak = max(peak, sum(
            x.nbytes for x in (q_dense, y_dense, rs_dense, c_dense, r_dense, singular_c)
        ))

    synchronize(q_cores)
    oracle_runtime = perf_counter() - oracle_t0
    return BoundedResidualResult(
        eta=int(eta), kappa=int(kappa), chi_sk=used_chi, ell=ell,
        n_power=int(n_power), sketch_kind=sketch_kind, q_cores=q_cores,
        sample_residual_cores=sample_r, residual_cores=residual_cores,
        sampled_cores=sampled_cores, cuts=cuts,
        reconstruction_error=reconstruction_error,
        residual_consistency_error=residual_consistency,
        delta_local=delta_local, delta_global=delta_global,
        delta_global_bound=delta_bound, projection_error_dense=projection_error,
        spectral_tail_dense=spectral_tail, projection_excess_dense=projection_excess,
        q_flat_rank=q_columns,
        max_q_vertical=max(max(q.shape[0], q.shape[3]) for q in q_cores),
        # ``Q^* C`` is the residual that is actually absorbed downstream.  Its
        # vertical rank is the product of the Q and original-column ranks and is
        # therefore the structural growth that the state benchmark must charge.
        max_residual_vertical=max(max(r.shape[0], r.shape[3]) for r in residual_cores),
        max_sample_residual_vertical=max(max(r.shape[0], r.shape[2]) for r in sample_r),
        residual_dims=residual_dims, matrix_mps_products=matrix_products,
        passes=1 + 2 * int(n_power), contraction_count=contraction_count,
        contraction_flops_estimate=int(contraction_flops),
        peak_allocated_bytes=int(peak), runtime_s=float(factor_runtime),
        dense_oracle_runtime_s=float(oracle_runtime),
    )


def score_projection_error(
    column: ColumnOperator,
    result: BoundedResidualResult,
    *,
    n_probes: int = 32,
    chi_score: int | None = None,
    rng: np.random.Generator | None = None,
    confidence: float = 90.0,
    n_bootstrap: int = 1000,
) -> ProjectionScore:
    """Score ``||(I-QQ*)C||_F / ||C||_F`` without materializing ``C``.

    Each fresh isotropic rMPS probe ``omega`` is sent through both ``C`` and the
    executed factorization ``Q (Q^* C)``.  The ratio of summed squared residual
    and output norms is a consistent Hutchinson estimator of the squared
    Frobenius ratio.  Construction and scoring RNGs are deliberately separate at
    the call site; this function never reuses ``result.sampled_cores``.

    The returned bootstrap interval resamples paired numerator/denominator
    contributions.  ``chi_score`` controls the high-order variance of the rMPS
    probes and defaults to ``max(result.chi_sk, column.lx)``.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be positive")
    if not (0.0 < confidence < 100.0):
        raise ValueError("confidence must lie strictly between 0 and 100")
    if len(result.q_cores) != column.lx:
        raise ValueError("factorization and column lengths must match")

    gen = np.random.default_rng() if rng is None else rng
    used_chi = max(int(result.chi_sk), column.lx) if chi_score is None else int(chi_score)
    if used_chi < 1:
        raise ValueError("chi_score must be positive")
    complex_valued = any(_is_complex(core) for core in column.cores)
    q_operator = ColumnOperator(result.q_cores)
    residual_operator = result.residual_operator
    numerator = np.empty(int(n_probes), dtype=float)
    denominator = np.empty(int(n_probes), dtype=float)
    peak_bond = 1
    contraction_flops = 0
    synchronize(column.cores)
    t0 = perf_counter()

    for probe in range(int(n_probes)):
        omega = rmps_cores(
            column.input_dims, used_chi, gen, complex_valued=complex_valued
        )
        contraction_flops += _mpo_mps_flops(column.cores, omega)
        exact = column.matvec_mps(omega)
        contraction_flops += _mpo_mps_flops(residual_operator.cores, omega)
        residual = residual_operator.matvec_mps(omega)
        contraction_flops += _mpo_mps_flops(q_operator.cores, residual)
        projected = q_operator.matvec_mps(residual)
        contraction_flops += (
            _mps_inner_flops(exact, exact)
            + _mps_inner_flops(projected, projected)
            + _mps_inner_flops(exact, projected)
        )
        exact_sq = float(block_mps_inner(exact, exact).real)
        projected_sq = float(block_mps_inner(projected, projected).real)
        overlap = block_mps_inner(exact, projected)
        numerator[probe] = max(exact_sq + projected_sq - 2.0 * overlap.real, 0.0)
        denominator[probe] = max(exact_sq, 0.0)
        peak_bond = max(
            peak_bond,
            max_mps_bond(omega),
            max_mps_bond(exact),
            max_mps_bond(residual),
            max_mps_bond(projected),
        )

    synchronize(column.cores)
    estimate = float(np.sqrt(np.sum(numerator) / max(np.sum(denominator), 1e-300)))
    if int(n_bootstrap) >= 2 and int(n_probes) >= 2:
        boot = np.empty(int(n_bootstrap), dtype=float)
        for sample in range(int(n_bootstrap)):
            indices = gen.integers(0, int(n_probes), size=int(n_probes))
            boot[sample] = np.sqrt(
                np.sum(numerator[indices])
                / max(np.sum(denominator[indices]), 1e-300)
            )
        alpha = (100.0 - float(confidence)) / 2.0
        standard_error = float(np.std(boot, ddof=1))
        ci_low, ci_high = map(float, np.percentile(boot, [alpha, 100.0 - alpha]))
    else:
        standard_error = float("nan")
        ci_low = ci_high = estimate

    return ProjectionScore(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=float(confidence),
        numerator_mean=float(np.mean(numerator)),
        denominator_mean=float(np.mean(denominator)),
        n_probes=int(n_probes),
        chi_score=used_chi,
        matrix_mps_products=3 * int(n_probes),
        contraction_count=6 * int(n_probes) * column.lx,
        contraction_flops_estimate=int(contraction_flops),
        peak_mps_bond=int(peak_bond),
        runtime_s=float(perf_counter() - t0),
    )


def apply_boundary_factorization(psi, j: int, result: BoundedResidualResult, split: str = "right"):
    """Compatibility bridge: insert ``Q`` and absorb ``Q* C`` without zip-up.

    The state-level implementation now lives beside the real isoTNS code and
    supports both boundary and interior columns.  Historical one-move callers
    keep this import and its uncompressed return semantics.
    """
    from rand_isopeps.real_isotns.column_bridge import insert_column_factorization

    return insert_column_factorization(
        psi,
        j,
        result.q_cores,
        result.residual_cores,
        split=split,
        inplace=False,
    )
