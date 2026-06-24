"""``ColumnOperator``: the whole active isoTNS column as one linear map.

The block sequential Moses move acts on the ``j``-th column ``C_j``. Locally it
splits each active tensor with the two-SVD isometric tensor-ring decomposition
(:mod:`rand_isopeps.moses.local_ring_decomp`). The *global* view instead fuses
the whole column into a single matrix

    C_j : X_j -> Y_j,

    X_j = (x) F^{r_i}     (the absorbed side: horizontal legs pushed into the
                           neighbouring column, optionally a block index p),
    Y_j = (x) F^{o_i}     (the retained side: physical legs + retained virtual
                           legs that stay in the new isometric column).

The exact column QR is ``C_j = Q_j R_j`` with ``Q_j`` isometric (the new column)
and ``R_j`` absorbed into ``C_{j+1}``. The Moses move is a structured, greedy,
local way to approximate this QR; the global sketched move
(:mod:`rand_isopeps.column.global_range`) approximates the *same* object directly
with a randomized range finder.

Representation -- the access model
----------------------------------
``C_j`` is stored as a vertical **MPO** with one core per row, in the repo's
``(ml, dout, din, mr)`` convention (left bond, retained/output leg, absorbed/input
leg, right bond), the same as
:func:`rand_isopeps.compression.mpo_mps_absorb.random_mpo`. This is faithful: a
real isoTNS active column, once its top/bottom environment is contracted, is an
MPO from absorbed legs (``din = r_i``) to retained legs (``dout = o_i``), with the
vertical MPO bond carrying the intra-column entanglement.

Two access modes share this one object so synthetic-now and real-later use the
same range finder and experiments:

* :meth:`materialize` -- dense ``(N_out, N_in)`` matrix for the tiny validation
  (exact SVD reference, dense ``C @ Omega``).
* :meth:`matvec_mps` -- the matrix-free MPO--MPS product ``omega -> C_j omega``
  for the access-model algorithm, returning an MPS (no ``d^{L}`` blow-up).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rand_isopeps.compression.mpo_mps_absorb import apply_mpo_to_mps, max_mps_bond, mps_to_vector
from rand_isopeps.synthetic.tensors import make_controlled_spectrum_matrix, random_complex

ColumnEnsemble = "gaussian | decay | identity_plus_noise"


def _as_tuple(value: int | tuple[int, ...], lx: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * lx
    out = tuple(int(v) for v in value)
    if len(out) != lx:
        raise ValueError(f"expected {lx} per-site dims, got {len(out)}")
    return out


@dataclass
class ColumnOperator:
    """A column map ``C_j`` as a vertical MPO of cores ``(ml, dout, din, mr)``.

    ``dout = o_i`` are the retained (output / codomain) legs, ``din = r_i`` the
    absorbed (input / domain) legs over which a probe is sketched, and the
    left/right bonds carry the intra-column entanglement (boundary bonds 1).
    """

    cores: list[np.ndarray]

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("a column operator needs at least one core")
        for i, c in enumerate(self.cores):
            if c.ndim != 4:
                raise ValueError(f"core {i} must be 4-index (ml, dout, din, mr), got ndim {c.ndim}")
        if self.cores[0].shape[0] != 1 or self.cores[-1].shape[3] != 1:
            raise ValueError("boundary MPO bonds must be 1 (left of first, right of last)")
        for i in range(len(self.cores) - 1):
            if self.cores[i].shape[3] != self.cores[i + 1].shape[0]:
                raise ValueError(f"MPO bond mismatch between cores {i} and {i + 1}")

    # --- shape bookkeeping ---------------------------------------------------
    @property
    def lx(self) -> int:
        return len(self.cores)

    @property
    def input_dims(self) -> tuple[int, ...]:
        """Absorbed-side factor dims ``(r_1, ..., r_Lx)`` -- the sketch domain."""
        return tuple(int(c.shape[2]) for c in self.cores)

    @property
    def output_dims(self) -> tuple[int, ...]:
        """Retained-side factor dims ``(o_1, ..., o_Lx)`` -- the new column's legs."""
        return tuple(int(c.shape[1]) for c in self.cores)

    @property
    def n_in(self) -> int:
        return int(np.prod(self.input_dims))

    @property
    def n_out(self) -> int:
        return int(np.prod(self.output_dims))

    @property
    def mpo_bond(self) -> int:
        return max((int(c.shape[3]) for c in self.cores[:-1]), default=1)

    # --- access modes --------------------------------------------------------
    def materialize(self) -> np.ndarray:
        """Dense ``(N_out, N_in)`` matrix, both legs flattened row-major over sites.

        The ordering matches :func:`rand_isopeps.linalg.rmps_sketch.rmps_to_vector`
        (row-major over sites) so a dense ``materialize() @ Omega`` agrees with the
        matrix-free :meth:`matvec` to floating-point roundoff.
        """
        work = self.cores[0][0]  # drop left boundary bond -> (dout0, din0, mr0)
        for core in self.cores[1:]:
            work = np.tensordot(work, core, axes=(-1, 0))  # (..., dout_i, din_i, mr_i)
        work = work[..., 0]  # drop trailing right boundary bond -> axes out0,in0,out1,in1,...
        lx = self.lx
        out_axes = list(range(0, 2 * lx, 2))
        in_axes = list(range(1, 2 * lx, 2))
        mat = np.transpose(work, out_axes + in_axes).reshape(self.n_out, self.n_in)
        return mat

    def matvec_mps(self, omega_cores: list[np.ndarray]) -> list[np.ndarray]:
        """Matrix-free ``C_j omega`` as an MPS (MPO--MPS product, the access model).

        ``omega_cores`` is an MPS over the absorbed legs (``din = r_i`` per site),
        e.g. from :func:`rand_isopeps.linalg.rmps_sketch.rmps_cores`. The returned
        MPS has output legs ``o_i`` and bond ``<= mpo_bond * sketch_bond``; no
        ``prod(o_i)``-length dense vector is ever formed.
        """
        return apply_mpo_to_mps(self.cores, omega_cores)

    def matvec(self, omega_cores: list[np.ndarray]) -> np.ndarray:
        """Dense ``C_j omega`` via the matrix-free path then a final contraction.

        Convenience for tiny cases / cross-checks; for the materialized validation
        use ``materialize() @ Omega`` directly.
        """
        return mps_to_vector(self.matvec_mps(omega_cores))

    def product_bond(self, omega_cores: list[np.ndarray]) -> int:
        """Max bond of the MPS ``C_j omega`` -- the cost the access model pays."""
        return max_mps_bond(self.matvec_mps(omega_cores))


def random_column_operator(
    lx: int,
    in_dim: int | tuple[int, ...],
    out_dim: int | tuple[int, ...],
    mpo_bond: int,
    rng: np.random.Generator,
    ensemble: str = "gaussian",
    decay: float = 0.6,
    identity_strength: float = 1.0,
    noise: float = 0.15,
    complex_valued: bool = True,
) -> ColumnOperator:
    """A synthetic column operator (vertical MPO) with a controllable spectrum.

    ``in_dim``/``out_dim`` are the per-site absorbed/retained leg dims (int =
    uniform, or a per-site tuple), ``mpo_bond`` the vertical MPO bond (the
    intra-column entanglement). The flat rank and singular spectrum of the
    materialized matrix are set by ``ensemble``:

    * ``gaussian`` -- iid Gaussian cores: a fairly flat spectrum, the stress case
      where a low-rank column approximation is genuinely lossy.
    * ``decay`` -- Gaussian cores whose right MPO-bond channels are geometrically
      suppressed (``decay ** m``), giving a decaying spectrum where a low-rank
      column range is a good approximation (the regime global sketching should
      win).
    * ``identity_plus_noise`` -- a near-identity column (output ``=`` input plus a
      small random perturbation); a well-conditioned, near-isometric control.

    Returns a :class:`ColumnOperator`; call ``.materialize()`` for the dense
    matrix or ``.matvec_mps()`` for the matrix-free product.
    """
    if ensemble not in ("gaussian", "decay", "identity_plus_noise"):
        raise ValueError(f"unknown column ensemble: {ensemble}")
    in_dims = _as_tuple(in_dim, lx)
    out_dims = _as_tuple(out_dim, lx)
    cores: list[np.ndarray] = []
    for i in range(lx):
        ml = 1 if i == 0 else mpo_bond
        mr = 1 if i == lx - 1 else mpo_bond
        shape = (ml, out_dims[i], in_dims[i], mr)
        scale = 1.0 / np.sqrt(out_dims[i] * max(ml, 1))
        if complex_valued:
            w = random_complex(shape, rng, scale=scale)
        else:
            w = scale * rng.standard_normal(shape)
        if ensemble == "decay" and mr > 1:
            w = w * (decay ** np.arange(mr))[None, None, None, :]
        elif ensemble == "identity_plus_noise":
            w = noise * w
            r = min(out_dims[i], in_dims[i])
            idx = np.arange(r)
            w[0, idx, idx, 0] += identity_strength  # diagonal identity channel
        cores.append(w)
    return ColumnOperator(cores)


def controlled_spectrum_column_matrix(
    out_dims: tuple[int, ...],
    in_dims: tuple[int, ...],
    rng: np.random.Generator,
    decay_kind: str = "exp",
    parameter: float = 4.0,
    complex_valued: bool = True,
) -> np.ndarray:
    """A *dense* column matrix ``C`` with a prescribed singular-value decay.

    Returns the materialized ``(prod(out_dims), prod(in_dims))`` matrix whose
    singular values decay as ``exp(-j/parameter)`` (``decay_kind="exp"``) or
    ``(j+1)^-parameter`` (``"power"``). The left/right singular vectors are random
    orthonormal, so the top-``r`` subspace is generic; only the *domain
    factorization* ``in_dims`` is structured, which is all the rMPS sketch needs
    (its factor count ``t = len(in_dims)`` is the axis where the ``chi_sk = 1``
    Kronecker limit fails).

    Unlike :func:`random_column_operator` this is dense-only -- it has no
    matrix-free MPO form -- so use it for the materialized validation when you
    want crisp control over the spectral decay (the low-rank-friendly regime
    where global sketching's advantage is theoretically visible). The companion
    ``factor_dims`` for the range finder is simply ``tuple(in_dims)``.
    """
    n_out = int(np.prod(out_dims))
    n_in = int(np.prod(in_dims))
    mat, _ = make_controlled_spectrum_matrix(
        n_out, n_in, rng, decay=decay_kind, parameter=parameter, complex_valued=complex_valued
    )
    return mat
