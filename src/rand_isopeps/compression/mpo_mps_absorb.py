"""synthetic R-column absorption as MPO-MPS product and compression."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from rand_isopeps.linalg.randomized_svd import SketchKind, rsvd_truncate, svd_truncate
from rand_isopeps.synthetic.tensors import random_complex

# "src" routes to the vendored successive-randomized-compression code
# (rand_isopeps.src_absorption). It is selectable but not in the default methods.
CompressionMethod = Literal["zipup_svd", "randomized", "src"]


@dataclass
class AbsorptionResult:
    method: CompressionMethod
    l_sites: int
    phys_dim: int
    mps_bond: int
    mpo_bond: int
    target_bond: int
    rel_error_exact: float
    runtime_s: float
    peak_product_bond: int
    final_max_bond: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "l_sites": self.l_sites,
            "phys_dim": self.phys_dim,
            "mps_bond": self.mps_bond,
            "mpo_bond": self.mpo_bond,
            "target_bond": self.target_bond,
            "rel_error_exact": self.rel_error_exact,
            "runtime_s": self.runtime_s,
            "peak_product_bond": self.peak_product_bond,
            "final_max_bond": self.final_max_bond,
        }


def random_mps(
    l_sites: int,
    phys_dim: int,
    bond_dim: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    tensors = []
    for i in range(l_sites):
        left = 1 if i == 0 else bond_dim
        right = 1 if i == l_sites - 1 else bond_dim
        tensors.append(random_complex((left, phys_dim, right), rng, scale=1.0 / np.sqrt(phys_dim * bond_dim)))
    vec = mps_to_vector(tensors)
    norm = np.linalg.norm(vec)
    if norm:
        tensors[0] = tensors[0] / norm
    return tensors


def random_mpo(
    l_sites: int,
    phys_dim: int,
    bond_dim: int,
    rng: np.random.Generator,
    identity_strength: float = 1.0,
    noise: float = 0.08,
) -> list[np.ndarray]:
    tensors = []
    for i in range(l_sites):
        left = 1 if i == 0 else bond_dim
        right = 1 if i == l_sites - 1 else bond_dim
        w = random_complex((left, phys_dim, phys_dim, right), rng, scale=noise / np.sqrt(max(bond_dim, 1)))
        idx = np.arange(phys_dim)
        w[0, idx, idx, 0] += identity_strength
        tensors.append(w)
    return tensors


def apply_mpo_to_mps(mpo: list[np.ndarray], mps: list[np.ndarray]) -> list[np.ndarray]:
    if len(mpo) != len(mps):
        raise ValueError("mpo and mps lengths must match")
    product = []
    for w, a in zip(mpo, mps):
        ml, dout, din, mr = w.shape
        dl, d, dr = a.shape
        if din != d:
            raise ValueError("mpo input physical dimension must match mps physical dimension")
        site = np.einsum("axyb,lyr->laxrb", w, a, optimize=True)
        product.append(site.reshape(dl * ml, dout, dr * mr))
    return product


def apply_mpo_adjoint_to_mps(mpo: list[np.ndarray], mps: list[np.ndarray]) -> list[np.ndarray]:
    """Apply the adjoint operator ``C^*`` (an MPO) to an MPS, matrix-free.

    ``C`` maps the input (absorbed) legs ``din`` to the output (retained) legs
    ``dout``; its adjoint ``C^*`` maps ``dout -> din``. As an MPO that is each core
    conjugated with the two physical legs swapped, ``(ml, dout, din, mr) ->
    (ml, din, dout, mr)``, with the bond structure unchanged. So ``mps`` here is an
    MPS over the *output* legs (``dout``) and the result is an MPS over the *input*
    legs (``din``) -- the access-model form of ``C^* y`` with no ``prod(din)``
    dense vector ever formed.
    """
    adjoint = [np.conj(w).transpose(0, 2, 1, 3) for w in mpo]
    return apply_mpo_to_mps(adjoint, mps)


def mps_norm_sq(mps: list[np.ndarray]) -> float:
    """``<psi|psi>`` of an MPS by the left-to-right transfer-matrix contraction.

    Cost ``O(L * d * bond^3)`` and, crucially, never forms the ``prod(d_i)`` dense
    vector -- the matrix-free norm used to score/normalize probes over the large
    input leg space."""
    env = np.ones((1, 1), dtype=np.result_type(*[s.dtype for s in mps]))
    for a in mps:  # a: (dl, d, dr)
        env = np.einsum("ab,aic,bid->cd", env, a.conj(), a, optimize=True)
    return float(np.abs(env[0, 0]))


def max_mps_bond(mps: list[np.ndarray]) -> int:
    return max(max(site.shape[0], site.shape[2]) for site in mps)


def mps_to_vector(mps: list[np.ndarray]) -> np.ndarray:
    state = mps[0][0, :, :]
    for site in mps[1:]:
        state = np.tensordot(state, site, axes=(-1, 0))
        state = state.reshape(-1, state.shape[-1])
    return state[:, 0]


def compress_mps(
    mps: list[np.ndarray],
    max_bond: int,
    method: CompressionMethod,
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    if max_bond <= 0:
        raise ValueError("max_bond must be positive")
    gen = np.random.default_rng() if rng is None else rng
    work = [site.copy() for site in mps]

    for i in range(len(work) - 1):
        dl, d, dr = work[i].shape
        mat = work[i].reshape(dl * d, dr)
        k = min(max_bond, mat.shape[0], mat.shape[1])
        if method == "zipup_svd":
            result = svd_truncate(mat, k)
        elif method == "randomized":
            result = rsvd_truncate(mat, k, oversample=oversample, n_power=n_power, rng=gen, sketch=sketch)
        else:
            raise ValueError(f"unknown compression method: {method}")

        work[i] = result.u.reshape(dl, d, result.rank)
        transfer = result.s[:, None] * result.vh
        work[i + 1] = np.tensordot(transfer, work[i + 1], axes=(1, 0))

    return work


def relative_vector_error(exact: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(exact)
    if denom == 0:
        return float(np.linalg.norm(approx))
    return float(np.linalg.norm(exact - approx) / denom)


def run_absorption_case(
    l_sites: int,
    phys_dim: int,
    mps_bond: int,
    mpo_bond: int,
    target_bond: int,
    methods: tuple[CompressionMethod, ...] = ("zipup_svd", "randomized"),
    oversample: int = 8,
    n_power: int = 1,
    sketch: SketchKind = "gaussian",
    seed: int | None = None,
) -> list[AbsorptionResult]:
    rng = np.random.default_rng(seed)
    mps = random_mps(l_sites, phys_dim, mps_bond, rng)
    mpo = random_mpo(l_sites, phys_dim, mpo_bond, rng)
    product = apply_mpo_to_mps(mpo, mps)
    exact = mps_to_vector(product)
    peak_product_bond = max_mps_bond(product)

    rows: list[AbsorptionResult] = []
    for method in methods:
        if method == "src":
            # Successive randomized compression: contract H@psi directly without
            # ever forming the inflated `product` MPS. Vendored from RandomMPOMPS.
            from .src_absorption import src_absorb

            res = src_absorb(mpo, mps, target_bond=target_bond, seed=(seed or 0) + 9973)
            runtime = res.runtime_s
            approx = res.vector
            final_bond = res.final_max_bond
        else:
            t0 = perf_counter()
            compressed = compress_mps(
                product,
                max_bond=target_bond,
                method=method,
                oversample=oversample,
                n_power=n_power,
                sketch=sketch,
                rng=rng,
            )
            runtime = perf_counter() - t0
            approx = mps_to_vector(compressed)
            final_bond = max_mps_bond(compressed)
        rows.append(
            AbsorptionResult(
                method=method,
                l_sites=l_sites,
                phys_dim=phys_dim,
                mps_bond=mps_bond,
                mpo_bond=mpo_bond,
                target_bond=target_bond,
                rel_error_exact=relative_vector_error(exact, approx),
                runtime_s=runtime,
                peak_product_bond=peak_product_bond,
                final_max_bond=final_bond,
            )
        )
    return rows
