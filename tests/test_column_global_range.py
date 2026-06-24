"""Column operator + global range finder: access-model agreement and the rMPS validation checks."""

import numpy as np

from rand_isopeps.column.global_range import global_column_range, reference_svd, sampled_bond_growth
from rand_isopeps.column.local_moses import local_column_qr
from rand_isopeps.column.operator import (
    ColumnOperator,
    controlled_spectrum_column_matrix,
    random_column_operator,
)
from rand_isopeps.linalg.rmps_sketch import rmps_cores, rmps_to_vector


# ---- ColumnOperator: the materialize / matrix-free access seam ----

def test_materialize_matches_matrix_free_matvec():
    # the dense materialization and the MPO--MPS product must agree to roundoff
    # (so synthetic-now and matrix-free-later share one object).
    for ens in ("gaussian", "decay", "identity_plus_noise"):
        c = random_column_operator(4, in_dim=2, out_dim=3, mpo_bond=4,
                                   rng=np.random.default_rng(1), ensemble=ens)
        mat = c.materialize()
        assert mat.shape == (c.n_out, c.n_in)
        probe = rmps_cores(c.input_dims, 3, np.random.default_rng(5))
        free = c.matvec(probe)
        dense = mat @ rmps_to_vector(probe)
        assert np.linalg.norm(free - dense) / np.linalg.norm(dense) < 1e-12


def test_operator_dims_and_validation():
    c = random_column_operator(3, in_dim=(2, 3, 2), out_dim=4, mpo_bond=5,
                               rng=np.random.default_rng(0))
    assert c.input_dims == (2, 3, 2) and c.output_dims == (4, 4, 4)
    assert c.n_in == 12 and c.n_out == 64 and c.mpo_bond == 5
    # a non-trivial boundary bond is rejected
    bad = [np.ones((2, 2, 2, 1)), np.ones((1, 2, 2, 1))]  # left boundary bond 2 != 1
    try:
        ColumnOperator(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---- global range finder: the mathematical-validation checks ----

def test_gaussian_recovers_full_rank_to_machine_precision():
    # with ell >= numerical rank, a dense Gaussian range recovers C exactly and Q is isometric.
    c = controlled_spectrum_column_matrix((3, 3, 3), (2, 2, 2, 2), np.random.default_rng(0),
                                          decay_kind="exp", parameter=2.0)
    n_in = c.shape[1]
    res = global_column_range(c, (2, 2, 2, 2), ell=n_in, sketch_kind="gaussian",
                              target_rank=n_in, rng=np.random.default_rng(1))
    assert res.rel_error < 1e-10
    assert res.isometry_defect < 1e-10
    assert res.q_cols == n_in


def test_isometry_defect_small_for_rmps():
    c = random_column_operator(5, 2, 3, 6, np.random.default_rng(2)).materialize()
    res = global_column_range(c, (2,) * 5, ell=10, chi_sk=4, sketch_kind="rmps",
                              target_rank=5, rng=np.random.default_rng(3))
    assert res.isometry_defect < 1e-10  # Q is orthonormal regardless of the probe distribution


def test_excess_over_eckart_young_is_nonnegative():
    # a rank-r truncation of the randomized SVD cannot beat the Eckart-Young optimum.
    c = controlled_spectrum_column_matrix((4, 4), (2, 2, 2), np.random.default_rng(4),
                                          decay_kind="power", parameter=1.5)
    ref = reference_svd(c)
    for kind, chi in (("gaussian", 0), ("rmps", 4), ("kron", 1)):
        res = global_column_range(c, (2, 2, 2), ell=6, chi_sk=chi, sketch_kind=kind,
                                  target_rank=3, rng=np.random.default_rng(5), ref_svd=ref)
        assert res.excess_error > -1e-9, (kind, res.excess_error)


def test_osi_improves_with_chi_sk_on_tall_column():
    # the paper's thesis: chi_sk=1 (Kronecker) is a poor subspace injection on a tall
    # column; raising chi_sk climbs toward the dense-Gaussian injectivity. Median over draws.
    lx = 8
    c = random_column_operator(lx, 2, 3, 6, np.random.default_rng(20)).materialize()
    fd = (2,) * lx
    ref = reference_svd(c)

    def med_osi(kind, chi):
        return float(np.median([
            global_column_range(c, fd, ell=8, chi_sk=chi, sketch_kind=kind, target_rank=4,
                                rng=np.random.default_rng(700 + t), ref_svd=ref).osi_sigma_min
            for t in range(12)
        ]))

    kron = med_osi("kron", 1)
    rmps_hi = med_osi("rmps", 8)
    gauss = med_osi("gaussian", 0)
    assert kron < rmps_hi <= gauss + 1e-9
    assert rmps_hi > 1.5 * kron  # a clear, not marginal, improvement


def test_reproducible_with_seed():
    c = random_column_operator(4, 2, 3, 5, np.random.default_rng(6)).materialize()
    a = global_column_range(c, (2, 2, 2, 2), ell=6, chi_sk=4, sketch_kind="rmps",
                            rng=np.random.default_rng(99))
    b = global_column_range(c, (2, 2, 2, 2), ell=6, chi_sk=4, sketch_kind="rmps",
                            rng=np.random.default_rng(99))
    assert a.rel_error == b.rel_error and a.osi_sigma_min == b.osi_sigma_min


def test_controlled_spectrum_decays():
    c = controlled_spectrum_column_matrix((3, 3), (2, 2, 2), np.random.default_rng(7),
                                          decay_kind="exp", parameter=2.0)
    s = np.linalg.svd(c, compute_uv=False)
    assert np.all(np.diff(s) <= 1e-12)  # non-increasing
    assert s[0] / s[-1] > 10.0          # a genuine decay


def test_sampled_bond_growth_equals_d_times_chi():
    c = random_column_operator(6, 2, 3, 5, np.random.default_rng(8), ensemble="gaussian")
    for chi in (1, 2, 4):
        bg = sampled_bond_growth(c, ell=4, chi_sk=chi, rng=np.random.default_rng(9))
        assert bg["max_product_bond"] == 5 * chi == bg["bound_d_times_chi"]


# ---- local Moses column QR baseline ----

def test_local_column_qr_reconstructs_at_full_rank():
    # an uncapped local sweep recovers C exactly and yields an output-isometric column.
    op = random_column_operator(5, 2, 3, 6, np.random.default_rng(0), ensemble="gaussian")
    res = local_column_qr(op, max_rank=10**6, randomized=False)
    assert res.rel_error < 1e-10
    assert res.isometry_defect < 1e-10


def test_local_column_qr_isometric_when_truncated():
    op = random_column_operator(5, 2, 3, 8, np.random.default_rng(1), ensemble="gaussian")
    for randomized in (False, True):
        res = local_column_qr(op, max_rank=4, randomized=randomized, rng=np.random.default_rng(2))
        assert res.isometry_defect < 1e-10  # the new column is isometric even when truncated
        assert 0.0 < res.rel_error < 1.0
        assert res.n_svd == op.lx  # one local SVD per row (the sequential cost)


def test_global_sketch_beats_greedy_local_at_matched_rank():
    # the fork: on a column with capturable flat rank, the one-shot global sketch is
    # closer to the Eckart-Young optimum than the greedy sequential local sweep.
    k = 6
    g_excess, l_excess = [], []
    for t in range(12):
        op = random_column_operator(5, 2, 3, 8, np.random.default_rng(40 + t), ensemble="gaussian")
        c = op.materialize()
        ref = reference_svd(c)
        floor = float(np.sqrt(np.sum((ref[1] / np.linalg.norm(ref[1]))[k:] ** 2)))
        g = global_column_range(c, op.input_dims, ell=k, sketch_kind="gaussian", target_rank=k,
                                rng=np.random.default_rng(3 * t), ref_svd=ref)
        l = local_column_qr(op, max_rank=k, randomized=False, reference=c)
        g_excess.append(g.rel_error - floor)
        l_excess.append(l.rel_error - floor)
    assert np.median(g_excess) < np.median(l_excess)  # global hugs the EY floor closer
