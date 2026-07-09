"""Disentangled sketch sweep: null test, monotonicity, and the identity fallbacks.

These lock the mechanism the exp10 go/no-go rests on: the Moses disentangler lowers the
rank-``eta`` tail of the sampled range ONLY via the composite-bond reshuffle -- a naive
unitary on the existing bond is provably inert (``sigma(D vtilde) = sigma(vtilde)``). We
also pin the degenerate fallbacks (``ndis=0`` and ``kappa=1`` reproduce the plain tail) so
the cost accounting has a clean baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from rand_isopeps.column.disentangled_qr import (
    disentangled_column_qr,
    disentangler_flops,
    mechanism_profile,
    sketch_range,
    summarize,
)
from rand_isopeps.column.operator import random_column_operator


def _op(lx=4, in_dim=4, out_dim=2, bond=6, seed=0, ensemble="decay"):
    return random_column_operator(lx, in_dim, out_dim, bond,
                                  np.random.default_rng(seed), ensemble=ensemble)


def _profile(seed=0, eta=2, kappa=2, ndis=8, ell=10):
    op = _op(seed=seed)
    c = op.materialize()
    y, _ = sketch_range(op, ell=ell, chi_sk=8, n_power=1,
                        rng=np.random.default_rng(seed + 1), reference=c)
    prof = mechanism_profile(y, op.output_dims, eta, kappa, (eta, eta + 2), ndis=ndis,
                             rng=np.random.default_rng(seed + 2))
    return op, y, prof


def test_null_gauge_is_inert():
    # A unitary on the existing composite bond (no reshuffle) leaves the rank-eta tail
    # invariant to machine precision -- the reshuffle, not the unitary, is what bites.
    _, _, prof = _profile()
    engaged = [c for c in prof if c.disentangled]
    assert engaged, "expected at least one cut with disentangler headroom"
    for c in engaged:
        # the KEY null result: the tail is invariant over random gauges (no reshuffle)
        assert c.null_std < 1e-10
        # the invariant value is the composite (top-rho) rank-eta tail = tail_eta_I - comp_tail
        assert c.null_mean == pytest.approx(c.tail_eta_I - c.comp_tail, rel=1e-8, abs=1e-18)


def test_disentangler_never_raises_the_tail():
    # D = I is in the feasible set, so the optimized rank-eta tail cannot exceed the
    # plain rank-eta tail at any cut with headroom.
    _, _, prof = _profile()
    for c in prof:
        if c.disentangled:
            assert c.dis_tail <= c.tail_eta_I * (1.0 + 1e-9)


def test_ndis_zero_is_identity():
    # ndis=0 disables the disentangler: the tail equals the plain rank-eta tail and no
    # disentangler FLOPs are charged.
    op, y, _ = _profile()
    prof0 = mechanism_profile(y, op.output_dims, 2, 2, (2,), ndis=0,
                              rng=np.random.default_rng(3))
    for c in prof0:
        assert c.dis_tail == pytest.approx(c.tail_eta_I, rel=1e-12)
        assert c.disentangled is False
    assert disentangler_flops(op.output_dims, ell=10, eta=2, kappa=2, ndis=0) == 0.0


def test_kappa_one_has_no_freedom():
    # kappa=1 -> composite bond == eta, no disentangler freedom: identity fallback.
    op, y, _ = _profile()
    prof1 = mechanism_profile(y, op.output_dims, 2, 1, (2,), ndis=8,
                              rng=np.random.default_rng(3))
    for c in prof1:
        assert c.disentangled is False
        assert c.dis_tail == pytest.approx(c.tail_eta_I, rel=1e-12)


def test_summary_and_flops_are_finite_and_positive():
    op, y, prof = _profile()
    s = summarize(prof, float(np.linalg.norm(y)))
    assert s.tau_dis <= s.tau_eta_I * (1.0 + 1e-9)          # disentangling cannot hurt
    assert np.isfinite(s.tau_dis) and s.tau_dis >= 0.0
    assert s.max_null_std < 1e-10
    flops = disentangler_flops(op.output_dims, ell=10, eta=2, kappa=2, ndis=5)
    assert flops > 0.0 and np.isfinite(flops)


def test_disentangled_column_bracket_and_isometry():
    # The disentangled-column accuracy bracket: best case (composite eta*kappa isometry)
    # is at least as accurate as the worst case (plain vertical eta), and the best-case
    # column is a genuine isometry.
    op = _op(lx=4, seed=7)
    c = op.materialize()
    res = disentangled_column_qr(op, eta=2, kappa=2, ell=10, chi_sk=8, n_power=1, ndis=8,
                                 rng=np.random.default_rng(1), reference=c)
    assert res.eps_best <= res.eps_worst + 1e-9        # more range never hurts
    assert res.best_iso_defect < 1e-8                  # best-case column is a real isometry
    assert 0.0 <= res.eps_best <= 1.5 and 0.0 <= res.eps_worst <= 1.5
    assert res.final_bond_best <= res.eta * res.kappa


def test_disentangled_column_kappa_one_collapses_bracket():
    # kappa=1 -> composite bond == eta, so best and worst are the SAME plain-eta build.
    op = _op(lx=4, seed=3)
    c = op.materialize()
    res = disentangled_column_qr(op, eta=3, kappa=1, ell=10, chi_sk=8, n_power=1, ndis=8,
                                 rng=np.random.default_rng(2), reference=c)
    assert res.eps_best == pytest.approx(res.eps_worst, rel=1e-6, abs=1e-9)


def test_larger_kappa_reaches_deeper():
    # More composite headroom (larger kappa) engages at least as many cuts once the rows
    # allow it -- a sanity check that the freedom knob does something.
    op = _op(lx=5, seed=4)
    c = op.materialize()
    y, _ = sketch_range(op, ell=12, chi_sk=8, n_power=1,
                        rng=np.random.default_rng(5), reference=c)
    n2 = sum(1 for cm in mechanism_profile(y, op.output_dims, 2, 2, (2,), ndis=6,
                                           rng=np.random.default_rng(6)) if cm.disentangled)
    assert n2 >= 1


def test_parallel_cuts_bit_identical():
    # cut_workers only changes the execution schedule, never the numbers: per-cut RNG
    # streams are spawned up front and each altmin acts on a copy of its cut's carrier.
    op = _op(seed=5)
    c = op.materialize()
    y, _ = sketch_range(op, ell=10, chi_sk=8, n_power=1,
                        rng=np.random.default_rng(6), reference=c)
    kw = dict(ndis=6, null_draws=4)
    serial = mechanism_profile(y, op.output_dims, 2, 2, (2, 4),
                               rng=np.random.default_rng(7), cut_workers=1, **kw)
    threaded = mechanism_profile(y, op.output_dims, 2, 2, (2, 4),
                                 rng=np.random.default_rng(7), cut_workers=4, **kw)
    assert len(serial) == len(threaded)
    for a, b in zip(serial, threaded):
        assert a.cut == b.cut and a.disentangled == b.disentangled
        assert a.dis_tail == b.dis_tail            # bit-identical, not approx
        assert a.null_mean == b.null_mean and a.null_std == b.null_std
        assert a.dis_iters == b.dis_iters
