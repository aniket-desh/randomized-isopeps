"""Phase-0 (0A/0B/0C) smoke tests for the real Moses Move per-stage control +
instrumentation. Skipped when quimb is not installed."""

import numpy as np
import pytest

pytest.importorskip("quimb")

import quimb.tensor as qtn  # noqa: E402

from rand_isopeps.isotns import RandSVD, moses_move  # noqa: E402  (back-compat shim path)
from rand_isopeps.real_isotns.instrument import MosesRandConfig, MosesStats  # noqa: E402
from rand_isopeps.real_isotns.moses_move import _as_config  # noqa: E402


def _peps(seed):
    psi = qtn.PEPS.rand(3, 3, bond_dim=3, phys_dim=2, seed=seed)
    psi.normalize()
    return psi


def _mm(psi, rand=None, stats=None):
    return moses_move(psi, j=2, chi=8, eta=2, cutoff=1e-12, Ndis=10,
                      sweep="up", split="left", renorm=False, rand=rand, stats=stats)


def test_as_config_shorthand():
    r = RandSVD(sketch="gaussian")
    cfg = _as_config(r)
    assert cfg.svd1 is r and cfg.disentangler_inner is r and cfg.svd2 is r
    assert _as_config(None) == MosesRandConfig()
    same = MosesRandConfig(svd2=r)
    assert _as_config(same) is same


def test_det_path_unchanged_by_config_plumbing():
    # rand=None and an all-None MosesRandConfig must both be the deterministic path.
    e1 = _mm(_peps(1), rand=None)
    e2 = _mm(_peps(1), rand=MosesRandConfig())
    assert np.allclose(e1, e2)


def test_stats_collects_per_stage_records():
    stats = MosesStats()
    _mm(_peps(2), rand=None, stats=stats)
    assert stats.n_moves == 1
    assert stats.records, "no splits recorded"
    stages = {r.stage for r in stats.records}
    assert stages <= {"svd1", "disentangler_inner", "svd2"}
    assert {"svd1", "svd2"} <= stages
    for r in stats.records:
        assert r.m > 0 and r.n > 0
        assert 0.0 < r.rho <= 1.0
        assert r.t_svd >= 0.0
    summ = stats.summary()
    assert summ["svd1"]["n_calls"] >= 1


def test_per_stage_toggle_isolates_stage():
    # all-deterministic: no stage adds oversampling (ell == k everywhere).
    sd = MosesStats()
    _mm(_peps(3), rand=None, stats=sd)
    assert all(r.ell == r.k for r in sd.records)

    # svd2-only randomized: svd1 stays deterministic (ell == k for every svd1 split).
    cfg = MosesRandConfig(svd2=RandSVD(sketch="gaussian", oversample=8,
                                       rng=np.random.default_rng(0)))
    s2 = MosesStats()
    errs = _mm(_peps(3), rand=cfg, stats=s2)
    assert errs is not None
    assert all(r.ell == r.k for r in s2.records if r.stage == "svd1")
    assert any(r.stage == "svd2" for r in s2.records)
