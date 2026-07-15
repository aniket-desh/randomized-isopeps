from __future__ import annotations

import json

import numpy as np
import scipy.linalg as la

from rand_isopeps.column.diagnostics import column_diagnostics, operator_cut_spectra
from rand_isopeps.column.operator import random_column_operator


def test_operator_cut_spectra_match_dense_vectorized_column():
    op = random_column_operator(
        3, (2, 3, 2), (3, 2, 2), 2, np.random.default_rng(12),
        complex_valued=True,
    )
    matrix = op.materialize().reshape(*op.output_dims, *op.input_dims)
    axes = [axis for site in range(op.lx) for axis in (site, op.lx + site)]
    local = matrix.transpose(axes)
    combined = tuple(o * i for o, i in zip(op.output_dims, op.input_dims))

    exact = []
    for cut in range(1, op.lx):
        exact.append(la.svdvals(local.reshape(np.prod(combined[:cut]), -1)))

    for found, wanted in zip(operator_cut_spectra(op), exact):
        assert np.allclose(found, wanted[:found.size], atol=1e-11)
        assert np.linalg.norm(wanted[found.size:]) < 1e-11


def test_column_diagnostics_dense_guard_and_serialized_cut_fields():
    op = random_column_operator(
        4, 2, 3, 2, np.random.default_rng(13), complex_valued=False
    )
    dense = column_diagnostics(op, eta=2, kappa=2, dense_max_elements=10_000_000)
    guarded = column_diagnostics(op, eta=2, kappa=2, dense_max_elements=1)
    assert dense["column_flat_r99"] >= 1
    assert np.isnan(guarded["column_flat_r99"])
    assert len(json.loads(guarded["operator_cut_r99"])) == op.lx - 1
    assert len(json.loads(guarded["operator_cut_tail_eta_kappa"])) == op.lx - 1


def test_cached_spectra_match_uncached_diagnostics():
    op = random_column_operator(
        4, 2, 3, 2, np.random.default_rng(14), complex_valued=False
    )
    flat = la.svdvals(op.materialize(), check_finite=False)
    spectra = operator_cut_spectra(op)
    uncached = column_diagnostics(
        op, eta=2, kappa=2, dense_max_elements=10_000_000
    )
    cached = column_diagnostics(
        op, eta=2, kappa=2, dense_max_elements=10_000_000,
        operator_spectra_cache=spectra, flat_singular_values=flat,
    )
    assert cached == uncached
