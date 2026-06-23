"""Cost model sanity: implementation-free FLOP / passes / peak-memory counts."""

from rand_isopeps.experiment_utils.cost_model import compare, gemm_flops, svd_flops


def test_flop_speedup_tracks_inverse_rho():
    # large min(m,n), small sketch width ell -> large FLOP-speedup ~ 1/rho; shrinks as ell grows
    m, n, k = 256, 4096, 8
    hi = compare(m, n, k, ell=8, n_power=0).flop_speedup    # rho ~ 8/256
    lo = compare(m, n, k, ell=64, n_power=0).flop_speedup   # rho ~ 64/256
    assert hi > lo > 1.0


def test_sparse_sketch_is_fewer_flops_than_gaussian():
    # the whole point of the fair comparison: sparse sketches do FEWER flops than a dense
    # Gaussian GEMM (the wall-clock gap is implementation, not algorithm).
    g = compare(128, 1024, 8, 8, n_power=0, sketch="gaussian").rand_flops
    s = compare(128, 1024, 8, 8, n_power=0, sketch="sparsestack").rand_flops
    assert s < g


def test_passes_and_peak_memory():
    c = compare(256, 4096, 8, 8, n_power=1)
    assert c.det_passes == 1 and c.rand_passes == 4   # rsvd reads A 2 + 2q times
    assert c.mem_ratio > 1.0                          # randomized has the smaller working set


def test_flop_primitives():
    assert gemm_flops(10, 20, 30) == 2 * 10 * 20 * 30
    assert svd_flops(100, 50) > 0
