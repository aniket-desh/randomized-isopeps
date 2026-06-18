# randomized-isopeps

Research code for **randomized linear algebra inside the block Moses Move** of
isoPEPS / block-isoPEPS: replacing selected deterministic SVD and compression
steps with randomized low-rank approximation without destroying approximate
isometry or accuracy.

`src/rand_isopeps/` is the reusable package; `experiments/` holds the separated
experiment suites. See [reports/synthetic_kernel_summary.md](reports/synthetic_kernel_summary.md)
for goals and results.

## structure

```
src/rand_isopeps/                   reusable package (pip install -e .)
  linalg/         randomized SVD (randomized_svd.py) + structured sketches (sketches.py)
  moses/          block Moses Move local kernels: local_ring_decomp, disentangler, als_ring, local_methods
  synthetic/      synthetic tensor ensembles (tensors.py) + column surrogates (column_moses, column_carrier)
  real_isotns/    real quimb isoTNS Moses Move (moses_move.py) + TEBD² (tebd2.py)   [needs quimb]
  compression/    R-column / MPO-MPS absorption (mpo_mps_absorb) + SRC (src_absorption, randommpomps/)
  experiment_utils/  aggregate, io, plotting, parallel
  tn_shapes.py    MosesDims (shared)
experiments/synthetic_kernels/scripts/   exp01..exp14 synthetic kernel experiments
experiments/real_moses_move/scripts/     exp01_real_moses_move.py (real quimb Moses Move)
experiments/tebd2/scripts/               (placeholder)
reports/                            curated summaries + figures (see reports/synthetic_kernel_summary.md)
outputs/                            gitignored raw CSV + per-run figures
tests/                              smoke tests
```

## install + run

```bash
pip install -e .            # core; add ".[quimb]" for the real Moses Move / TEBD²
python experiments/synthetic_kernels/scripts/exp01_local_first_vs_second_svd.py --quick
```

```bash
# from the repo root; add --quick for a fast smoke run
python experiments/synthetic_kernels/scripts/exp01_local_first_vs_second_svd.py --ensemble noisy_ring   # also --ensemble powerlaw
python experiments/synthetic_kernels/scripts/exp02_column_error_accumulation.py
python experiments/synthetic_kernels/scripts/exp03_r_column_absorption.py
python experiments/synthetic_kernels/scripts/exp04_tiny_full_isopeps_validation.py
python experiments/synthetic_kernels/scripts/exp05_disentangler_ablation.py
python experiments/synthetic_kernels/scripts/exp06_svd_spectra.py
```

The first SVD (`rho1 ~ 1/d`) vs second SVD (`rho2 ~ 1/eta`) randomization
question is stress-tested across physical dimension `d` (faceted grids) and on
the dimensionless rank fraction `rho = (k+oversample)/min(m,n)`:

```bash
python experiments/synthetic_kernels/scripts/exp07_rank_fraction_phase_diagram.py     # rho vs speedup, colored by excess error
python experiments/synthetic_kernels/scripts/exp08_min_time_to_accuracy.py            # best valid speedup after tuning s,q
python experiments/synthetic_kernels/scripts/exp09_svd1_microbenchmark.py             # SVD1 stage-timing breakdown
python experiments/synthetic_kernels/scripts/exp10_rand_first_downstream_disentangler.py   # does rand SVD1 pollute the gauge?
python experiments/synthetic_kernels/scripts/exp11_true_column_carrier.py --ensemble flat  # true recursive upward carrier; also --ensemble decay
```

exp01/05/06 facet over `--ds` (physical dimension); pass e.g. `--ds 2 3 4 6 8`.

Structured sketching *inside* the disentangler (the sketch `Omega` is approximate; the
gauge `Q` stays exactly unitary). New sketch kinds `sparsestack` / `khatri_rao` via
`src/rand_isopeps/linalg/sketches.py`:

```bash
python experiments/synthetic_kernels/scripts/exp12_gauge_preserving_sketched_disentangler.py  # sketched search/SVD2 preserves the gauge (~1e-14)
python experiments/synthetic_kernels/scripts/exp13_sketch_overfitting_stress.py              # frozen sketch overfits; fresh resampling escapes
python experiments/synthetic_kernels/scripts/exp14_structured_final_svd2.py                  # khatri-rao/sparsestack final SVD2 after a good gauge
```

The **real** isoTNS Moses Move (not a synthetic kernel) lives in
`src/rand_isopeps/real_isotns/moses_move.py` (quimb-based, adapted from Dektor's
`isoTNS_sampling`): the vertical carrier *and* the sideways R-column zip-up
absorption. `RandSVD(sketch=...)` swaps the **local tensor-ring SVDs** (first
SVD, disentangler search, second SVD) for our randomized range finder; the
R-column zip-up absorption stays quimb's deterministic compress for now (a known
insertion point, studied in exp03).

```bash
python experiments/real_moses_move/scripts/exp01_real_moses_move.py   # det vs randomized SVD on the real Moses move
```

`src/rand_isopeps/real_isotns/tebd2.py` is a minimal isoTNS imaginary-time
**TEBD²** ground-state solver built on that Moses Move (1D TEBD along the
orthogonality column + Moses Move to shift the center). Randomization plugs in
via the same `RandSVD`:

```python
from rand_isopeps.tebd import tfi_ham, imaginary_time
from rand_isopeps.isotns import RandSVD
import quimb.tensor as qtn
ham = tfi_ham(3, 3, j=1.0, g=2.5)
psi = qtn.PEPS.rand(3, 3, bond_dim=2, seed=1)
psi, energies = imaginary_time(psi, ham, taus=[(0.1, 40), (0.03, 40), (0.01, 40)],
                               steps=None, chi=8, eta=8, rand=RandSVD(sketch="sparsestack"))
# 3x3 TFI(g=2.5): converges to the ED ground energy -23.836 (~0.05% err), det and randomized alike
```

Curated plots are committed under `reports/figures/<suite>/`; raw CSV and per-run
figures land in `outputs/` (gitignored).

## notes

- The old flat imports still work via shims: `from rand_isopeps.isotns import RandSVD, moses_move`,
  `from rand_isopeps.tebd import tfi_ham, imaginary_time`, and
  `from rand_isopeps.randomized_svd import rsvd_truncate` all still resolve
  (they re-export from the new submodules).
- `--sketch {gaussian,rademacher,countsketch}`, plus `--workers` / `--blas-threads`
  for parallelism.
- The Riemannian disentangler (`exp05` `B_riem`) needs `pip install pymanopt`.
- The real Moses Move (`src/rand_isopeps/real_isotns/moses_move.py`, real_moses_move
  `exp01`) needs `pip install quimb` (or `pip install -e ".[quimb]"`); it is an
  **optional** dependency — the core randomized-NLA modules and the synthetic
  kernels exp01–14 import without it.
- SRC (successive randomized compression) is vendored under
  `src/rand_isopeps/compression/randommpomps/`; its optional C++ incremental QR
  builds with `bash src/rand_isopeps/compression/randommpomps/build_incrementalqr.sh`.
```
