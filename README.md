# randomized-isopeps

Research code for **randomized linear algebra inside the block Moses Move** of
isoPEPS / block-isoPEPS: replacing selected deterministic SVD and compression
steps with randomized low-rank approximation without destroying approximate
isometry or accuracy.

`rand_isopeps/` is the shared library; `synthetic/` is the experiment suite.
See [synthetic/summary.md](synthetic/summary.md) for goals and results.

## run

```bash
# from the repo root; add --quick for a fast smoke run
python3 synthetic/experiments/exp1_local_first_vs_second_svd.py --ensemble noisy_ring   # also --ensemble powerlaw
python3 synthetic/experiments/exp2_column_error_accumulation.py
python3 synthetic/experiments/exp3_r_column_absorption.py
python3 synthetic/experiments/exp4_tiny_full_isopeps_validation.py
python3 synthetic/experiments/exp5_disentangler_ablation.py
python3 synthetic/experiments/exp6_svd_spectra.py
```

The first SVD (`rho1 ~ 1/d`) vs second SVD (`rho2 ~ 1/eta`) randomization
question is stress-tested across physical dimension `d` (faceted grids) and on
the dimensionless rank fraction `rho = (k+oversample)/min(m,n)`:

```bash
python3 synthetic/experiments/exp7_rank_fraction_phase_diagram.py     # rho vs speedup, colored by excess error
python3 synthetic/experiments/exp8_min_time_to_accuracy.py            # best valid speedup after tuning s,q
python3 synthetic/experiments/exp9_svd1_microbenchmark.py             # SVD1 stage-timing breakdown
python3 synthetic/experiments/exp10_rand_first_downstream_disentangler.py   # does rand SVD1 pollute the gauge?
python3 synthetic/experiments/exp11_true_column_carrier.py --ensemble flat  # true recursive upward carrier; also --ensemble decay
```

exp1/5/6 facet over `--ds` (physical dimension); pass e.g. `--ds 2 3 4 6 8`.

Structured sketching *inside* the disentangler (the sketch `Omega` is approximate; the
gauge `Q` stays exactly unitary). New sketch kinds `sparsestack` / `khatri_rao` via
`rand_isopeps/sketches.py`:

```bash
python3 synthetic/experiments/exp12_gauge_preserving_sketched_disentangler.py  # sketched search/SVD2 preserves the gauge (~1e-14)
python3 synthetic/experiments/exp13_sketch_overfitting_stress.py              # frozen sketch overfits; fresh resampling escapes
python3 synthetic/experiments/exp14_structured_final_svd2.py                  # khatri-rao/sparsestack final SVD2 after a good gauge
```

The **real** isoTNS Moses Move (not a synthetic kernel) lives in
`rand_isopeps/isotns.py` (quimb-based, adapted from Dektor's `isoTNS_sampling`):
the vertical carrier *and* the sideways R-column zip-up absorption. `RandSVD(sketch=...)`
swaps the **local tensor-ring SVDs** (first SVD, disentangler search, second SVD)
for our randomized range finder; the R-column zip-up absorption stays quimb's
deterministic compress for now (a known insertion point, studied in exp3).

```bash
python3 synthetic/experiments/exp15_real_moses_move.py   # det vs randomized SVD on the real Moses move
```

`rand_isopeps/tebd.py` is a minimal isoTNS imaginary-time **TEBD²** ground-state
solver built on that Moses Move (1D TEBD along the orthogonality column + Moses
Move to shift the center). Randomization plugs in via the same `RandSVD`:

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

Curated plots are committed under `synthetic/figures/`; raw CSV and per-run
figures land in `synthetic/outputs/` (gitignored).

## notes

- `--sketch {gaussian,rademacher,countsketch}`, plus `--workers` / `--blas-threads`
  for parallelism.
- The Riemannian disentangler (`exp5` `B_riem`) needs `pip install pymanopt`.
- The real Moses Move (`rand_isopeps/isotns.py`, `exp15`) needs `pip install quimb`;
  it is an **optional** dependency — the core randomized-NLA modules and exp1–14
  import without it.
- SRC (successive randomized compression) is vendored under
  `rand_isopeps/randommpomps/`; its optional C++ incremental QR builds with
  `bash rand_isopeps/randommpomps/build_incrementalqr.sh`.
