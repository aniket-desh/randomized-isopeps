# synthetic_kernels

Synthetic, kernel-level experiments on the block Moses Move randomized SVDs.
Each script isolates a randomized-linear-algebra insertion point inside one
local two-SVD Moses-Move step (and its column/absorption surrogates) using small
numpy kernels, and measures whether randomization is accuracy-neutral and where
it pays off in cost. The randomized-NLA core and exp01–14 need only numpy/scipy.

## Running

Install the package once, then run any script (the `--quick` flag runs a small,
fast sweep):

```
pip install -e .
python experiments/synthetic_kernels/scripts/<name>.py --quick
```

Outputs (CSVs and figures) land in `outputs/synthetic_kernels/` (gitignored).

## Scripts

- `exp01_local_first_vs_second_svd.py` — local two-SVD experiment: randomize the first SVD, the second, both, or neither, and compare reconstruction error, isometry, and timing (with an excess-error panel and stress ensembles).
- `exp02_column_error_accumulation.py` — measure how local approximation errors accumulate down a product-column surrogate of height `Lx` for each randomization mode.
- `exp03_r_column_absorption.py` — R-column / MPO–MPS compression: compare zip-up SVD, randomized SVD, and SRC (successive randomized compression) for absorbing the residual column.
- `exp04_tiny_full_isopeps_validation.py` — tiny explicit isometry validation: confirm an isometric center preserves norm and has a vanishing isometry defect across the center-dimension sweep.
- `exp05_disentangler_ablation.py` — six-way ablation of the unitary disentangler before the second SVD (no-D baseline, alt-min, Riemannian, randomized SVD2, sketched search, ALS), with the second-SVD spectrum as the headline diagnostic.
- `exp06_svd_spectra.py` — SVD spectrum diagnostic: plot the singular values entering each of the two local SVDs (with the `k1`, `k2` truncation ranks marked) to explain the exp1 result across ensembles.
- `exp07_rank_fraction_phase_diagram.py` — collapse a large `(chi, eta, d, oversample, ensemble)` sweep onto the rank fraction `rho = (k+s)/min(m,n)`, plotting stage speedup vs `rho` colored by excess error.
- `exp08_min_time_to_accuracy.py` — fairest test: for each tensor and mode, sweep oversample/power and report the fastest configuration whose error is within tolerance of deterministic (best valid time-to-accuracy speedup).
- `exp09_svd1_microbenchmark.py` — strip to just the first-SVD matrix and time deterministic vs randomized stage by stage, to show the weak `d=2` first-SVD gain is algorithmic (`rho1 ~ 1/d`), not reshape overhead.
- `exp10_rand_first_downstream_disentangler.py` — does a randomized first SVD pollute the downstream disentangler? Compare four pipelines on reconstruction error, second-SVD tail, gauge iterations, and defects.
- `exp11_true_column_carrier.py` — true recursive upward-carrier column Moses Move (replaces the exp2 surrogate): carry each row's residual into the next and compute the final column error exactly via MPS transfer-matrix overlaps.
- `exp12_gauge_preserving_sketched_disentangler.py` — show structured sketches (gaussian/countsketch/sparsestack/khatri-rao) in the disentangler search or final SVD2 keep the gauge exactly unitary while matching the exact alt-min tail.
- `exp13_sketch_overfitting_stress.py` — fixed-sketch overfitting stress test: compare {fresh, fixed} × {validated, unvalidated} against exact alt-min, demonstrating a frozen sketch overfits and a fresh sketch per step is the effective fix.
- `exp14_structured_final_svd2.py` — after an exact disentangler, truncate the gauged second-SVD matrix with exact / gaussian / sparsestack / Khatri–Rao SVD2 and show all structured sketches match the deterministic SVD2.
