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
python3 synthetic/experiments/exp1_local_first_vs_second_svd.py --chi 4 --etas 4 6 8 10 --trials 12
python3 synthetic/experiments/exp2_column_error_accumulation.py
python3 synthetic/experiments/exp3_r_column_absorption.py
python3 synthetic/experiments/exp4_tiny_full_isopeps_validation.py
python3 synthetic/experiments/exp5_disentangler_ablation.py
python3 synthetic/experiments/exp6_svd_spectra.py
```

Curated plots are committed under `synthetic/figures/`; raw CSV and per-run
figures land in `synthetic/outputs/` (gitignored).

## notes

- `--sketch {gaussian,rademacher,countsketch}`, plus `--workers` / `--blas-threads`
  for parallelism.
- The Riemannian disentangler (`exp5` `B_riem`) needs `pip install pymanopt`.
- SRC (successive randomized compression) is vendored under
  `rand_isopeps/randommpomps/`; its optional C++ incremental QR builds with
  `bash rand_isopeps/randommpomps/build_incrementalqr.sh`.
