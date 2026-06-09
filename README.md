# randomized-isopeps

Research code for **randomized linear algebra inside the block Moses Move** of
isoPEPS / block-isoPEPS. The goal is to replace selected deterministic SVD and
compression steps with structured randomized low-rank approximation without
destroying approximate isometry or energy-relevant accuracy.

## layout

```text
randomized-isopeps/
  rand_isopeps/      shared library: rSVD backends, tensor-ring decomposition,
                     disentangler + ALS, MPO-MPS absorption, synthetic ensembles,
                     plotting/IO/parallel
  synthetic/         synthetic quick-experiment suite
    experiments/     exp1..exp5 driver scripts
    figures/         curated result plots (committed)
    outputs/         scratch CSV data + timestamped figures (gitignored)
    summary.md       goals + results writeup of the experiments
  docs/              reference papers (local only, gitignored)
```

`rand_isopeps/` is the reusable core and is imported by every experiment suite.
Additional suites (beyond `synthetic/`) live as sibling directories at the repo
root and reuse the same package.

The randomized targets follow the grouped block-Moses dimensions:

```text
n1 = chi * eta * d
n2 = eta * p
n3 = chi * eta
k1 = chi * eta      (first local SVD rank)
k2 = eta            (second local SVD rank)
```

## synthetic experiments

Run from the repo root; outputs are written next to the suite, under
`synthetic/outputs/{data,figures}/`, regardless of the working directory.

```bash
# quick smoke runs
python3 synthetic/experiments/exp1_local_first_vs_second_svd.py --quick
python3 synthetic/experiments/exp2_column_error_accumulation.py --quick
python3 synthetic/experiments/exp3_r_column_absorption.py --quick
python3 synthetic/experiments/exp4_tiny_full_isopeps_validation.py --quick
python3 synthetic/experiments/exp5_disentangler_ablation.py --quick

# larger first pass
python3 synthetic/experiments/exp1_local_first_vs_second_svd.py --chi 4 --etas 4 6 8 10 --p 2 --trials 3
python3 synthetic/experiments/exp2_column_error_accumulation.py --chi 4 --eta 8 --lx-values 2 4 6 8 10 --trials 3
python3 synthetic/experiments/exp3_r_column_absorption.py --chi 4 --etas 4 6 8 10 --l-sites 8 --trials 3
python3 synthetic/experiments/exp5_disentangler_ablation.py --chi 3 --etas 3 4 5 6 --p 2 --trials 3
```

exp1–exp4 isolate where randomized SVD enters (see `synthetic/summary.md`). exp5
adds the **disentangler** before the second SVD and compares: no disentangler,
alternating-min and Riemannian (pymanopt) disentanglers, a randomized final SVD,
a *sketched* disentangler search, and tensor-ring ALS (random / warm-started).

Each run writes raw CSV and timestamped PDF figures under `synthetic/outputs/`
(gitignored, regenerated every run). The **curated** plots that tell the story
live under `synthetic/figures/` and are committed so collaborators can see
results without pulling data. Plotting uses matplotlib with the non-interactive
`Agg` backend via `rand_isopeps/plotting.py`.

The Riemannian disentangler (`exp5` method `B_riem`) needs `pymanopt`
(`pip install pymanopt`); the alternating-minimization disentangler is the
dependency-free default.

Experiment scripts accept `--workers` and `--blas-threads`. By default,
`--workers 0` chooses a conservative local process count and `--blas-threads 1`
prevents BLAS oversubscription inside each worker. For a single large case where
BLAS threading is preferable, run with `--workers 1 --blas-threads 0`.

Randomized runs accept `--sketch gaussian`, `--sketch rademacher`, or
`--sketch countsketch`. Gaussian is the correctness baseline; CountSketch is the
first sparse sketching option.

## current modeling choices

The exp1–exp4 local tensor-ring decomposition intentionally omits the optional
disentangler, isolating the two SVDs as randomized low-rank approximation
targets. exp5 reintroduces the disentangler as a unitary gauge on the bond
before the second SVD (`rand_isopeps/disentangler.py`) and studies optimizing it
exactly, via a randomized/sketched objective, and against tensor-ring ALS
(`rand_isopeps/als_ring.py`).

The column experiment is a product-column surrogate: it measures accumulated
local Moses Move approximation error without building a full PEPS or a full
sequential carrier network.

The absorption experiment applies a synthetic MPO to an MPS, then compares
deterministic zip-up-style local SVD compression with randomized local SVD
compression. It is not yet a full SRC implementation.
