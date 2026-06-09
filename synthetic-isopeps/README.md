# synthetic randomized isoPEPS kernels

This repo is a small synthetic testbed for randomized insertion points in the block Moses Move:

- local tensor-ring first SVD
- local tensor-ring second SVD
- R-column absorption as MPO-MPS product compression

The implementation follows the grouped dimensions from the block-isoPEPS excited-state paper:

```text
n1 = chi * eta * d
n2 = eta * p
n3 = chi * eta
k1 = chi * eta
k2 = eta
```

## quick smoke runs

```bash
python3 experiments/exp1_local_first_vs_second_svd.py --quick
python3 experiments/exp2_column_error_accumulation.py --quick
python3 experiments/exp3_r_column_absorption.py --quick
python3 experiments/exp4_tiny_full_isopeps_validation.py --quick
```

Figures are written as PDF files to `outputs/figures/` and CSV data to `outputs/data/`. Plotting uses matplotlib (non-interactive `Agg` backend) via `rand_isopeps/plotting.py`.

## larger first pass

```bash
python3 experiments/exp1_local_first_vs_second_svd.py --chi 4 --etas 4 6 8 10 --p 2 --trials 3
python3 experiments/exp2_column_error_accumulation.py --chi 4 --eta 8 --lx-values 2 4 6 8 10 --trials 3
python3 experiments/exp3_r_column_absorption.py --chi 4 --etas 4 6 8 10 --l-sites 8 --trials 3
```

Experiment scripts accept `--workers` and `--blas-threads`. By default, `--workers 0` chooses a conservative local process count and `--blas-threads 1` prevents BLAS oversubscription inside each worker. For a single large case where BLAS threading is preferable, run with `--workers 1 --blas-threads 0`.

Randomized runs accept `--sketch gaussian`, `--sketch rademacher`, or `--sketch countsketch`. Gaussian is the correctness baseline; CountSketch is the first sparse sketching option.

## current modeling choices

The local tensor-ring decomposition intentionally omits the optional nonlinear disentangler. This isolates the two SVDs as randomized low-rank approximation targets.

The column experiment is a product-column surrogate: it measures accumulated local Moses Move approximation error without building a full PEPS or a full sequential carrier network.

The absorption experiment applies a synthetic MPO to an MPS, then compares deterministic zip-up-style local SVD compression with randomized local SVD compression. It is not yet a full SRC implementation.
