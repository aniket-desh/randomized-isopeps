# real_moses_move

The real isoTNS block Moses Move (not the synthetic numpy kernels). This suite
runs the genuine block Moses Move on a quimb PEPS via `rand_isopeps.isotns` and
swaps the local tensor-ring SVDs for a randomized range finder, to check that the
accuracy-neutrality established synthetically holds end-to-end in the real
algorithm.

## Requirements

Needs the optional `quimb` dependency:

```
pip install -e ".[quimb]"
```

## Running

```
python experiments/real_moses_move/scripts/exp01_real_moses_move.py --quick
```

Outputs land in `outputs/real_moses_move/` (gitignored).

## Scripts

- `exp01_real_moses_move.py` — **E1, stage ablation.** One real Moses move on a 3×3 PEPS (exact contraction), isolating *which* local tensor-ring SVD is randomized via the per-stage `MosesRandConfig`: `det`, `RSVD1`, `RSVD2`, `sketch-Q`, `sketch-Q+RSVD2`, `all-rand`. Headline metric is the represented-state error `1 - |<psi0|psi>|`; paired over PEPS-instance × sketch seeds with median+IQR bands, plus a per-stage rank-fraction panel (ρ1 vs ρ2 from `MosesStats`). Finding: randomizing the low-rank `svd2`/disentangler stages is accuracy-neutral, while `svd1` is full-rank (ρ1≈1) at these settings so `RSVD1`/`all-rand` only add a degradation tail. `--sketch` fixes the sketch family; `--instances`, `--sketch-seeds` set the statistics.
