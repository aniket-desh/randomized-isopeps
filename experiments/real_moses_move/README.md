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
- `exp02_svd2_time_to_accuracy.py` — **E2-kernel, matched-accuracy time-to-accuracy.** The *speed* question E1 couldn't answer (E1 was overhead-bound). Captures the actual second-SVD matrix from a real deterministic Moses move (`MosesStats(keep_arrays=("svd2",))`) and measures the matched-accuracy speedup `T_det / min_{s,q : ε_rand ≤ (1+tol)·ε_det} T_rand(s,q)` — dial the randomized method to just match deterministic accuracy, then compare wall-clock. The ρ2 lever is the **source bond** (carrier `n2 ≈ bond/4`, so ρ2 ≈ 4/bond); sweeping it into ρ2≪1 makes the dense SVD dominate. No PEPS contraction (kernel benchmark) so no memory wall. Finding: as ρ2 → 0.06, gaussian rsvd reaches a **~31× matched-accuracy speedup**; this is the kernel (svd2-stage) speedup, not end-to-end (the full move is Amdahl-bounded by deterministic absorption + bookkeeping + full-rank svd1).
