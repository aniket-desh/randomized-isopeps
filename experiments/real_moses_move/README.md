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

- `exp01_real_moses_move.py` — deterministic vs randomized SVD (gaussian / sparsestack / countsketch) inside the real quimb isoTNS Moses Move: sweep the vertical truncation bond `eta` and compare the represented-state error `1 - |<psi0|psi>|` after a Moses move.
