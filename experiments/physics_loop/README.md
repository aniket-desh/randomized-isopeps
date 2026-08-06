# Physics loop

This is the active phase-two experiment. One runner applies a fixed number of
imaginary-time updates and records the Rayleigh quotient after every update:

```text
state -> exp(-tau H) state -> normalize / Rayleigh-Ritz -> measure -> repeat
```

For a normalized state, the exact small-system diagnostic is

```text
lambda = x* H x
r      = Hx - lambda x
||r||² = <H²> - lambda².
```

The residual certifies an eigenstate, not specifically the ground state, so the
small runs also record exact low-energy references. For large PEPS the runner
records the contracted energy and leaves the residual as unavailable rather
than materializing the full state.

Small PEPS runs also maintain two dense references: exact `exp(-tau H)` and the
exact ordered bond-gate product used by `tebd_iteration`. Their two infidelities
separate Suzuki--Trotter error from tensor/sketch truncation error.

## Modes

- `dense_exact` stores the `2^N` state vector, keeps `H` sparse, and applies
  `scipy.sparse.linalg.expm_multiply`. It is the accuracy oracle.
- `dense_trotter` splits `H` into horizontal/vertical even/odd layers and uses a
  palindromic second-order Suzuki--Trotter step.
- `peps_full` applies the same bond-level palindromic product with no bond cutoff.
  Its bond dimension grows, so it is only a small-PEPS baseline.
- `peps_local` uses the deterministic sequential Moses move with explicit gate,
  column, and absorption caps.
- `peps_sketch` replaces that column move with the whole-column rMPS sketch,
  interior-column insertion, and explicit zip-up absorption.

The old sketch distributions, diagnostic grids, cost models, and comparison
plots remain in the phase-one experiment directories. The physics runner does
not import any of their storage or parallel-experiment infrastructure.

## Local smoke runs

```bash
python experiments/physics_loop/run.py --mode dense_exact --lx 2 --ly 3 \
  --stage 0.1:10 --stage 0.03:20 --states 3

python experiments/physics_loop/run.py --mode peps_local --lx 3 --ly 3 \
  --stage 0.05:5 --stage 0.02:10 --chi 8 --eta 4 \
  --gate-bond 8 --absorption-bond 8

python experiments/physics_loop/run.py --mode peps_sketch --lx 3 --ly 3 \
  --stage 0.05:5 --stage 0.02:10 --ell 8 --eta 4 --kappa 2 --chi-sk 4 \
  --gate-bond 8 --absorption-bond 8
```

Every line is one JSON record. Pass `--output outputs/physics_loop/run.jsonl`
to save the same records directly. NERSC arrays parallelize independent modes,
Hamiltonians, and seeds; a single trajectory remains one simple sequential loop.
