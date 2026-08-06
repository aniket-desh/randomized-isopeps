# randomized isoPEPS

Physics-first research code for imaginary-time evolution and low-energy
eigenstates with isoPEPS. The active question is whether a whole-column random
MPS sketch can replace the sequential Moses move while retaining the same
physical accuracy at lower cost.

The repository now has two explicit phases:

```text
src/rand_isopeps/
  physics/          sparse exact oracles, H layers, Rayleigh residual/Ritz, one loop
  sketching/        small public facade for the rMPS column sketch
  real_isotns/      quimb gates, local Moses move, sketched global move, PEPS loop

experiments/
  physics_loop/     active fixed-iteration eigenvalue experiments
  sketching/        index to the preserved phase-one sketching laboratory

nersc/jobs/
  physics_small_array.slurm
  physics_large_peps.slurm
```

The older `column/`, `linalg/`, `moses/`, `synthetic/`, `compression/`, and
`experiment_utils/` modules remain available for the earlier experiment suites.
They hold useful comparison evidence and compatibility APIs, but the active
physics runner does not depend on their experiment storage or plotting layers.

## Install

```bash
python -m pip install -e ".[quimb,test]"
```

## Run the physics loop

Dense exact evolution stores a `2^N` state vector but keeps the Hamiltonian
sparse:

```bash
python experiments/physics_loop/run.py --mode dense_exact --lx 2 --ly 3 \
  --stage 0.1:10 --stage 0.03:20 --states 3
```

Small PEPS comparisons use the same initial product state and symmetric gate
order:

```bash
python experiments/physics_loop/run.py --mode peps_full --lx 2 --ly 2 --iterations 2
python experiments/physics_loop/run.py --mode peps_local --lx 3 --ly 3 --eta 4 --chi 8
python experiments/physics_loop/run.py --mode peps_sketch --lx 3 --ly 3 \
  --ell 8 --eta 4 --kappa 2 --chi-sk 4
```

Each line is one JSON record. Small systems report the exact Rayleigh quotient,
`||Hx-lambda x||`, variance, and exact low-energy reference. Larger PEPS report
the contracted energy and explicitly mark the full-state residual unavailable.
See [`experiments/physics_loop/README.md`](experiments/physics_loop/README.md) for
mode semantics and [`nersc/README.md`](nersc/README.md) for Perlmutter setup.

## Preserved sketching work

The method-development studies are intentionally retained:

- `experiments/synthetic_kernels/`: local randomized-SVD and structured-sketch
  controls.
- `experiments/column_sketch/`: whole-column rMPS accuracy, mechanism, state
  insertion, and cost experiments.
- `experiments/real_moses_move/`: randomized versus deterministic local Moses
  moves on real quimb tensor networks.

Their old imports still resolve through compatibility modules. New physics code
should use `rand_isopeps.physics`, `rand_isopeps.sketching`, and
`rand_isopeps.real_isotns.physics_loop`.
