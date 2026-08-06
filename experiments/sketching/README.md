# Phase-one sketching laboratory

The earlier method-development experiments are preserved in place because they
remain the evidence for sketch choice, failure modes, and comparisons:

- `experiments/synthetic_kernels/` tests local randomized-SVD and structured
  sketch primitives on controlled tensors.
- `experiments/column_sketch/` tests the whole-column rMPS range finder,
  factorization accuracy, state insertion, and cost.
- `experiments/real_moses_move/` compares deterministic and randomized local
  Moses moves on real quimb states.

New code should enter those implementations through `rand_isopeps.sketching`.
The active eigenvalue experiments live in `experiments/physics_loop/` and do not
depend on the historical experiment utilities.
