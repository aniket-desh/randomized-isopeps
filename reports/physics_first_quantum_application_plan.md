# Physics-first quantum application plan

**Active research contract — August 5, 2026.** The next paper is a tensor-network
and many-body-physics paper. The rMPS sketch is one compression/column-factorization
backend inside a minimal imaginary-time iteration, not the subject of a new
randomized-linear-algebra theory paper. Accuracy, Trotter correctness, truncation,
bond growth, and eigenstate convergence come first; optimized wall time comes later.

This plan supersedes `reports/rmps_peps_plan.md` and `paper/NEXT_EXPERIMENTS.md` as
the active sequence, but those documents remain historical validation records. Do
not delete their code, tests, or negative results. The active physics path should
simply stop importing most of them.

## Implementation status — August 5, 2026

The first functional slice now exists:

- `rand_isopeps.physics` owns the sparse dense oracle, four checkerboard layers,
  exact and Strang imaginary-time steps, Rayleigh residual/variance, block
  Rayleigh--Ritz, and the one fixed-iteration loop.
- `rand_isopeps.sketching` is the small phase-two entry point to the validated
  rMPS sketch. The phase-one alternatives remain available to their old tests and
  experiments without leaking into the physics runner.
- `real_isotns.column_bridge` now unfuses an interior column's
  `physical x away` output, absorbs the residual, and zip-compresses the next
  column with explicit cap/cutoff controls. Exact `0->1->2->1->0` movement on a
  2x3 PEPS preserves fidelity to roundoff.
- `tebd_iteration` supports untruncated PEPS, deterministic local Moses, and the
  iterative rMPS column backend under one symmetric gate schedule.
- `experiments/physics_loop/run.py` writes streaming JSONL for dense ground and
  block-excited-state references plus single-state PEPS trajectories. Shared and
  full-node Slurm arrays are prepared but not submitted.

The remaining scientific work is numerical: run the 2x2/2x3/3x3 accuracy matrix,
calibrate truncation settings, then run paired 4x4 trajectories. PEPS `p>1`, a
scalable large-PEPS residual, and QR/Rayleigh overlap remain later milestones.

## 1. The one loop we are trying to build

For one state, one iteration should read conceptually as

```text
psi <- apply one explicit Suzuki-Trotter macro-step
psi <- canonicalize/compress each affected column with METHOD
psi <- normalize
(energy, residual, bonds, truncation losses) <- measure(psi)
```

`METHOD` has only two values during the first study:

```text
local       existing deterministic Moses move
sketch      bounded-residual rMPS column QR, followed by controlled absorption
```

Everything else is a validation mode or a later optimization. Gaussian sketches,
Kronecker controls, OSI curves, power-iteration grids, alternate disentanglers, and
large Hamiltonian surveys should not appear in the first physics script.

For a block of `p` states, the eventual excited-state loop is block subspace
iteration followed by Rayleigh-Ritz:

\[
Y_k \approx e^{-\tau H}X_k,
\qquad G_k=Y_k^\dagger Y_k,
\qquad Q_k=Y_kG_k^{-1/2},
\]

\[
T_k=Q_k^\dagger H Q_k=S_k\Theta_kS_k^\dagger,
\qquad X_{k+1}=Q_kS_k,
\qquad R_k=HX_{k+1}-X_{k+1}\Theta_k.
\]

The eigenvalues are the diagonal entries of `Theta`; the norm of each column of
`R` is its eigenpair residual. For `p=1`, the Gram orthogonalization is just
normalization and `T` is the scalar Rayleigh quotient. This reduction is why the
single-state loop is the right correctness harness, even though the eventual paper
story is excited states.

There are two different QRs and they must not be conflated:

- **Block QR** orthogonalizes the `p` trial states before Rayleigh-Ritz.
- **Column QR** factors an active isoPEPS column, `C_j \approx Q_jR_j`, so the
  orthogonality center can move without losing a controlled tensor-network form.

Our rMPS sketch replaces the second operation. Dense block subspace iteration and
Rayleigh--Ritz are implemented as the excited-state oracle, while the PEPS path is
still `p=1`; a block leg and block-aware column move are required before we can
claim PEPS excited-state computation.

## 2. The physics and diagnostics

### Imaginary time and the Hamiltonian split

For a ground-state filter the sign is

\[
\psi_{k+1}=\frac{e^{-\tau H}\psi_k}{\|e^{-\tau H}\psi_k\|}.
\]

In an eigenbasis, each coefficient is multiplied by `exp(-tau E_n)`, so the
lowest-energy component wins. `exp(+tau H)` would instead amplify the largest
eigenvalue unless the Hamiltonian were sign-reversed.

For a nearest-neighbor square lattice, use an explicit four-color decomposition

\[
H=H_{h,e}+H_{h,o}+H_{v,e}+H_{v,o}.
\]

Horizontal bonds are colored by the parity of their left column; vertical bonds by
the parity of their upper row. Bonds within one color are disjoint, so their gates
commute and can be applied independently. A second-order macro-step is the
palindromic Strang product

\[
e^{-\tau H}
\approx
e^{-\frac\tau2 H_1}
e^{-\frac\tau2 H_2}
e^{-\frac\tau2 H_3}
e^{-\tau H_4}
e^{-\frac\tau2 H_3}
e^{-\frac\tau2 H_2}
e^{-\frac\tau2 H_1}.
\]

The existing alternating left/right sweep is not this product. A dense 2x2 audit
showed error ratios approaching four when `tau` was halved, which diagnoses an
`O(tau^2)` local error, i.e. first-order behavior. We must either implement the
explicit palindromic schedule or label the current method first-order. The plan
below implements and tests the palindromic schedule.

### Rayleigh quotient and residual

For a possibly unnormalized vector `x`,

\[
\lambda(x)=\frac{x^\dagger Hx}{x^\dagger x},
\qquad r=Hx-\lambda x.
\]

For normalized `x` and Hermitian `H`,

\[
\|r\|_2^2
=x^\dagger H^2x-\lambda^2
=\|Hx\|_2^2-\lambda^2
=\operatorname{Var}_x(H).
\]

The stable dense computation is to form `y = H @ x`, then compute
`norm(y - lambda*x)` directly. Report both the absolute residual and a fixed-scale
relative residual such as

\[
\rho=\frac{\|r\|_2}{\|H\|_{\rm bound}\|x\|_2},
\]

where the same exact norm or local-term norm bound is used for every method. Do not
divide only by `|lambda|`, which becomes unstable if the spectrum is shifted near
zero.

If `H = sum_a H_a`, the energy is additive,

\[
\lambda=\sum_a x^\dagger H_ax,
\]

but the squared norm is not:

\[
\|Hx\|^2
=\sum_{a,b}(H_ax)^\dagger(H_bx)
=\sum_{a,b}x^\dagger H_a^\dagger H_bx.
\]

All cross terms matter. The even/odd split makes the exponential cheap; it does not
make the variance a sum of independent per-layer variances. On small systems form
`sum_a H_a x` exactly. A future large-PEPS sketch must apply one common linear
sketch to every `H_a x` and to `x`, then form the sketched residual; independent
sketches would destroy the cross terms.

A small residual means the Rayleigh quotient is close to *some* eigenvalue. It does
not by itself prove that the state is the ground state, so small systems also need
ED energy/overlap and block runs need state ordering plus orthogonality.

## 3. What “dense,” “untruncated,” and “truncated” mean

For `N` qubits, a dense state stores all `2^N` amplitudes. It does not require a
dense `2^N x 2^N` Hamiltonian: build `H` sparsely from the local terms and use
`scipy.sparse.linalg.expm_multiply`.

| lattice | sites | complex128 state | complex128 dense H |
|---|---:|---:|---:|
| 2x2 | 4 | 256 B | 4 KiB |
| 2x3 | 6 | 1 KiB | 64 KiB |
| 3x3 | 9 | 8 KiB | 4 MiB |
| 4x4 | 16 | 1 MiB | 64 GiB |
| 5x5 | 25 | 512 MiB | prohibitive |

Thus 2x2, 2x3, and 3x3 are straightforward exact oracles. A 4x4 state vector and
sparse Hamiltonian are still useful for scoring even though a dense Hamiltonian is
not. Beyond that, tensor-network or stochastic diagnostics are required.

Applying a two-site gate and refactorizing a PEPS generally increases a bond rank.
An SVD

\[
A=U\Sigma V^\dagger
\]

is **untruncated** when every nonzero singular value is retained. The represented
state is preserved, but bonds can grow rapidly. It is **truncated** when only the
leading rank or singular values are retained. At a canonical cut, the squared
discarded Frobenius weight is `sum_{i>r} sigma_i^2` and is the local approximation
loss.

The present code has several separate truncation points:

1. a gate split capped at `eta`;
2. the first Moses SVD capped at `chi*eta`;
3. the second Moses SVD capped at `eta`;
4. a Procrustes split with quimb's implicit `1e-10` cutoff;
5. residual-column zip-up with another implicit `1e-10` cutoff.

The same `cutoff` also has different meanings at these sites. The new loop needs an
explicit policy with separate gate, Moses, and absorption caps/cutoffs. An
“untruncated” run must fail if any cap is hit or any discarded tail exceeds the
oracle tolerance; setting one argument to zero is not enough.

### Why reinserting a column grows bonds

The sketched factorization builds `C_j \approx Q_jR_j`. If the active column MPO has
vertical bond `D`, the sketch bond is `chi_sk`, and the retained caps are `eta` and
`kappa`, then:

- one temporary sampled MPS has bond at most `D*chi_sk`;
- stacking `ell` samples can temporarily reach `ell*D*chi_sk`;
- the returned `Q` has vertical bond at most `eta` and sideways residual legs at
  most `kappa`;
- `R = Q^dagger C` has vertical bond at most `eta*D`;
- absorbing `R` into a neighbor of vertical bond `B` creates an exact product bond
  as large as `B*eta*D` before compression.

The sketch itself does not permanently enlarge the PEPS. Exact absorption does.
Zip/SVD compression controls that growth and introduces a distinct physical error,
which must be logged separately from sketch projection and column-rounding errors.

## 4. What the repository already has

| Existing code | Keep for the active loop | Current limitation |
|---|---|---|
| `real_isotns/tebd2.py` | Hamiltonians, legacy local preparation, energy | retained first-order API; the new symmetric step lives in `physics_loop.py` |
| `real_isotns/moses_move.py` | deterministic local column backend | absorption controls are explicit; complete cutoff-rank accounting is still incomplete |
| `real_isotns/instrument.py` | temporary correctness counters | reported tail is at the nominal cap, not always the actual cutoff rank |
| `column/from_quimb.py` | extract a center column as an MPO | caller must track the center explicitly |
| `column/operator.py` | dense and matrix-free column access | validation primitive, not a physics loop |
| `column/bounded_residual.py` | the one active rMPS sketch backend | large diagnostic object is hidden by `rand_isopeps.sketching` |
| `exp11_invariant_one_move.py` | retained phase-one state benchmark | sparse H and dense-state helpers now delegate to `rand_isopeps.physics` |

`bounded_residual_column_qr` is the actual insertable sketch method. The older
`global_range`, `structured_qr`, and `disentangled_qr` paths remain useful evidence,
but they should not be selectable from the first physics runner.

The automated oracle now reproduces the same ordered 2x2 gate product with a
full-rank local Moses sweep to below `1e-10` infidelity, and the full-range rMPS
bridge preserves a complete 2x3 interior round trip to roundoff. The prepared
small-system array extends those checks to the full 2x2/2x3/3x3 matrix; those
experiments have not been submitted yet.

## 5. Minimal code contract

Do not rewrite or delete the research repository. Add a narrow facade and make the
new experiment import only that facade.

### `src/rand_isopeps/physics/`

```python
rayleigh_residual(H, x) -> dict
checkerboard_layers(ham) -> dict
exact_imaginary_step(H, x, tau) -> x_next
trotter_imaginary_step(layers, x, tau) -> x_next
rayleigh_ritz(H, X) -> dict
run_iterations(state, *, iterations, update, measure) -> (state, history)
```

This file owns exact dense math, the four-color partition, and p=1/block
diagnostics. Move the reusable sparse-H and scale-safe dense-state helpers out of
`exp11` rather than copying them.

### `src/rand_isopeps/real_isotns/physics_loop.py`

```python
tebd_iteration(psi, ham, tau, *, column_backend,
               gate_options, column_options, rng=None) -> (psi, metrics)
```

One iteration has one documented imaginary-time increment and one explicit gate
order. The shared `run_iterations` function never stops early during comparisons,
so every method does the same amount of work.

### `src/rand_isopeps/real_isotns/global_move.py`

```python
rmps_column_move(psi, j, *, eta, kappa, ell, chi_sk,
                 absorption_max_bond, absorption_cutoff) -> (psi, metrics)
```

This is the only sketch call visible to the loop. Internally it does

```text
from_quimb_column
-> bounded_residual_column_qr
-> insert Q
-> absorb R
-> explicit zip/compress
-> return an ordinary next-column-extractable PEPS
```

The interior bridge and full boundary-to-boundary sweep are implemented. The
local backend remains a thin call to `moses_move`; both backends receive explicit
gate, column, and absorption controls as plain dictionaries rather than a large
configuration class.

### One experiment script

Use one runner, `experiments/physics_loop/run.py`, with modes

```text
dense_exact
dense_trotter
peps_full
peps_local
peps_sketch
```

It should write one tidy row per iteration and method. Do not fork five experiment
frameworks.

## 6. Two-week sequence

### Week 1 — make one iteration physically trustworthy

1. **Extract the dense oracle.** Build sparse `H`, exact `expm_multiply`, the
   four-color term split, Rayleigh quotient, direct residual, variance, and ED
   reference for 2x2, 2x3, and 3x3. Also implement dense block subspace iteration
   for `p=2` as the mathematical excited-state reference, without PEPS yet.

2. **Fix the Trotter definition.** Implement the explicit Strang schedule and test
   that every bond appears exactly once in the four layers, no layer overlaps, and
   the one-step error scales as `O(tau^3)` under step halving.

3. **Make “untruncated” real.** Remove implicit Procrustes and zip cutoffs, separate
   gate/Moses/absorption settings, and log actual retained ranks and discarded
   weights. The run must abort if an alleged full-rank path discards anything.

4. **Run the no-truncation oracle.** Starting from the same product state, compare
   the dense ordered-gate product and PEPS result after every gate group and after
   the whole macro-step on 2x2, 2x3, and 3x3. Require near-roundoff state fidelity,
   norm, energy, and residual agreement.

**Week-1 deliverable:** one command produces a table/plot of exact evolution,
dense Strang, and certified-untruncated PEPS for one iteration. No sketch result is
needed to call Week 1 successful.

### Week 2 — controlled truncation, then one sketch backend

5. **Deterministic truncation study.** On the same initial states and gate order,
   sweep small explicit bond caps. A minimal grid is `eta in {2,4,8}` with `chi`
   chosen generously enough that the first SVD is not the accidental bottleneck.
   Record state infidelity to dense Strang, energy error, residual error, norm drift,
   per-stage discarded weight, and the maximum bond after each operation.

6. **Finish one reusable global move.** Zip/compress the exact residual into the
   neighbor with an explicit cap and cutoff, then prove that the next column can be
   extracted and moved. Validate one full boundary-to-boundary sweep on 2x3 and
   3x3 before attempting 4x4.

7. **Swap the backend, not the loop.** Run exactly one fixed sketch configuration
   first: provisionally `kappa=2`, `chi_sk=4`, `ell=eta+4`, and no power iteration.
   Compare it with deterministic local Moses at the same retained bond and
   absorption policy. Vary one parameter only if the baseline fails.

8. **Run fixed trajectories.** Use the same provisional schedule for every method,
   for example ten macro-steps at each `tau` in `(0.1, 0.03, 0.01)`, refined after
   the dense pilot. Record the first residual thresholds crossed but do not stop
   early. Start with TFIM at `g=3.5`; use one random-state control. Add a critical
   or Heisenberg case only if the first comparison is clean.

9. **Cross the dense-H wall carefully.** Use 4x4 with a dense state/sparse-H scoring
   oracle when feasible. For 5x5 and larger, begin with energy, norm, observables,
   bond growth, and exact structural invariants; treat a sketched large-state
   residual as a separate method that must first calibrate against 3x3/4x4.

**Week-2 deliverable:** convergence and truncation curves for local versus one rMPS
backend, with the error separated into Trotter, tensor-network truncation, sketch,
and residual-absorption components. Wall time is recorded as engineering context,
not yet the headline.

### Pilot statistics

This is an integration pilot, not a paper-scale sweep. Use four preparation seeds
and three sketch seeds for the chosen sketch configuration, paired on the same
initial state. Plot individual trajectories plus median and interval. Scale to the
larger historical protocol only after the method passes all correctness gates.

## 7. Acceptance gates

- The Hamiltonian layer sum equals the original sparse Hamiltonian to roundoff.
- Each layer contains disjoint supports and the Strang step shows the advertised
  order under `tau` halving.
- `lambda`, the direct residual, and the energy variance agree algebraically on
  dense normalized states.
- Certified-untruncated PEPS matches the same dense gate product to `<=1e-10`
  infidelity and reports zero discarded weight at every stage.
- Lowering a cutoff or increasing a cap makes truncated trajectories approach the
  certified full-rank trajectory.
- Local and sketch methods start from the identical state and use identical gates,
  `tau`, step count, and absorption cap.
- The global move returns a standard PEPS from which the next column can be
  extracted; leaving parallel residual bonds is not an iterative implementation.
- State infidelity, energy, residual, local observables, norm, and isometry all pass
  before speed is discussed.
- Small-system variational energies never fall below the ED ground energy.
- A block excited-state result is not claimed until `X^dagger X`, Rayleigh-Ritz,
  state ordering, and every column of `HX-XTheta` are measured.

## 8. Paper reframe

The main paper should follow the block-isoPEPS/excited-state structure:

1. **Physics problem.** Lowest `p` eigenstates of 2D local Hamiltonians and why
   imaginary-time/subspace iteration needs repeated tensor-network compression.
2. **Tensor-network algorithm.** Isometric PEPS, the orthogonality column, explicit
   Suzuki-Trotter layers, TEBD2, block orthogonalization, and Rayleigh-Ritz.
3. **Sketched column move.** One algorithm box for `C_j Omega`, structured QR,
   bounded residual, and absorption. Explain the sketch in a paragraph, not a new
   convergence-theory section.
4. **Cost model.** Give actual tensor/matrix shapes and leading operation counts in
   `d, chi, eta, p, kappa, ell, chi_sk`; report passes, peak intermediate bonds, and
   memory. Avoid a sprawling Gaussian-theory argument.
5. **Numerics.** Dense/ED validation, Trotter order, full versus truncated bonds,
   ground/excited energy and residual convergence, observables, bond growth, and
   eventually matched-accuracy speed.

The OSI analysis, Gaussian/Kronecker comparisons, synthetic phase diagrams, and
prior negative cost results belong in an appendix or prior-work validation note.
They justify that the sketch primitive is sane; they are not the new paper's main
claim.

The first claim ladder is deliberately narrow:

- **After the dense and full-rank gates:** the physics loop and error decomposition
  are correct.
- **After small truncated PEPS:** the sketched move preserves the same physical
  convergence at controlled bond dimensions.
- **After a real `p>1` implementation:** the method computes multiple low-lying
  states with controlled block residuals.
- **After fair optimization:** the sketched implementation improves time or memory
  at matched physical accuracy in a reproducible regime.

Do not attribute a wall-clock improvement to sketching if it comes from optimizing
only the sketch implementation. Both backends need comparable kernels and the same
accuracy target before that becomes a scientific speed claim.

## 9. The later QR/Rayleigh pipeline

The whiteboard pipeline is a later scheduling optimization. Column `j+1` cannot be
factored until `R_j` has been absorbed and compressed into it. A Rayleigh
calculation can overlap the next QR only if it reads an immutable state snapshot or
precomputed contraction environments; otherwise the energy sum mixes different
states.

The safe eventual schedule is

```text
finish/zip column j
freeze the data needed for its Rayleigh contribution
run RQ(j) in parallel with QR(updated column j+1)
```

or, more simply, double-buffer outer iterations: compute diagnostics for immutable
`psi_k` while constructing `psi_{k+1}`. The concurrent implementation must match a
serial reference before any pipeline speedup is reported. QR/SVD and contraction
are both BLAS-heavy, so concurrency can also lose to resource contention; wall
time, not an assumed overlap diagram, decides whether it helps.
