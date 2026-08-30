# paper campaign

The campaign is accuracy-first: dense or published references gate the physical
claims, and timing becomes a paper claim only after the CPU/CuPy parity pilot passes.
The builders currently emit 2,763 deterministic tasks across seven families.

| family | tasks | purpose |
|---|---:|---|
| `gaussian_limit` | 420 CPU | reproduce the RMPS Gaussian-limit variance and Nyström grids, then compare paired subspace errors |
| `column_moves` | 504 CPU | random-raw versus Moses-prepared size sweeps, justified local/global sketches, one-factor RMPS tuning, and global-only controlled spectra |
| `isometry` | 384 CPU | isometry preservation for every valid column method; exact full-rank certification is restricted to 2x3 and larger cells are rank-saturated or truncated |
| `physics` | 748 mixed | dense and block correctness, Dektor Figures 2-4 and Table 2 cells, Hamiltonian robustness, and separate `chi`, `eta`, `ell`, `chi_sk`, `kappa`, and `p` sweeps |
| `gpu_pilot` | 12 CPU + 12 GPU | mandatory representative backend parity gate |
| `gpu_crossover` | 324 CPU + 324 GPU | optional broad timing crossover after promotion |
| `references` | 35 CPU | exact small-cell calibration and paired paper-scale energy/orthogonality convergence cells |

Build the manifests from the repository root:

```bash
python experiments/paper_campaign/build_manifests.py
```

Each complete family gets a `campaign_revision` before it is split, and every task
also carries a hash of the Python runtime sources. Task IDs include both values, so a
changed implementation creates a new immutable campaign instead of reusing old output.

## scientific contracts

The random-versus-Moses comparison uses raw bond-2 random PEPS through column size 7,
where all dense controls are safe. Prepared physical columns at larger input bond use
only matrix-free RMPS/Kronecker and local sequential candidates; dense global sketches
are labeled `materialized_only` and limited to feasible cells. Synthetic controlled
spectra are global-only because they do not have an MPO factorization for a valid local
Moses comparison.
The local baseline uses `ndis=0`; a separately labeled Riemannian-Renyi Moses
comparator uses 30 disentangling sweeps, so optimization work is never folded into the
deterministic baseline.

The `p > 1` correctness ladder uses the same `shared_block_oracle` initializer for dense
and PEPS paths. The block methods compute one shared column QR for the ordered states;
each measurement solves the projected generalized Rayleigh-Ritz problem and rotates the
open block index into ascending Ritz-energy order. The method does not claim a
tangent-space solver. The Dektor-style block loop alternates forward and reverse sweeps,
so successive iterations do not all begin at the same corner.
Within each Dektor cell and the dedicated block-size sweep, every `p` value slices the
same parent random block before QR. Those curves therefore compare nested initial
subspaces rather than unrelated random starts; unrelated Hamiltonian, lattice, and bond
cells retain independent seeds.
The `correctness` figure requires every single-state and block-oracle task and facets each
Hamiltonian, lattice, and block size separately before comparing energy errors with its
paired dense exact evolution.

Large PEPS energies are measured at contraction bonds 64 and 128. The higher-bond value
is canonical only when their configured difference passes `measurement_converged`;
accuracy plots reject unconverged final cells. Paper-scale MPS references use an
explicit energy-and-orthogonality contract because an uncompressed paper-bond residual
would exceed memory. Two nested high-bond runs must agree, every state must complete two
energy-stable sweeps at the declared final bond and retain that bond, every exact MPS
overlap gate must pass, and unrestricted excited states use a bond-64 projector with at
most `1e-4` compression infidelity. This is convergence evidence rather than a claimed
paper-scale residual certificate. Heisenberg references explicitly target total-Sz 0
for `alpha=0` and total-Sz 1 for `alpha=1`.

The Dektor Table 2 and tangent-space comparator are published literature data, never
locally executed evidence. Their values and provenance live in
`references/dektor_table_2_v1.json`; executed block results remain separately labeled.
All eight published reference energies must also agree with independent 4x4 exact
diagonalization within the manifest-bound relative tolerance before Table 2 physics can
run or the comparison figure can render.

## staged execution

1. Build and dry-run the submission plan. The summary reports task counts by family,
   hardware, and resource class, plus maximum requested concurrent allocations. Add
   `--pilot-largest` to select and print one maximum-scale task per resource class before
   committing an entire family.
2. Run the complete `gpu_pilot` family on both CPU and GPU, then run the exact small-cell
   reference calibration, including every Table 2 field value, in stage 1.
   The pilot needs both backends for pairing; submitting only its GPU half cannot pass.
   Power-iteration pilots compare the complete matvec-adjoint-matvec chain across backends.
3. After every exact stage-1 task validates, submit paper-energy reference stage 2.
   Export references and validate the GPU pilot against the exact full-family
   manifest used for that run. The submitter checks artifact schema, code revision,
   complete pilot IDs, and reference cell/state/sector coverage before any dependent job.
4. Run the accuracy families and physics campaign. Treat `gpu_crossover` as optional
   performance data after accuracy and parity are established.

```bash
python nersc/submit_paper_campaign.py

python nersc/submit_paper_campaign.py \
  --family physics --hardware cpu --pilot-largest

python nersc/submit_paper_campaign.py \
  --family gpu_pilot --submit

python nersc/submit_paper_campaign.py \
  --family references --reference-stage 1 --hardware cpu --submit

python nersc/submit_paper_campaign.py \
  --family references --reference-stage 2 --hardware cpu --submit

python experiments/paper_campaign/export_references.py \
  /global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign \
  /global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign/references.json

python experiments/paper_campaign/validate_gpu_pilot.py \
  /global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign \
  /global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign/gpu_gate.json \
  --manifest <immutable-gpu-pilot-family-manifest-printed-at-submission>
```

On `--submit`, full-family and split-array manifests are copied to
`OUTPUT_ROOT/manifests` with content-addressed names. Queued arrays reference those CFS
snapshots, so rebuilding a local manifest cannot change a queued array index.

Run one task locally with an explicit zero-based index:

```bash
python experiments/paper_campaign/run_task.py \
  experiments/paper_campaign/manifests/column_moves.jsonl --task-index 0 \
  --checkpoint-root "$PSCRATCH/rand_isopeps/paper_campaign/checkpoints"
```

Results are immutable task JSONL files under `outputs/<manifest_hash>/tasks/`. An all-ok
retry is a no-op; a failed immutable record is terminal for that manifest, so corrected
code requires a new manifest revision.

Render completed studies directly from the durable result root:

```bash
python experiments/paper_campaign/plot_results.py \
  /global/cfs/cdirs/m4926/risopeps/outputs/paper_campaign \
  --output-dir experiments/paper_campaign/figures --figure all
```

The CLI checks the exact current task subset consumed by each requested figure, rejects
failed or stale revisions and incomplete comparison grids, and refuses accuracy figures
whose selected preparation or B/2B contraction did not converge. A completed small
figure can therefore render before unrelated large studies finish, without accepting a
partial version of its own data.

The Iris screenshot shows about 253 personal CPU node-hours and 1,428 personal GPU
node-hours remaining. Task counts do not determine actual charges: record pilot walltimes
for each resource class before asserting that the full campaign fits. At the 24-hour
request ceiling, the complete plan would consume 2,943.75 CPU and 2,778 GPU node-hours, so
it does not fit either personal pool under that deliberately pessimistic bound. The
submitter's CPU and GPU array throttles apply independently to every resource-class shard;
use its summed peak-rate and requested-walltime ceiling before raising them or submitting
several families at once.
