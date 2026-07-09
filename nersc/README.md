# Running `randomized-isopeps` on NERSC Perlmutter

A crash course + workflow briefing for running this project's experiments on **NERSC
Perlmutter** (LBL). It translates the Compute-Canada **Trillium** Slurm workflow used on
the Vector project (kept verbatim in [`trillium_reference/`](trillium_reference/)) to
NERSC, and adapts it to the fact that **`rand_isopeps` is CPU / BLAS-bound** (numpy, scipy,
optional quimb — no GPU). Written to be followed by a human *or* an agent: every command is
copy-paste, and §9 is an agent operating guide.

> The two clusters both run **Slurm**, so the *shape* of the workflow carries over. What
> changes is how you select nodes (constraint + QOS, not partitions), how you charge
> (CPU vs GPU accounts), the module/environment stack, the scratch purge, and that you
> **must launch with `srun`**. The cheat sheet in §4 is the whole translation in one table.

---

## 0. TL;DR

```bash
ssh <user>@perlmutter.nersc.gov            # login (MFA: password+OTP)
cd $HOME && git clone https://github.com/aniket-desh/randomized-isopeps && cd randomized-isopeps
PROJECT=mXXXX bash nersc/setup_env.sh      # one-time: build the conda env on the Common FS
# edit nersc/templates/cpu_job.slurm (set --account, the conda prefix, the experiment line)
sbatch nersc/templates/cpu_job.slurm       # submit a CPU job
squeue --me                                # watch it
```

- This project runs **CPU jobs** (`-C cpu`), not GPU. It's dominated by dense/randomized
  SVDs, QRs and matmuls, so the lever is **BLAS threads**, not accelerators.
- Parameter sweeps (instances × sketch seeds × bond) → **Slurm job arrays** and the
  **`shared`** QOS (partial node, cheap). See §5–6.

---

## 1. Perlmutter in one screen

- **Login:** `ssh <user>@perlmutter.nersc.gov`. Login nodes are for editing, building
  environments, `sbatch`, and *small* tests — **not** heavy compute. They **have internet**
  (compute nodes do not).
- **Node types:**
  - **CPU node** — 2× AMD EPYC 7763 (Milan) = **128 physical cores**, 512 GB. Selected with
    `-C cpu`. *This is what you want for `rand_isopeps`.*
  - **GPU node** — 1× AMD EPYC (64 cores) + **4× NVIDIA A100**. Selected with `-C gpu`.
- **Allocations are split:** CPU-node hours and GPU-node hours are separate pools. **CPU
  jobs charge to `<project>` (e.g. `m1234`); GPU jobs charge to `<project>_g` (e.g.
  `m1234_g`).** Find your exact project codes in the **Iris** portal (iris.nersc.gov) or:
  ```bash
  sacctmgr -p show assoc user=$USER format=account,qos
  ```

---

## 2. Filesystems — where things go (this is the #1 thing people get wrong)

| Filesystem | Path | Use it for | Backed up | Purged? |
|---|---|---|---|---|
| **Home** | `/global/homes/<u>/<user>` = `$HOME` | code, scripts, dotfiles | yes | no (small quota) |
| **Common** | `/global/common/software/<project>` | **conda environments**, software | yes | no |
| **CFS** | `/global/cfs/cdirs/<project>` | datasets + results to **keep** | yes | no |
| **Scratch** | `/pscratch/sd/<l>/<user>` = `$PSCRATCH` | **run jobs here**, big/fast I/O | **no** | **yes (~8 wks untouched)** |

**Rules of thumb**
- Put the **conda env on Common** (fast parallel FS + big quota; a torch/quimb env will blow
  your `$HOME` quota).
- Keep **code** in `$HOME` (or CFS); **run from `$PSCRATCH`** (fast, large).
- **Copy anything you want to keep off `$PSCRATCH` to CFS** — scratch is purged and *not*
  backed up. Never leave the only copy of a result on scratch.

---

## 3. First-time setup (once, on a login node)

Compute nodes have **no outbound internet**, so all `pip`/`conda` installs happen on a login
node. That's exactly what [`setup_env.sh`](setup_env.sh) does:

```bash
PROJECT=mXXXX bash nersc/setup_env.sh
```

which is, in essence:
```bash
module load python                                   # NERSC's conda base
conda create -y --prefix /global/common/software/$PROJECT/rand-isopeps-env \
    python=3.11 pip numpy scipy matplotlib threadpoolctl
conda activate /global/common/software/$PROJECT/rand-isopeps-env
pip install -e ".[quimb]"                            # package + real-isoTNS Moses-move extra
```
In every job script you then just `module load python && conda activate <that prefix>`.

---

## 4. Trillium → NERSC cheat sheet (the translation)

| Concept | Trillium (Compute Canada) | NERSC Perlmutter |
|---|---|---|
| pick a node | `--partition=compute_full_node` | **`-C cpu`** (or `-C gpu`) |
| queue / limits | *(implied by partition)* | **`-q regular`** \| `debug` (≤30 min) \| `shared` (partial node) \| `interactive` |
| account | `--account=rrg-aspuru` | **`-A <project>`** (CPU) / **`-A <project>_g`** (GPU) |
| CPUs | `--cpus-per-task=8` | `-c <cores>` (up to **128** on a CPU node) |
| GPUs | `--gpus-per-node=4` | `--gpus-per-node=4` **and** `-C gpu` *(GPU jobs only)* |
| modules | `module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack` | **`module load python`** (+ `cudatoolkit` only for GPU) |
| environment | `source ~/envs/gqs/bin/activate` (venv) | **`conda activate <prefix on Common>`** |
| working dir | `$SCRATCH/...` | **`$PSCRATCH/...`** |
| launch the app | `python script.py` | **`srun -c <cores> --cpu-bind=cores python script.py`** |
| interactive | `salloc ...` | `salloc -N 1 -C cpu -q interactive -t 1:00:00 -A <project>` |
| watch queue | `squeue -u $USER` | `squeue --me` (or `sqs`) |
| cancel / history | `scancel`, `sacct` | `scancel <id>`, `sacct -j <id>` |

**Four conceptual shifts that trip people up**
1. **Partitions → constraint + QOS.** NERSC chooses the hardware with `-C cpu/gpu` and the
   queue/limits with `-q`. There is no `--partition`.
2. **Always `srun`.** Even a single-node job launches its program with `srun` so Slurm binds
   cores (and GPUs) correctly. The Trillium scripts call `python …` directly; on NERSC that
   under-uses the node and misbinds threads.
3. **CPU vs GPU accounts.** GPU jobs must charge to the `_g` account or they're rejected.
4. **Scratch is aggressively purged.** Move results to CFS.

---

## 5. Job templates ([`templates/`](templates/))

Edit the `--account` and the conda-env prefix, point the `srun … python …` line at your
experiment, then `sbatch`.

| Template | When | Key flags |
|---|---|---|
| [`cpu_job.slurm`](templates/cpu_job.slurm) | **default** — one experiment on a full CPU node | `-C cpu -q regular -N 1 -c 128` |
| [`array_sweep.slurm`](templates/array_sweep.slurm) | a sweep (seeds/instances/bond), many small tasks | `-q shared --array=0-N -c 16` |
| [`shared_small.slurm`](templates/shared_small.slurm) | a quick / small run without burning a full node | `-q shared -c 8` |
| [`gpu_job.slurm`](templates/gpu_job.slurm) | **reference only** — if you ever add a torch/cupy backend | `-C gpu -A …_g --gpus-per-node=4` |

Example experiment line (real isoTNS Moses move, 20 PEPS instances):
```bash
srun -n 1 -c $SLURM_CPUS_PER_TASK --cpu-bind=cores \
  python experiments/real_moses_move/scripts/exp01_real_moses_move.py --instances 20
```

---

## 6. BLAS threading (the performance knob for this project)

`rand_isopeps` is SVD/QR/matmul-bound, so speed comes from **BLAS/OpenMP threads**, not GPUs.
Every template exports the thread counts consistently:
```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
```
Two regimes:
- **One big run:** give the whole node to one task (`-c 128`, threads = 128). Best for a
  single large PEPS / long benchmark.
- **A sweep:** run **many tasks with few threads each** (e.g. 8 tasks × 16 threads on one
  node, or a job array under `-q shared`). Dense SVDs stop scaling well past ~16–32 threads,
  so packing independent runs gives far more throughput than one 128-thread job. The
  experiment scripts already expose the sweep axes (`--instances`, `--sketch-seeds`, `--seed`).
- The package uses `threadpoolctl`, so it respects these env vars; keep them in sync.

---

## 7. Running the workflow

The Trillium reference was a 5-stage **GPU ML** pipeline (`01_generate_data → 02_train →
03_sample → 04_evaluate → 05_plot`), each a separate `sbatch`, sometimes chained. For
`rand_isopeps` the analog is simpler — there's no training, just **run experiment → collect →
plot**:

```bash
# 1. code + scratch workspace
cd $PSCRATCH && git clone https://github.com/aniket-desh/randomized-isopeps && cd randomized-isopeps
# (or keep code in $HOME and `cd $PSCRATCH/run && ln -s $HOME/randomized-isopeps/* .`)

# 2. submit
sbatch $HOME/randomized-isopeps/nersc/templates/cpu_job.slurm     # -> Submitted batch job 12345

# 3. monitor
squeue --me
tail -f risopeps-12345.out

# 4. keep the results (scratch is purged!)
cp -r outputs /global/cfs/cdirs/mXXXX/risopeps/
```

**Chaining stages** (if one depends on another), the NERSC equivalent of the Trillium
`regression_extrap.sh` orchestrator:
```bash
jid=$(sbatch --parsable stage_a.slurm)
sbatch --dependency=afterok:$jid stage_b.slurm
```

---

## 8. Interactive debugging

Grab a compute node interactively to iterate quickly (use the experiments' `--quick` flags):
```bash
salloc -N 1 -C cpu -q interactive -t 1:00:00 -A mXXXX
# …lands you on a CPU node…
module load python && conda activate /global/common/software/mXXXX/rand-isopeps-env
export OMP_NUM_THREADS=32
python experiments/real_moses_move/scripts/exp01_real_moses_move.py --quick
```

---

## 9. Agent operating guide

For an automated agent driving NERSC:
- **Never run heavy compute on a login node.** Everything real goes through `sbatch`.
- **Submit:** `sbatch job.slurm` → prints `Submitted batch job <jobid>` (use `--parsable`
  to get just the id).
- **Poll:** `squeue --me` — the job is done when it's gone from the queue. Don't busy-wait
  hard; poll every ~30–60 s.
- **Outcome:** `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode` — `COMPLETED`
  + `0:0` is success; `FAILED`/`TIMEOUT`/`OUT_OF_MEMORY` tell you what to fix.
- **Logs:** the `-o/-e` files, default `<jobname>-<jobid>.out` / `.err`.
- **No internet on compute nodes:** any dependency install is a *separate login-node* step
  (or bake it into `setup_env.sh`). A job that tries to `pip install` will hang/fail.
- **Purge:** copy `outputs/` from `$PSCRATCH` to CFS as the last step of a run.
- **Pick the QOS by size:** `-q debug` (≤30 min) for a fast smoke test, `-q regular` for a
  real run, `-q shared` for anything that doesn't need a whole node.
- **Most common failures:** wrong account (CPU `m1234` vs GPU `m1234_g`); missing `-C cpu`;
  forgot `srun`; env in `$HOME` instead of Common (quota/perf); results left on purged scratch.

---

## 10. Gotchas vs. Trillium (quick list)

- No `--partition` → use `-C cpu|gpu` + `-q …`.
- Must launch with `srun` (binding).
- GPU jobs charge to the `_g` account.
- `$PSCRATCH` is purged and not backed up → stage results to CFS.
- Compute nodes have **no internet** → install on login nodes.
- `module load python` (conda) instead of `StdEnv/2023 … + venv`.
- Confirm charge factors / QOS limits in Iris — they differ from Compute Canada.

---

## References (read these; verify against current docs)

- Jobs overview — https://docs.nersc.gov/jobs/
- Batch-script examples — https://docs.nersc.gov/jobs/examples/
- Running jobs on Perlmutter — https://docs.nersc.gov/systems/perlmutter/running-jobs/
- Python on Perlmutter — https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
- Filesystems — https://docs.nersc.gov/filesystems/
- QOS / job policy — https://docs.nersc.gov/jobs/policy/
