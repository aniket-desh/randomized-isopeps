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
ssh aniketd@perlmutter.nersc.gov           # login (MFA: password+OTP)
cd $HOME && git clone https://github.com/aniket-desh/randomized-isopeps && cd randomized-isopeps
PROJECT=mXXXX bash nersc/setup_env.sh      # one-time: build the conda env on the Common FS
# edit nersc/templates/cpu_job.slurm (set --account, the conda prefix, the experiment line)
sbatch nersc/templates/cpu_job.slurm       # submit a CPU job
squeue --me                                # watch it
```

**Repeatable loop** (deploy code → run → publish figures → pull data → re-plot locally) is
scripted in `nersc/{deploy,publish,pull_data}.sh` — see **§11**. Network steps (`git
pull/push`) run on a login node; the job only runs and stages results.

- This project runs **CPU jobs** (`-C cpu`), not GPU. It's dominated by dense/randomized
  SVDs, QRs and matmuls.
- **The levers are two *script* flags, not env vars: `--workers` and `--blas-threads`.** The
  default worker count is laptop-capped (4–12), so **you must pass `--workers` on a big node**
  or you'll idle 116+ cores. See §6 — this is the one thing to get right.
- Parameter sweeps (Lx × states × seeds, or a batch of different ideas) → **one node with
  `--workers ~120`**, or a **Slurm job array** on the **`shared`** QOS. See §5–6.
- Large-Lx runs are **memory-bound** (dense column ~`16·2^Lx·8^Lx` bytes); the worker count
  is auto-capped by RAM. See §6.1.

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
In every job script you then `module load python && source activate <that prefix>` — use
`source activate` (or source the conda shell hook) in **batch** scripts; a bare
`conda activate` can raise `CommandNotFoundError` in a non-interactive shell.

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
| [`cpu_job.slurm`](templates/cpu_job.slurm) | **default** — one experiment on a full CPU node | `-C cpu -q regular -N 1 -c 128` + `--workers` |
| [`array_sweep.slurm`](templates/array_sweep.slurm) | **batch many ideas at once** — one array task per config | `-q shared --array=0-N` + a `CONFIGS[]` list |
| [`shared_small.slurm`](templates/shared_small.slurm) | a quick / small run without burning a full node | `-q shared -c 16` |
| [`gpu_job.slurm`](templates/gpu_job.slurm) | **reference only** — if you ever add a torch/cupy backend | `-C gpu -A …_g --gpus-per-node=4` |

Example experiment line — the current frontier is the `column_sketch` end-to-end cost sweep,
which is process-parallel (**pass `--workers`**; §6):
```bash
srun -n 1 -c $SLURM_CPUS_PER_TASK --cpu-bind=cores \
  python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
    --lxs 3 4 5 --states random tfim@3.5 tfim@3.04 --workers 96 --blas-threads 1
```
A quick smoke test (the real isoTNS Moses move):
```bash
srun -n 1 -c $SLURM_CPUS_PER_TASK --cpu-bind=cores \
  python experiments/real_moses_move/scripts/exp01_real_moses_move.py --instances 20 --workers 96
```
For a large-Lx single column, flip to the big-linalg regime: `--lxs 7 --workers 1 --blas-threads 128`.

---

## 6. Workers vs. threads, and memory (read this — it's what wastes a node)

`rand_isopeps` experiments do **not** run as one big multi-threaded process. Each script
spawns a **pool of worker processes** (`ProcessPoolExecutor`, one trial per worker) and
**pins BLAS to a single thread inside each worker** (via `threadpoolctl`). Two *script* flags
control this, and they matter more than any env var:

- `--workers N` — number of independent trials in flight (worker processes).
- `--blas-threads T` — BLAS/OpenMP threads **inside each worker** (default **1**).

Because the scripts call `threadpool_limits(--blas-threads)` at runtime, they **override**
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`. So `export OMP_NUM_THREADS=128` does essentially nothing
here — the real knob is `--blas-threads`, not the env var. (The templates still export the
env vars as a floor for any stray library, but the script wins.)

**The default worker count is laptop-conservative and will idle a NERSC node.** With
`--workers 0` (the default) the resolver caps at **4** for the `column_sketch` suite
([`auto_worker_count`](../src/rand_isopeps/experiment_utils/parallel.py)) and **12** for
`real_moses_move` — regardless of core count. On a 128-core node that leaves 116+ cores idle.
**You must pass `--workers` explicitly.**

Two regimes — pick per job:

| Regime | When | Flags |
|---|---|---|
| **Process-parallel** (default) | a sweep: many independent trials (Lx × states × seeds) | `--workers ~120 --blas-threads 1` |
| **One big linalg** | a single large-Lx column, very few trials | `--workers 1 --blas-threads 128` |

Dense SVDs stop scaling past ~16–32 threads, so for sweeps the many-1-thread-workers regime
beats one fat 128-thread job. Reserve big-linalg for a single column so large its dense form
dominates the node (§6.1).

### 6.1 Memory & large Lx (the real reason for the node)

The heavy jobs are the `column_sketch` sweeps (`exp08/09/10`): a dense column is materialized
for scoring, and it grows as ~`16 · 2^Lx · 8^Lx` bytes.

| Lx | dense column | + state prep |
|---|---|---|
| 6 | 0.27 GB | ~0.8 GB |
| 7 | 4.3 GB | ~13 GB |
| 8 | 69 GB | approaches the whole 512 GB node |

At large Lx a **single trial** can eat tens of GB, so:

- **`auto_worker_count` caps `--workers` by memory automatically**: `workers × (est bytes/trial)
  ≤ 0.6 × node RAM`. This is what stops N workers each holding a big column from OOM-ing (it
  caused a 108 GB crash locally — 4 workers × a ~27 GB trial). On the 512 GB node the ceiling
  is higher, but at large Lx the cap may hand you far fewer than 120 workers. **That's correct
  — don't force `--workers` past it.**
- The scripts carry explicit guards: `exp08/09/10` take `--lxs` (which Lx to run);
  `exp10_etaq_sweep` has a hard `--max-dense-gb` refusal (it is single-process — no `--workers`).

**Rule of thumb:** as Lx climbs, *lower* `--workers` (or let the memory cap do it) and *raise*
`--blas-threads`. At Lx=8, `--workers 1 --blas-threads 128`.

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
module load python && source activate /global/common/software/mXXXX/rand-isopeps-env
python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py --quick --workers 16
```
(`--quick` shrinks the sweep; still pass `--workers` — the default is 4.)

---

## 9. On-node operating guide

This is the checklist for whoever is **in the NERSC login session** (you by hand — SSH needs
MFA — or an agent that has an authenticated session). The **local analysis agent does not log
in**: it works only over git (`git fetch origin results`), so its half of the loop is §11.

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
- **Purge:** the templates stage `outputs/` to CFS at the end of a run (scratch is purged).
- **Publish when the job's done:** `bash nersc/publish.sh` — pushes figures + result data to
  the `results` branch so the local agent can pick them up over git (§11).
- **Pick the QOS by size:** `-q debug` (≤30 min) for a fast smoke test, `-q regular` for a
  real run, `-q shared` for anything that doesn't need a whole node.
- **Always set `--workers` on the experiment command** (default is 4–12 → idle node). Use
  `--workers ~cores --blas-threads 1` for a sweep; `--workers 1 --blas-threads 128` for one
  big-Lx column. Don't set both high — that oversubscribes (§6).
- **On `OUT_OF_MEMORY`:** lower `--workers` and/or `--lxs` — a single large-Lx trial can be
  tens of GB (§6.1). The script's `auto_worker_count` memory cap should prevent this, but a
  forced `--workers` overrides it.
- **Most common failures:** wrong account (CPU `m1234` vs GPU `m1234_g`); missing `-C cpu`;
  forgot `srun`; forgot `--workers` (node idles); env in `$HOME` instead of Common
  (quota/perf); results left on purged scratch.

---

## 10. Gotchas vs. Trillium (quick list)

- No `--partition` → use `-C cpu|gpu` + `-q …`.
- Must launch with `srun` (binding).
- GPU jobs charge to the `_g` account.
- `$PSCRATCH` is purged and not backed up → stage results to CFS.
- Compute nodes have **no internet** → install on login nodes.
- `module load python` (conda) instead of `StdEnv/2023 … + venv`; in a **batch** script
  prefer `source activate <prefix>` (or source the conda hook) — bare `conda activate` can
  fail in a non-interactive shell.
- Parallelism is via **`--workers`/`--blas-threads` script flags**, not `OMP_NUM_THREADS`
  (which the scripts override). Default workers = 4–12 → **pass `--workers`**.
- Large Lx is **memory-bound**; worker count is auto-capped by RAM. Don't force it past the cap.
- Confirm charge factors / QOS limits in Iris — they differ from Compute Canada.

---

## 11. The git round-trip: deploy → run → publish → analyze

The loop is **git-based on purpose**, because of the division of labor:

- **You drive NERSC by hand** (SSH needs MFA). You `deploy`, `sbatch`, and `publish` inside
  your authenticated login session.
- **The local agent never logs into NERSC** — it works only over **git**, which needs no MFA.
  So results have to arrive by `git`, and they do: `publish.sh` pushes **figures *and* the
  (small) result CSVs** to a `results` branch, and the agent gets everything with one
  `git fetch`. No rsync, no MFA, no waiting on you.

One hard rule: **`git pull`/`git push` run on a login node, never inside a job** — compute
nodes have no internet, so a job that pushes just hangs. The network steps bracket the
`sbatch`; they aren't part of it.

```
  YOU, local (Mac)        YOU, NERSC login node (MFA)         compute node
  ────────────────        ───────────────────────────         ────────────
  edit + git push    ─▶   bash nersc/deploy.sh [branch]    (git pull; reinstall if deps moved)
                          sbatch nersc/templates/…      ─▶  job runs, writes outputs/, → CFS
                          bash nersc/publish.sh             (figures + data → 'results' branch)
                                    │
  AGENT, local (no MFA):           ▼
  git fetch origin results  ◀── everything is on the results branch → assess, design next run
  └──────────────────────────────── repeat ──────────────────────────────────┘
```

- **`nersc/deploy.sh [branch]`** *(login node)* — clones into `$PSCRATCH/randomized-isopeps`
  if missing (e.g. after a scratch purge), else fast-forwards; reinstalls only if
  `pyproject.toml` changed. Override the location with `RISOPEPS_DIR`.
- **`nersc/publish.sh ["msg"]`** *(login node, after the job)* — regenerates the curated
  figures (`curate_figures.py`) and pushes `reports/figures/` **+ `outputs/`** to an orphan
  `results` branch (a separate worktree, so your run checkout is untouched; never merged to
  `main`, so `main` stays data-free). The result CSVs are tiny (~12 MB for months of runs), so
  they ride git. **Size guard:** if `outputs/` exceeds `DATA_MAX_MB` (default **200**), it
  publishes figures only and tells you to use `pull_data.sh`.
- **`nersc/pull_data.sh`** *(local)* — the **escape hatch** for a run too big for git: rsyncs
  the raw `outputs/` from CFS to `./nersc-data/` (gitignored), with a `MAX_GB` guard (default
  **1 GB**, `FORCE=1` to override). Needs an authenticated NERSC session — either you run it,
  or the agent can if your `sshproxy` cert is still live (~24 h). Not needed for normal runs.

**Agent, locally:** to read the latest results without disturbing the working branch,
`git fetch origin results` then either browse on GitHub or add a read-only view:
`git worktree add ../risopeps-results origin/results`.

**Two setup notes:**
- **Login-node push needs cached credentials.** The repo is public, but *pushing* the
  `results` branch needs auth on Perlmutter. Easiest: an SSH remote
  (`git remote set-url origin git@github.com:aniket-desh/randomized-isopeps`) with an SSH key
  on NERSC, or a GitHub PAT via `git config --global credential.helper store` + one manual push.
- **CFS staging is still worth keeping.** The templates' `cp -ru outputs/. $CFS_OUT/` line
  backs results up off purge-prone scratch and feeds `pull_data.sh` for the big-run case; set
  `$CFS_OUT` and `pull_data.sh`'s `REMOTE_DATA` to the same
  `/global/cfs/cdirs/<project>/risopeps/outputs` path.

---

## References (read these; verify against current docs)

- Jobs overview — https://docs.nersc.gov/jobs/
- Batch-script examples — https://docs.nersc.gov/jobs/examples/
- Running jobs on Perlmutter — https://docs.nersc.gov/systems/perlmutter/running-jobs/
- Python on Perlmutter — https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
- Filesystems — https://docs.nersc.gov/filesystems/
- QOS / job policy — https://docs.nersc.gov/jobs/policy/
