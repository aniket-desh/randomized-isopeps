# Trillium reference scripts (Compute Canada)

The original Slurm scripts from the Vector project's runs on **Trillium** (SciNet /
Compute Canada), kept **verbatim** as the source material for the NERSC translation in
[`../README.md`](../README.md). They do **not** run on NERSC as-is — see the cheat sheet
in that file for the line-by-line mapping.

These are a 5-stage **GPU ML** pipeline (a Heisenberg GCN-transformer), each stage a
standalone `sbatch`:

| Script | Stage |
|---|---|
| `01_generate_data.sh` | shadow/data generation |
| `02_train_transformer.sh` | train the GCN-transformer (4 GPUs) |
| `03_sample_transformer.sh` | sample from the trained model |
| `04_evaluate.sh` | evaluate properties |
| `05_plot.sh` | plots |
| `regression_extrap.sh` | a parameterized, **chained** orchestrator (datagen → eval) that builds `#SBATCH` headers in heredocs and submits with dependencies — the pattern §7 of the briefing translates to `sbatch --dependency=afterok:<id>` |

**Compute-Canada specifics to notice** (all translated in `../README.md` §4):
- `#SBATCH --partition=compute_full_node`  → NERSC `-C cpu`/`-C gpu` + `-q …`
- `#SBATCH --account=rrg-aspuru`          → NERSC `-A <project>` / `<project>_g`
- `#SBATCH --gpus-per-node=4`             → same flag, but NERSC also needs `-C gpu`
- `module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a` → NERSC `module load python` (+ `cudatoolkit` for GPU)
- `source $HOME/envs/gqs/bin/activate` (virtualenv) → NERSC `conda activate <prefix on Common>`
- `cd $SCRATCH/generative-quantum-states` → NERSC `cd $PSCRATCH/...`
- programs launched as `python …` (no `srun`) → NERSC launches with `srun -c … --cpu-bind=cores …`

The full per-run script zoo (dozens of `_datagen_*`, `_eval_*`, `_comp_diag_*` variants)
lives in the Vector `fermionic-shadow-regressor` repo under `models/slurm/`; these six are
the representative, canonical set.
