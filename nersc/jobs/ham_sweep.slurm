#!/bin/bash
# HAMILTONIAN SURVEY (study 3 of the NERSC TODO): does disentangled-global prefer some
# physical systems? One array task per (instrument x state), on the paper-aligned ladder
#   TFIM g = 1.0 / 2.0 / 3.04(critical) / 3.5   (Dektor-Chi-Van Beeumen-Yang + Zaletel-Pollmann)
#   heis (the paper's hard case) | xxz@0.5, xxz@1.5 (hardness dial) | compass (frustrated)
# Instruments: exp04 spectrum diagnostic (the hypothesis: win tracks low effective rank),
# exp09 end-to-end cost + eta-to-match-local, exp10 3-way at fixed eta=4 with the PARALLEL
# per-cut disentanglers (--cut-workers; results bit-identical, see disentangled_qr).
#
#   sbatch nersc/jobs/ham_sweep.slurm          # from the repo root on $PSCRATCH
#
#SBATCH --job-name=ham-sweep
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1                     # shared REQUIRES --nodes=1 per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # ~256 GB shared cgroup -- headroom for the memory note below
#SBATCH --time=12:00:00
#SBATCH --array=0-18%8                # = number of CONFIGS below; %8 caps concurrency
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

module load python
source activate /global/common/software/m4926/rand-isopeps-env

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # floor; scripts re-pin BLAS

# Worker math / MEMORY (learned the hard way, 2026-07-09): --workers is the concurrent memory
# knob (each worker holds a column materialize + prep). The first run OOM-killed tfim@1.0 at
# --workers 8: g=1.0 lands in a high-effective-rank (cat) column whose Lx=6 factorize spikes far
# past the low-rank states. CAUTION: auto_worker_count reads the NODE's 512 GB (sysconf), NOT the
# shared cgroup (~4 GB/core), so it can't protect you here -- --workers is EXPLICIT and CONSERVATIVE.
# 64 cores -> ~256 GB cgroup; 4 workers -> ~64 GB headroom each (fits the cat spike), 4x{blas|cut}
# threads fill the cores. prep pinned to eta=4 (~3 GB); default prep-eta 6 (~25 GB) is lx_ceiling's.
S09="--lxs 4 5 6 --prep-eta 4 --eta-targets 4 --seeds 4 --workers 4 --blas-threads 16"
S10="--lxs 4 5 6 --kappa 2 --seeds 4 --workers 4 --cut-workers 4 --blas-threads 4"

CONFIGS=(
  # --- exp04: column spectrum diagnostic, all states at once (cheap) -------------------
  "experiments/column_sketch/scripts/exp04_real_column_spectrum.py --lxs 4 5 6 --seeds 4 --workers 4 --blas-threads 16 --states random tfim@1.0 tfim@2.0 tfim@3.04 tfim@3.5 heis xxz@0.5 xxz@1.5 compass"
  # --- exp09: end-to-end cost + eta-to-match, one state per task -----------------------
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states tfim@1.0"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states tfim@2.0"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states tfim@3.04"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states tfim@3.5"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states heis"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states xxz@0.5"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states xxz@1.5"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py $S09 --states compass"
  # --- exp10: mechanism + 3-way at fixed eta=4, one state per task ---------------------
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states tfim@1.0"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states tfim@2.0"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states tfim@3.04"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states tfim@3.5"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states heis"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states xxz@0.5"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states xxz@1.5"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py $S10 --states compass"
  # --- parallel-disentangler timing pair: same work, cut-workers 1 vs 4 (compare .out) -
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py --lxs 5 6 --kappa 2 --seeds 2 --workers 1 --cut-workers 1 --blas-threads 1 --states tfim@3.5"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py --lxs 5 6 --kappa 2 --seeds 2 --workers 1 --cut-workers 4 --blas-threads 1 --states tfim@3.5"
)

CMD=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$CMD" ]; then echo "no CONFIGS entry for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

cd "$SLURM_SUBMIT_DIR"
echo "[ham-sweep $SLURM_ARRAY_TASK_ID] $(date +%T) python $CMD"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores python $CMD
echo "[ham-sweep $SLURM_ARRAY_TASK_ID] $(date +%T) done"

# Stage to CFS (purge-safe); publish once the WHOLE array is done: bash nersc/publish.sh
CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"
mkdir -p "$CFS_OUT" && cp -ru outputs/. "$CFS_OUT"/ 2>/dev/null || true
