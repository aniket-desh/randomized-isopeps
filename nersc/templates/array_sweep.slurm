#!/bin/bash
# Batch many ideas at once: a Slurm job array where each array task runs ONE config from the
# CONFIGS list below. Perfect for firing off a pile of experiments before a maintenance window
# -- edit CONFIGS, set --array=0-(N-1), sbatch once.
#     sbatch nersc/templates/array_sweep.slurm
#
#SBATCH --job-name=risopeps-batch
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=shared                  # partial node, charged fractionally by cores
#SBATCH --nodes=1                     # shared REQUIRES --nodes=1 per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32            # cores per array task (shared gives ~4 GB/core -> ~128 GB)
#SBATCH --time=06:00:00
#SBATCH --array=0-2%8                 # ==EDIT== 0-(N-1); %8 = at most 8 tasks running at once
#SBATCH --output=%x-%A_%a.out         # %A=array job id, %a=array index
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu   # job start/fail emails

module load python
source activate /global/common/software/m4926/rand-isopeps-env   # source (not conda): batch-safe

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # floor; script re-pins BLAS

# ---- one line per idea; each is a full "script.py <args>" -------------------------------
# Keep each config's --workers <= --cpus-per-task (32 here). Bump memory-heavy large-Lx runs
# to the full-node cpu_job.slurm instead of shared. See nersc/README.md §6.
CONFIGS=(
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py --lxs 3 4 5 --states random     --workers 32 --blas-threads 1"
  "experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py --lxs 3 4 5 --states tfim@3.04   --workers 32 --blas-threads 1"
  "experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py --lxs 4 5 --kappa 2         --workers 24 --blas-threads 1"
  # ==EDIT== add more ideas here, then set --array=0-(count-1) above.
)
# -----------------------------------------------------------------------------------------

CMD=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$CMD" ]; then echo "no CONFIGS entry for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

# Run from wherever you submitted (your repo checkout); use a $PSCRATCH checkout for big runs.
cd "$SLURM_SUBMIT_DIR"
echo "[array $SLURM_ARRAY_TASK_ID] python $CMD"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores python $CMD

# Tasks share one timestamped outputs/ tree (unique slugs, no collisions). Stage to CFS so
# nersc/pull_data.sh can fetch it; -u = only copy newer, cheap when tasks overlap.
CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"        # keepers survive the scratch purge
mkdir -p "$CFS_OUT" && cp -ru outputs/. "$CFS_OUT"/ 2>/dev/null || true
