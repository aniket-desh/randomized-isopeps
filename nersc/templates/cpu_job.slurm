#!/bin/bash
# Default template: one rand_isopeps experiment on a full Perlmutter CPU node.
# Edit the ==EDIT== lines, then:  sbatch nersc/templates/cpu_job.slurm
#
#SBATCH --job-name=risopeps
#SBATCH --account=m4926               # CPU project (GPU jobs use m4926_g)
#SBATCH --constraint=cpu              # CPU node: 2x AMD EPYC 7763 = 128 cores, 512 GB
#SBATCH --qos=regular                 # regular | debug (<=30 min) | shared (partial node)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # ONE task; the script forks its own worker pool
#SBATCH --cpus-per-task=128           # give the task every core; --workers spends them
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu   # job start/fail emails

module load python
# Use `source activate` (not `conda activate`) in a batch shell -- the bare form can raise
# CommandNotFoundError in a non-interactive job.
source activate /global/common/software/m4926/rand-isopeps-env   # env on Common (batch-safe: source, not conda)

# These are only a floor: each worker re-pins BLAS to --blas-threads via threadpoolctl, so
# the script's flags below are what actually decide threading. Kept for stray libraries.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- parallelism: the two knobs that matter (see nersc/README.md §6) --------------------
# Sweep (many independent trials): many 1-thread workers.   Big single-Lx column: WORKERS=1.
# Default WORKERS=0 caps at 4 (column_sketch) / 12 (real_moses) -- that idles the node.
WORKERS=96          # ==EDIT== ~cores for a sweep; drop toward 1 as Lx grows (memory)
BLAS_THREADS=1      # ==EDIT== 1 for a sweep; 128 for a single big-linalg column
# -----------------------------------------------------------------------------------------

# Run from scratch (fast, large). Assumes a checkout at $PSCRATCH/randomized-isopeps.
cd "$PSCRATCH/randomized-isopeps"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
    --lxs 3 4 5 --states random tfim@3.5 tfim@3.04 \
    --workers "$WORKERS" --blas-threads "$BLAS_THREADS"   # ==EDIT== experiment + args

# Stage results to CFS: scratch is PURGED and not backed up. Scripts write everything under
# a single top-level outputs/ tree (outputs/<suite>/{data,figures}). This is what
# nersc/pull_data.sh rsyncs back to your Mac.
CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"        # keepers survive the scratch purgemkdir -p "$CFS_OUT" && cp -ru outputs/. "$CFS_OUT"/ 2>/dev/null || true
