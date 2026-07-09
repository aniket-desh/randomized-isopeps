#!/bin/bash
# Small / quick run that doesn't need a whole node: 'shared' QOS, a few cores.
# Cheapest way to run a --quick benchmark or a single small experiment.
#     sbatch nersc/templates/shared_small.slurm
#
#SBATCH --job-name=risopeps-quick
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1                     # shared REQUIRES --nodes=1 (still only charged for the cores below)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python
source activate /global/common/software/m4926/rand-isopeps-env   # source (not conda): batch-safe

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # floor; script re-pins BLAS

# Run from wherever you submitted (your repo checkout); $PSCRATCH for big runs, $HOME is fine here.
cd "$SLURM_SUBMIT_DIR"

# --workers still needed even for a --quick run (default is 4). Keep it <= --cpus-per-task.
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
    --quick --workers "$SLURM_CPUS_PER_TASK" --blas-threads 1
