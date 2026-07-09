#!/bin/bash
# Small / quick run that doesn't need a whole node: 'shared' QOS, a few cores.
# Cheapest way to run a --quick benchmark or a single small experiment.
#     sbatch nersc/templates/shared_small.slurm
#
#SBATCH --job-name=risopeps-quick
#SBATCH --account=mXXXX               # ==EDIT==
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python
conda activate /global/common/software/mXXXX/rand-isopeps-env   # ==EDIT==

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$PSCRATCH/randomized-isopeps"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/real_moses_move/scripts/exp01_real_moses_move.py --quick
