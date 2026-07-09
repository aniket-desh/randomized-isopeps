#!/bin/bash
# Job array for a parameter sweep (one array task per value), on the 'shared' QOS so
# each task uses only part of a node and is charged fractionally. Ideal for the
# instance/seed sweeps the experiment scripts expose.
#     sbatch nersc/templates/array_sweep.slurm
#
#SBATCH --job-name=risopeps-sweep
#SBATCH --account=mXXXX               # ==EDIT==
#SBATCH --constraint=cpu
#SBATCH --qos=shared                  # partial node; do NOT also set --nodes with shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # cores per array task
#SBATCH --time=02:00:00
#SBATCH --array=0-4                   # one task per sweep value (5 seeds here)
#SBATCH --output=%x-%A_%a.out         # %A=array job id, %a=array index
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@lbl.gov       # ==EDIT==

module load python
conda activate /global/common/software/mXXXX/rand-isopeps-env   # ==EDIT==

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Map the array index -> a sweep value.
SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$PSCRATCH/randomized-isopeps"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/real_moses_move/scripts/exp01_real_moses_move.py --sketch-seeds 1 --seed "$SEED"

# Each task writes its own outputs/ subdir; copy them to CFS after the array finishes.
