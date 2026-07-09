#!/bin/bash
# Default template: one rand_isopeps experiment on a full Perlmutter CPU node.
# Edit the three ==EDIT== lines, then:  sbatch nersc/templates/cpu_job.slurm
#
#SBATCH --job-name=risopeps
#SBATCH --account=mXXXX               # ==EDIT== your NERSC CPU project (no _g for CPU jobs)
#SBATCH --constraint=cpu              # CPU node: 2x AMD EPYC 7763 = 128 cores, 512 GB
#SBATCH --qos=regular                 # regular | debug (<=30 min) | shared (partial node)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128           # whole node to one BLAS-threaded task
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@lbl.gov       # ==EDIT== your email

module load python
conda activate /global/common/software/mXXXX/rand-isopeps-env   # ==EDIT== env prefix

# BLAS/OpenMP threads (this project is SVD/QR/matmul-bound; threadpoolctl honours these).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run from scratch (fast, large). Assumes a checkout at $PSCRATCH/randomized-isopeps.
cd "$PSCRATCH/randomized-isopeps"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/real_moses_move/scripts/exp01_real_moses_move.py --instances 20

# Keep results: scratch is PURGED and not backed up.
# cp -r outputs /global/cfs/cdirs/mXXXX/risopeps/
