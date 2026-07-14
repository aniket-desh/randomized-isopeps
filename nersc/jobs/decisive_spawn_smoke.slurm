#!/bin/bash
# Cluster-specific guard for fork-after-BLAS worker crashes. The exact benchmark
# is dependency-gated on this cheap test by submit_decisive_suite.sh.
#SBATCH --job-name=column-spawn-smoke
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

set -euo pipefail
module load python
source activate /global/common/software/m4926/rand-isopeps-env
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  pytest -q tests/test_parallel.py
