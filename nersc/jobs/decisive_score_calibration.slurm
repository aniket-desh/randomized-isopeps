#!/bin/bash
# Dense calibration of fresh rMPS projection scoring at every feasible Lx.
# Submitted by nersc/submit_decisive_suite.sh.
#SBATCH --job-name=column-score-cal
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

set -euo pipefail
module load python
source activate /global/common/software/m4926/rand-isopeps-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QUIMB_NUMBA_CACHE=False PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/column_sketch/scripts/exp12_projection_score_calibration.py \
    --lxs 2 3 4 5 6 \
    --ensembles gaussian decay identity_plus_noise \
    --eta 4 --kappa 2 --chi-sk 4 --ell 8 --n-power 0 --ndis 5 \
    --score-chis 1 2 4 8 0 --score-probes-grid 16 64 \
    --problem-seeds 4 --score-seeds 4 --bootstrap 500 \
    --workers 8 --blas-threads 8

CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"
mkdir -p "$CFS_OUT"
cp -ru outputs/. "$CFS_OUT"/
