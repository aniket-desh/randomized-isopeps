#!/bin/bash
# Headline exact 4x4 invariant benchmark: all requested states, 16x4 seeds at one
# defensible operating point. Parameter curves belong to the separate saturation
# stage. Dense projection scoring is not repeated here; exp12 calibrates the
# independent estimator, while exact Frobenius residuals are available per row.
#SBATCH --job-name=column-exact-headline
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
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

CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"
stage_results() {
  mkdir -p "$CFS_OUT"
  cp -ru outputs/. "$CFS_OUT"/
}
trap stage_results EXIT

srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/column_sketch/scripts/exp11_invariant_one_move.py \
    --lx 4 --ly 4 \
    --states random tfim@1.0 tfim@2.0 tfim@3.04 tfim@3.5 \
             heis xxz@0.5 xxz@1.5 compass \
    --bond 2 --chi 8 --prep-eta 8 \
    --eta-grid 4 --kappa-grid 1 2 --chi-sk-grid 8 \
    --ell-oversampling-grid 4 --n-power-grid 0 --ndis 10 \
    --taus 0.3 0.1 0.03 0.01 --energy-rtol 1e-6 \
    --stable-sweeps 3 --min-sweeps-per-tau 3 --max-sweeps-per-tau 40 \
    --prep-seeds 16 --sketch-seeds 4 \
    --score-probes 0 \
    --exact-max-sites 16 --ed-max-sites 16 --boundary-bond 64 \
    --workers 4 --blas-threads 32
