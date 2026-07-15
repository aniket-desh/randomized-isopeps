#!/bin/bash
# Matrix-free size scaling. Each completed size is copied to CFS immediately;
# Lx=10/12 are feasibility probes and a failure there preserves all earlier rows.
#SBATCH --job-name=column-mf-scaling
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

run_size() {
  local lx="$1" prep_seeds="$2" sketch_seeds="$3" score_probes="$4"
  local chi_grid="8"
  local score_chi="8"
  if [ "$lx" -gt 8 ]; then chi_grid="8 $lx"; fi
  if [ "$lx" -gt 8 ]; then score_chi="$lx"; fi
  echo "[Lx=$lx] $(date -u +%FT%TZ) start"
  srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    python experiments/column_sketch/scripts/exp11_invariant_one_move.py \
      --lx "$lx" --ly 4 --states random tfim@3.04 heis \
      --methods local_det local_rsvd2 global_rmps_plain global_rmps_bounded global_kron \
      --bond 2 --chi 8 --prep-eta 6 --eta-grid 4 \
      --kappa-grid 1 2 4 --chi-sk-grid $chi_grid \
      --ell-grid 8 --n-power-grid 0 --ndis 10 \
      --taus 0.3 0.1 0.03 0.01 --energy-rtol 1e-6 \
      --stable-sweeps 3 --min-sweeps-per-tau 3 --max-sweeps-per-tau 40 \
      --prep-seeds "$prep_seeds" --sketch-seeds "$sketch_seeds" \
      --score-chi "$score_chi" --score-probes "$score_probes" --score-bootstrap 200 \
      --exact-max-sites 16 --ed-max-sites 16 --boundary-bond 96 \
      --workers 4 --blas-threads 16
  stage_results
  echo "[Lx=$lx] $(date -u +%FT%TZ) done"
}

run_size 5 8 4 32
run_size 6 8 4 32
run_size 7 4 4 24
run_size 8 4 4 24
run_size 10 2 4 16
run_size 12 1 4 16
