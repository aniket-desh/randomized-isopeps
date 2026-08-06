#!/bin/bash
# Full-node 4x4 phase-two trajectories. Accuracy first; no pipeline overlap yet.
# Submit from the repository root: sbatch nersc/jobs/physics_large_peps.slurm
#SBATCH --job-name=physics-large
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=08:00:00
#SBATCH --array=0-7%2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

set -euo pipefail
module load python
source activate /global/common/software/m4926/rand-isopeps-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(
  "peps_local tfim@3.04 0"
  "peps_sketch tfim@3.04 0"
  "peps_local heis 0"
  "peps_sketch heis 0"
  "peps_local tfim@3.04 1"
  "peps_sketch tfim@3.04 1"
  "peps_local heis 1"
  "peps_sketch heis 1"
)

read -r MODE HAMILTONIAN SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT="outputs/physics_loop/large-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.jsonl"
CFS_OUT=/global/cfs/cdirs/m4926/risopeps/outputs/physics_loop

stage_result() {
  if [[ -f "$OUT" ]]; then
    mkdir -p "$CFS_OUT"
    cp "$OUT" "$CFS_OUT/"
  fi
}
trap stage_result EXIT

COMMON=(--lx 4 --ly 4 --hamiltonian "$HAMILTONIAN" --seed "$SEED"
        --stage 0.05:5 --stage 0.02:10 --stage 0.01:10
        --eta 4 --gate-bond 8 --absorption-bond 8
        --energy-bond 32 --exact-max-sites 12 --blas-threads 32 --output "$OUT")
if [[ "$MODE" == peps_local ]]; then
  METHOD=(--chi 8 --ndis 0)
else
  METHOD=(--ell 8 --kappa 2 --chi-sk 4 --ndis 0)
fi

cd "$SLURM_SUBMIT_DIR"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/physics_loop/run.py --mode "$MODE" "${COMMON[@]}" "${METHOD[@]}"
