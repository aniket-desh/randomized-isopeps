#!/bin/bash
# Dense or small-PEPS accuracy array. One trajectory per shared-node task.
# Submit from the repository root: sbatch nersc/jobs/physics_small_array.slurm
#SBATCH --job-name=physics-small
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --array=0-14%6
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

set -euo pipefail
module load python
source activate /global/common/software/m4926/rand-isopeps-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(
  "dense_exact --lx 2 --ly 2 --states 3 --stage 0.10:10 --stage 0.03:20"
  "dense_trotter --lx 2 --ly 2 --states 3 --stage 0.10:10 --stage 0.03:20"
  "dense_exact --lx 2 --ly 3 --states 3 --stage 0.10:10 --stage 0.03:20"
  "dense_trotter --lx 2 --ly 3 --states 3 --stage 0.10:10 --stage 0.03:20"
  "dense_exact --lx 3 --ly 3 --states 3 --stage 0.08:10 --stage 0.02:20"
  "dense_trotter --lx 3 --ly 3 --states 3 --stage 0.08:10 --stage 0.02:20"
  "peps_full --lx 2 --ly 2 --iterations 3 --tau 0.01 --abort-bond 256"
  "peps_local --lx 2 --ly 2 --stage 0.05:5 --stage 0.02:10 --chi 8 --eta 4 --gate-bond 8 --absorption-bond 8"
  "peps_sketch --lx 2 --ly 2 --stage 0.05:5 --stage 0.02:10 --ell 8 --eta 4 --kappa 2 --chi-sk 4 --gate-bond 8 --absorption-bond 8"
  "peps_full --lx 2 --ly 3 --iterations 2 --tau 0.02 --abort-bond 256"
  "peps_local --lx 2 --ly 3 --stage 0.05:5 --stage 0.02:10 --chi 8 --eta 4 --gate-bond 8 --absorption-bond 8"
  "peps_sketch --lx 2 --ly 3 --stage 0.05:5 --stage 0.02:10 --ell 8 --eta 4 --kappa 2 --chi-sk 4 --gate-bond 8 --absorption-bond 8"
  "peps_full --lx 3 --ly 3 --iterations 1 --tau 0.01 --abort-bond 256"
  "peps_local --lx 3 --ly 3 --stage 0.03:5 --stage 0.01:5 --chi 8 --eta 4 --gate-bond 8 --absorption-bond 8"
  "peps_sketch --lx 3 --ly 3 --stage 0.03:5 --stage 0.01:5 --ell 8 --eta 4 --kappa 2 --chi-sk 4 --gate-bond 8 --absorption-bond 8"
)

read -r -a RUN_ARGS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODE=${RUN_ARGS[0]}
RUN_ARGS=("${RUN_ARGS[@]:1}")
OUT="outputs/physics_loop/small-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.jsonl"
CFS_OUT=/global/cfs/cdirs/m4926/risopeps/outputs/physics_loop

stage_result() {
  if [[ -f "$OUT" ]]; then
    mkdir -p "$CFS_OUT"
    cp "$OUT" "$CFS_OUT/"
  fi
}
trap stage_result EXIT

cd "$SLURM_SUBMIT_DIR"
srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python experiments/physics_loop/run.py --mode "$MODE" "${RUN_ARGS[@]}" \
  --hamiltonian tfim@3.04 --blas-threads 16 --output "$OUT"
