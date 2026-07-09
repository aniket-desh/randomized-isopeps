#!/bin/bash
# Lx CEILING (study 2 of the NERSC TODO): push the column size past the Mac's limit and find
# where the node gives out. Full CPU node, staged by the STAGE env var:
#
#   sbatch nersc/jobs/lx_ceiling.slurm                          # STAGE=6 (default)
#   sbatch --export=ALL,STAGE=7 nersc/jobs/lx_ceiling.slurm     # after 6 succeeds
#   sbatch --export=ALL,STAGE=8 -t 24:00:00 nersc/jobs/lx_ceiling.slurm   # the probe
#
# Dense-column sizing (16*2^Lx*8^Lx B): Lx=6 0.27 GB | Lx=7 4.3 GB | Lx=8 69 GB. The dense
# budget is env-tunable (RAND_ISOPEPS_DENSE_MAX_GB, default 2): above it the QR runs
# MATRIX-FREE with the Hutchinson eps estimator. Stage 7 raises it to keep an exact dense
# reference; stage 8 raises it to 220 so the probe genuinely materializes -- TIMEOUT/OOM
# there IS the measurement. TEBD prep is the other axis: ~3 GB/worker at eta=4,
# ~25 GB/worker at eta=6 (exp09's default prep-eta 6 is fine here on 512 GB).
#
#SBATCH --job-name=lx-ceiling
#SBATCH --account=m4926
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniket4@illinois.edu

module load python
source activate /global/common/software/m4926/rand-isopeps-env

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # floor; scripts re-pin BLAS

cd "$SLURM_SUBMIT_DIR"
STAGE="${STAGE:-6}"
STATES="tfim@3.5 tfim@3.04 heis"          # carry the survey's easy/critical/hard trio upward

case "$STAGE" in
  6)  # first beyond-Mac point: dense everywhere, prep@eta6 ~25 GB/worker -> 12 workers
      srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
          --lxs 4 5 6 --states $STATES --seeds 4 --workers 12 --blas-threads 2
      srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py \
          --lxs 4 5 6 --states $STATES --kappa 2 --seeds 4 \
          --workers 12 --cut-workers 4 --blas-threads 1
      ;;
  7)  # dense reference at Lx=7 (4.3 GB col -> ~13 GB peak/trial + prep) -> 10 workers
      export RAND_ISOPEPS_DENSE_MAX_GB=16
      srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
          --lxs 6 7 --states $STATES --seeds 3 --workers 10 --blas-threads 4
      srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py \
          --lxs 6 7 --states $STATES --kappa 2 --seeds 3 \
          --workers 8 --cut-workers 4 --blas-threads 1
      ;;
  8)  # THE PROBE: dense Lx=8 column (69 GB, ~210 GB peak) -- one trial, whole node's BLAS.
      # Submit with -t 24:00:00. A TIMEOUT or OOM here is a result, not a failure.
      export RAND_ISOPEPS_DENSE_MAX_GB=220
      srun -n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python experiments/column_sketch/scripts/exp09_end_to_end_propagated_cost.py \
          --lxs 8 --states tfim@3.5 --seeds 1 --workers 1 --blas-threads 128
      ;;
  *)  echo "unknown STAGE=$STAGE (want 6|7|8)" >&2; exit 1 ;;
esac

# Stage to CFS (purge-safe); then: bash nersc/publish.sh
CFS_OUT="/global/cfs/cdirs/m4926/risopeps/outputs"
mkdir -p "$CFS_OUT" && cp -ru outputs/. "$CFS_OUT"/ 2>/dev/null || true
