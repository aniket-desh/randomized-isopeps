#!/bin/bash
#SBATCH --job-name=heis-sample
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute_full_node
#SBATCH --account=rrg-aspuru
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketrd

# ── Config ──────────────────────────────────────────────────────
RESULTS_DIR=${RESULTS_DIR:?Set RESULTS_DIR to the training results directory}
SNAPSHOTS=${SNAPSHOTS:-20000}
SEED=${SEED:-42}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"
echo "=== Sampling from trained model ==="
echo "  results_dir=$RESULTS_DIR, snapshots=$SNAPSHOTS"

python heisenberg_sample_transformer.py \
    --results-dir "$RESULTS_DIR" \
    --snapshots "$SNAPSHOTS" \
    --seed "$SEED"

echo "=== Sampling complete ==="
