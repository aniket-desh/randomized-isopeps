#!/bin/bash
#SBATCH --job-name=heis-eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
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
K=${K:-1}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"
echo "=== Evaluating properties ==="
echo "  results_root=$RESULTS_DIR, snapshots=$SNAPSHOTS, k=$K"

python heisenberg_evaluate_properties.py \
    --results-root "$RESULTS_DIR" \
    --snapshots "$SNAPSHOTS" \
    --k "$K"

echo "=== Evaluation complete ==="
