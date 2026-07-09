#!/bin/bash
#SBATCH --job-name=heis-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute_full_node
#SBATCH --account=rrg-aspuru
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketrd

# ── Config ──────────────────────────────────────────────────────
ROWS=${ROWS:-4}
COLS=${COLS:-4}
TRAIN_SIZE="${ROWS}x${COLS}"
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1000}
ITERATIONS=${ITERATIONS:-100}
TF_ARCH=${TF_ARCH:-transformer_l4_d128_h4}
GCN_ARCH=${GCN_ARCH:-gcn_proj_3_16}
SEED=${SEED:-0}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"

echo "=== Training GCN-Transformer on ${TRAIN_SIZE} Heisenberg ==="
echo "  arch=$TF_ARCH, gcn=$GCN_ARCH, samples=$TRAIN_SAMPLES, iters=$ITERATIONS"

python heisenberg_train_transformer.py \
    --train-size "$TRAIN_SIZE" \
    --train-samples "$TRAIN_SAMPLES" \
    --iterations "$ITERATIONS" \
    --tf-arch "$TF_ARCH" \
    --gcn-arch "$GCN_ARCH" \
    --seed "$SEED"

echo "=== Training complete ==="
