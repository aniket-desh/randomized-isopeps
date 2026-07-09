#!/bin/bash
#SBATCH --job-name=heis-datagen
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --account=rrg-aspuru
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketrd

# ── Config ──────────────────────────────────────────────────────
ROWS=${ROWS:-4}
COLS=${COLS:-4}
NH_TRAIN=${NH_TRAIN:-80}
NH_TEST=${NH_TEST:-20}
SHOTS=${SHOTS:-1000}
SEED=${SEED:-42}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

# Work from scratch (compute nodes can only write to $SCRATCH)
WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"

echo "=== Generating ${ROWS}x${COLS} Heisenberg data ==="
echo "  nh_train=$NH_TRAIN, nh_test=$NH_TEST, shots=$SHOTS"

python heisenberg_generate_data.py \
    --rows "$ROWS" \
    --cols "$COLS" \
    --nh_train "$NH_TRAIN" \
    --nh_test "$NH_TEST" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --device default.qubit

echo "=== Data generation complete ==="
