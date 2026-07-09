#!/bin/bash
#SBATCH --job-name=heis-plot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:15:00
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
SPLIT=${SPLIT:-test}
MODEL_PROPS_DIR=${MODEL_PROPS_DIR:-}

# ── Environment ─────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 scipy-stack/2024a
source "$HOME/envs/gqs/bin/activate"

WORKDIR="$SCRATCH/generative-quantum-states"
cd "$WORKDIR"
echo "=== Generating plots ==="

PLOT_CMD="python heisenberg_plot_data.py --rows $ROWS --cols $COLS --split $SPLIT --save"

if [ -n "$MODEL_PROPS_DIR" ]; then
    PLOT_CMD="$PLOT_CMD --model-props-dir $MODEL_PROPS_DIR"
fi

eval "$PLOT_CMD"

echo "=== Plotting complete ==="
