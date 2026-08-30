#!/bin/bash
# zero-based paper campaign array with one gpu trajectory per allocation
#SBATCH --job-name=paper-gpu
#SBATCH --account=m4926_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --requeue
#SBATCH --licenses=scratch,cfs
#SBATCH --open-mode=append
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail
module load python cudatoolkit/12.9
# the shared environment is created by setup_env.sh on perlmutter
# shellcheck disable=SC1091
source activate "${ENV_PREFIX:-/global/common/software/m4926/rand-isopeps-env}"

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}"
# the deployment root supplies the shared array launcher
# shellcheck disable=SC1091
source "$repo_root/nersc/jobs/paper_campaign_task.sh"
run_paper_campaign_task gpu
