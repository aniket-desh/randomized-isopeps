#!/bin/bash
# REFERENCE ONLY. The current rand_isopeps stack is CPU / BLAS-bound (numpy, scipy,
# quimb) — use cpu_job.slurm. Keep this template for when/if you add a GPU backend
# (torch/cupy). It's also the closest analog to the Trillium GPU scripts in
# ../trillium_reference/, so it shows the GPU-side translation.
#
# Note the two GPU-specific differences vs the CPU templates:
#   * account is the "_g" allocation
#   * constraint is gpu, and you must explicitly request GPUs
#
#SBATCH --job-name=risopeps-gpu
#SBATCH --account=mXXXX_g             # ==EDIT== GPU jobs charge to the _g allocation
#SBATCH --constraint=gpu              # Perlmutter GPU node: 1x EPYC (64c) + 4x A100
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python cudatoolkit
conda activate /global/common/software/mXXXX/rand-isopeps-env   # ==EDIT==

cd "$PSCRATCH/randomized-isopeps"
srun -n 1 -c 32 --cpu-bind=cores -G 4 --gpu-bind=none \
  python <your_gpu_experiment>.py                              # ==EDIT==
