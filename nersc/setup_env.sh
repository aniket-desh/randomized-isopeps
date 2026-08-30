#!/bin/bash
# create the shared environment on a login node, where pip and conda have network access.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your NERSC project code, e.g. m1234}"
ENV_PREFIX="/global/common/software/${PROJECT}/rand-isopeps-env"

module load python cudatoolkit/12.9                   # nersc's conda base and cuda 12
# load conda's shell functions for this non-interactive script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y --prefix "$ENV_PREFIX" \
      python=3.11 pip numpy scipy matplotlib threadpoolctl
fi
conda activate "$ENV_PREFIX"

# install quimb and the cuda 12 cupy wheel on the login node.
pip install -e ".[quimb,gpu,riemannian]"

echo
echo "=== env ready ==="
echo "  prefix: $ENV_PREFIX"
echo "  in job scripts:  module load python && source activate $ENV_PREFIX"
