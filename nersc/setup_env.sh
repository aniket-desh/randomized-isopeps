#!/bin/bash
# One-time environment setup for randomized-isopeps on NERSC Perlmutter.
#
# RUN ON A LOGIN NODE (compute nodes have no outbound internet, so pip/conda must
# happen here). From the repo root:
#
#     PROJECT=mXXXX bash nersc/setup_env.sh
#
# Builds the conda env on the *Common* filesystem (fast parallel FS + big quota;
# a torch/quimb env would blow your $HOME quota). Every job script then just does
#     module load python && conda activate "$ENV_PREFIX"
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your NERSC project code, e.g. m1234}"
ENV_PREFIX="/global/common/software/${PROJECT}/rand-isopeps-env"

module load python                                    # NERSC's Anaconda base

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y --prefix "$ENV_PREFIX" \
      python=3.11 pip numpy scipy matplotlib threadpoolctl
fi
# `conda activate` needs conda's shell hook; `module load python` sets it up.
conda activate "$ENV_PREFIX"

# Editable install of the package. The ".[quimb]" extra pulls quimb+autoray for the
# real isoTNS block Moses-move experiments; drop it if you only need synthetic kernels.
pip install -e ".[quimb]"

echo
echo "=== env ready ==="
echo "  prefix: $ENV_PREFIX"
echo "  in job scripts:  module load python && conda activate $ENV_PREFIX"
