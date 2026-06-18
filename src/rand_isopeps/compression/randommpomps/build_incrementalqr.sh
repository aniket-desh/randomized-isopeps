#!/usr/bin/env bash
# Build the optional C++ incremental-QR pybind11 extension used by SRC
# (random_contraction_inc). Mirrors RandomMPOMPS/setup_QR.sh but without the
# upstream's hardcoded conda/homebrew paths: it auto-detects pybind11, the
# Python extension suffix, and an OpenBLAS install that provides cblas.h +
# lapacke.h. Produces libincrementalqr<EXT_SUFFIX> next to this script.
#
# Usage:  bash rand_isopeps/randommpomps/build_incrementalqr.sh
# The pure-Python fallback is used automatically if this is not built.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/incrementalqr.cpp"

PYBIND_INCLUDES="$(python3 -m pybind11 --includes)"
EXT_SUFFIX="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="$DIR/libincrementalqr${EXT_SUFFIX}"

# Locate OpenBLAS (provides cblas.h, lapacke.h, and LAPACK/BLAS symbols).
if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
    OPENBLAS="$(brew --prefix openblas)"
elif [ -d /opt/homebrew/opt/openblas ]; then
    OPENBLAS=/opt/homebrew/opt/openblas
elif [ -d /usr/local/opt/openblas ]; then
    OPENBLAS=/usr/local/opt/openblas
else
    OPENBLAS="${OPENBLAS:-/usr}"
fi
echo "pybind11 includes: ${PYBIND_INCLUDES}"
echo "ext suffix:        ${EXT_SUFFIX}"
echo "openblas prefix:   ${OPENBLAS}"

CXX="${CXX:-c++}"
COMMON=(-O3 -Wall -std=c++14 -fPIC ${PYBIND_INCLUDES} -I"${OPENBLAS}/include" "${SRC}" -L"${OPENBLAS}/lib" -lopenblas -o "${OUT}")

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: bundle with dynamic_lookup for Python symbols; rpath to libopenblas.
    "${CXX}" -shared -undefined dynamic_lookup -Wl,-rpath,"${OPENBLAS}/lib" "${COMMON[@]}"
else
    "${CXX}" -shared -Wl,-rpath,"${OPENBLAS}/lib" "${COMMON[@]}"
fi

echo "built: ${OUT}"
python3 -c "import sys; sys.path.insert(0, '${DIR}'); import libincrementalqr; print('verify: libincrementalqr imports OK ->', [f for f in dir(libincrementalqr) if not f.startswith('__')])"
