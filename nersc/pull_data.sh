#!/bin/bash
# Pull RAW experiment data (the gitignored outputs/ CSVs) from NERSC to THIS machine, so you
# can re-configure / regenerate plots locally. RUN LOCALLY (not on NERSC).
#
#   bash nersc/pull_data.sh
#
# Data never enters git -- this is the out-of-band channel for it. There is a size guard:
# the pull is refused if the remote data exceeds MAX_GB (default 1 GB) unless you FORCE=1.
# Pull from CFS (backed up, not purged); make sure your job staged results there (the
# `cp -r outputs /global/cfs/...` line in the templates).
set -euo pipefail

NERSC_USER="${NERSC_USER:-aniketd}"
REMOTE_HOST="${REMOTE_HOST:-perlmutter.nersc.gov}"
# ==EDIT== the project code (mXXXX). Where your jobs stage outputs on CFS:
REMOTE_DATA="${REMOTE_DATA:-/global/cfs/cdirs/mXXXX/risopeps/outputs/}"
LOCAL_DEST="${LOCAL_DEST:-./nersc-data/}"          # gitignored (see .gitignore)
MAX_GB="${MAX_GB:-1}"

remote="${NERSC_USER}@${REMOTE_HOST}"
echo "== sizing ${remote}:${REMOTE_DATA} =="
bytes="$(ssh "$remote" "du -sb '${REMOTE_DATA}' 2>/dev/null | cut -f1" || echo 0)"
if [ -z "$bytes" ] || [ "$bytes" -eq 0 ]; then
  echo "ERROR: remote path is empty or unreachable: ${remote}:${REMOTE_DATA}" >&2
  echo "       (did the job stage outputs to CFS? check the path / your ssh login)" >&2
  exit 1
fi
max_bytes=$(( MAX_GB * 1000 * 1000 * 1000 ))
human="$(awk -v b="$bytes" 'BEGIN{printf "%.2f", b/1e9}')"
echo "== remote data is ${human} GB (guard: ${MAX_GB} GB) =="
if [ "$bytes" -gt "$max_bytes" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "REFUSING: ${human} GB exceeds ${MAX_GB} GB. Re-run with FORCE=1 to pull anyway," >&2
  echo "          or narrow REMOTE_DATA to a subdir, or keep it on CFS and plot on NERSC." >&2
  exit 1
fi

mkdir -p "$LOCAL_DEST"
echo "== rsync -> ${LOCAL_DEST} =="
rsync -avz --progress "${remote}:${REMOTE_DATA}" "$LOCAL_DEST"
echo "== pulled ${human} GB -> ${LOCAL_DEST} (gitignored). Re-plot locally from here. =="
