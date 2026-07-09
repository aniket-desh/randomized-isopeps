#!/bin/bash
# ESCAPE HATCH: pull RAW experiment data from NERSC to THIS machine over rsync. RUN LOCALLY.
#
#   bash nersc/pull_data.sh
#
# Normally you DON'T need this: publish.sh already puts figures + the (small) result CSVs on
# the 'results' branch, so a plain `git fetch origin results` gives the local agent everything
# -- no NERSC login, no MFA. Use this script only when a run is too big for git (publish.sh
# skipped the data because outputs/ > DATA_MAX_MB). It rsyncs from CFS to ./nersc-data/
# (gitignored) with a MAX_GB guard (default 1 GB, FORCE=1 to override). Needs an authenticated
# NERSC session (rsync over ssh -> MFA, or a live ~24 h sshproxy cert). Make sure the job
# staged results to CFS (the `cp -ru outputs/. $CFS_OUT/` line in the templates).
set -euo pipefail

NERSC_USER="${NERSC_USER:-aniketd}"
REMOTE_HOST="${REMOTE_HOST:-perlmutter.nersc.gov}"
# Where jobs stage outputs on CFS (matches the templates CFS_OUT):
REMOTE_DATA="${REMOTE_DATA:-/global/cfs/cdirs/m4926/risopeps/outputs/}"
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
