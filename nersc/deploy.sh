#!/bin/bash
# Deploy the latest code onto NERSC. RUN ON A LOGIN NODE (needs internet; compute nodes
# have none). Idempotent: clones the repo into $PSCRATCH if it isn't there (e.g. after a
# scratch purge), otherwise fast-forwards it. Reinstalls only if dependencies changed.
#
#   ssh aniketd@perlmutter.nersc.gov
#   bash ~/randomized-isopeps/nersc/deploy.sh [branch]     # default branch: main
#
# Override the checkout location with RISOPEPS_DIR (default: $PSCRATCH/randomized-isopeps,
# so big outputs land on the fast/large filesystem -- see nersc/README.md §2).
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run deploy.sh on a LOGIN node -- compute nodes have no internet." >&2
  exit 1
fi

REPO_URL="${RISOPEPS_URL:-https://github.com/aniket-desh/randomized-isopeps}"
# Default to $PSCRATCH (big outputs on the fast FS); clean error if run off Perlmutter.
WORKDIR="${RISOPEPS_DIR:-${PSCRATCH:?set PSCRATCH (on Perlmutter) or pass RISOPEPS_DIR}/randomized-isopeps}"
BRANCH="${1:-main}"

if [ ! -d "$WORKDIR/.git" ]; then
  echo "== cloning $REPO_URL -> $WORKDIR (fresh / purged) =="
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "== deploying '$BRANCH' into $WORKDIR =="
before="$(git rev-parse HEAD 2>/dev/null || echo none)"
git fetch --prune origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
after="$(git rev-parse HEAD)"

# Re-point the env's EDITABLE install at THIS checkout. Without this, `import rand_isopeps`
# resolves to wherever setup_env.sh last ran pip (e.g. a stale $HOME clone) and jobs run old
# library code against new scripts. --no-deps makes it a fast (~s) idempotent re-link; full
# dependency resolution only when pyproject.toml actually moved.
if [ "$before" != "$after" ] && git diff --name-only "$before" "$after" | grep -qx 'pyproject.toml'; then
  echo "== pyproject.toml changed -> refresh quimb, gpu, and riemannian extras =="
  pip install -e ".[quimb,gpu,riemannian]"
else
  pip install -e . --no-deps --quiet
fi

echo "== ready @ ${after:0:9}  ($WORKDIR) =="
echo "   next:  sbatch nersc/templates/cpu_job.slurm   (or array_sweep.slurm)"
