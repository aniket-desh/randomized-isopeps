#!/bin/bash
# Publish this run's FIGURES + result data + Slurm logs (.out/.err) to the 'results' branch, so
# everything is available locally with a single `git fetch` -- no NERSC login / MFA needed on the
# local side. RUN ON A LOGIN NODE (needs internet; compute nodes have none).
#
#   bash nersc/publish.sh ["commit message"]
#
# The result CSVs are tiny (KB-MB), so they ride git by default. A size guard keeps anything
# huge OUT of git: if outputs/ exceeds DATA_MAX_MB (default 200) it publishes figures only and
# tells you to fetch the data with nersc/pull_data.sh instead. Uses a separate git worktree,
# so the checkout you run jobs from is never disturbed; 'results' is an orphan branch (never
# merged to main -- main stays data-free).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run publish.sh on a LOGIN node -- compute nodes have no internet." >&2
  exit 1
fi

MSG="${1:-results: $(date -u +%Y-%m-%dT%H:%MZ) on $(hostname -s)}"
DATA_MAX_MB="${DATA_MAX_MB:-200}"
WT="$(dirname "$ROOT")/rand-isopeps-results"   # results worktree, sibling of the main checkout

# 1) Regenerate the curated, tracked figures from the latest run outputs. curate_figures imports
#    rand_isopeps, so ensure the env is active -- a fresh login shell won't have it (this is why
#    a bare `python` gave a SyntaxError). Activate only if the import is missing; an already-active
#    env wins. Env prefix is optional (RISOPEPS_ENV overrides the default m4926 prefix).
if ! python -c "import rand_isopeps" >/dev/null 2>&1; then
  module load python >/dev/null 2>&1 || true
  source activate "${RISOPEPS_ENV:-/global/common/software/m4926/rand-isopeps-env}" >/dev/null 2>&1 || true
fi
echo "== regenerating curated figures =="
python experiments/column_sketch/scripts/curate_figures.py \
  || echo "!! curate_figures failed; publishing existing reports/figures as-is"

# 2) Decide whether the raw data is small enough to ride git (compare in KB -- exact).
include_data=1
if [ -d outputs ]; then
  data_kb=$(du -sk outputs | cut -f1)
  max_kb=$(( DATA_MAX_MB * 1024 ))
  data_mb=$(awk -v k="$data_kb" 'BEGIN{printf "%.1f", k/1024}')
  if [ "$data_kb" -gt "$max_kb" ]; then
    include_data=0
    echo "!! outputs/ is ${data_mb} MB > DATA_MAX_MB=${DATA_MAX_MB} MB -> FIGURES ONLY."
    echo "   fetch the raw data locally with nersc/pull_data.sh instead."
  else
    echo "== outputs/ is ${data_mb} MB -> including data on the results branch =="
  fi
else
  include_data=0
fi

# 3) Ensure a worktree bound to 'results' exists (create the orphan branch on first run).
git fetch origin results >/dev/null 2>&1 || true
if ! git worktree list --porcelain | grep -q "^worktree $WT$"; then
  if git show-ref --verify --quiet refs/remotes/origin/results; then
    git worktree add "$WT" results
  else
    echo "== first run: creating orphan 'results' branch =="
    git worktree add --detach "$WT"
    ( cd "$WT" && git checkout --orphan results && git rm -rfq . >/dev/null 2>&1 || true )
  fi
fi

# 4) Sync into the worktree, mirroring repo paths (so local analysis uses the same paths).
#    mkdir first: rsync only creates the FINAL dest component, so a nested dest whose parent
#    ('reports/') is absent on a freshly-wiped orphan branch fails to mkdir. Pre-create both.
mkdir -p "$WT/reports/figures"
rsync -a --delete "$ROOT/reports/figures/" "$WT/reports/figures/"
if [ "$include_data" -eq 1 ]; then
  mkdir -p "$WT/outputs"
  rsync -a --delete "$ROOT/outputs/" "$WT/outputs/"
else
  rm -rf "$WT/outputs"
fi

# 4b) Also carry the Slurm job logs (.out/.err, written to the submit dir = repo root). Small text,
#     but invaluable after the fact: energy trajectories, per-task timings, OOM traces, the reason a
#     task failed. Mirror the current root-level set (RISOPEPS_LOGDIR overrides the search dir).
logdir="${RISOPEPS_LOGDIR:-$ROOT}"
rm -rf "$WT/logs"
if compgen -G "$logdir"/*.out >/dev/null 2>&1 || compgen -G "$logdir"/*.err >/dev/null 2>&1; then
  mkdir -p "$WT/logs"
  find "$logdir" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -exec cp {} "$WT/logs/" \;
  echo "== carrying $(find "$WT/logs" -type f | wc -l | tr -d ' ') Slurm log(s) -> results/logs/ =="
fi

# 5) Commit + push (quiet no-op if nothing changed).
git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet; then
  echo "== no changes to publish =="
  exit 0
fi
git -C "$WT" commit -q -m "$MSG"
git -C "$WT" push origin results
what="figures"; [ "$include_data" -eq 1 ] && what="figures + data"
[ -d "$WT/logs" ] && what="$what + logs"
echo "== published ${what} -> origin/results =="
echo "   locally:  git fetch origin results   (then analyze from the 'results' ref; logs in logs/)"
