#!/bin/bash
# Publish curated FIGURES to the 'results' branch so you can view them locally (or on
# GitHub). RUN ON A LOGIN NODE (needs internet). Figures only -- raw data never enters git;
# pull that with nersc/pull_data.sh (see nersc/README.md §11).
#
#   bash nersc/publish.sh ["commit message"]
#
# Uses a separate git worktree for the 'results' branch, so the checkout you run jobs from
# is never disturbed. First run creates the branch (orphan); later runs append a snapshot.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run publish.sh on a LOGIN node -- compute nodes have no internet." >&2
  exit 1
fi

MSG="${1:-results: $(date -u +%Y-%m-%dT%H:%MZ) on $(hostname -s)}"
WT="$(dirname "$ROOT")/rand-isopeps-results"   # results worktree, sibling of the main checkout

# 1) Regenerate the curated, tracked figures from the latest run outputs.
echo "== regenerating curated figures =="
python experiments/column_sketch/scripts/curate_figures.py \
  || echo "!! curate_figures failed; publishing existing reports/figures as-is"

# 2) Ensure a worktree bound to 'results' exists (create the branch on first run).
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

# 3) Sync ONLY figures into the results worktree (never the raw outputs/ CSVs).
rsync -a --delete "$ROOT/reports/figures/" "$WT/figures/"

# 4) Commit + push (quiet no-op if nothing changed).
git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet; then
  echo "== no figure changes to publish =="
  exit 0
fi
git -C "$WT" commit -q -m "$MSG"
git -C "$WT" push origin results
echo "== published figures -> origin/results =="
echo "   view locally:  git fetch origin results && git checkout origin/results -- ."
