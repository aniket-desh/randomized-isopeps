#!/bin/bash
# Submit every stage of paper/NEXT_EXPERIMENTS.md. Calibration can run alongside
# the main chain because it writes a separate suite. Exact -> saturation ->
# scaling are serialized because they append to one conflict-safe CSV. ``afterok``
# prevents a downstream stage from spending allocation when its prerequisite
# failed; partial rows are persisted and a failed stage can be resumed explicitly.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

calibration_id="$(sbatch --parsable nersc/jobs/decisive_score_calibration.slurm)"
smoke_id="$(sbatch --parsable nersc/jobs/decisive_spawn_smoke.slurm)"
exact_id="$(sbatch --parsable --dependency="afterok:${smoke_id}" \
  nersc/jobs/decisive_column_exact.slurm)"
saturation_id="$(sbatch --parsable --dependency="afterok:${exact_id}" \
  nersc/jobs/decisive_column_saturation.slurm)"
scaling_id="$(sbatch --parsable --dependency="afterok:${saturation_id}" \
  nersc/jobs/decisive_column_scaling.slurm)"

echo "calibration: $calibration_id"
echo "spawn smoke: $smoke_id"
echo "exact:       $exact_id (afterok:$smoke_id)"
echo "saturation:  $saturation_id (afterok:$exact_id)"
echo "scaling:     $scaling_id (afterok:$saturation_id)"
echo "monitor:     squeue --me"
echo "accounting:  sacct -j ${calibration_id},${smoke_id},${exact_id},${saturation_id},${scaling_id} --format=JobID,JobName,State,Elapsed,ExitCode"
