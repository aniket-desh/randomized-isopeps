#!/bin/bash
# Submit every stage of paper/NEXT_EXPERIMENTS.md. Calibration can run alongside
# the main chain because it writes a separate suite. Exact -> saturation ->
# scaling are serialized because they append to one conflict-safe CSV. ``afterany``
# lets later stages preserve and extend partial data even if an earlier ceiling
# times out; failures remain explicit in sacct and the published logs.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

calibration_id="$(sbatch --parsable nersc/jobs/decisive_score_calibration.slurm)"
exact_id="$(sbatch --parsable nersc/jobs/decisive_column_exact.slurm)"
saturation_id="$(sbatch --parsable --dependency="afterany:${exact_id}" \
  nersc/jobs/decisive_column_saturation.slurm)"
scaling_id="$(sbatch --parsable --dependency="afterany:${saturation_id}" \
  nersc/jobs/decisive_column_scaling.slurm)"

echo "calibration: $calibration_id"
echo "exact:       $exact_id"
echo "saturation:  $saturation_id (afterany:$exact_id)"
echo "scaling:     $scaling_id (afterany:$saturation_id)"
echo "monitor:     squeue --me"
echo "accounting:  sacct -j ${calibration_id},${exact_id},${saturation_id},${scaling_id} --format=JobID,JobName,State,Elapsed,ExitCode"
