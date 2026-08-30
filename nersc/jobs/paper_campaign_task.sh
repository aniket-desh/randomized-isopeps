#!/bin/bash

run_paper_campaign_task() {
  local expected_hardware="${1:?pass cpu or gpu}"
  : "${MANIFEST:?set MANIFEST to a paper campaign jsonl manifest}"
  : "${OUTPUT_ROOT:?set OUTPUT_ROOT to a durable cfs directory}"
  : "${SLURM_ARRAY_TASK_ID:?run this script as a slurm array}"
  : "${SLURM_ARRAY_JOB_ID:?run this script as a slurm array}"
  : "${SLURM_JOB_ID:?run this script inside a slurm allocation}"
  : "${SLURM_CPUS_PER_TASK:?request cpus per task}"
  : "${PSCRATCH:?PSCRATCH is unavailable}"

  local repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}}"
  repo_root="$(cd "$repo_root" && pwd -P)"
  if [[ "$MANIFEST" != /* ]]; then
    MANIFEST="$repo_root/$MANIFEST"
  fi
  [[ -f "$MANIFEST" ]] || { echo "manifest does not exist: $MANIFEST" >&2; return 2; }
  case "$OUTPUT_ROOT" in
    /global/cfs/cdirs/*) ;;
    *) echo "OUTPUT_ROOT must be a durable cfs path" >&2; return 2 ;;
  esac

  mkdir -p "$OUTPUT_ROOT"
  local run_root="$PSCRATCH/rand_isopeps/paper_campaign/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
  local checkpoint_root="${CHECKPOINT_ROOT:-$PSCRATCH/rand_isopeps/paper_campaign/checkpoints}"
  mkdir -p "$run_root/tmp" "$run_root/cache" "$run_root/cupy" "$checkpoint_root"
  cd "$run_root"

  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  export TMPDIR="$run_root/tmp"
  export XDG_CACHE_HOME="$run_root/cache"
  export MPLCONFIGDIR="$run_root/cache/matplotlib"
  export CUPY_CACHE_DIR="$run_root/cupy"
  export PYTHONUNBUFFERED=1
  export QUIMB_NUMBA_CACHE=False
  local physical_threads=$(( (SLURM_CPUS_PER_TASK + 1) / 2 ))
  export OMP_NUM_THREADS="$physical_threads"
  export OPENBLAS_NUM_THREADS="$physical_threads"
  export MKL_NUM_THREADS="$physical_threads"

  local task_hardware
  task_hardware="$(python -c 'from rand_isopeps.campaign import select_task; import sys; print(select_task(sys.argv[1], int(sys.argv[2]))["resources"]["hardware"])' "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
  if [[ "$task_hardware" != "$expected_hardware" ]]; then
    echo "task requests $task_hardware hardware, but this is the $expected_hardware launcher" >&2
    return 2
  fi

  local srun_args=(-n 1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores)
  if [[ "$expected_hardware" == "gpu" ]]; then
    srun_args+=(--gpus-per-task=1)
  fi

  export STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-1800}"
  # scheduler variables expand inside the launched allocation
  # shellcheck disable=SC2016
  local launch=': "${SLURM_JOB_END_TIME:?SLURM_JOB_END_TIME is unavailable}"
stop_after=$((SLURM_JOB_END_TIME - $(date +%s) - STOP_GRACE_SECONDS))
(( stop_after > 0 )) || exit 75
exec python "$@" --stop-after-seconds "$stop_after"'

  set +e
  srun "${srun_args[@]}" bash -c "$launch" bash \
    "$repo_root/experiments/paper_campaign/run_task.py" "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" --checkpoint-root "$checkpoint_root"
  local status=$?
  set -e
  if (( status == 75 )); then
    scontrol requeue "$SLURM_JOB_ID"
    return 0
  fi
  return "$status"
}
