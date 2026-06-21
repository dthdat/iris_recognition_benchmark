#!/usr/bin/env bash
set -u

KAGGLE_USER="${KAGGLE_USER:-datdasdasd}"
DATASET_SOURCE="${DATASET_SOURCE:-sondosaabed/casia-iris-thousand}"
DATASET_ROOT="${DATASET_ROOT:-/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand/CASIA-Iris-Thousand}"
ACCELERATOR="${ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${TIMEOUT:-21600}"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"

RUNS=(
  "b1_arciris_nomask"
  "b3_arciris_softmask"
  "b4_mobilenet_softmask"
  "ours_iresnet_msff_softmask"
)

mkdir -p results logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a logs/auto_kaggle_queue.log
}

status_run() {
  local run_id="$1"
  python tools/kaggle_status.py --run-id "$run_id" --kaggle-user "$KAGGLE_USER" 2>&1
}

submit_run() {
  local run_id="$1"
  log "Submitting $run_id"
  python tools/kaggle_submit.py \
    --run-id "$run_id" \
    --kaggle-user "$KAGGLE_USER" \
    --dataset-source "$DATASET_SOURCE" \
    --dataset-root "$DATASET_ROOT" \
    --accelerator "$ACCELERATOR" \
    --timeout "$TIMEOUT"
}

collect_run() {
  local run_id="$1"
  log "Collecting $run_id"
  python tools/kaggle_collect.py \
    --run-id "$run_id" \
    --kaggle-user "$KAGGLE_USER"
}

verify_splits() {
  for f in \
    splits/train_subjects.csv \
    splits/val_subjects.csv \
    splits/test_subjects.csv \
    splits/train_images.csv \
    splits/val_images.csv \
    splits/test_images.csv
  do
    if [ ! -s "$f" ]; then
      log "Missing split file: $f"
      return 1
    fi
  done
  return 0
}

wait_for_run() {
  local run_id="$1"

  while true; do
    out="$(status_run "$run_id")"
    rc=$?

    echo "$out" | tee "logs/status_${run_id}.log"

    if [ $rc -ne 0 ]; then
      log "Status command failed for $run_id. Probably temporary Kaggle/network issue. Retrying after sleep."
      sleep "$SLEEP_SECONDS"
      continue
    fi

    lower="$(echo "$out" | tr '[:upper:]' '[:lower:]')"

    if echo "$lower" | grep -Eq "complete|completed|success|succeeded"; then
      log "$run_id appears completed."
      return 0
    fi

    if echo "$lower" | grep -Eq "fail|failed|error|cancel"; then
      log "$run_id appears failed. Stopping queue."
      return 1
    fi

    log "$run_id still running/queued. Sleeping ${SLEEP_SECONDS}s."
    sleep "$SLEEP_SECONDS"
  done
}

for run_id in "${RUNS[@]}"; do
  log "======== RUN: $run_id ========"

  if [ "$run_id" != "b1_arciris_nomask" ]; then
    if ! verify_splits; then
      log "b1 splits not present. Refusing to start $run_id."
      exit 2
    fi
  fi

  submit_run "$run_id"
  if ! wait_for_run "$run_id"; then
    log "Run failed: $run_id"
    {
      echo "# Monitor report"
      echo
      echo "- Failed run: $run_id"
      echo "- Time: $(date)"
      echo "- Last status log: logs/status_${run_id}.log"
      echo "- Action: queue stopped. Inspect Kaggle logs and patch manually."
    } > results/MONITOR_REPORT.md
    exit 1
  fi

  collect_run "$run_id"

  if [ "$run_id" = "b1_arciris_nomask" ]; then
    if ! verify_splits; then
      log "b1 completed but split CSVs missing. Stopping queue."
      exit 3
    fi
    log "b1 split CSVs verified. Remaining runs may continue."
  fi

  log "Finished $run_id"
done

log "All planned runs completed."
python experiments/aggregate_results.py || true

{
  echo "# Monitor report"
  echo
  echo "- Status: all planned Kaggle runs completed or collection attempted."
  echo "- Time: $(date)"
  echo "- Runs:"
  printf '  - %s\n' "${RUNS[@]}"
} > results/MONITOR_REPORT.md
