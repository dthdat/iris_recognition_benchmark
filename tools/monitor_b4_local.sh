#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="b4_mobilenet_softmask"
KAGGLE_USER="datdasdasd"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/monitor_b4_local.log"
LOCK_FILE="$LOG_DIR/monitor_b4_local.lock"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '[%s] b4 monitor is already running\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG_FILE"
    exit 0
fi

printf '[%s] started b4 monitor (interval=%ss)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$INTERVAL_SECONDS" >>"$LOG_FILE"

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    status="$(python tools/kaggle_status.py --run-id "$RUN_ID" --kaggle-user "$KAGGLE_USER" 2>&1)"
    status_rc=$?
    printf '[%s] rc=%s %s\n' "$timestamp" "$status_rc" "$status" >>"$LOG_FILE"

    if printf '%s' "$status" | grep -q 'KernelWorkerStatus.COMPLETE'; then
        printf '[%s] terminal success; collecting outputs\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG_FILE"
        if python tools/kaggle_collect.py --run-id "$RUN_ID" --kaggle-user "$KAGGLE_USER" >>"$LOG_FILE" 2>&1; then
            metrics="$ROOT/runs/$RUN_ID/test_metrics.json"
            if [[ -f "$metrics" ]]; then
                printf '[%s] verified %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$metrics" >>"$LOG_FILE"
                exit 0
            fi
            printf '[%s] ERROR: test_metrics.json is missing after collection\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG_FILE"
            exit 1
        fi
        printf '[%s] ERROR: output collection failed\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG_FILE"
        exit 1
    fi

    if printf '%s' "$status" | grep -Eq 'KernelWorkerStatus\.(ERROR|CANCELLED)|FAILED'; then
        printf '[%s] terminal failure; collecting logs and stopping\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG_FILE"
        python tools/kaggle_collect.py --run-id "$RUN_ID" --kaggle-user "$KAGGLE_USER" >>"$LOG_FILE" 2>&1 || true
        exit 1
    fi

    sleep "$INTERVAL_SECONDS"
done
