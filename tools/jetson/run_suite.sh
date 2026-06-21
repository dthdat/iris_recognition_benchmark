#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${1:?usage: run_suite.sh BENCH_DIR}"
TOOLS_DIR="${2:?usage: run_suite.sh BENCH_DIR TOOLS_DIR}"
CHECKPOINT="${CHECKPOINT:-$HOME/Documents/thesis/best_model.pth}"
INPUTS_NPY="${INPUTS_NPY:-$BENCH_DIR/deployment_inputs.npy}"
FP32_ENGINE="${FP32_ENGINE:-$BENCH_DIR/engines/iris_iresnet50_msff_fp32.engine}"
FP16_ENGINE="${FP16_ENGINE:-$BENCH_DIR/engines/iris_iresnet50_msff_fp16.engine}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
TRIAL_SECONDS="${TRIAL_SECONDS:-120}"
TRIALS="${TRIALS:-5}"
BACKENDS="${BACKENDS:-pytorch_fp32 tensorrt_fp32 tensorrt_fp16}"

mkdir -p "$BENCH_DIR/raw" "$BENCH_DIR/parity"
if [[ "${SKIP_POWER_SETUP:-0}" != "1" ]]; then
  sudo -n true || { echo "Run sudo -v before this script." >&2; exit 2; }
  sudo nvpmodel -m 0
  sudo jetson_clocks
fi

cool_down() {
  local attempts=0
  while true; do
    local raw
    raw="$(cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null || echo 0)"
    if [[ "$raw" -lt 50000 ]]; then
      return
    fi
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge 120 ]]; then
      echo "Jetson did not cool below 50C within 20 minutes." >&2
      exit 3
    fi
    sleep 10
  done
}

run_backend() {
  local backend="$1"
  local artifact_arg="$2"
  cool_down
  local tegra_log="$BENCH_DIR/raw/${backend}_tegrastats.log"
  local window="$BENCH_DIR/raw/${backend}_window.json"
  : > "$tegra_log"
  tegrastats --stop 2>/dev/null || true
  stdbuf -oL tegrastats --interval 500 | while IFS= read -r line; do
    printf '%s|%s\n' "$(date +%s.%N)" "$line"
  done >> "$tegra_log" &
  local tegra_pid=$!
  local idle_start active_start active_end
  idle_start="$(date +%s.%N)"
  sleep 60
  active_start="$(date +%s.%N)"
  python3 "$TOOLS_DIR/benchmark_backend.py" \
    --backend "$backend" \
    $artifact_arg \
    --output "$BENCH_DIR/raw/${backend}.json" \
    --parity-output "$BENCH_DIR/parity/${backend}.npy" \
    --inputs-npy "$INPUTS_NPY" \
    --warmup-seconds "$WARMUP_SECONDS" \
    --trial-seconds "$TRIAL_SECONDS" \
    --trials "$TRIALS"
  active_end="$(date +%s.%N)"
  kill "$tegra_pid" 2>/dev/null || true
  wait "$tegra_pid" 2>/dev/null || true
  tegrastats --stop 2>/dev/null || true
  printf '{"idle_start": %s, "active_start": %s, "active_end": %s}\n' \
    "$idle_start" "$active_start" "$active_end" > "$window"
}

for backend in $BACKENDS; do
  case "$backend" in
    pytorch_fp32)
      run_backend pytorch_fp32 "--checkpoint $CHECKPOINT --artifact $CHECKPOINT"
      ;;
    tensorrt_fp32)
      run_backend tensorrt_fp32 "--engine $FP32_ENGINE --artifact $FP32_ENGINE"
      ;;
    tensorrt_fp16)
      run_backend tensorrt_fp16 "--engine $FP16_ENGINE --artifact $FP16_ENGINE"
      ;;
    *)
      echo "Unknown backend: $backend" >&2
      exit 4
      ;;
  esac
done
