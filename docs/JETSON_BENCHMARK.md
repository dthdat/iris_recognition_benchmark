# Jetson Nano B01 benchmark

This benchmark compares the notebook's trained IResNet50-MSFF embedding model in three deployment modes without retraining:

1. PyTorch FP32
2. TensorRT FP32
3. TensorRT FP16

The scripts under `tools/jetson/` are Python 3.6 compatible where they execute on JetPack 4.6. Engines, checkpoints, ONNX files, test images, SSH material, and raw logs stay outside Git.

## Reproduction contract

- Device: Jetson Nano B01, MAXN, fixed clocks, active cooling.
- Shape: `1x1x64x512` input and 512-D embedding output.
- Batch/streams: batch 1, one stream.
- Timing: 60-second idle window, 30-second warm-up, five 120-second trials per backend.
- Correctness: deterministic seed-42 tensors; PyTorch FP32 is the numerical reference.
- Power: `tegrastats` rail data when the platform exposes `POM_5V_IN`.
- Thermal rule: begin below 50°C and reject throttled trials.

## Safety

`build_engines.sh` writes versioned engines and preserves existing engines. `run_suite.sh` requires a pre-authorized `sudo` timestamp, selects MAXN/fixed clocks, and never edits the live application configuration. Unsupported INT8/FP8/FP4 modes are documented without fabricated values.
