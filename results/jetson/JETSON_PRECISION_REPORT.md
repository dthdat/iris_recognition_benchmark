# Jetson Nano B01 Precision Comparison

Measured on the same notebook IResNet50-MSFF model at batch size 1 in MAXN mode.

| Backend | Artifact (MiB) | Network median (ms) | End-to-end median (ms) | p95 (ms) | Throughput (FPS) | Active power (W) | Energy (mJ/inf) | Cosine vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch_fp32 | 170.60 | 168.553 | 250.436 | 169.116 | 5.925 | n/a | n/a | 1.000000 |
| tensorrt_fp32 | 290.12 | 126.331 | 208.215 | 126.862 | 7.900 | n/a | n/a | 1.000000 |
| tensorrt_fp16 | 145.54 | 76.339 | 158.223 | 76.702 | 13.059 | n/a | n/a | 0.999940 |

## Speedups

- TensorRT FP32 vs PyTorch FP32: **1.33x**
- TensorRT FP16 vs TensorRT FP32: **1.65x**
- TensorRT FP16 vs PyTorch FP32: **2.21x**
- Shared preprocessing median: **81.883 ms** across 127 valid deployment images (3 failures).

## Unsupported precision modes

The device reports fast FP16 support and no fast INT8 support. TensorRT 8.2 on the Nano does not provide FP8 or FP4 execution. Those modes are reported as unsupported rather than assigned fabricated measurements.

## Accuracy boundary

This report validates numerical embedding parity on deterministic inputs. The original notebook records test EER 3.864%, AUC 0.9918, and TAR@0.1% FAR 84.56%. Quantized CASIA metrics require the frozen image split and are not inferred from latency or parity measurements.
