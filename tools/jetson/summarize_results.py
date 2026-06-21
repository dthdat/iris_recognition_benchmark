#!/usr/bin/env python3
"""Create compact CSV/Markdown/JSON summaries from Jetson raw results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


BACKENDS = ["pytorch_fp32", "tensorrt_fp32", "tensorrt_fp16"]
POWER_RE = re.compile(r"POM_5V_IN\s+(\d+)/(\d+)")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
TEMP_RE = re.compile(r"(?:GPU|CPU)@([0-9.]+)C")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tegra(path: Path, window: dict):
    idle_power = []
    active_power = []
    active_ram = []
    active_temp = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            stamp_text, payload = line.split("|", 1)
            stamp = float(stamp_text)
        except ValueError:
            continue
        power_match = POWER_RE.search(payload)
        power = float(power_match.group(1)) if power_match else None
        if window["idle_start"] <= stamp < window["active_start"] and power is not None:
            idle_power.append(power)
        if window["active_start"] <= stamp <= window["active_end"]:
            if power is not None:
                active_power.append(power)
            ram_match = RAM_RE.search(payload)
            if ram_match:
                active_ram.append(float(ram_match.group(1)))
            active_temp.extend(float(value) for value in TEMP_RE.findall(payload))
    return {
        "idle_power_mw": float(np.mean(idle_power)) if idle_power else None,
        "active_power_mw": float(np.mean(active_power)) if active_power else None,
        "active_power_p95_mw": float(np.percentile(active_power, 95)) if active_power else None,
        "peak_ram_mb": max(active_ram) if active_ram else None,
        "peak_temp_c": max(active_temp) if active_temp else None,
    }


def cosine_rows(raw_dir: Path):
    reference = np.load(raw_dir.parent / "parity" / "pytorch_fp32.npy")
    rows = {}
    for backend in BACKENDS:
        candidate = np.load(raw_dir.parent / "parity" / f"{backend}.npy")
        dots = np.sum(reference * candidate, axis=1)
        denom = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
        cosine = dots / np.maximum(denom, 1e-12)
        error = np.abs(reference - candidate)
        rows[backend] = {
            "cosine_median": float(np.median(cosine)),
            "cosine_min": float(np.min(cosine)),
            "mean_abs_error": float(np.mean(error)),
            "max_abs_error": float(np.max(error)),
        }
    return rows


def fmt(value, places=3):
    return "n/a" if value is None else f"{value:.{places}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parity = cosine_rows(args.raw_dir)
    preprocessing_path = args.raw_dir / "preprocessing.json"
    preprocessing = load_json(preprocessing_path) if preprocessing_path.is_file() else {}
    preprocessing_median = preprocessing.get("latency_median_ms")
    rows = []
    for backend in BACKENDS:
        result = load_json(args.raw_dir / f"{backend}.json")
        window = load_json(args.raw_dir / f"{backend}_window.json")
        power = parse_tegra(args.raw_dir / f"{backend}_tegrastats.log", window)
        throughput = float(np.mean([row["throughput_fps"] for row in result["trials"]]))
        active_w = power["active_power_mw"] / 1000.0 if power["active_power_mw"] is not None else None
        idle_w = power["idle_power_mw"] / 1000.0 if power["idle_power_mw"] is not None else None
        rows.append({
            "backend": backend,
            "artifact_bytes": result["artifact_bytes"],
            "compute_median_ms": result["overall_compute"]["median_ms"],
            "latency_mean_ms": result["overall_transfer_included"]["mean_ms"],
            "latency_median_ms": result["overall_transfer_included"]["median_ms"],
            "latency_p95_ms": result["overall_transfer_included"]["p95_ms"],
            "latency_p99_ms": result["overall_transfer_included"]["p99_ms"],
            "end_to_end_median_ms": result["overall_transfer_included"]["median_ms"] + preprocessing_median if preprocessing_median is not None else None,
            "throughput_fps": throughput,
            "process_peak_rss_mb": result["process_peak_rss_kb"] / 1024.0,
            "system_peak_ram_mb": power["peak_ram_mb"],
            "idle_power_w": idle_w,
            "active_power_w": active_w,
            "active_power_p95_w": power["active_power_p95_mw"] / 1000.0 if power["active_power_p95_mw"] is not None else None,
            "energy_mj_per_inference": active_w * 1000.0 / throughput if active_w is not None else None,
            "dynamic_energy_mj_per_inference": max(0.0, active_w - idle_w) * 1000.0 / throughput if active_w is not None and idle_w is not None else None,
            "peak_temp_c": power["peak_temp_c"],
            **parity[backend],
        })

    fields = list(rows[0].keys())
    with (args.output_dir / "precision_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "precision_comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    reference = rows[0]
    fp32 = rows[1]
    fp16 = rows[2]
    lines = [
        "# Jetson Nano B01 Precision Comparison",
        "",
        "Measured on the same notebook IResNet50-MSFF model at batch size 1 in MAXN mode.",
        "",
        "| Backend | Artifact (MiB) | Network median (ms) | End-to-end median (ms) | p95 (ms) | Throughput (FPS) | Active power (W) | Energy (mJ/inf) | Cosine vs PyTorch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {backend} | {size:.2f} | {median:.3f} | {end_to_end} | {p95:.3f} | {fps:.3f} | {power} | {energy} | {cosine:.6f} |".format(
            backend=row["backend"], size=row["artifact_bytes"] / 1048576.0,
            median=row["latency_median_ms"], end_to_end=fmt(row["end_to_end_median_ms"]),
            p95=row["latency_p95_ms"], fps=row["throughput_fps"],
            power=fmt(row["active_power_w"]), energy=fmt(row["energy_mj_per_inference"]),
            cosine=row["cosine_median"]))
    lines.extend([
        "",
        "## Speedups",
        "",
        "- TensorRT FP32 vs PyTorch FP32: **{:.2f}x**".format(reference["latency_median_ms"] / fp32["latency_median_ms"]),
        "- TensorRT FP16 vs TensorRT FP32: **{:.2f}x**".format(fp32["latency_median_ms"] / fp16["latency_median_ms"]),
        "- TensorRT FP16 vs PyTorch FP32: **{:.2f}x**".format(reference["latency_median_ms"] / fp16["latency_median_ms"]),
        "- Shared preprocessing median: **{} ms** across {} valid deployment images ({} failures).".format(
            fmt(preprocessing_median), preprocessing.get("valid_images", "n/a"), preprocessing.get("failed_images", "n/a")),
        "",
        "## Unsupported precision modes",
        "",
        "The device reports fast FP16 support and no fast INT8 support. TensorRT 8.2 on the Nano does not provide FP8 or FP4 execution. Those modes are reported as unsupported rather than assigned fabricated measurements.",
        "",
        "## Accuracy boundary",
        "",
        "This report validates numerical embedding parity on deterministic inputs. The original notebook records test EER 3.864%, AUC 0.9918, and TAR@0.1% FAR 84.56%. Quantized CASIA metrics require the frozen image split and are not inferred from latency or parity measurements.",
        "",
    ])
    (args.output_dir / "JETSON_PRECISION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
