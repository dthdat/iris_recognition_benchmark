from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate_results import PLANNED_RUNS
from src.io_utils import ensure_dir, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quantization comparison plan table without retraining.")
    parser.add_argument("--runs-dir", default=os.environ.get("IRIS_OUTPUT_DIR", "runs"))
    parser.add_argument("--results-dir", default=os.environ.get("IRIS_RESULTS_DIR", "results"))
    return parser.parse_args()


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llll}",
        "\\hline",
        "Run & Precision & Status & Notes \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(f"{row['run_id']} & {row['precision']} & {row['status']} & {row['notes']} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = ensure_dir(args.results_dir)
    rows = []
    for run_id in PLANNED_RUNS:
        run_dir = runs_dir / run_id
        has_ckpt = (run_dir / "best_model.pth").is_file()
        has_onnx = any(run_dir.glob("*_embedding.onnx"))
        for precision in ["FP32", "FP16 TensorRT", "INT8 TensorRT"]:
            if precision == "FP32":
                status = "ready" if has_ckpt else "pending_checkpoint"
                notes = "PyTorch/ONNX baseline measurement; no calibration needed."
            elif precision == "FP16 TensorRT":
                status = "ready_after_onnx" if has_onnx else "export_onnx_first"
                notes = "Use TensorRT FP16 on Jetson if supported by the deployed layers."
            else:
                status = "requires_calibration" if has_onnx else "export_onnx_first"
                notes = "Requires representative polar-strip calibration data; INT4 is optional/simulated only."
            rows.append({"run_id": run_id, "precision": precision, "status": status, "notes": notes})
    write_csv_rows(results_dir / "quantization_summary.csv", rows, ["run_id", "precision", "status", "notes"])
    write_tex(results_dir / "quantization_summary.tex", rows)
    print(f"Wrote quantization plan to {results_dir}")


if __name__ == "__main__":
    main()
