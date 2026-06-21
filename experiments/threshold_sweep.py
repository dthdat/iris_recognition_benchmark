from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate_results import PLANNED_RUNS
from src.io_utils import ensure_dir, read_csv_rows, write_csv_rows
from src.metrics import rates_at_threshold


THRESHOLDS = [0.30, 0.35, 0.40, 0.43, 0.45, 0.50, 0.52, 0.55, 0.60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fixed cosine thresholds from saved score distributions.")
    parser.add_argument("--runs-dir", default=os.environ.get("IRIS_OUTPUT_DIR", "runs"))
    parser.add_argument("--results-dir", default=os.environ.get("IRIS_RESULTS_DIR", "results"))
    return parser.parse_args()


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Run & Threshold & FAR (\\%) & FRR (\\%) & TAR (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['run_id']} & {row['threshold']:.2f} & {row['far']*100:.2f} & {row['frr']*100:.2f} & {row['tar']*100:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = ensure_dir(args.results_dir)
    rows = []
    for run_id in PLANNED_RUNS:
        path = runs_dir / run_id / "score_distribution_test.csv"
        if not path.is_file():
            continue
        score_rows = read_csv_rows(path)
        scores = np.array([float(r["score"]) for r in score_rows], dtype=np.float32)
        is_genuine = np.array([int(r["is_genuine"]) for r in score_rows], dtype=np.int32)
        for threshold in THRESHOLDS:
            rates = rates_at_threshold(scores, is_genuine, threshold)
            rows.append({"run_id": run_id, "threshold": threshold, **rates})
    write_csv_rows(results_dir / "threshold_sweep.csv", rows, ["run_id", "threshold", "far", "frr", "tar"])
    write_tex(results_dir / "threshold_sweep.tex", rows)
    print(f"Wrote {len(rows)} threshold-sweep rows to {results_dir}")


if __name__ == "__main__":
    main()
