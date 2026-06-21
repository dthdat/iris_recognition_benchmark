from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate_results import PLANNED_RUNS
from src.io_utils import ensure_dir, read_json, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all-template matching against same-eye-side-only matching.")
    parser.add_argument("--runs-dir", default=os.environ.get("IRIS_OUTPUT_DIR", "runs"))
    parser.add_argument("--results-dir", default=os.environ.get("IRIS_RESULTS_DIR", "results"))
    return parser.parse_args()


def evaluate(npz_path: Path, threshold: float, same_side_only: bool) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    embeds = data["embeddings"]
    labels = data["labels"]
    eyes = data["eyes"] if "eyes" in data else np.array([""] * len(labels))
    scores = embeds @ embeds.T
    np.fill_diagonal(scores, -np.inf)
    accepted = 0
    correct = 0
    false_accept = 0
    for i in range(len(labels)):
        candidates = np.arange(len(labels))
        if same_side_only:
            candidates = candidates[eyes == eyes[i]]
            candidates = candidates[candidates != i]
            if len(candidates) == 0:
                continue
        best_idx = int(candidates[np.argmax(scores[i, candidates])])
        if scores[i, best_idx] >= threshold:
            accepted += 1
            if labels[best_idx] == labels[i]:
                correct += 1
            else:
                false_accept += 1
    n = max(1, len(labels))
    return {
        "accept_rate": accepted / n,
        "correct_accept_rate": correct / n,
        "false_accept_rate": false_accept / n,
    }


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Run & Mode & Accept (\\%) & Correct Accept (\\%) & False Accept (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['run_id']} & {row['mode']} & {row['accept_rate']*100:.2f} & "
            f"{row['correct_accept_rate']*100:.2f} & {row['false_accept_rate']*100:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = ensure_dir(args.results_dir)
    rows = []
    for run_id in PLANNED_RUNS:
        npz_path = runs_dir / run_id / "test_embeddings.npz"
        metrics_path = runs_dir / run_id / "test_metrics.json"
        if not npz_path.is_file() or not metrics_path.is_file():
            continue
        threshold = float(read_json(metrics_path).get("selected_threshold", 1.0))
        rows.append({"run_id": run_id, "mode": "all_templates", **evaluate(npz_path, threshold, same_side_only=False)})
        rows.append({"run_id": run_id, "mode": "same_eye_side_only", **evaluate(npz_path, threshold, same_side_only=True)})
    write_csv_rows(results_dir / "same_side_matching.csv", rows, ["run_id", "mode", "accept_rate", "correct_accept_rate", "false_accept_rate"])
    write_tex(results_dir / "same_side_matching.tex", rows)
    print(f"Wrote {len(rows)} same-side rows to {results_dir}")


if __name__ == "__main__":
    main()
