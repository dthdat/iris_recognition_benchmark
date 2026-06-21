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


MARGINS = [0.00, 0.03, 0.05, 0.08, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep best-vs-second identity margins from saved embeddings.")
    parser.add_argument("--runs-dir", default=os.environ.get("IRIS_OUTPUT_DIR", "runs"))
    parser.add_argument("--results-dir", default=os.environ.get("IRIS_RESULTS_DIR", "results"))
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold; defaults to test selected_threshold.")
    return parser.parse_args()


def run_margin_eval(npz_path: Path, threshold: float, margin: float) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    embeds = data["embeddings"]
    labels = data["labels"]
    scores = embeds @ embeds.T
    np.fill_diagonal(scores, -np.inf)
    accepted = 0
    correct_accept = 0
    false_accept = 0
    reject = 0
    for i in range(len(labels)):
        order = np.argsort(scores[i])[::-1]
        best_idx = int(order[0])
        best_score = float(scores[i, best_idx])
        different = order[labels[order] != labels[best_idx]]
        second_diff = float(scores[i, different[0]]) if len(different) else -np.inf
        accept = best_score >= threshold and (best_score - second_diff) >= margin
        if accept:
            accepted += 1
            if labels[best_idx] == labels[i]:
                correct_accept += 1
            else:
                false_accept += 1
        else:
            reject += 1
    n = max(1, len(labels))
    return {
        "accept_rate": accepted / n,
        "correct_accept_rate": correct_accept / n,
        "false_accept_rate": false_accept / n,
        "reject_rate": reject / n,
    }


def write_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Run & Margin & Accept (\\%) & Correct Accept (\\%) & False Accept (\\%) & Reject (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['run_id']} & {row['margin']:.2f} & {row['accept_rate']*100:.2f} & "
            f"{row['correct_accept_rate']*100:.2f} & {row['false_accept_rate']*100:.2f} & {row['reject_rate']*100:.2f} \\\\"
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
        threshold = args.threshold
        if threshold is None:
            threshold = float(read_json(metrics_path).get("selected_threshold", 1.0))
        for margin in MARGINS:
            rows.append({"run_id": run_id, "threshold": threshold, "margin": margin, **run_margin_eval(npz_path, threshold, margin)})
    write_csv_rows(
        results_dir / "margin_sweep.csv",
        rows,
        ["run_id", "threshold", "margin", "accept_rate", "correct_accept_rate", "false_accept_rate", "reject_rate"],
    )
    write_tex(results_dir / "margin_sweep.tex", rows)
    print(f"Wrote {len(rows)} margin-sweep rows to {results_dir}")


if __name__ == "__main__":
    main()
