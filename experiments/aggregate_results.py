from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir, read_json, write_csv_rows


PLANNED_RUNS = [
    "b1_arciris_nomask",
    "b3_arciris_softmask",
    "b4_mobilenet_softmask",
    "ours_iresnet_msff_softmask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline metrics into thesis tables.")
    parser.add_argument("--runs-dir", default=os.environ.get("IRIS_OUTPUT_DIR", "runs"))
    parser.add_argument("--results-dir", default=os.environ.get("IRIS_RESULTS_DIR", "results"))
    return parser.parse_args()


def metric_value(data: dict[str, Any] | None, key: str) -> Any:
    if data is None:
        return ""
    return data.get(key, "")


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in PLANNED_RUNS:
        run_dir = runs_dir / run_id
        metrics_path = run_dir / "test_metrics.json"
        val_path = run_dir / "val_metrics.json"
        test_metrics = read_json(metrics_path) if metrics_path.is_file() else None
        val_metrics = read_json(val_path) if val_path.is_file() else None
        status = "complete" if test_metrics is not None else "pending"
        rows.append(
            {
                "run_id": run_id,
                "status": status,
                "val_eer": metric_value(val_metrics, "eer"),
                "test_eer": metric_value(test_metrics, "eer"),
                "test_auc": metric_value(test_metrics, "auc"),
                "test_tar_at_01far": metric_value(test_metrics, "tar_at_01far"),
                "val_selected_threshold": metric_value(test_metrics, "selected_threshold"),
                "test_far_at_val_threshold": metric_value(test_metrics, "selected_far"),
                "test_frr_at_val_threshold": metric_value(test_metrics, "selected_frr"),
                "genuine_mean": (test_metrics or {}).get("genuine_stats", {}).get("mean", ""),
                "impostor_mean": (test_metrics or {}).get("impostor_stats", {}).get("mean", ""),
            }
        )
    return rows


def format_float(value: Any, pct: bool = False) -> str:
    if value == "" or value is None:
        return "pending"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pct:
        return f"{v * 100:.2f}" if 0 <= v <= 1 else f"{v:.2f}"
    return f"{v:.4f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Run | Status | Test EER (%) | AUC | TAR@0.1%FAR (%) | FAR@ValThr (%) | FRR@ValThr (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {status} | {eer} | {auc} | {tar} | {far} | {frr} |".format(
                run_id=row["run_id"],
                status=row["status"],
                eer=format_float(row["test_eer"]),
                auc=format_float(row["test_auc"]),
                tar=format_float(row["test_tar_at_01far"], pct=True),
                far=format_float(row["test_far_at_val_threshold"], pct=True),
                frr=format_float(row["test_frr_at_val_threshold"], pct=True),
            )
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Run & EER (\\%) & AUC & TAR@0.1\\%FAR (\\%) & FAR@ValThr (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['run_id']} & {format_float(row['test_eer'])} & {format_float(row['test_auc'])} & "
            f"{format_float(row['test_tar_at_01far'], pct=True)} & {format_float(row['test_far_at_val_threshold'], pct=True)} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis(path: Path, rows: list[dict[str, Any]]) -> None:
    complete = [r for r in rows if r["status"] == "complete" and r["test_eer"] != ""]
    lines = ["# Baseline Analysis", ""]
    if len(complete) < len(PLANNED_RUNS):
        lines.extend(
            [
                "No final scientific conclusions are written yet because one or more planned runs are missing.",
                "",
                "Pending runs should be completed and collected from Kaggle before thesis conclusions are finalized.",
                "",
            ]
        )
    if complete:
        strongest = min(complete, key=lambda r: float(r["test_eer"]))
        lines.extend(
            [
                f"Strongest baseline so far: `{strongest['run_id']}` by lowest test EER.",
                "",
                "Comparisons to fill after all four runs are complete:",
                "- Soft mask vs no mask: compare `b3_arciris_softmask` against `b1_arciris_nomask`.",
                "- MobileFaceNet Jetson acceptability: compare `b4_mobilenet_softmask` accuracy and model size against thesis constraints.",
                "- MSFF contribution: compare `ours_iresnet_msff_softmask` against `b3_arciris_softmask`.",
                "- Final thesis model: choose only after all four complete test metrics are present.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Strongest baseline: pending.",
                "Soft mask improvement: pending.",
                "MobileFaceNet Jetson acceptability: pending.",
                "IResNet50-MSFF improvement: pending.",
                "Final thesis model choice: pending.",
                "",
            ]
        )
    lines.extend(
        [
            "Limitations:",
            "- Results use pair-sampled open-set verification metrics, so confidence improves with larger pair counts.",
            "- The frozen split must be reused for all four runs.",
            "- Quantization and Jetson latency should be reported from exported checkpoints, not retraining.",
            "",
            "Next steps:",
            "- Continue b4 only through the tested, single-run, stop-on-failure workflow in `docs/HANDOFF.md`.",
            "- Run the final planned model only after b4 succeeds and its real test metrics are verified.",
            "- Re-run this aggregation script only after a real completed result is collected.",
            "- Run threshold, margin, same-side, and quantization analyses from saved checkpoints only.",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = ensure_dir(args.results_dir)
    rows = collect_rows(runs_dir)
    fieldnames = [
        "run_id",
        "status",
        "val_eer",
        "test_eer",
        "test_auc",
        "test_tar_at_01far",
        "val_selected_threshold",
        "test_far_at_val_threshold",
        "test_frr_at_val_threshold",
        "genuine_mean",
        "impostor_mean",
    ]
    write_csv_rows(results_dir / "summary_baselines.csv", rows, fieldnames)
    write_markdown(results_dir / "summary_baselines.md", rows)
    write_tex(results_dir / "summary_baselines.tex", rows)
    write_analysis(results_dir / "BASELINE_ANALYSIS.md", rows)
    print(f"Wrote baseline summaries to {results_dir}")


if __name__ == "__main__":
    main()
