from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from kaggle_submit import kernel_slug


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and collect one Kaggle iris run output.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kaggle-user", required=True)
    return parser.parse_args()


def merge_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)


def find_first(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def print_metrics(path: Path) -> None:
    import json

    if not path.is_file():
        print("test_metrics.json not found in downloaded output yet.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    print("Key test metrics")
    print(f"  EER: {data.get('eer', 'n/a')}")
    print(f"  AUC: {data.get('auc', 'n/a')}")
    print(f"  TAR@0.1%FAR: {data.get('tar_at_01far', 'n/a')}")
    print(f"  FAR@val-threshold: {data.get('selected_far', 'n/a')}")
    print(f"  FRR@val-threshold: {data.get('selected_frr', 'n/a')}")


def main() -> None:
    args = parse_args()
    kernel = f"{args.kaggle_user}/{kernel_slug(args.run_id)}"
    out_dir = ROOT / "kaggle_outputs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(out_dir), "-o"], check=True)

    downloaded_run = out_dir / "runs" / args.run_id
    if downloaded_run.is_dir():
        merge_copy(downloaded_run, ROOT / "runs" / args.run_id)
        merge_copy(downloaded_run, ROOT / "results" / "kaggle" / args.run_id / "run")

    downloaded_results = out_dir / "results"
    if downloaded_results.is_dir():
        merge_copy(downloaded_results, ROOT / "results" / "kaggle" / args.run_id / "results")

    downloaded_splits = out_dir / "splits"
    if downloaded_splits.is_dir():
        merge_copy(downloaded_splits, ROOT / "splits")
        print(f"Collected frozen split CSVs into {ROOT / 'splits'}")
    else:
        print("No splits/ folder found in downloaded output.")

    metrics_path = downloaded_run / "test_metrics.json"
    if not metrics_path.is_file():
        found = find_first(out_dir, "test_metrics.json")
        if found is not None:
            metrics_path = found
    print_metrics(metrics_path)
    print(f"Downloaded Kaggle output to {out_dir}")


if __name__ == "__main__":
    main()
