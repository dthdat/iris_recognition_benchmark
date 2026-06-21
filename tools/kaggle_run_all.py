from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from kaggle_submit import PLANNED_RUNS, kernel_slug


RUN_IDS = [
    "b1_arciris_nomask",
    "b3_arciris_softmask",
    "b4_mobilenet_softmask",
    "ours_iresnet_msff_softmask",
]
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit and collect the four planned Kaggle iris runs.")
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def submit(run_id: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "kaggle_submit.py"),
        "--run-id",
        run_id,
        "--kaggle-user",
        args.kaggle_user,
        "--dataset-source",
        args.dataset_source,
        "--dataset-root",
        args.dataset_root,
        "--accelerator",
        args.accelerator,
        "--timeout",
        str(args.timeout),
    ]
    result = run(cmd)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Submit failed for {run_id}")


def wait_for_completion(run_id: str, args: argparse.Namespace) -> bool:
    kernel = f"{args.kaggle_user}/{kernel_slug(run_id)}"
    while True:
        result = run(["kaggle", "kernels", "status", kernel])
        output = result.stdout or ""
        print(output)
        lower = output.lower()
        if any(word in lower for word in ["complete", "succeeded", "success"]):
            return True
        if any(word in lower for word in ["error", "failed", "cancel", "stopped"]):
            return False
        print(f"Waiting {args.poll_seconds}s before checking {run_id} again.")
        time.sleep(args.poll_seconds)


def collect(run_id: str, args: argparse.Namespace) -> None:
    result = run(
        [
            sys.executable,
            str(ROOT / "tools" / "kaggle_collect.py"),
            "--run-id",
            run_id,
            "--kaggle-user",
            args.kaggle_user,
        ]
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Collect failed for {run_id}")


def main() -> None:
    args = parse_args()
    if args.parallel:
        for idx, run_id in enumerate(RUN_IDS, start=1):
            print(f"Submitting {idx}/{len(RUN_IDS)}: {run_id}")
            submit(run_id, args)
        print("Parallel submission requested; collect outputs manually or rerun without --parallel after completion.")
        return

    for idx, run_id in enumerate(RUN_IDS, start=1):
        remaining = len(RUN_IDS) - idx
        print(f"Starting run {idx}/{len(RUN_IDS)}: {run_id}. Remaining after this: {remaining}")
        try:
            submit(run_id, args)
            ok = wait_for_completion(run_id, args)
            if not ok:
                raise RuntimeError(f"Kaggle run failed: {run_id}")
            collect(run_id, args)
        except Exception as exc:
            print(f"Run {run_id} failed: {exc}")
            if not args.continue_on_fail:
                raise
    print("All planned Kaggle runs processed.")


if __name__ == "__main__":
    main()
