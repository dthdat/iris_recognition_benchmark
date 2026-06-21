from __future__ import annotations

import argparse
import subprocess

from kaggle_submit import kernel_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one Kaggle iris run status.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kaggle-user", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = f"{args.kaggle_user}/{kernel_slug(args.run_id)}"
    subprocess.run(["kaggle", "kernels", "status", kernel], check=True)


if __name__ == "__main__":
    main()
