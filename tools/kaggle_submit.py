from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANNED_RUNS = {
    "b1_arciris_nomask",
    "b3_arciris_softmask",
    "b4_mobilenet_softmask",
    "ours_iresnet_msff_softmask",
}
FORBIDDEN_NAMES = {".kaggle", "kaggle.json", "access_token", "access-token"}
PROJECT_ARCHIVE_ITEMS = ("src", "experiments", "requirements.txt", "splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally submit a Kaggle kernel bundle for one iris run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--dataset-source", required=True, help="Kaggle dataset source, e.g. USERNAME/DATASET_SLUG.")
    parser.add_argument("--dataset-root", required=True, help="Kaggle input path, e.g. /kaggle/input/DATASET_SLUG.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def kernel_slug(run_id: str) -> str:
    return "iris-" + run_id.replace("_", "-")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def copy_existing_splits(bundle: Path) -> int:
    split_src = ROOT / "splits"
    split_dst = bundle / "splits"
    count = 0
    if not split_src.is_dir():
        return count
    split_dst.mkdir(parents=True, exist_ok=True)
    for csv_path in split_src.glob("*.csv"):
        shutil.copy2(csv_path, split_dst / csv_path.name)
        count += 1
    return count


def project_archive_literal(bundle: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item_name in PROJECT_ARCHIVE_ITEMS:
            item_path = bundle / item_name
            if item_path.is_file():
                archive.write(item_path, item_path.relative_to(bundle).as_posix())
                continue
            if not item_path.is_dir():
                continue
            for file_path in sorted(path for path in item_path.rglob("*") if path.is_file()):
                archive.write(file_path, file_path.relative_to(bundle).as_posix())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, width=88))


def write_run_script(bundle: Path, run_id: str, dataset_root: str) -> None:
    archive_literal = project_archive_literal(bundle)
    script = f'''from __future__ import annotations

import base64
import io
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch


RUN_ID = "{run_id}"
DATASET_ROOT = "{dataset_root}"
BUNDLE_ROOT = Path(__file__).resolve().parent
WORKING = Path("/kaggle/working")
PROJECT_ARCHIVE_B64 = (
{archive_literal}
)
PROJECT_ROOT = BUNDLE_ROOT


def has_subject_layout(root):
    return (root / "000" / "L").is_dir() and (root / "000" / "R").is_dir()


def resolve_kaggle_dataset_root(configured_root):
    root = Path(configured_root)
    if has_subject_layout(root):
        return root

    print("Configured dataset root is not usable:", root)
    input_root = Path("/kaggle/input")
    candidates = []
    if input_root.is_dir():
        for path in input_root.rglob("000"):
            parent = path.parent
            if has_subject_layout(parent):
                candidates.append(parent)

    if not candidates:
        raise FileNotFoundError(
            "Could not find CASIA-Iris-Thousand subject folders under /kaggle/input. "
            "Expected a directory containing 000/L and 000/R."
        )

    candidates = sorted(candidates, key=lambda path: (len(path.parts), str(path)))
    print("Discovered dataset root:", candidates[0])
    return candidates[0]


def project_has_code(root):
    return (root / "experiments" / "train.py").is_file() and (root / "src").is_dir()


def extract_project_archive(dst):
    dst.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(PROJECT_ARCHIVE_B64)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        dst_resolved = dst.resolve()
        for member in archive.infolist():
            target = (dst / member.filename).resolve()
            if not target.is_relative_to(dst_resolved):
                raise RuntimeError("Unsafe archive member path: " + member.filename)
        archive.extractall(dst)


def materialize_project():
    if project_has_code(BUNDLE_ROOT):
        print("Using project root:", BUNDLE_ROOT)
        return BUNDLE_ROOT

    try:
        entries = sorted(path.name for path in BUNDLE_ROOT.iterdir())
    except FileNotFoundError:
        entries = []
    print("Project files not found directly under BUNDLE_ROOT:", BUNDLE_ROOT)
    print("BUNDLE_ROOT entries:", entries)

    for dst in (BUNDLE_ROOT / "project", Path("/tmp/iris_project")):
        try:
            if not project_has_code(dst):
                if dst.exists():
                    shutil.rmtree(dst)
                extract_project_archive(dst)
            if project_has_code(dst):
                print("Extracted project root:", dst)
                return dst
        except OSError as exc:
            print(f"Cannot extract project to {{dst}}: {{exc}}")

    raise RuntimeError("Could not materialize bundled project files.")


def run(cmd):
    if not project_has_code(PROJECT_ROOT):
        raise RuntimeError(f"Project root does not contain experiments/train.py: {{PROJECT_ROOT}}")
    print("cwd:", Path.cwd(), flush=True)
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def run_training(config_path):
    cmd = [
        "python",
        "experiments/train.py",
        "--config",
        config_path,
    ]
    run(cmd)


def copy_bundled_splits(project_root):
    src = project_root / "splits"
    dst = WORKING / "splits"
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        print("No bundled splits found; training will create frozen splits if needed.")
        return
    count = 0
    for item in src.glob("*.csv"):
        shutil.copy2(item, dst / item.name)
        count += 1
    print(f"Copied {{count}} bundled split CSVs to {{dst}}")


def main():
    global PROJECT_ROOT
    PROJECT_ROOT = materialize_project()
    os.chdir(PROJECT_ROOT)
    print("Changed working directory to:", Path.cwd())

    dataset_root = resolve_kaggle_dataset_root(DATASET_ROOT)
    os.environ["IRIS_DATASET_ROOT"] = str(dataset_root)
    os.environ["IRIS_OUTPUT_DIR"] = str(WORKING / "runs")
    os.environ["IRIS_RESULTS_DIR"] = str(WORKING / "results")
    os.environ["IRIS_SPLIT_DIR"] = str(WORKING / "splits")

    print("Python:", sys.version.replace("\\n", " "))
    print("Platform:", platform.platform())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset root:", os.environ["IRIS_DATASET_ROOT"])
    print("Run ID:", RUN_ID)

    copy_bundled_splits(PROJECT_ROOT)
    config_path = f"experiments/configs/{{RUN_ID}}.yaml"
    run_training(config_path)
    run(["python", "experiments/evaluate.py", "--run", f"/kaggle/working/runs/{{RUN_ID}}"])
    run(["python", "experiments/aggregate_results.py"])
    print("Outputs saved under /kaggle/working/runs, /kaggle/working/results, and /kaggle/working/splits")


if __name__ == "__main__":
    main()
'''
    (bundle / "run_one_config.py").write_text(script, encoding="utf-8")


def write_metadata(bundle: Path, run_id: str, kaggle_user: str, dataset_source: str) -> None:
    slug = kernel_slug(run_id)
    metadata = {
        "id": f"{kaggle_user}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": "run_one_config.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_source],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (bundle / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_bundle(bundle: Path) -> None:
    bad_paths = []
    for path in bundle.rglob("*"):
        parts = set(path.parts)
        if any(name in parts or path.name == name for name in FORBIDDEN_NAMES):
            bad_paths.append(path.relative_to(bundle))
    if bad_paths:
        joined = "\n".join(str(p) for p in bad_paths)
        raise RuntimeError(f"Forbidden credential-like files found in bundle paths:\n{joined}")


def main() -> None:
    args = parse_args()
    for name in ["run_id", "kaggle_user", "dataset_source", "dataset_root", "accelerator"]:
        if not str(getattr(args, name, "")).strip():
            raise SystemExit(f"--{name.replace('_', '-')} is required and cannot be empty")
    if args.run_id not in PLANNED_RUNS:
        raise SystemExit(f"run_id must be one of: {sorted(PLANNED_RUNS)}")
    config_path = ROOT / "experiments" / "configs" / f"{args.run_id}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    bundle = ROOT / ".kaggle_bundle" / args.run_id
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    copy_tree(ROOT / "src", bundle / "src")
    copy_tree(ROOT / "experiments", bundle / "experiments")
    if (ROOT / "requirements.txt").is_file():
        shutil.copy2(ROOT / "requirements.txt", bundle / "requirements.txt")
    copied_splits = copy_existing_splits(bundle)
    write_run_script(bundle, args.run_id, args.dataset_root)
    write_metadata(bundle, args.run_id, args.kaggle_user, args.dataset_source)
    validate_bundle(bundle)

    print(f"Bundle ready: {bundle}")
    print(f"Run ID: {args.run_id}")
    print(f"Kernel: {args.kaggle_user}/{kernel_slug(args.run_id)}")
    print(f"Bundled split CSVs: {copied_splits}")
    if args.dry_run:
        print("Dry run only; not submitting to Kaggle.")
        return

    cmd = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(bundle),
        "-t",
        str(args.timeout),
        "--accelerator",
        args.accelerator,
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
