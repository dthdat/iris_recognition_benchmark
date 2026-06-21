"""Fix #3: the generated Kaggle runner must pass --resume-state and perform a
single bounded restart, so a transient OOM kill resumes instead of restarting
from scratch."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import kaggle_submit  # noqa: E402


def _make_minimal_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "src").mkdir(parents=True)
    (bundle / "experiments").mkdir(parents=True)
    (bundle / "splits").mkdir(parents=True)
    (bundle / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    return bundle


def test_generated_runner_passes_resume_state(tmp_path):
    bundle = _make_minimal_bundle(tmp_path)
    kaggle_submit.write_run_script(bundle, "b4_mobilenet_softmask", "/kaggle/input/x")
    script = (bundle / "run_one_config.py").read_text(encoding="utf-8")
    assert "--resume-state" in script


def test_generated_runner_has_single_bounded_restart(tmp_path):
    bundle = _make_minimal_bundle(tmp_path)
    kaggle_submit.write_run_script(bundle, "b4_mobilenet_softmask", "/kaggle/input/x")
    script = (bundle / "run_one_config.py").read_text(encoding="utf-8")
    # exactly one retry path (a single except-driven restart), not an unbounded loop
    assert "CalledProcessError" in script
    assert script.count("except subprocess.CalledProcessError") == 1
    assert "while" not in script.split("def run_training")[1].split("def ")[0]
