# Iris Recognition Benchmark

Reproducible open-set iris-verification baselines derived from `iris_baseline.ipynb`. The repository contains the training/evaluation code, four fixed experiment configurations, frozen subject-exclusive splits, Kaggle automation, and compact reports. It intentionally excludes datasets, checkpoints, raw Kaggle outputs, generated bundles, and local logs.

## Current status

| Run | Model | Mask | State | Test EER | Test AUC | TAR @ 0.1% FAR |
|---|---|---|---|---:|---:|---:|
| `b1_arciris_nomask` | IResNet50 | none | complete | 3.4944% | 0.99209 | 88.8451% |
| `b3_arciris_softmask` | IResNet50 | soft angular | complete | 3.5915% | 0.99193 | 88.2568% |
| `b4_mobilenet_softmask` | MobileFaceNet | soft angular | blocked after partial training | — | — | — |
| `ours_iresnet_msff_softmask` | IResNet50 + MSFF | soft angular | not submitted | — | — | — |

These numbers are copied from collected `test_metrics.json` files. No metric is reported for an incomplete run. See [the monitor report](results/MONITOR_REPORT.md) for the execution history and [the handoff guide](docs/HANDOFF.md) before continuing.

Despite its historical run ID, the current b4 config uses `model_name: mobilefacenet`. Treat the YAML files as the source of truth.

## Repository layout

```text
src/                    dataset, preprocessing, masks, models, losses, metrics
experiments/            train/evaluate/analysis entry points and fixed configs
tools/                  Kaggle bundling, submission, monitoring, collection
splits/                 frozen 700/100/200 subject train/val/test split
results/                compact reports and summary tables only
docs/                   automation and continuation documentation
iris_baseline.ipynb     protected original notebook
AGENTS.md               mandatory constraints for coding agents
CLAUDE.md               Claude Code entry point
```

## Setup

Python 3.10+ is recommended. Install PyTorch and TorchVision for the machine's CUDA/CPU platform first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install torch and torchvision using https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

The CASIA-Iris-Thousand dataset is not included. Set `IRIS_DATASET_ROOT` to the directory that directly contains `000/`, `001/`, and the other subject folders:

```text
CASIA-Iris-Thousand/
  000/L/*.jpg
  000/R/*.jpg
  001/L/*.jpg
  001/R/*.jpg
```

Run a one-epoch local smoke test only when the dataset is available:

```bash
IRIS_DATASET_ROOT=/path/to/CASIA-Iris-Thousand \
python experiments/train.py \
  --config experiments/configs/b1_arciris_nomask.yaml \
  --max-epochs 1
```

The next owner may continue b4 after completing the resume-path tests and dry-run checklist in [HANDOFF.md](docs/HANDOFF.md). Use one controlled submission at a time, stop on failure, and never invent missing metrics. [Kaggle automation](docs/KAGGLE_AUTOMATION.md) contains the exact workflow.

## Collaborating

Read [CONTRIBUTING.md](CONTRIBUTING.md). Codex reads `AGENTS.md` automatically when operating in this directory. Claude Code should start from `CLAUDE.md`, which points to the same constraints and handoff.

After all four models complete, continue with the analysis and deployment work in [ROADMAP.md](docs/ROADMAP.md).

This repository does not currently declare an open-source license. Public visibility does not grant reuse rights beyond GitHub's terms; add an explicit license if broader reuse is intended.
