# Codex Report

> Historical implementation report. For the current run state and continuation instructions, use `docs/HANDOFF.md`. Some execution-state statements below predate the completed b1/b3 runs and the blocked b4 attempts.

## Files Created Or Modified

Created the reproducible framework under `src/`, experiment entrypoints under `experiments/`, Kaggle automation under `tools/`, and documentation under `docs/`.

Key files:

- `src/data.py`, `src/preprocessing.py`, `src/masks.py`, `src/models.py`, `src/losses.py`, `src/metrics.py`, `src/train_utils.py`, `src/io_utils.py`
- `experiments/train.py`, `experiments/evaluate.py`, `experiments/aggregate_results.py`, `experiments/export_onnx.py`
- `experiments/threshold_sweep.py`, `experiments/margin_sweep.py`, `experiments/same_side_eval.py`, `experiments/quantization_plan.py`
- `experiments/configs/b1_arciris_nomask.yaml`
- `experiments/configs/b3_arciris_softmask.yaml`
- `experiments/configs/b4_mobilenet_softmask.yaml`
- `experiments/configs/ours_iresnet_msff_softmask.yaml`
- `tools/kaggle_submit.py`, `tools/kaggle_status.py`, `tools/kaggle_collect.py`, `tools/kaggle_run_all.py`
- `docs/KAGGLE_AUTOMATION.md`, `.gitignore`, `requirements.txt`

The original `iris_baseline.ipynb` was preserved.

## Extracted From Notebook

- Dataset layout: CASIA-Iris-Thousand subject folders with `L/` and `R/` eye folders.
- Label unit: subject plus eye side, for example `001_L`.
- Split: subject-exclusive train/open-val/test with 70%/10%/20% subjects and seed `42`.
- Preprocessing: grayscale eye image, Hough pupil detection, IDO-first iris/limbus detection with Hough fallback, Daugman rubber-sheet normalization to `64x512`, radial crop `0.10..0.87`.
- Masking: optional soft angular/butterfly mask with randomized train mask bank.
- Model/loss: IResNet50-MSFF backbone, ArcFace head, cosine verification.
- Training defaults: batch size `128`, epochs `40`, AdamW, learning rate `0.001`, weight decay `0.0003`, ArcFace scale `64.0`, margin `0.25`, dropout `0.35`, AMP enabled.
- Checkpoint selection: open-validation EER, not closed-set train accuracy.
- Metrics: EER, AUC, TAR@0.1%FAR, validation-selected target-FAR threshold, FAR/FRR at selected threshold, score distributions, ROC and histogram plots.
- ONNX export: embedding network only, not the full segmentation/normalization pipeline.

## Dataset Root

The config default is `./data/CASIA-Iris-Thousand`, but environment variable `IRIS_DATASET_ROOT` overrides it.

Expected layout:

```text
CASIA-Iris-Thousand/
  SUBJECT_ID/
    L/
      *.jpg
    R/
      *.jpg
```

Kaggle command examples use:

```bash
--dataset-root /kaggle/input/DATASET_SLUG
```

If the dataset is nested deeper, pass the full nested path that directly contains subject folders.

## Smoke Test

Do not run full training first. If the dataset exists locally:

```bash
IRIS_DATASET_ROOT=/path/to/CASIA-Iris-Thousand \
python experiments/train.py --config experiments/configs/b1_arciris_nomask.yaml --max-epochs 1
```

If the local dataset is missing, the script exits with a clear `dataset_root` error.

## Submit One Kaggle Run

First create a dry-run bundle:

```bash
python tools/kaggle_submit.py \
  --run-id b1_arciris_nomask \
  --kaggle-user USERNAME \
  --dataset-source USERNAME/DATASET_SLUG \
  --dataset-root /kaggle/input/DATASET_SLUG \
  --accelerator NvidiaTeslaT4 \
  --timeout 21600 \
  --dry-run
```

Then submit:

```bash
python tools/kaggle_submit.py \
  --run-id b1_arciris_nomask \
  --kaggle-user USERNAME \
  --dataset-source USERNAME/DATASET_SLUG \
  --dataset-root /kaggle/input/DATASET_SLUG \
  --accelerator NvidiaTeslaT4 \
  --timeout 21600
```

## Check Status

```bash
python tools/kaggle_status.py --run-id b1_arciris_nomask --kaggle-user USERNAME
```

## Collect Outputs

```bash
python tools/kaggle_collect.py --run-id b1_arciris_nomask --kaggle-user USERNAME
```

The collector also copies Kaggle-created `splits/` CSVs into the local `splits/` folder. Future dry-run or submit bundles include those frozen split CSVs automatically.

## Run All Four Planned Experiments

```bash
python tools/kaggle_run_all.py \
  --kaggle-user USERNAME \
  --dataset-source USERNAME/DATASET_SLUG \
  --dataset-root /kaggle/input/DATASET_SLUG \
  --accelerator NvidiaTeslaT4
```

The full sequence is four trainings only. Estimated total budget is about 8-10 Kaggle GPU hours, leaving buffer from the remaining credit.

## Dependencies

`requirements.txt` lists Python dependencies that may be missing outside Kaggle:

- `pyyaml`
- `opencv-python-headless`
- `scikit-learn`
- `matplotlib`
- `onnx`

PyTorch and TorchVision are expected from the Kaggle GPU image or local environment.

PyTorch and TorchVision availability depends on the collaborator's environment. Run the compile/import checks in `CONTRIBUTING.md`; full training requires a compatible PyTorch installation, dataset, and accelerator.

## Assumptions

- The b4 lightweight baseline uses MobileFaceNet with a global depthwise convolution adapted to the 64x512 polar feature geometry.
- Full training defaults to 40 epochs, matching the notebook source config.
- The first successful Kaggle run may create the frozen split if no local split CSVs exist yet.
- Later Kaggle bundles reuse local frozen split CSVs after `kaggle_collect.py` downloads them.
- No quantization results are fabricated; quantization scripts create plan/summary tables from available checkpoints.

## Warning

No fake results were generated. Summary and analysis files must be regenerated after real Kaggle outputs are collected.
