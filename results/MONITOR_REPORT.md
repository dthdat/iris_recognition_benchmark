# Monitor report

> Public report only. Raw Kaggle outputs, checkpoints, bundles, run directories, and local logs are intentionally excluded from Git.

- Current phase: b1 complete; frozen splits collected and verified.
- Completed run: `b1_arciris_nomask`
- Kaggle status: `KernelWorkerStatus.COMPLETE`
- Collected outputs: `kaggle_outputs/b1_arciris_nomask`, `runs/b1_arciris_nomask`, `results/kaggle/b1_arciris_nomask`
- Frozen split CSVs verified:
  - `splits/train_subjects.csv`
  - `splits/val_subjects.csv`
  - `splits/test_subjects.csv`
  - `splits/train_images.csv`
  - `splits/val_images.csv`
  - `splits/test_images.csv`
- b1 test metrics:
  - EER: 3.4944025588302465
  - AUC: 0.9920946507876629
  - TAR@0.1%FAR: 0.8884509938313914
  - Val-selected threshold: 0.39895081520080566
  - FAR/FRR at val-selected threshold: 0.0007882111034955448 / 0.12080191912268677
- Next action after b1 was completed: submit and monitor `b3_arciris_softmask`; stop on failure.

Notes:
- `experiments/train.py` prints per-epoch metrics to stdout, but the epoch metrics print does not use `flush=True`.
- Future-only logging patch proposal saved at `results/FUTURE_EPOCH_LOGGING_PATCH.md`.
- No changes were made to the running b1 Kaggle job.

## b3_arciris_softmask

- Kaggle status: `KernelWorkerStatus.COMPLETE`
- Outputs collected: `kaggle_outputs/b3_arciris_softmask`, `runs/b3_arciris_softmask`, `results/kaggle/b3_arciris_softmask`
- Frozen split CSVs: verified present locally after collection
- Test metrics:
  - EER: 3.5915010281014395
  - AUC: 0.9919270796224412
  - TAR@0.1%FAR: 0.8825679689284899
  - Val-selected threshold: 0.40976735949516296
  - FAR/FRR at val-selected threshold: 0.00039981722641078363 / 0.14176376513593786
- Next action after b3 was completed: submit and monitor `b4_mobilenet_softmask`; stop on failure.

## b4_mobilenet_softmask

- Kaggle status: `KernelWorkerStatus.ERROR`
- Queue state: stopped. `ours_iresnet_msff_softmask` was not submitted.
- Outputs collected: `kaggle_outputs/b4_mobilenet_softmask`, partial `runs/b4_mobilenet_softmask`, `results/kaggle/b4_mobilenet_softmask`
- Frozen split CSVs: verified present locally and downloaded from b4 output.
- Missing final artifact: `runs/b4_mobilenet_softmask/test_metrics.json`
- Failure point: `experiments/train.py` died before evaluation, so no test metrics were produced.
- Kaggle log evidence:
  - Dataset root discovery succeeded.
  - GPU detection succeeded.
  - Six bundled split CSVs were copied to `/kaggle/working/splits`.
  - Training started with `python experiments/train.py --config experiments/configs/b4_mobilenet_softmask.yaml`.
  - The runner reported `subprocess.CalledProcessError: Command '['python', 'experiments/train.py', '--config', 'experiments/configs/b4_mobilenet_softmask.yaml']' died with <Signals.SIGKILL: 9>.`
- Partial training evidence:
  - `training_log.csv` exists and contains epochs 1-4.
  - Last logged epoch: 4
  - Last logged validation EER: 16.55264070264648
  - Last logged validation AUC: 0.9065489184695936
  - Last logged TAR@0.1%FAR: 0.18201779729573558
- Dry-run after failure: passed; bundle generation still succeeds and bundles 6 split CSVs.
- Diagnosis: real Kaggle runtime failure during training, not a Kaggle CLI `ConnectionResetError` and not the previous bundle `cwd` or dataset-root infrastructure bug.
- Automatic action taken: stopped queue and did not retry b4.

## Final audit

- Current blocking run: `b4_mobilenet_softmask`
- Latest checked Kaggle status: `KernelWorkerStatus.ERROR`
- Downstream run state: `ours_iresnet_msff_softmask` was not submitted by this queue path; no local `.kaggle_bundle`, `kaggle_outputs`, or `runs` directory exists for it.
- Frozen split CSVs remain present locally.
- `experiments/train.py` epoch metrics logging was inspected: per-epoch metrics are printed to stdout, but the print call does not include `flush=True`.
- Non-invasive future patch proposal: `results/FUTURE_EPOCH_LOGGING_PATCH.md`
- Queue decision: stop automatic progression until b4 failure is explicitly addressed.

The later sections supersede this intermediate audit.

## b4 retry patch

- Failure reason under investigation: previous b4 run was killed by `SIGKILL: 9` inside `experiments/train.py` after 4 logged epochs.
- Infrastructure-only patch applied before retry:
  - Added `flush=True` to training/preload progress prints so Kaggle receives live epoch and preload output.
  - Released validation embedding/metric buffers after each epoch with `del`, `gc.collect()`, and `torch.cuda.empty_cache()`.
- Scientific configs unchanged: model, split logic, metrics, epoch count, batch size, and config YAML values were not changed.
- Local verification before retry:
  - `python -m py_compile experiments/train.py src/data.py` passed.
  - b4 Kaggle bundle dry-run passed and bundled 6 split CSVs.
- Action: submit one corrected b4 retry and monitor; stop again if it fails.

## b4 retry result

- Retry status: `KernelWorkerStatus.ERROR`
- Retry collection: completed; no `test_metrics.json` was produced.
- Retry failure point: `experiments/train.py` was killed again with `SIGKILL: 9` during training.
- Retry progress before kill:
  - Last logged epoch: 4
  - Last logged validation EER: 15.980584768288452
  - Last logged validation AUC: 0.9071168281274847
  - Last logged TAR@0.1%FAR: 0.15428175199352825
- The retry confirms that the prior b4 failure was not fixed by the infrastructure-only stdout flush and post-validation memory cleanup patch.
- Queue state: stopped again. `ours_iresnet_msff_softmask` was not submitted.
- No further automatic b4 retries should be submitted without explicit approval, because the project rule forbids repeated failed retries.

## b4 baseline-stack retry patch

- User instruction: keep the baseline stack; do not change the model.
- Verified config: `experiments/configs/b4_mobilenet_softmask.yaml` remains `model_name: mobilenet_v2`.
- Failure pattern: previous b4 retry reached epoch 4, then produced no stdout until Kaggle killed `experiments/train.py` with `SIGKILL: 9`.
- Infrastructure-only patch applied:
  - Added batch-level heartbeat logging inside each training epoch.
  - Added explicit validation-start and metrics-start log lines.
  - Existing model, split logic, metrics, epoch count, batch size, and YAML scientific values remain unchanged.
- Local verification:
  - `python -m py_compile experiments/train.py src/data.py` passed.
  - b4 Kaggle bundle dry-run passed and bundled 6 split CSVs.
- Action: submit one b4 baseline-stack run and monitor; stop if it fails.

## b4 version 4 result

- Final status: `KernelWorkerStatus.ERROR`
- User instruction after failure: stop and leave the next fix for later.
- Baseline stack remained unchanged: `model_name: mobilenet_v2`.
- The training child was killed by `SIGKILL` during epoch 5 after batch 50/108.
- The bounded runner restart activated and found the resume checkpoint saved after epoch 4.
- Resume then failed before epoch 5 restarted:
  - `TypeError: RNG state must be a torch.ByteTensor`
  - Failure location: `restore_rng_state()` calling `torch.set_rng_state(state["torch"])`
- Collected artifacts include:
  - `runs/b4_mobilenet_softmask/resume_state.pth`
  - Partial `training_log.csv` through epoch 4
  - Validation metrics and best-model checkpoints
- Missing artifact: `test_metrics.json`
- Queue state: stopped. No further patches, retries, or downstream submissions were made.

## Public handoff state (2026-06-21)

- The current source includes a defensive conversion/fallback for serialized RNG state restoration, but it has not been validated by a successful resumed Kaggle run.
- The current generated runner does not pass `--resume-state`; this must be reconciled and dry-run-tested before any resubmission.
- b4 remains blocked and has no test metrics.
- `ours_iresnet_msff_softmask` remains not submitted.
- Repository policy now requires explicit owner approval before another Kaggle submission.
- Continuation procedure: `docs/HANDOFF.md`.
