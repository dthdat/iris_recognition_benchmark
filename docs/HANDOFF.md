# Engineering handoff

This is the continuation document for a human collaborator, Codex, or Claude Code. Read `AGENTS.md` first; its restrictions are mandatory.

## What is complete

- The protected source notebook remains in `iris_baseline.ipynb`.
- Reusable implementation exists under `src/` and `experiments/`.
- Four fixed YAML configurations exist under `experiments/configs/`.
- Frozen subject-exclusive splits are committed under `splits/`: 700 train subjects/14,000 images, 100 validation subjects/2,000 images, and 200 test subjects/4,000 images.
- b1 and b3 completed on Kaggle and their test metrics are recorded in `README.md` and `results/MONITOR_REPORT.md`.
- Compact reports and plots are retained. Checkpoints and raw run/Kaggle directories remain local-only and are ignored.

## Exact current blocker

`b4_mobilenet_softmask` is incomplete. The current YAML uses MobileFaceNet even though the historical run ID says `mobilenet`.

The latest collected attempt was killed by `SIGKILL` during epoch 5. A bounded process restart then found a checkpoint saved after epoch 4, but resume failed while restoring PyTorch RNG state because the deserialized value was not accepted as a `torch.ByteTensor`.

The current `experiments/train.py` contains a defensive RNG-state conversion and warning-based fallback (`as_cpu_byte_tensor` and `restore_rng_state`). This code has not been validated by a successful resumed Kaggle run. Also verify that the generated Kaggle runner actually passes `--resume-state` and implements only the intended bounded restart; the current `tools/kaggle_submit.py` training call does not pass that option.

Do not claim the blocker is fixed until the resume path has been tested. The next collaborator may own b4 end-to-end, but only through the gated sequence below. Never skip directly to submission and never infer success from partial validation metrics.

## Required b4 continuation sequence

1. Create a focused unit test for resume serialization/restoration using CPU tensors. It should save a minimal checkpoint, load it through `safe_torch_load`, restore Python/NumPy/PyTorch RNG state, and prove training can advance from `next_epoch` without the dataset.
2. Reconcile `tools/kaggle_submit.py` with the intended bounded-restart behavior. Keep retry count bounded to one child-process restart; never create a kernel resubmission loop.
3. Run `python -m compileall -q src experiments tools`.
4. Generate a b4 dry-run bundle. Confirm it contains all six frozen split CSVs, no credential files, the unchanged b4 scientific config, the correct project `cwd`, `/kaggle/working` outputs, and the intended resume argument/restart bound.
5. Update `results/MONITOR_REPORT.md` with the local verification only. Do not change scientific status or add metrics.
6. Submit one corrected b4 kernel. Poll with a shell script/status command rather than an agent reasoning loop.
7. At the terminal state, collect output once. If it failed, document the failure and stop without resubmitting.
8. If b4 succeeded, require `test_metrics.json`, verify its values from the file, regenerate summaries, and commit compact reports only.
9. Only after b4 succeeds, submit `ours_iresnet_msff_softmask` sequentially. Apply the same stop-on-failure and real-artifact rules.
10. After all four real results exist, follow `docs/ROADMAP.md`.

## Kaggle boundary

Use the single-run submit/status/collect tools described in `docs/KAGGLE_AUTOMATION.md`. Do not use the all-runs or parallel queue from the current state. Never request, print, copy, or commit Kaggle credentials. A failed terminal state always stops progression.

## How to use Codex

Open Codex at the repository root. A safe first prompt is:

> Read AGENTS.md and docs/HANDOFF.md. Own the b4 infrastructure recovery without changing scientific configuration. First add a CPU resume-state test, patch the bounded runner resume path, compile, and inspect a dry-run bundle. Only then submit one b4 attempt. Poll cheaply, collect once, stop on failure, and never fabricate metrics. If all four runs eventually complete, follow docs/ROADMAP.md.

Codex should automatically discover `AGENTS.md`. Still state the one-attempt and stop-on-failure boundaries explicitly in the prompt.

## How to use Claude Code

Open Claude Code at the repository root. `CLAUDE.md` is the entry point. Use the same first task as above and require a diff plus test output before accepting changes. Review tool permissions carefully; local `.claude/` settings are intentionally excluded from Git.

## Verification checklist before any pull request

- `iris_baseline.ipynb` is unchanged.
- `git diff --check` passes.
- Python compile checks pass.
- No credential-like or private environment file is tracked.
- No `.pth`, `.onnx`, `.engine`, raw Kaggle output, generated bundle, run directory, or log is tracked.
- All six split CSVs exist and are non-empty.
- Any reported metric points to a real collected result.
- Documentation distinguishes completed, partial, blocked, and not-submitted runs.
- Scientific changes, if explicitly approved, are called out separately from infrastructure changes.
- Any Kaggle submission was preceded by passing preflight checks and was not an automatic failed retry.
