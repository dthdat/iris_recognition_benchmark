# Iris Thesis Baseline Automation Rules

Project goal:
Build a reproducible iris-recognition baseline framework and run exactly 4 planned Kaggle trainings:
1. b1_arciris_nomask
2. b3_arciris_softmask
3. b4_mobilenet_softmask
4. ours_iresnet_msff_softmask

Hard rules:
- Do not delete or overwrite iris_baseline.ipynb.
- Do not fabricate metrics.
- Do not run more than the 4 planned full trainings.
- Do not submit all Kaggle jobs until b1 succeeds and split CSVs are collected.
- Do not commit or bundle secrets: kaggle.json, access_token, ~/.kaggle, .kaggle/.
- Kaggle outputs must be under /kaggle/working.
- Project code in Kaggle script kernels may live under /kaggle/src, so generated runners must cd to BUNDLE_ROOT or use absolute BUNDLE_ROOT paths.
- Dataset root must point to the folder that directly contains subject folders like 000, 001, 002.
- If a Kaggle run fails, inspect logs, patch, dry-run, then resubmit only the failed run.
- After the first successful Kaggle run, collect splits/ locally and reuse those frozen split CSVs for all future runs.

Current state (2026-06-21):
- b1 and b3 completed and their frozen splits/results were collected.
- b4 is incomplete after repeated SIGKILL failures. Its latest bounded restart then failed while restoring a serialized PyTorch RNG state.
- The current training code contains a defensive RNG-state conversion, but that resume path has not been validated by a successful Kaggle run.
- The current generated runner must be audited because its training command does not pass `--resume-state`.
- `ours_iresnet_msff_softmask` has not been submitted.
- A collaborator may continue b4 using the gated workflow in `docs/HANDOFF.md`: local test, infrastructure-only patch, compile, dry-run, bundle inspection, then one controlled submission.
- Stop on failure. Do not automatically resubmit or continue to the final run.
- See `docs/HANDOFF.md` for the exact continuation procedure.

Preferred continuation workflow:
1. Test the b4 resume path locally without full training.
2. Patch only the b4 resume/runner infrastructure issue and audit the runner.
3. Compile-check, dry-run, and inspect the b4 bundle.
4. Submit one corrected b4 attempt, poll cheaply, and collect once.
5. Stop on failure; on success verify `test_metrics.json` before the final run.
6. Run the final planned model only after b4 succeeds, then follow `docs/ROADMAP.md`.

# Low-cost Kaggle automation mode

Use the cheapest/smallest model for monitoring.
Do not use subagents for monitoring.
Prefer shell scripts for polling instead of reasoning loops.

Allowed automatic actions:
- Run Kaggle status commands using shell polling.
- Build and inspect a dry-run bundle.
- Submit one corrected b4 attempt after all preflight checks pass.
- Collect b4 once after it reaches a terminal state.
- After b4 succeeds and real test metrics exist, submit and collect the final planned run sequentially.
- Verify split CSVs and results files.
- Run local compile/unit checks that do not start full training.

Not allowed:
- Do not submit remaining runs before b1 succeeds and split CSVs exist.
- Do not submit multiple failed retries in a loop.
- Do not use subagents unless user explicitly requests it.
- Do not change scientific configs, model architecture, metrics, split logic, or epoch count.
- Do not touch or print Kaggle token/access_token/kaggle.json.
- Do not fabricate metrics.
- Do not delete iris_baseline.ipynb.

If a run fails:
- Stop automatic run queue.
- Try to collect logs/output.
- Patch only obvious infrastructure bugs.
- Dry-run the failed bundle.
- Write results/MONITOR_REPORT.md.
