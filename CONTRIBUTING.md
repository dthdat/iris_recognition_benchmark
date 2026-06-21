# Contributing

## Start here

1. Read `AGENTS.md` and `docs/HANDOFF.md` completely.
2. Check `git status` and preserve unrelated work.
3. Create a branch from the current default branch.
4. Keep scientific changes separate from infrastructure fixes.
5. Run the checks listed below and describe exactly what was and was not tested.

## Development checks

Checks that do not require the private dataset or a GPU:

```bash
python -m compileall -q src experiments tools
```

Before b4, add a focused resume-state test using synthetic/CPU data. Then generate a dry-run bundle and inspect it before any real submission. A collaborator with configured Kaggle access may submit exactly one corrected b4 attempt, monitor it with the shell polling workflow, and collect it once. Stop on failure; never create an automatic resubmission loop.

## Pull requests

- Explain the problem, implementation, and verification.
- State whether configs, architecture, metrics, split logic, or epoch count changed.
- Never commit datasets, weights, raw Kaggle outputs, bundles, logs, credentials, or local agent settings.
- Never invent or round-trip missing metrics into reports.
- Keep the original notebook unchanged unless the owner explicitly authorizes a notebook edit.
- Keep b4 infrastructure fixes separate from later scientific/ablation work.

## Result reporting

Only report metrics read from collected machine-generated result files. Incomplete runs remain `pending` or `blocked`. Record the source run, threshold selection method, and split identity with any new result.
