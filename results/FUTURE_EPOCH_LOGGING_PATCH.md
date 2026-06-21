# Future epoch logging patch proposal

> Archived proposal: the current `experiments/train.py` already uses flushed progress output and batch heartbeats. Do not apply this patch again. See `docs/HANDOFF.md` for the current b4 blocker.

Current evidence:
- `experiments/train.py` prints per-epoch metrics to stdout with a line beginning `Epoch {epoch+1:02d}/{config['epochs']}`.
- The epoch metrics print does not pass `flush=True`.
- The current Kaggle job must not be modified while running.

Recommended future-only patch:

```diff
diff --git a/experiments/train.py b/experiments/train.py
--- a/experiments/train.py
+++ b/experiments/train.py
@@
         print(
             f"Epoch {epoch+1:02d}/{config['epochs']} | loss {train_loss:.4f} acc {train_acc:.4f} | "
             f"open-val EER {val_metrics['eer']:.3f}% AUC {val_metrics['auc']:.4f} "
-            f"TAR@0.1%FAR {val_metrics['tar_at_01far']*100:.2f}% | {epoch_seconds:.1f}s"
+            f"TAR@0.1%FAR {val_metrics['tar_at_01far']*100:.2f}% | {epoch_seconds:.1f}s",
+            flush=True,
         )
```

Optional same-category additions for future runs:
- Add `flush=True` to startup context prints in `experiments/train.py`.
- Add `flush=True` to the `Saved best EER model`, `Early stopping`, and `Training complete` prints.

Do not apply this to a submitted/running Kaggle job. Apply before generating the next Kaggle bundle if live epoch visibility is needed.
