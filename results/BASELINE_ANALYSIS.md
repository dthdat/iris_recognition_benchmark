# Baseline Analysis

No final scientific conclusions are written yet because one or more planned runs are missing.

Pending runs should be completed and collected from Kaggle before thesis conclusions are finalized.

Strongest baseline so far: `b1_arciris_nomask` by lowest test EER.

Comparisons to fill after all four runs are complete:
- Soft mask vs no mask: compare `b3_arciris_softmask` against `b1_arciris_nomask`.
- MobileFaceNet Jetson acceptability: compare `b4_mobilenet_softmask` accuracy and model size against thesis constraints.
- MSFF contribution: compare `ours_iresnet_msff_softmask` against `b3_arciris_softmask`.
- Final thesis model: choose only after all four complete test metrics are present.

Limitations:
- Results use pair-sampled open-set verification metrics, so confidence improves with larger pair counts.
- The frozen split must be reused for all four runs.
- Quantization and Jetson latency should be reported from exported checkpoints, not retraining.

Next steps:
- Continue b4 only through the tested, single-run, stop-on-failure workflow in `docs/HANDOFF.md`.
- Run the final planned model only after b4 succeeds and its real test metrics are verified.
- Re-run this aggregation script only after a real completed result is collected.
- Run threshold, margin, same-side, and quantization analyses from saved checkpoints only.
