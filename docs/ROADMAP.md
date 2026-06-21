# Roadmap after all four models

Start this roadmap only when all four planned runs have real collected `test_metrics.json` files on the same frozen split. Missing results must remain missing; never estimate them from validation logs.

## 1. Freeze and audit the comparison

- Re-run `experiments/aggregate_results.py` and verify every table value against its source JSON.
- Record commit hash, config, split files, Kaggle kernel version, environment versions, and checkpoint checksum for each run.
- Compare b1 vs b3 for the mask effect, b3 vs the final MSFF model for the architecture effect, and b4 for the lightweight accuracy/size tradeoff.
- Add uncertainty or repeated-pair-sampling analysis before making strong thesis claims.

## 2. Verification analysis from saved checkpoints

- Run threshold sensitivity with `experiments/threshold_sweep.py`.
- Run same-side/eye-side evaluation with `experiments/same_side_eval.py`.
- Run margin analysis with `experiments/margin_sweep.py` only when it operates on saved artifacts and does not silently retrain.
- Report ROC, EER, TAR at the target FAR, FAR/FRR at the validation-selected threshold, and genuine/impostor distributions.
- Check demographic or acquisition-condition limitations supported by available metadata; do not invent unavailable subgroup labels.

## 3. Deployment artifacts

- Export the chosen embedding network with `experiments/export_onnx.py`.
- Validate ONNX output parity against PyTorch on a fixed sample batch.
- Generate the quantization plan with `experiments/quantization_plan.py`; distinguish planned, simulated, and measured results.
- Benchmark FP32/FP16/INT8 only on the actual target hardware. Record latency distribution, throughput, peak memory, model size, power mode, warm-up, and software versions.
- Keep preprocessing/segmentation latency separate from embedding-network latency, then report end-to-end latency as well.

## 4. Jetson evaluation

- Build the TensorRT engine on the target Jetson rather than committing the engine binary.
- Measure batch size 1 latency after warm-up and report median plus tail latency.
- Confirm accuracy after each export/quantization step using the frozen test protocol.
- Select the final model using both verification accuracy and deployment constraints, not EER alone.

## 5. Reproducibility and thesis delivery

- Add automated unit tests for split loading, preprocessing shapes, mask determinism, model output normalization, metric edge cases, checkpoint resume, and export parity.
- Add a clean-environment smoke workflow that never launches full training.
- Produce the final results table, ablation table, failure analysis, limitations, and reproducibility appendix.
- Document dataset licensing/access separately; do not redistribute CASIA images.
- Publish compact reports and code only. Keep credentials, datasets, raw Kaggle outputs, weights, ONNX/TensorRT binaries, and machine-specific logs outside Git.

## Definition of done

- Four verified test results exist on the identical frozen split.
- Every headline number has a machine-readable source artifact.
- Exported-model parity and target-device measurements are reproducible.
- Scientific conclusions distinguish measured evidence from hypotheses and future work.
