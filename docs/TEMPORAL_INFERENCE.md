# Temporal ONNX inference (experimental)

The live pipeline can optionally run a three-class temporal ONNX model after
pose estimation, tracking, and per-track feature extraction. No trained model
weights are bundled, and this integration is **not** a validated safety detector.

## Interface and safeguards

The ONNX graph must satisfy the existing contract in `training.py`:

- Input `features`: float32 tensor `[batch, sequence_length, feature_width]`
- Output `logits`: float32 tensor `[batch, 3]`
- Class order: `normal, falling, fallen`
- Model metadata: `model_contract_version=1`,
  `dataset_schema_version=1`, `class_names=normal,falling,fallen`
- `feature_width = 10 + 3 * keypoint_count` (61 with 17 keypoints)

At startup, the adapter verifies graph shape, metadata, and a finite smoke
output. Runtime windows must have exactly the configured sequence length, a
currently observed person, and at least 80% observed samples. Incomplete
windows produce `unknown` with no confidence. Malformed or nonfinite model
outputs raise an explicit error rather than becoming `normal`.

The per-person record includes a softmax score and
`reason=onnx_temporal_unvalidated`. This score is **not calibrated** and
must not be interpreted as a validated probability of a real-world fall.
The frame-level state remains `unknown` and
`fall_detection_available=false` until independent validation.

## Local invocation

Install `pip install -e ".[headless,model,test]"` and provide your own
compatible YOLO pose export and temporal model:

```bash
fall-detection-live --video local_video.mp4 \
  --model models/pose.onnx \
  --temporal-model models/temporal_state.onnx \
  --training-config configs/training.toml \
  --output outputs/temporal.jsonl --headless
```

The training sequence length and keypoint count must match
`configs/features.toml`. The output file must not already exist.
Omit `--temporal-model` to use the existing unvalidated heuristic baseline.

## Verification

```bash
python -m unittest tests/test_temporal_inference.py -v
python -m unittest discover -s tests -v
```

The regression tests inject synthetic logits and pose-feature windows; they
verify interface behavior only, **not** accuracy, sensitivity, latency, or
clinical suitability. Real-video evaluation with authorized footage,
subject-disjoint splits, and separately audited alert behavior is still needed.
