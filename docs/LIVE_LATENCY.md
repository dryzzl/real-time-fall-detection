# Bounded live-processing latency profiling

Use the opt-in profiler to measure processing stages for each decoded frame:

```bash
fall-detection-live --video path/to/authorized-video.mp4 --model path/to/pose.onnx \
  --headless --output outputs/live.jsonl --latency-report outputs/live-latency.json
python -m fall_detection.live_stress --frames 150 --people 4 \
  --width 160 --height 120 --profile-latency
```

The live CLI writes the latency report as a **new** JSON file (no overwrite).
The synthetic stress CLI embeds the same profile in its JSON report.
The optional profiler keeps only the most recent 2,048 frame samples,
with a separate total frame count, preventing unbounded growth in a long run.
Each stage reports minimum, mean, nearest-rank p50, p95, p99, and maximum
in milliseconds. An empty run has null latency statistics.

Stage definitions:

- `pose_tracking_features_ms`: pose estimator, person tracking, and temporal features.
- `classification_ms`: heuristic or optional temporal ONNX classification.
- `record_emit_ms`: JSON record construction and emission callback.
- `total_ms`: the sum of the three processing stages, measured before UI callbacks.

**Not included:** frame acquisition, video decoding, UI preview, notification
delivery, or camera-to-alert end-to-end latency. Synthetic stress inputs use
fake upright pose observations, not camera frames or trained models. These
figures are diagnostics only, not fall-detection accuracy, safety validation,
or proof that the system meets real-time performance on edge devices.
Performance depends on hardware, input size, models, and runtime environment.

Run regression and synthetic integration checks with:

```bash
python -m unittest tests.test_live_latency tests.test_live_stress -v
python -m fall_detection.live_stress --frames 150 --people 4 --width 160 --height 120 --profile-latency
```
