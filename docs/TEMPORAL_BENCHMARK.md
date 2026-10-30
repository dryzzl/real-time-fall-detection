# Temporal ONNX CPU microbenchmark

This tool measures the **Python feature-encoding and ONNX Runtime CPU classification**
path for one already-complete synthetic temporal feature window. It does **not**
measure video capture, YOLO pose estimation, tracking, UI rendering, camera-to-alert
latency, detection accuracy, or safety suitability.

## Reproducible CI fixture

Install optional dependencies: python -m pip install -e ".[headless,model,test]"

Run an actual ONNX Runtime session against a generated, contract-valid **untrained**
constant-logit graph:

    python -m fall_detection.temporal_benchmark --synthetic-fixture --iterations 100 --warmup 10

The fixture uses constant logits and cannot detect falls. Its only purpose is to
test that the model interface, CPU runtime, feature encoding, and timing/reporting
work. All input windows are synthetic and observed, so the unknown-state gating
path is covered separately in temporal adapter tests.

## Benchmark your own compatible model

    python -m fall_detection.temporal_benchmark --model models/temporal_state.onnx --training-config configs/training.toml --iterations 500 --warmup 30 --report outputs/temporal_benchmark.json

The supplied model must satisfy the strict input/output/metadata contract in
[TEMPORAL_INFERENCE.md](TEMPORAL_INFERENCE.md). The benchmark verifies the model
before timing. The output includes its SHA-256, input shape, state counts, and
nearest-rank min/p50/p95/p99/max per-inference milliseconds, excluding warmup.
Results are platform- and load-dependent. The reported times include Python
feature encoding and ONNX Runtime inference, but **exclude** video and pose.

For an optional performance gate add --max-p95-ms 50. A measured p95 above
that value exits with code 1; invalid inputs and missing model files exit 2.
The report path is created exclusively and existing files are never overwritten.
Do not set hard timing thresholds in shared CI without accounting for runner
variance.

## Safety and interpretation

- A synthetic constant model's output is not an accuracy or recall measurement.
- An arbitrary user-supplied model is not presumed trained or validated.
- Softmax outputs are uncalibrated research scores, not validated probabilities.
- The benchmark never sets fall_detection_available=true.
- Real-world evaluation requires authorized, unseen videos, appropriate subject
  separation, camera-angle variation, and an end-to-end latency study.

Run python -m unittest tests/test_temporal_benchmark.py -v to verify
deterministic timing math, input validation, report safety, and actual CPU ONNX
fixture inference.
