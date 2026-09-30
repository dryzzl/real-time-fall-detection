# Architecture

The completed pipeline is intended to follow this flow:

`video source → pose estimator → person tracks → temporal features → state classifier → local alerts / overlay`

Only the source contract, synthetic source, stream runner, and prediction interface exist in milestone 1.

## Current contracts

- `FramePacket`: sequential index, source-relative seconds, dimensions, source identifier, and immutable interleaved BGR bytes. The buffer length must match the dimensions.
- `Prediction`: an explicit `FallState`, optional confidence, and a machine-readable reason. `unknown` is distinct from `normal`; missing inference must never imply that a scene is safe.
- `Predictor`: accepts a frame and returns a prediction. The foundation implementation always returns unavailable. A later orchestration layer will carry multiple tracked people and separate per-person histories.
- `run_pipeline`: consumes one frame at a time, checks temporal order, optionally paces replay, and emits metadata records without image bytes.

Source time and wall time are different. Features will use source timestamps; runtime diagnostics use a monotonic clock. Synthetic throughput is not representative of future model latency.

## Planned choices

- OpenCV for local video/webcam I/O and optional display.
- ONNX Runtime for pose inference on CPU, with explicitly configured model input/output contracts.
- A simple, documented tracking baseline before any stronger tracking integration.
- An explainable temporal baseline so the program has a testable detection path before a learned classifier is available.
- Optional temporal-model training/export with subject/session-separated evaluation. Real model quality remains contingent on suitable, authorized data.
- Local JSONL events and overlays. No footage uploads or external notification service by default.

## Technical references

- [OpenCV VideoCapture](https://docs.opencv.org/4.10.0/d8/dfe/classcv_1_1VideoCapture.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

The later integration milestones must verify the exact dependency versions and model formats they use.
