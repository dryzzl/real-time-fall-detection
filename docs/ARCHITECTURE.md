# Architecture

The completed pipeline is intended to follow this flow:

`video source → pose estimator → person tracks → temporal features → state classifier → local alerts / overlay`

Milestone 2 includes synthetic input, OpenCV webcam/local-video capture, the stream runner, the prediction interface, and optional desktop preview. Pose inference and downstream detection stages remain unimplemented.

## Current contracts

- `FramePacket`: sequential index, source-relative seconds, dimensions, source identifier, explicit timestamp basis, and immutable interleaved BGR bytes. The buffer length must match the dimensions.
- `Prediction`: an explicit `FallState`, optional confidence, and a machine-readable reason. `unknown` is distinct from `normal`; missing inference must never imply that a scene is safe.
- `Predictor`: accepts a frame and returns a prediction. The foundation implementation always returns unavailable. A later orchestration layer will carry multiple tracked people and separate per-person histories.
- `run_pipeline`: consumes one frame at a time, checks temporal order, optionally paces replay, and emits metadata records without image bytes. An optional per-frame callback receives the frame and prediction after pacing and emission; returning false stops without reading another frame.

Source time and wall time are different. Features will use source timestamps; runtime diagnostics use a monotonic clock. Synthetic throughput is not representative of future model latency.

## Capture lifecycle

`open_capture(CaptureConfig(...))` owns the OpenCV resource in a context manager. Its iterator reads one native-size BGR frame at a time. Callers must keep iteration inside the context, including when they stop early. The CLI uses an `ExitStack` so capture, preview, and output close on normal completion, errors, and interruption. The output file is acquired before camera access, so an existing output cannot trigger camera capture.

Camera timestamps use monotonic acquisition time. Video positions are relative to the first decoded position; if they are invalid or stop increasing, all remaining frames use estimated FPS timing. Every JSONL frame names its timestamp basis. FPS estimates cannot reconstruct variable-frame-rate timing. Video frame-count metadata detects some premature decode stops; EOF versus a silent decode failure remains ambiguous when that metadata is unavailable.

Preview is optional and loaded only when requested. It displays the pipeline's current prediction and reason, supports Q/Esc and window close, and never substitutes a normal state for unavailable inference. Headless mode makes no GUI calls. Camera-driver behavior and physical display support require local checks beyond mocked lifecycle tests.

## Planned choices
- ONNX Runtime for pose inference on CPU, with explicitly configured model input/output contracts.
- A simple, documented tracking baseline before any stronger tracking integration.
- An explainable temporal baseline so the program has a testable detection path before a learned classifier is available.
- Optional temporal-model training/export with subject/session-separated evaluation. Real model quality remains contingent on suitable, authorized data.
- Local JSONL events and overlays. No footage uploads or external notification service by default.

## Technical references

- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [OpenCV capture properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
- [OpenCV HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)
- [OpenCV Python package variants](https://pypi.org/project/opencv-python/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

The later integration milestones must verify the exact dependency versions and model formats they use.
