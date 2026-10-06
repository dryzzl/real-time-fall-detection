# Architecture

The completed pipeline is intended to follow this flow:

`video source → pose estimator → person tracks → temporal features → state classifier → local alerts / overlay`

Milestone 7 includes synthetic input, OpenCV webcam/local-video capture, a CPU ONNX pose adapter, deterministic person tracking, normalized temporal feature windows, an explainable state baseline, temporal dataset/split contracts, and a verified future ONNX export boundary. Persistent alerting, evaluation, and end-to-end runtime integration remain later milestones. No trained temporal artifact is present.

## Current contracts

- `FramePacket`: sequential index, source-relative seconds, dimensions, source identifier, explicit timestamp basis, and immutable interleaved BGR bytes. The buffer length must match the dimensions.
- `Prediction`: an explicit `FallState`, optional confidence, and a machine-readable reason. `unknown` is distinct from `normal`; missing inference must never imply that a scene is safe.
- `Predictor`: accepts a frame and returns a prediction. The foundation implementation always returns unavailable. Tracking and feature extraction remain separate because no state classifier exists yet.
- `TrackSnapshot`: a process-local track ID, frame index, age, hit/miss counts, predicted box, and an optional current pose detection. `detection=None` explicitly represents a missed observation.
- `PoseFeatureSample`: timestamped box/motion values, box-relative keypoints, confidence mask, and explicit observation availability for one tracked person.
- `TemporalFeatureWindow`: one bounded, independently owned sample sequence with an observed count and readiness flag. Readiness is data availability, not a fall or safety state.
- `StateDecision`: a per-track `FallState`, machine-readable reason, explicit evidence, and no fabricated probability. `unknown` covers missing, insufficient, low-confidence, and ambiguous evidence.
- `TemporalSequence` / `SplitManifest`: fixed-shape labeled features with pseudonymous subject/session groups, authorization references, a dataset fingerprint, and deterministic subject-separated assignments.
- `TemporalModelContract`: a CPU-verified ONNX input/output and metadata boundary for future injected training backends; compatibility is not model quality.
- `run_pipeline`: consumes one frame at a time, checks temporal order, optionally paces replay, and emits metadata records without image bytes. An optional per-frame callback receives the frame and prediction after pacing and emission; returning false stops without reading another frame.

Source time and wall time are different. Features use source timestamps; runtime diagnostics use a monotonic clock. Synthetic throughput is not representative of future model latency.

## Capture lifecycle

`open_capture(CaptureConfig(...))` owns the OpenCV resource in a context manager. Its iterator reads one native-size BGR frame at a time. Callers must keep iteration inside the context, including when they stop early. The CLI uses an `ExitStack` so capture, preview, and output close on normal completion, errors, and interruption. The output file is acquired before camera access, so an existing output cannot trigger camera capture.

Camera timestamps use monotonic acquisition time. Video positions are relative to the first decoded position; if they are invalid or stop increasing, all remaining frames use estimated FPS timing. Every JSONL frame names its timestamp basis. FPS estimates cannot reconstruct variable-frame-rate timing. Video frame-count metadata detects some premature decode stops; EOF versus a silent decode failure remains ambiguous when that metadata is unavailable.

Preview is optional and loaded only when requested. It displays the pipeline's current prediction and reason, supports Q/Esc and window close, and never substitutes a normal state for unavailable inference. Headless mode makes no GUI calls. Camera-driver behavior and physical display support require local checks beyond mocked lifecycle tests.

## Planned choices
- Local JSONL events and overlays. No footage uploads or external notification service by default.

## Technical references

- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [OpenCV capture properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
- [OpenCV HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)
- [OpenCV Python package variants](https://pypi.org/project/opencv-python/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

The later integration milestones must verify the exact dependency versions and model formats they use.

## Pose boundary

`OnnxPoseEstimator` is deliberately separate from the fall-state `Predictor`. It turns one `FramePacket` into zero or more immutable pose detections; it does not classify safety or a fall. `PoseTrackingStage` now composes that boundary with `PersonTracker` before later feature and state stages consume per-person histories.

The adapter lazily imports NumPy and ONNX Runtime, uses only `CPUExecutionProvider`, validates graph metadata, and supports the raw single-batch YOLO11-pose layout documented in [MODEL.md](MODEL.md). Preprocessing records a reversible letterbox transform. Decoding maps boxes and keypoints back to clipped source-frame coordinates, filters invalid/low-confidence rows, and applies class-agnostic NMS. Missing weights, dependencies, incompatible graph metadata, and execution errors fail explicitly; none become an empty normal scene.

## Tracking lifecycle

`PersonTracker` canonicalizes detection order, predicts active boxes/keypoints with smoothed constant velocity, scores gated pairs using center distance, IoU, and confident-keypoint distance, and applies deterministic greedy assignment. New detections receive monotonically increasing IDs. Unmatched tracks remain active as explicit missing snapshots through `max_missed_frames`, then expire. Skipped frame indices count toward expiry.

The tracker is bounded by the number of recently active people and retains only the last detection and velocity per track. IDs reset with the process and are association handles, not real-world identities. See [TRACKING.md](TRACKING.md) for configuration, deterministic behavior, and limitations including the absence of appearance re-identification or global assignment.

## Temporal feature lifecycle

`TemporalFeatureBank` consumes the complete active `TrackSnapshot` set with its corresponding `FramePacket`. It normalizes box geometry to frame dimensions, pose geometry to each detection box, and center motion to source time. Confidence masks distinguish missing joints from valid zero-valued coordinates. Missing track observations produce missing samples rather than repeated poses; motion is not bridged across missing samples or excessive time gaps.

Each active track owns a fixed-size deque. Histories are independent, include missing slots, and are deleted after the tracker retires an ID, so storage is bounded by active people times configured window size. `PoseTrackingFeatureStage` composes pose tracking and feature extraction without exposing a fall state. See [FEATURES.md](FEATURES.md) for the exact fields, validation rules, readiness semantics, and synthetic smoke check.

## Explainable state lifecycle

`TemporalStateClassifier` consumes one complete feature window and returns a `StateDecision`. It first rejects missing, unready, low-confidence, or non-contiguous evidence as `unknown`. Settled lying posture maps to `fallen`; a recent upright-to-wider transition with sufficient downward displacement and speed maps to `falling`; stable upright posture maps to `normal`; every other case remains `unknown`.

`TemporalStateStage` composes an arbitrary feature stage with deterministic, track-ordered classification. Decisions expose the relevant thresholds and measurements but keep confidence null because the rules have not been statistically calibrated. The existing `run_pipeline` default still uses `UnavailablePredictor`; this boundary prevents a library-level synthetic baseline from being presented as working end-to-end video detection. See [STATE_BASELINE.md](STATE_BASELINE.md).

## Temporal training and export boundary

`training.py` converts exact-length `TemporalFeatureWindow` values into a versioned numeric layout with time offsets, observation flags, normalized box/motion values, box-relative keypoints, and explicit joint masks. The same validator reads JSONL sequences and rejects partial, duplicate, non-finite, mislabeled, or structurally incompatible data. `unknown` is excluded from supervised targets.

Splits are grouped by pseudonymous subject, so every session and sequence for that subject stays in one partition. Deterministic hashing with the configured seed makes assignments independent of file order. The manifest preserves the dataset SHA-256, schema, feature and class order, and assignments. A training plan stays blocked when data is absent or the training split lacks a supervised class.

`TemporalModelExporter` is an injection boundary rather than a bundled training framework. Its output is staged, loaded on CPU, checked for exact feature/logit shapes and schema metadata, exercised with a finite smoke tensor, and only then published without overwriting an existing artifact. The default CLI does not invoke a trainer. No dataset or learned model exists in the repository, so model availability and end-to-end fall detection remain false. See [TRAINING.md](TRAINING.md).
