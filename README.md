# Real-Time Fall Detection

A Python project for detecting falls from video through pose estimation, motion tracking, and temporal analysis.

**Development status: milestone 8 of 10 — persistent local alert lifecycle.** This version streams synthetic frames, a local video, or a webcam; provides pose/tracking/feature/state libraries; validates temporal training/export contracts; and now supports latched per-person alerts with local acknowledgment, guarded reset, cooldown, duplicate suppression, JSONL transition logs, and preview overlays. No authorized labeled dataset or trained temporal artifact is present, and default streaming predictions remain `unknown` with `detection_not_implemented`.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/dryzzl/real-time-fall-detection.git
cd real-time-fall-detection
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Choose **one** installation:

```bash
# Desktop capture, preview, and pose inference
python -m pip install -e ".[video,model]"
```

```bash
# Server/CI capture and pose inference, without GUI dependencies
python -m pip install -e ".[headless,model]"
```

The OpenCV variants share the `cv2` namespace. Do not install both in the same environment; create a fresh virtual environment to switch. The supported range currently excludes OpenCV 4.14 because its Linux wheel crashed during import in the milestone environment; 4.13 passed the full suite. `python -m pip install -e .` still supports the synthetic demo without third-party runtime dependencies. The `model` extra installs CPU ONNX Runtime and NumPy.

## Run

```bash
fall-detection status
fall-detection demo --frames 60
fall-detection demo --config configs/default.toml --realtime --output outputs/demo.jsonl

# Local webcam; Q/Esc, window close, or Ctrl+C stops capture
fall-detection capture --camera 0 --preview

# Bounded webcam session without a window
fall-detection capture --camera 0 --headless --max-frames 150 --output outputs/camera.jsonl

# Replay an existing local video at source speed
fall-detection capture --video data/sample.mp4 --preview --realtime

# Decode as fast as possible without a window
fall-detection capture --video data/sample.mp4 --headless --output outputs/video.jsonl

# After supplying an authorized raw YOLO11-pose ONNX export
fall-detection pose-check --config configs/pose.toml
fall-detection pose-check --config configs/pose.toml --smoke

# Dependency-free tracking regression check
fall-detection track-smoke --config configs/tracker.toml

# Dependency-free temporal-feature regression check
fall-detection feature-smoke --config configs/features.toml

# Dependency-free temporal-state sequence check
fall-detection state-smoke --config configs/state.toml

# Validate the training schema/split implementation without training
fall-detection training-check --config configs/training.toml --smoke

# Honest current readiness check; reports the missing authorized dataset
fall-detection training-check --config configs/training.toml

# Validate a future temporal model's ONNX graph and CPU smoke inference
fall-detection temporal-model-check --config configs/training.toml \
  --model models/temporal_state.onnx

# Dependency-free persistent-alert lifecycle check
fall-detection alert-smoke --config configs/alerts.toml

# Same check with a new local metadata-only event journal
fall-detection alert-smoke --config configs/alerts.toml \
  --event-log outputs/alert-smoke.jsonl
```

`python -m fall_detection` is an alternative to the installed command. Camera index `0` usually selects the default webcam; another index may be needed. Video paths must name existing local files. Streaming URLs are not supported. Decoded frames retain their native dimensions. `configs/default.toml` configures the synthetic demo only.

Capture defaults to headless mode. Without `--max-frames`, videos run to their end and cameras run until interrupted. `--realtime` paces video or synthetic replay; cameras already run live. Preview needs a local graphical session and the desktop extra. A headless build or missing Linux display produces an actionable error.

Output files must be new; existing files are never overwritten. JSONL contains metadata, not image bytes. Webcam/video pixels stay in process memory or the requested local preview; the application does not record or upload footage. Completed records survive interruption or failure. Exit code `0` means a successful stop, `2` an input/runtime error, and `130` Ctrl+C. Errors and interruption do not emit a success summary.

Example video frame record:

```json
{"event":"frame","source":"video","frame_index":0,"timestamp_seconds":0.0,"timestamp_basis":"video_position","width":640,"height":480,"state":"unknown","confidence":null,"reason":"detection_not_implemented"}
```

A final summary reports processed frames, source duration, elapsed time, and pipeline throughput. These are execution diagnostics, not fall-detection performance metrics.

## Pose model contract

`pose-check` validates a single-input/single-output float ONNX graph, forces `CPUExecutionProvider`, and optionally runs a black test frame through preprocessing, inference, and output decoding. The adapter performs aspect-preserving bilinear letterboxing, BGR-to-RGB conversion, float32 NCHW normalization, coordinate restoration, confidence filtering, and class-agnostic NMS. It accepts raw YOLO11-pose output with 17 COCO keypoints by default; thresholds, input dimensions, keypoint count, and model path are configurable in `configs/pose.toml` or through CLI overrides.

The repository does not contain or download weights. Ultralytics offers YOLO11 models under AGPL-3.0 and Enterprise licenses; select an authorized model and suitable license before use. See [the exact model/output contract and licensing notes](docs/MODEL.md). `PoseTrackingStage` composes an estimator with the tracker, but it does not produce fall states.

## Person tracking

The tracking baseline combines smoothed constant-velocity prediction, normalized center distance, predicted-box overlap, and confident-keypoint distance. IDs are assigned deterministically, survive a configurable number of missed frames, and expire rather than silently reviving stale people. Missed tracks emit `detection=None` so later stages can distinguish an absent observation from an unchanged pose. Parameters are validated from `configs/tracker.toml`.

`track-smoke` runs two synthetic people through a clean crossing, reverses detector order between frames, and recovers one deliberately missed observation. This validates deterministic code behavior only. The baseline is not BoT-SORT and has no appearance model, camera-motion compensation, global assignment, or cross-session identity. Identity switches remain possible in crowded scenes, abrupt motion, or long occlusion. See [the tracking contract and limitations](docs/TRACKING.md).

## Temporal pose and motion features

Each active track has an independent fixed-size history. Observed samples contain frame-normalized box center/size, box aspect ratio, source-time center velocity, and box-relative keypoint coordinates with a separate confidence mask. Low-confidence joints become explicit missing values rather than zero coordinates. A missed detection becomes an explicit missing sample; recovery does not invent motion across the gap. Histories for expired tracker IDs are removed.

Window length, expected keypoint count, confidence threshold, readiness minimum, and maximum motion interval are validated from `configs/features.toml`. `feature-smoke` exercises two tracks, a masked joint, a missed observation, recovery, and bounded JSON output. No feature window is a fall state, and this milestone does not make fall detection available. See [the temporal feature contract](docs/FEATURES.md).

## Explainable state baseline

`TemporalStateClassifier` applies configurable rules to each ready feature window. A rapid downward center shift plus a widening posture is `falling`; a wide posture that remains still for the settled duration is `fallen`; a stable upright posture is `normal`. Missing, low-confidence, short, interrupted, or ambiguous evidence is always `unknown`. Each decision includes the observed measurements and a reason string.

The baseline returns `confidence: null` because its thresholds have not been calibrated as probabilities. `state-smoke` checks standing, lying, descent, and missing sequences without video or model weights. The baseline is available as a library stage, but end-to-end fall detection remains unavailable in `demo` and `capture` until later integration and real-input evaluation. See [the state rules, configuration, and limitations](docs/STATE_BASELINE.md).

## Temporal-model preparation

Training records use a versioned, fixed-shape JSONL contract with pseudonymous subject/session groups, an authorization reference, one of the three supervised labels (`normal`, `falling`, `fallen`), and explicit missing-data masks. The deterministic splitter keeps every sequence and session for a subject in exactly one of train, validation, or test and fingerprints the complete dataset in the generated manifest. `unknown` is never converted into a supervised target.

The repository defines an injected training/export interface but does not choose a trainer or fabricate weights. A future export must expose the configured `[batch, sequence_length, feature_width]` float input, three ordered logits, required schema metadata, and a finite CPU smoke result. Dataset files, footage, and model artifacts remain excluded from Git. See [the full dataset, split, and temporal ONNX contract](docs/TRAINING.md).

## Persistent local alerts

`AlertManager` requires a configurable continuous `fallen` interval before opening an alert. Once opened, that incident remains latched through later `normal` or `unknown` decisions. Local acknowledgment records that the alert was seen but does not clear it. Reset is allowed only after acknowledgment and a current `normal` observation; cooldown then suppresses duplicates until another `normal` decision explicitly rearms the track.

Transition events can be written to a validated append-only local JSONL journal with per-event flush. Records contain state metadata only—never images, keypoints, credentials, recipients, or network destinations. Alert snapshots provide pending/active/acknowledged banners to the optional preview. The smoke command never overwrites an existing log and performs no external action. See [the lifecycle, event schema, overlay, and safety limitations](docs/ALERTS.md).

## Timing and capture limitations

| Timestamp basis | Meaning |
| --- | --- |
| `synthetic_fps` | Frame index divided by configured synthetic FPS |
| `camera_monotonic` | Time since the first successful camera read, using a monotonic clock; not sensor exposure time |
| `video_position` | Decoder position in seconds, relative to the first decoded frame |
| `video_fps` | Estimated timing from reported FPS after missing, repeated, or backward video positions |
| `video_fallback_fps` | Estimated timing from `--fallback-fps` (default 30) when both positions and reported FPS are unusable |

Video timing switches permanently to FPS estimates when positions become unreliable. Those estimates cannot recover true variable-frame-rate timing. FPS values outside `(0, 240]` are treated as unusable. A camera read failure, an undecodable first frame, a decoder exception, or termination before the reported video frame count is an error. When a file has no usable frame count, OpenCV cannot reliably distinguish ordinary EOF from a silent decode failure after valid frames. Codec and camera-driver support depend on the local OpenCV backend; a blocking device driver can delay interruption.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers configuration, frame contracts, stream ordering, capture cleanup, model metadata, letterbox transforms, output decoding, person tracking, temporal features, state decisions, dataset validation, subject-separated splits, ONNX export safeguards, alert persistence, acknowledgment/reset/cooldown, duplicate suppression, local journal integrity, preview banners, and invalid inputs. Test extras generate an MJPG video and tiny constant-output ONNX contract fixtures in temporary directories. Synthetic tracking/features/state/training/alert records and constant graphs contain no real people or trained weights and are not quality measurements. CI runs on Linux and Windows with Python 3.11 and 3.12.

Physical webcam/display validation is separate from automated tests. On a local desktop, run the preview and bounded-camera commands above; verify that frames appear, Q/Esc and window close stop cleanly, Ctrl+C releases the device, and the camera can reopen. Also check that disconnecting it reports an error. **These hardware checks have not been performed in this development environment.** See [the progress log](docs/PROGRESS.md), [roadmap](docs/ROADMAP.md), and [architecture](docs/ARCHITECTURE.md).

## Scope

This is a development prototype, not a medical or emergency-response device. There are no accuracy, recall, or clinical-validation claims for this repository. The synthetic sources and generated test video do not depict a person or a fall. State, training-contract, and alert-lifecycle smoke tests prove code behavior, not real-world quality. The project has no external notification or emergency-service integration. A trained temporal model requires suitable authorized labeled sequences; the current artifact is explicitly blocked rather than replaced by synthetic training or invented results.
