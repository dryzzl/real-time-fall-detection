# Development progress

## Milestone 1 — foundation

Date: 2026-09-29

Status: complete (1 of 10 planned milestones).

Implemented:

- Python package and CLI with `status` and `demo` commands.
- Strict TOML configuration and validated frame/prediction data contracts.
- Deterministic synthetic BGR frames and a bounded-memory streaming pipeline.
- Optional realtime pacing, JSONL output, safe handling of existing output files.
- Tests for data validity, temporal order, CLI behavior, and unavailable inference.
- Architecture notes and the ten-milestone roadmap.

Current limitations: no webcam input, pose model, tracking, fall classifier, or alerting. Demo predictions are always `unknown`. No real-world detection metrics are available.

Next milestone: implement and test local camera/video input and shutdown behavior.

## Validation evidence

Environment: Python 3.12.14 on Linux.

- `python -m pip install --no-build-isolation --no-deps -e .`: passed using the available build backend.
- `python -m unittest discover -s tests -v`: 16 tests passed.
- `python -m fall_detection status`: reported milestone 1/10 and detection unavailable.
- `python -m fall_detection demo --frames 3 --output outputs/day1-smoke.jsonl`: produced three frame records and one summary.
- `python -m fall_detection demo --frames 4 --fps 30 --realtime --output outputs/day1-paced.jsonl`: passed; replay spanned 0.1 source seconds.
- Installed console entry point verified inside a virtual environment. The base shell did not include the user scripts directory on PATH; the module entry point and virtual environment both work.

GitHub Actions is configured for Python 3.11 and 3.12 on Linux and Windows. Remote results must be checked separately after publication.

## Milestone 2 — webcam and local-video capture

Date: 2026-09-30

Status: complete against the milestone's automated acceptance gates (2 of 10 planned milestones). Physical camera and desktop-display checks remain unperformed and are documented separately.

Implemented:

- `capture.py`: optional OpenCV dependency loading, validated camera/local-file selection, native BGR frame streaming, bounded runs, decoder error handling, and context-managed release.
- Source-relative decoder timestamps with an explicit, permanent FPS fallback on invalid/repeated/backward positions; camera acquisition uses a monotonic clock. Frame contracts and JSONL include the timestamp basis.
- `preview.py`: optional desktop window showing the current prediction/reason, Q/Esc and window-close controls, and cleanup on failure. Headless capture makes no display calls.
- CLI `capture` command, `--max-frames`, `--fallback-fps`, replay pacing, output protection, and interruption handling. Prediction remains `unknown`; no detector or model was added.
- Separate optional desktop/headless dependencies, version `0.1.0.dev2`, updated status, installation/usage guidance, lifecycle notes, and hardware-check instructions.
- `tests/test_capture.py`: failure/cleanup/timing/preview tests plus a real generated-video decode through a subprocess CLI. CI now installs the headless extra so the decoder test runs on all four configured OS/Python combinations.

Validation performed on Linux with Python 3.12.14, OpenCV 4.14.0 (headless wheel 4.14.0.94), and NumPy 2.3.5:

- `.venv/bin/python -m pip install 'opencv-python-headless>=4.10,<5'`: passed.
- `.venv/bin/python -m pip install --no-build-isolation --no-deps -e .`: passed; installed version `0.1.0.dev2`.
- `.venv/bin/python -m pip install --no-build-isolation -e '.[headless]'`: passed with the declared optional dependencies resolved.
- `.venv/bin/python -m unittest discover -s tests -q`: **39 tests passed, none skipped**. Includes a five-frame MJPG fixture written to a temporary directory, decoded by the CLI, checked for dimensions, timestamps, unavailable predictions, and summary, then reopened.
- `PYTHONPATH=src python -m unittest discover -s tests -q` in the base environment without OpenCV: 38 passed; the one real decoder test explicitly skipped. The dependency-free foundation remains usable.
- Generated a separate 12-frame, 96×64 MJPG fixture at 15 FPS locally, then ran `.venv/bin/fall-detection capture --video outputs/day2-generated.avi --headless --realtime --output outputs/day2-smoke.jsonl`: passed, 12 frames, 0.733333 source seconds, approximately 0.734 elapsed seconds. These are replay diagnostics only; the fixture contains colored bars, not falls. Generated media and logs are excluded from Git.
- `.venv/bin/fall-detection status`: reported milestone 2/10, version `0.1.0.dev2`, and fall detection unavailable.

Limitations and blockers:

- No physical webcam or graphical desktop was used. Camera timing/disconnection and preview controls are covered with test doubles; real hardware/display verification must be performed locally using the README checklist. This does not block building the next adapter milestone.
- Video EOF can be indistinguishable from a silent decoder failure when frame-count metadata is unavailable. Estimated FPS timestamps do not recover variable-frame-rate timing. Blocking camera drivers may delay interruption.
- Pose weights, pose inference, tracking, fall classification, alerts, and real-world detection metrics remain unavailable. No private footage or model artifacts were uploaded.

Next milestone: ONNX pose adapter with explicit model input/output contracts, preprocessing/coordinate-transform tests, missing-weight behavior, and model licensing documentation.

Final cleanup review also added a release-failure test: an unsuccessful resource release returns an error without emitting a success summary. The final replay smoke check used `outputs/day2-final-smoke.jsonl` and again completed all 12 frames.

Remote CI for this commit is checked after publication; the result is reported with the commit link.

## Milestone 3 — ONNX pose adapter

Date: 2026-10-01

Status: complete against the milestone's adapter acceptance gates (3 of 10 planned milestones). A production model was intentionally not bundled; real model selection, licensing, and quality validation remain explicit external requirements.

Implemented:

- `pose.py`: strict pose configuration, lazy optional dependencies, CPU-only ONNX Runtime session loading, graph metadata checks, and actionable missing dependency/model failures.
- Aspect-preserving bilinear letterbox preprocessing from immutable BGR bytes to contiguous normalized RGB float32 NCHW tensors. The exact rounded x/y scales and padding are retained for inverse transforms.
- Raw YOLO11-pose output decoding for `[1, 5+K*3, N]` and transposed layouts, with non-finite/invalid row rejection, configurable confidence filtering, source-coordinate clipping, 17-keypoint COCO defaults, and class-agnostic NMS.
- `pose-check` CLI with TOML and command-line overrides. Optional `--smoke` runs one black contract-test frame through preprocessing, inference, and decoding without implying accuracy.
- `configs/pose.toml`, version `0.1.0.dev3`, updated capability status, architecture notes, and a dedicated model contract/licensing guide.
- Separate `model` and `test` extras. CI now installs headless capture, ONNX Runtime, and ONNX test tooling on all four OS/Python combinations.
- OpenCV desktop/headless ranges capped below 4.14 after the 4.14.0.94 Linux wheel produced a native bus error during import in this environment. Version 4.13.0.92 passed all capture and pose tests.

Validation performed on Linux with Python 3.12.14, NumPy 2.3.5, ONNX Runtime 1.30.0, ONNX 1.23.1, and OpenCV 4.13.0:

- `.venv/bin/python -m pip install --no-build-isolation -e '.[headless,model,test]'`: passed using the declared extras.
- `.venv/bin/python -m pip check`: passed with no broken requirements.
- `.venv/bin/python -m unittest discover -s tests -q`: **54 tests passed, none skipped**.
- `.venv/bin/python -m unittest tests.test_pose.OnnxRuntimeSmokeTests.test_real_cpu_session_and_cli_smoke -v`: passed. The test generated a tiny constant-output ONNX graph in a temporary directory, opened a real CPU inference session, ran a black frame through the adapter, decoded one contract fixture, and exercised `pose-check --smoke` in a subprocess. It contains no trained weights and is not evidence of detection quality.
- `PYTHONPATH=src python -m fall_detection status`: passed without importing model or capture dependencies; reported milestone 3/10 and fall detection unavailable.
- `.venv/bin/fall-detection pose-check --config configs/pose.toml` without weights: returned exit code 2 with an explicit missing-model error and no inference or safety claim.

Limitations and blockers:

- No YOLO11-pose weights were downloaded, committed, or evaluated. Ultralytics documents YOLO11 code/models under AGPL-3.0 and Enterprise licensing; the user must select an authorized model and license before real inference. External published metrics are not results for this repository.
- Only the documented raw, single-input/single-output float layout is supported. Embedded-NMS/end-to-end exports and incompatible keypoint layouts fail closed.
- The adapter is not connected to streaming fall-state output yet. A successfully decoded pose is not a tracked person and cannot produce `normal`, `falling`, or `fallen` without later milestones.
- Physical webcam/display checks from milestone 2 remain unperformed. No footage, weights, credentials, or external messages were used.

Next milestone: deterministic person association with stable IDs and stale-track expiry, including crossing, missed-frame, and expiry tests plus documented baseline limitations.

Remote CI for this commit is checked after publication; the result is reported with the commit link.

## Milestone 4 — deterministic person tracking

Date: 2026-10-02

Status: complete against the milestone's deterministic baseline acceptance gates (4 of 10 planned milestones). This is a transparent local association baseline, not a claim of production tracking quality.

Implemented:

- `tracking.py`: validated tracker configuration, canonical detection ordering, process-local stable IDs, constant-velocity box/keypoint prediction, gated association costs, deterministic greedy matching, and reset support.
- Association combines resolution-normalized center distance, predicted-box IoU, and mean confident-keypoint distance. Velocity updates use frame-index gaps and configurable smoothing.
- Explicit `TrackSnapshot` values carry age, hit count, missed-frame count, predicted box, and an optional current detection. A missed observation is `detection=None`; stale keypoints are never presented as current.
- Tracks remain eligible across at most `max_missed_frames` intervening frames, then expire. Reappearance after expiry receives a new monotonically increasing ID. Skipped frame indices count toward the same boundary.
- `PoseTrackingStage` composes any pose estimator with the tracker without inventing a fall state.
- `configs/tracker.toml`, dependency-free `track-smoke` CLI, version `0.1.0.dev4`, updated capability status, architecture notes, and dedicated tracking contract/limitation documentation.
- Fifteen tracking tests cover reversed detection order, two-person crossing, exact overlap with pose disambiguation, missed observations, recovery, skipped indices, expiry, far detections, low-confidence keypoint fallback, reset, invalid inputs, stage composition, and CLI smoke output.

Validation performed on Linux with Python 3.12.14:

- `.venv/bin/python -m pip install --no-build-isolation -e '.[headless,model,test]'`: passed; installed version `0.1.0.dev4` with the existing milestone dependencies.
- `.venv/bin/python -m pip check`: passed with no broken requirements.
- `.venv/bin/python -m unittest discover -s tests -v`: **69 tests passed, none skipped**.
- `PYTHONPATH=src python -m unittest tests.test_tracking -v`: 15 tracking tests passed using the dependency-free code path.
- `.venv/bin/fall-detection track-smoke --config configs/tracker.toml`: passed across six synthetic frames. IDs 1 and 2 crossed from opposite sides without swapping; ID 2 emitted `null` for one missed frame and recovered. The output retained `fall_detection_available: false`.
- `.venv/bin/fall-detection status`: reported milestone 4/10, version `0.1.0.dev4`, fall detection unavailable, and temporal pose features next.

Limitations and blockers:

- Matching is deterministic greedy assignment, not BoT-SORT or another evaluated production tracker. It has no appearance embedding, camera-motion compensation, Kalman uncertainty, global assignment, or cross-session re-identification. Stable behavior in the synthetic crossing does not guarantee correct identity in real footage.
- IDs are ephemeral association handles, not biometric identities. Sudden motion, long occlusion, crowded overlap, detector jitter, or similar people can still cause identity switches or new tracks.
- No authorized pose weights or real recordings were used. Tracking quality, latency on a real model, and real-world fall performance remain unmeasured. Physical webcam/display checks also remain pending.
- No blocker prevents the next code milestone. Authorized weights and footage will be required later for real-world validation.

Next milestone: bounded per-person pose/motion feature windows with normalization, low-confidence and missing-observation handling, and independent-history tests.

Remote CI for this commit is checked after publication; the result is reported with the commit link.

## Milestone 5 — bounded pose and motion features

Date: 2026-10-03

Status: complete against the milestone's deterministic feature-contract acceptance gates (5 of 10 planned milestones). No state classifier or detection-quality claim is included.

Implemented:

- `features.py`: strict feature configuration; frame-normalized box center/size; box aspect ratio; source-time normalized center velocity; box-relative keypoint coordinates; and explicit confidence masks.
- Low-confidence joints become `(None, None)` and false mask entries. Missed track observations become fully missing samples rather than stale or zero-filled poses. Motion is unavailable for a new track, after missing data, or across an excessive source-time gap.
- `TemporalFeatureBank`: independent fixed-size deques per active track, strictly ordered frame/timestamp validation, deterministic track ordering, configurable readiness, reset support, and immediate history cleanup when a tracker ID expires.
- `PoseTrackingFeatureStage` composes tracking with temporal features without assigning a fall state.
- `configs/features.toml`, dependency-free `feature-smoke` CLI, version `0.1.0.dev5`, updated status/architecture/README, and a dedicated feature contract document.
- Nineteen feature tests cover configuration, translation/scale normalization, frame geometry, confidence boundaries, missing observations, recovery, source-time velocity, long gaps, independent histories, fixed window bounds, readiness, retirement, transactional validation, JSON serialization, reset, invalid contracts, stage composition, and smoke output.

Validation performed on Linux with Python 3.12.14:

- `.venv/bin/python -m pip install --no-build-isolation -e '.[headless,model,test]'`: passed after installing the declared `setuptools>=68` build requirement into the new virtual environment; installed version `0.1.0.dev5`.
- `.venv/bin/python -m pip check`: passed with no broken requirements.
- `.venv/bin/python -m unittest discover -s tests -v`: **88 tests passed, none skipped**.
- `PYTHONPATH=src python -m unittest tests.test_features -v`: **19 feature tests passed** through the dependency-free code path.
- `.venv/bin/fall-detection feature-smoke --config configs/features.toml`: passed across six synthetic frames. Both tracks retained independent six-sample histories, one low-confidence joint was masked, and track 2 recorded one false observation before recovery. Output retained `fall_detection_available: false`.
- `.venv/bin/fall-detection status`: reported milestone 5/10, version `0.1.0.dev5`, fall detection unavailable, and the explainable temporal state baseline next.
- `.venv/bin/python -m compileall -q src tests`: passed.

Limitations and blockers:

- Normalization and window behavior are contract tests, not measured feature usefulness. No authorized weights or real recordings were used; real pose noise, occlusion behavior, tracking identity switches, and feature distributions remain unmeasured.
- The feature bank expects the complete active tracker snapshot set on every processed frame. Keypoint layout and count must match the configured pose model. Histories are process-local and intentionally disappear when a track expires.
- `ready` means enough usable observations are present; it is not a `normal`, `falling`, or `fallen` prediction. Fall detection remains unavailable and pipeline output stays `unknown`.
- No blocker prevents the next code milestone. Authorized weights and representative footage will be required for later real-world validation.

Next milestone: an explainable, configurable temporal baseline that distinguishes standing, lying, descent, and missing observations as `normal`, `falling`, `fallen`, or `unknown` in deterministic sequence tests.

Remote CI for this commit is checked after publication; the result is reported with the commit link.

## Milestone 6 — explainable temporal state baseline

Date: 2026-10-04

Status: complete against the milestone's deterministic sequence acceptance gates (6 of 10 planned milestones). The classifier is available for validated feature windows; default video/camera pipeline output remains unavailable until later integration.

Implemented:

- `state.py`: strict threshold configuration, per-track `StateDecision` and `StateEvidence` contracts, full feature-window validation, deterministic ordering, and `TemporalStateStage` composition.
- Conservative decision order: missing/unready/low-confidence/interrupted evidence stays `unknown`; settled lying posture becomes `fallen`; rapid descent with concurrent posture widening becomes `falling`; stable upright posture becomes `normal`; all other poses remain `unknown`.
- Configurable thresholds for contiguous observations, confident-joint fraction, upright/lying aspect ratios, stable/transition speed, center drop, aspect-ratio change, transition interval, and settled duration.
- Evidence records explain each branch using current posture/motion, transition displacement/change, or settled duration. Confidence remains `None` because the deterministic thresholds are not calibrated probabilities.
- `configs/state.toml`, dependency-free `state-smoke` CLI, version `0.1.0.dev6`, updated capability status/architecture/README, and a dedicated state baseline document.
- Twenty state tests cover configuration, standing, settled lying, descent, missing observations, readiness, pose-confidence gates, interrupted histories, ambiguous poses, necessary multi-signal fall evidence, configurable thresholds, state precedence, deterministic multi-person ordering, malformed contracts, stage composition, finite JSON, and CLI smoke output.

Validation performed on Linux with Python 3.12.14:

- `.venv/bin/python -m pip install --no-build-isolation -e '.[headless,model,test]'`: passed after installing the declared `setuptools>=68` build requirement into the new virtual environment; installed version `0.1.0.dev6`.
- `.venv/bin/python -m pip check`: passed with no broken requirements.
- `.venv/bin/python -m unittest discover -s tests -v`: **108 tests passed, none skipped**.
- `PYTHONPATH=src python -m unittest tests.test_state -v`: **20 state tests passed** through the dependency-free code path.
- `.venv/bin/fall-detection state-smoke --config configs/state.toml`: passed. Final synthetic states were standing `normal`, lying `fallen`, descent `falling`, and missing `unknown`; all confidences remained null and `fall_detection_available` remained false.
- `.venv/bin/fall-detection status`: reported milestone 6/10, version `0.1.0.dev6`, the baseline classifier available, end-to-end fall detection unavailable, and training/export preparation next.
- `.venv/bin/python -m compileall -q src tests`: passed.

Limitations and blockers:

- The rules rely on detector box geometry, tracked center motion, and confidence masks. Camera motion, viewpoint, cropping, furniture, crouching, detector jitter, and identity switches can produce misleading evidence.
- Synthetic sequence tests validate branch behavior only. No authorized real footage or production pose weights were used, thresholds were not tuned, and no accuracy, recall, precision, latency, or clinical claim is available.
- The default `demo` and `capture` commands still use `UnavailablePredictor`; a library-level decision over handcrafted feature windows is not presented as end-to-end real-time detection.
- No blocker prevents the next scaffold milestone. User-authorized labeled sequences will be required to train or evaluate a temporal model; without them, the dataset-dependent model artifact must remain blocked.

Next milestone: define the dataset and split contracts, build local temporal-model training/export interfaces, test data handling, and record the learned model artifact as blocked unless authorized labeled data is supplied.

Remote CI for this commit is checked after publication; the result is reported with the commit link.
