# Ten development milestones

Each milestone represents approximately one tenth of the planned work, not a measured percentage of code or model accuracy. Complete one meaningful milestone per development session. Run relevant checks and record evidence before marking it complete; do not use empty commits to simulate progress.

| Milestone | Target date | Scope | Acceptance check |
| --- | --- | --- | --- |
| 1 | 2026-09-29 | Package, configuration, frame contracts, streaming runner, synthetic input, tests | Install succeeds; CLI demo emits ordered frames and honest unavailable states; test suite passes |
| 2 | 2026-09-30 | OpenCV webcam and local-video capture, clean shutdown, timestamps, preview/headless modes | Test a generated video; cover open/read failures and resource cleanup; document physical-camera checks separately |
| 3 | 2026-10-01 | ONNX pose adapter, preprocessing, output decoding, configurable model path | Test tensor shapes, coordinate transforms, empty detections, and missing weights; document supported model and licensing |
| 4 | 2026-10-02 | Person association, stable track IDs, stale-track expiry | Deterministic multi-person tests for crossing, missed frames, and expiry; clearly label baseline tracking limitations |
| 5 | 2026-10-03 | Pose/motion features and bounded per-person temporal windows | Test normalization, low-confidence keypoints, missing data, and independent histories |
| 6 | 2026-10-04 | Explainable temporal baseline: normal, falling, fallen, unknown | Sequence tests distinguish standing, lying, a descent, and missing observations; thresholds are configurable |
| 7 | 2026-10-05 | Dataset contract, temporal-model training/export scaffold, evaluation split design | Test data handling and export interfaces; train/evaluate only if authorized labeled data is available; otherwise record the model artifact as blocked |
| 8 | 2026-10-06 | Persistent alert lifecycle, cooldown/acknowledgment, local event logs and overlay | Test persistence and reset; prevent duplicate events; no external messages or emergency-service calls |
| 9 | 2026-10-07 | Replay evaluation, latency measurements, regression cases | Generate a reproducible report from actual inputs; keep synthetic tests separate from real-world quality measurements |
| 10 | 2026-10-08 | Integration hardening, installation guide, demo instructions, final review | Verify the supported runtime path end-to-end, document remaining blockers, and summarize what is ready |

## Development rules

1. Read the current repository and progress log first; preserve other contributors' changes.
2. Implement the earliest incomplete milestone, or resolve its concrete blocker. Do not skip testing to meet a date.
3. Keep the default runtime local and CPU-compatible. Add optional dependencies only as their features arrive.
4. Keep real footage, credentials, unlicensed data, and large model files out of Git. Obtain and document authorization and licenses for datasets and weights.
5. An ONNX runtime adapter is not a trained classifier. Never present placeholder weights, synthetic test scores, or inherited project claims as measured detection quality.
6. Update the progress log with changed files, commands actually run, outcomes, and blockers. Commit only real completed work; do not backdate history.
7. Once all milestones are complete, review and report rather than adding filler work. If an external resource is missing, provide a precise request and retain the last working path.
