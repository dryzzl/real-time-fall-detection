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
