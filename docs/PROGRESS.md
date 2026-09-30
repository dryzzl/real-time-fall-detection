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
