# Real-Time Fall Detection

A Python project for detecting falls from video through pose estimation, motion tracking, and temporal analysis.

**Development status: milestone 1 of 10 — foundation.** This version runs a synthetic frame stream through a tested pipeline. Camera capture, pose estimation, tracking, and fall classification are planned; they are not implemented yet. The demo always returns `unknown` with `detection_not_implemented`.

## Run the first milestone

Requires Python 3.11 or newer. The foundation has no third-party runtime dependencies.

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

Install and run:

```bash
python -m pip install -e .
fall-detection status
fall-detection demo --frames 60
fall-detection demo --config configs/default.toml --realtime --output outputs/demo.jsonl
```

`python -m fall_detection` is an alternative to the installed command. Output files must be new; existing files are never silently overwritten. Realtime mode paces the synthetic stream by its timestamps. Without it, the demo runs as fast as the foundation can process frames.

Example frame record:

```json
{"event":"frame","source":"synthetic","frame_index":0,"timestamp_seconds":0.0,"width":320,"height":180,"state":"unknown","confidence":null,"reason":"detection_not_implemented"}
```

A final summary reports processed frames, source duration, elapsed time, and pipeline throughput. These are execution diagnostics, not fall-detection performance metrics. Pixel data is never written to these JSONL records.

## What works today

- Validated TOML configuration and command-line overrides.
- Immutable BGR frame and prediction contracts with timestamp validation.
- A deterministic, bounded-memory synthetic source with a moving colored bar.
- A streaming pipeline, an injectable prediction interface, and optional realtime pacing.
- JSONL output, explicit unavailable status, and a standard-library test suite.
- A ten-milestone development roadmap and an auditable progress log.

The synthetic source does not depict a person or a fall. No camera, recording, cloud service, trained weights, or private footage is accessed by this version.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover malformed frames, invalid configuration, non-finite values, output protection, stream ordering, incremental consumption, realtime pacing, and CLI behavior. See [the roadmap](docs/ROADMAP.md), [architecture](docs/ARCHITECTURE.md), and [progress log](docs/PROGRESS.md) for the remaining work and acceptance checks.

## Scope

This is a development prototype, not a medical or emergency-response device. There are no accuracy, recall, or clinical-validation claims for this repository. Later model evaluation must use authorized data and document the dataset, split, measurements, and limitations. A trained temporal model will require suitable labeled sequences; missing data or weights will be reported as blockers rather than replaced by invented results.
