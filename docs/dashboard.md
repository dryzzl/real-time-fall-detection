# Local fall-detection monitoring dashboard

This read-only, loopback-only full-stack demonstration serves **metadata**, not video frames, from the JSONL produced by the existing `fall-detection` and `fall-detection-live` commands. It initializes from a bounded recent-event snapshot, then polls incremental events every second and displays reported states, track counts, frame counts, and recent events. The browser's frame counter is the number of frames loaded in the current session, not the all-time total in a long log.

## Installation

```bash
python -m pip install -e .
```

No new third-party dependency is required for the dashboard itself.

## Synthetic smoke demonstration

Terminal 1:

```bash
fall-detection demo --frames 5000 --realtime --output outputs/synthetic-dashboard.jsonl
```

Terminal 2:

```bash
fall-detection-dashboard --log outputs/synthetic-dashboard.jsonl --port 8765
```

Open http://127.0.0.1:8765. This is a **synthetic metadata demo** only: `unknown` is the correct unvalidated state and must not be portrayed as a detected fall.

## Live pose pipeline

If you have an authorized camera or video and a compatible YOLO11-Pose ONNX file:

```bash
fall-detection-live --camera 0 --model models/yolo11n-pose.onnx --output outputs/live.jsonl --preview
fall-detection-dashboard --log outputs/live.jsonl --port 8765
```

Run these commands in separate terminals. The app can show tracked people and experimental state estimates, but this does not establish fall-detection accuracy. The API never returns raw frames, model binaries, keypoints, or user-selected files. The monitor is restricted to 127.0.0.1.

## API

- `GET /` serves the dashboard.
- `GET /api/health` checks log availability.
- `GET /api/snapshot?limit=100` reads the newest complete JSONL records from a bounded 2 MiB tail window, returning a continuation cursor and flags for truncated history or an incomplete trailing line. This avoids replaying an entire long-running log on browser refresh.
- `GET /api/events?cursor=0&limit=100` returns at most 200 sanitized metadata records per request and a cursor. A rotated/truncated log resets the cursor.

## Tests and stress benchmarks

```bash
python -m unittest discover -s tests -p test_dashboard.py -v
python scripts/benchmark_dashboard.py --events 50000 --batch-size 200
```

The benchmark uses entirely synthetic JSONL events and measures reader throughput/memory only, **not** webcam-to-alert latency, inference speed, classifier quality, or real-world stress performance.

## Safety limitations

All state predictions remain research-only until a trained and independently evaluated temporal model is available. This dashboard is not a medical or emergency notification system. Do not run it on an untrusted network.
