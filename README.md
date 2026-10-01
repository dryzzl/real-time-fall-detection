# Real-Time Fall Detection

A Python project for detecting falls from video through pose estimation, motion tracking, and temporal analysis.

**Development status: milestone 2 of 10 — webcam and local-video input.** This version streams synthetic frames, a local video, or a webcam through a tested pipeline, with optional desktop preview. Pose estimation, tracking, and fall classification are not implemented yet. Every prediction returns `unknown` with `detection_not_implemented`.

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
# Desktop, including a preview window
python -m pip install -e ".[video]"
```

```bash
# Server or CI, without GUI dependencies
python -m pip install -e ".[headless]"
```

The OpenCV variants share the `cv2` namespace. Do not install both extras in the same environment; create a fresh virtual environment to switch. `python -m pip install -e .` still supports the synthetic demo without third-party runtime dependencies.

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
```

`python -m fall_detection` is an alternative to the installed command. Camera index `0` usually selects the default webcam; another index may be needed. Video paths must name existing local files. Streaming URLs are not supported. Decoded frames retain their native dimensions. `configs/default.toml` configures the synthetic demo only.

Capture defaults to headless mode. Without `--max-frames`, videos run to their end and cameras run until interrupted. `--realtime` paces video or synthetic replay; cameras already run live. Preview needs a local graphical session and the desktop extra. A headless build or missing Linux display produces an actionable error.

Output files must be new; existing files are never overwritten. JSONL contains metadata, not image bytes. Webcam/video pixels stay in process memory or the requested local preview; the application does not record or upload footage. Completed records survive interruption or failure. Exit code `0` means a successful stop, `2` an input/runtime error, and `130` Ctrl+C. Errors and interruption do not emit a success summary.

Example video frame record:

```json
{"event":"frame","source":"video","frame_index":0,"timestamp_seconds":0.0,"timestamp_basis":"video_position","width":640,"height":480,"state":"unknown","confidence":null,"reason":"detection_not_implemented"}
```

A final summary reports processed frames, source duration, elapsed time, and pipeline throughput. These are execution diagnostics, not fall-detection performance metrics.

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

The suite covers configuration, frame contracts, stream ordering, pacing, output protection, capture open/read failures, timestamp fallbacks, early stops, interruption, resource cleanup, and preview controls. With either OpenCV extra installed, it also generates an MJPG video in a temporary directory and decodes it through the CLI. Without OpenCV, that integration test is explicitly skipped. CI installs the headless extra and runs on Linux and Windows with Python 3.11 and 3.12.

Physical webcam/display validation is separate from automated tests. On a local desktop, run the preview and bounded-camera commands above; verify that frames appear, Q/Esc and window close stop cleanly, Ctrl+C releases the device, and the camera can reopen. Also check that disconnecting it reports an error. **These hardware checks have not been performed in this development environment.** See [the progress log](docs/PROGRESS.md), [roadmap](docs/ROADMAP.md), and [architecture](docs/ARCHITECTURE.md).

## Scope

This is a development prototype, not a medical or emergency-response device. There are no accuracy, recall, or clinical-validation claims for this repository. The synthetic sources and generated test video do not depict a person or a fall. Later model evaluation must use authorized data and document the dataset, split, measurements, and limitations. A trained temporal model will require suitable labeled sequences; missing data or weights will be reported as blockers rather than replaced by invented results.
