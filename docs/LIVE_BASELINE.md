# Live pose-to-temporal baseline

The `fall-detection-live` command now connects ONNX pose inference, the repository's local multi-person tracker, bounded temporal features, and explainable heuristic states for webcam or local-video input. This is **not** a trained or validated fall detector; no emergency alert is emitted. The global state remains `unknown` and `fall_detection_available` remains false.

Install: `python -m pip install -e ".[headless,model]"`

Video: `fall-detection-live --video demo/example.mp4 --model models/yolo11n-pose.onnx --max-frames 300 --output outputs/live.jsonl`

Webcam: `fall-detection-live --camera 0 --model models/yolo11n-pose.onnx`

A compatible licensed raw YOLO11-Pose ONNX export must be supplied locally. The command emits JSONL frame records with per-person track IDs, observed flags, bounded window sizes, and heuristic states/evidence. It does not serialize raw pixels or keypoint coordinates. Output files are created exclusively, never overwritten.

The local tracker is not BoT-SORT. The baseline does not report measured real-world detection accuracy. Synthetic tests only validate integration behavior.

Run tests: `python -m unittest discover -s tests -p test_live.py -v`.

Next: visual overlays, trained temporal ONNX model integration, and validation on authorized subject-separated real videos.
