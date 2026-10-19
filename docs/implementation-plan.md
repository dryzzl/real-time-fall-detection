# End-to-end fall detection implementation plan

This branch tracks the remaining integration work while preserving the existing, tested source tree on `main`.

## Current system

The repository already provides camera/video capture, pose ONNX parsing, tracking, temporal features, state logic, alert event persistence, schema validation, and evaluation checks. The README accurately identifies the primary gap: an authorized labeled dataset and trained temporal model are not yet available, and live classification remains `unknown`.

## Deliverable stages

1. Wire live pose detections, persistent person tracks, pose features, and rolling sequence buffers into the streaming CLI.
2. Implement actual three-class LSTM training and ONNX export using the project's established temporal input schema.
3. Add end-to-end ONNX Runtime classification per tracked person, with consistent normalization, class labels and model metadata.
4. Connect state filtering and persistent ground-contact alerts to the real predictions, ensuring that unknown or invalid observations are not treated as verified falls.
5. Add synthetic-only integration fixtures and regression tests; segregate them clearly from any real accuracy claims.
6. Validate video-level held-out splits and benchmark the complete camera-to-alert latency, not only model inference.

## Scientific limitations

Accuracy 98.2%, macro F1 97.8%, falling recall 95.3%, and false-positive rate 1.5% are supplied targets/earlier reported preliminary results. They are **not independently verified by this repository** without the matching dataset and trained weights. The research prototype is not a certified emergency or medical safety system.

## Deployment architecture

For the ONNX-only deployment path, ensure the entire model inference path uses ONNX Runtime, NumPy, OpenCV and a tracker compatible with the strict runtime dependencies. Ultralytics BoT-SORT may be used during offline training/data extraction, but should not silently be claimed to be present in the Torch-free path if a different tracker is used.
