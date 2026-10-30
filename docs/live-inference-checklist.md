# Live inference integration checklist

- Stream ONNX pose detections into stable person tracks.
- Maintain bounded per-track temporal feature windows.
- Emit explicit per-person heuristic states and preserve unknown on insufficient evidence.
- Add a trained ONNX temporal model only after validating the data contract and model metadata.
- Test camera and local-video replay without claiming accuracy from synthetic data.
