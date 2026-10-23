# Local annotated pose preview

The live CLI accepts `--preview` for a local window showing observed person boxes, masked COCO keypoints, stable track IDs and **unvalidated heuristic** states. Missing observations do not get stale boxes. The banner states that no validated fall alert is available.

Install: `python -m pip install -e ".[video,model]"` (choose desktop OpenCV, not headless).

Video: `fall-detection-live --video demo/example.mp4 --model models/yolo11n-pose.onnx --preview`

Webcam: `fall-detection-live --camera 0 --model models/yolo11n-pose.onnx --preview`

Press Q/Esc or close the window. Without `--preview`, the command is headless and only emits JSONL metadata, never video frames or raw keypoint coordinates. A real ONNX pose model, authorized video, and desktop are still required to verify a full visual demonstration; synthetic unit tests do not establish fall-detection accuracy.
