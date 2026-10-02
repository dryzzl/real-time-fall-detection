# Supported pose model contract

Milestone 3 supports a **raw, single-batch Ultralytics YOLO11-pose ONNX export** through ONNX Runtime on CPU. The repository does not include or download model weights. A model file must be supplied deliberately at `models/yolo11n-pose.onnx`, another TOML path, or `--model PATH`.

This adapter is intentionally strict. A file loading successfully as ONNX does not mean its outputs have the expected meaning.

## Input

- Exactly one `tensor(float)` input.
- Shape `[1, 3, H, W]`, with a fixed batch of 1 or dynamic batch metadata. Channels must be 3.
- Static `H` and `W` must match the configured `input_height` and `input_width`; dynamic spatial dimensions use the configured values.
- Frames are decoded as BGR bytes, bilinearly resized with aspect ratio preserved, padded with value 114, changed to RGB, normalized to `[0, 1]`, and arranged as contiguous float32 NCHW.
- Letterbox padding and the actual rounded x/y scales are retained so boxes and keypoints can be mapped back to source pixels.

## Output

- Exactly one `tensor(float)` output.
- Raw output shape `[1, 5 + K*3, N]` or `[1, N, 5 + K*3]`.
- For the configured default `K=17`, there are 56 attributes per candidate: `center_x, center_y, width, height, box_confidence`, followed by 17 `x, y, keypoint_confidence` triples.
- Coordinates must be in input-image pixels, before reversing letterboxing. Embedded-NMS/exported end-to-end layouts are not accepted.
- Non-finite rows, confidence outside `[0, 1]`, candidates below the configured threshold, and degenerate boxes are ignored. Coordinates are clipped to the source frame. Class-agnostic NMS uses the configured IoU threshold.
- The default 17-keypoint ordering is the COCO person convention: nose; left/right eyes; left/right ears; left/right shoulders, elbows, wrists, hips, knees, and ankles.

Run a contract check after placing a model:

```bash
fall-detection pose-check --config configs/pose.toml
fall-detection pose-check --config configs/pose.toml --smoke
```

`--smoke` runs one black test frame through the graph and reports its pose count. It confirms that preprocessing, ONNX execution, and decoding complete; it is not an accuracy, safety, or real-world detection test. The pose adapter is not yet connected to tracking or fall classification, so `fall_detection_available` remains false.

## Export and licensing

Ultralytics documents YOLO11-pose as supporting inference and export, including the `yolo11n-pose.pt` variant. Export a raw ONNX model at the same image size as the config; after export, use `pose-check` to verify the actual graph rather than relying on its filename. A typical Ultralytics command is:

```bash
yolo export model=yolo11n-pose.pt format=onnx imgsz=640 batch=1 dynamic=False nms=False
```

Export behavior can change between Ultralytics releases. If the output includes embedded NMS or does not contain 56 raw attributes, this adapter rejects it.

Ultralytics states that YOLO11 code and models are offered under **AGPL-3.0 and Enterprise licenses**. The exported model retains applicable licensing obligations; choose and document a license appropriate for the intended use before placing weights in this project. Do not commit weights to Git. ONNX Runtime is separately distributed under the MIT license. Dataset rights and privacy obligations are separate from model and runtime licensing.

Primary references:

- [Ultralytics YOLO11 models and licensing](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics pose task and ONNX export](https://docs.ultralytics.com/tasks/pose/)
- [Ultralytics export arguments](https://docs.ultralytics.com/modes/export/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ONNX Runtime repository and license](https://github.com/microsoft/onnxruntime)

No model quality figures in external documentation are inherited as results for this repository. Real performance remains unmeasured until an authorized model and evaluation dataset are selected and tested here.
