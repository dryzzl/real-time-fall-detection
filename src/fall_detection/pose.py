"""CPU ONNX adapter for raw Ultralytics YOLO11-pose exports."""

from dataclasses import asdict, dataclass, fields
import importlib
from math import isfinite
from pathlib import Path
import tomllib
from typing import Any

from .contracts import FramePacket


class PoseError(RuntimeError):
    """Pose configuration, model validation, or inference failed."""


def _load_module(name: str, extra: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise PoseError(f'{name} is unavailable; install the model extra: pip install -e ".[model]"') from error


@dataclass(frozen=True)
class PoseConfig:
    model_path: Path = Path("models/yolo11n-pose.onnx")
    input_width: int = 640
    input_height: int = 640
    keypoint_count: int = 17
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, Path) or self.model_path.suffix.lower() != ".onnx":
            raise ValueError("model_path must be a local .onnx path")
        for name in ("input_width", "input_height"):
            value = getattr(self, name)
            if type(value) is not int or not 32 <= value <= 4096:
                raise ValueError(f"{name} must be an integer between 32 and 4096")
        if type(self.keypoint_count) is not int or not 1 <= self.keypoint_count <= 100:
            raise ValueError("keypoint_count must be an integer between 1 and 100")
        for name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")


def load_pose_config(path: Path | None = None, **overrides: object) -> PoseConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"pose"} or not isinstance(document["pose"], dict):
            raise ValueError("pose configuration must contain only a [pose] table")
        settings.update(document["pose"])
    allowed = {field.name for field in fields(PoseConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError("unknown pose settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    model_path = settings.get("model_path")
    if model_path is not None:
        if not isinstance(model_path, (str, Path)):
            raise ValueError("model_path must be a path string")
        settings["model_path"] = Path(model_path)
    return PoseConfig(**settings)


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int

    def source_xy(self, x: float, y: float) -> tuple[float, float]:
        source_x = min(max((x - self.pad_left) / self.scale_x, 0.0), self.source_width - 1.0)
        source_y = min(max((y - self.pad_top) / self.scale_y, 0.0), self.source_height - 1.0)
        return source_x, source_y


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseDetection:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[Keypoint, ...]


def _resize_bilinear(image, width: int, height: int, np):
    source_height, source_width = image.shape[:2]
    x = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    x = np.clip(x, 0, source_width - 1)
    y = np.clip(y, 0, source_height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0)[None, :, None]
    wy = (y - y0)[:, None, None]
    horizontal = image[:, x0] * (1.0 - wx) + image[:, x1] * wx
    return horizontal[y0] * (1.0 - wy) + horizontal[y1] * wy


def preprocess_frame(frame: FramePacket, config: PoseConfig):
    """Return a normalized RGB NCHW float32 tensor and reversible letterbox transform."""
    np = _load_module("numpy", "model")
    pixels = np.frombuffer(frame.bgr, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    scale = min(config.input_width / frame.width, config.input_height / frame.height)
    resized_width = max(1, min(config.input_width, round(frame.width * scale)))
    resized_height = max(1, min(config.input_height, round(frame.height * scale)))
    pad_left = (config.input_width - resized_width) // 2
    pad_top = (config.input_height - resized_height) // 2
    resized = _resize_bilinear(pixels.astype(np.float32), resized_width, resized_height, np)
    canvas = np.full((config.input_height, config.input_width, 3), 114.0, dtype=np.float32)
    canvas[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0, dtype=np.float32)
    transform = LetterboxTransform(
        frame.width, frame.height, config.input_width, config.input_height,
        resized_width / frame.width, resized_height / frame.height, pad_left, pad_top,
    )
    return tensor, transform


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def decode_output(output: Any, transform: LetterboxTransform, config: PoseConfig) -> tuple[PoseDetection, ...]:
    """Decode raw `[1, 5 + K*3, N]` (or transposed) YOLO pose output."""
    np = _load_module("numpy", "model")
    values = np.asarray(output)
    attributes = 5 + config.keypoint_count * 3
    if values.ndim != 3 or values.shape[0] != 1:
        raise PoseError(f"pose output must have rank 3 and batch 1; received {values.shape}")
    if values.shape[1] == attributes:
        rows = values[0].T
    elif values.shape[2] == attributes:
        rows = values[0]
    else:
        raise PoseError(f"pose output must contain {attributes} attributes; received {values.shape}")
    candidates: list[PoseDetection] = []
    for row in rows:
        if not np.isfinite(row).all():
            continue
        confidence = float(row[4])
        if confidence < config.confidence_threshold or not 0 <= confidence <= 1:
            continue
        center_x, center_y, width, height = map(float, row[:4])
        x1, y1 = transform.source_xy(center_x - width / 2, center_y - height / 2)
        x2, y2 = transform.source_xy(center_x + width / 2, center_y + height / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        keypoints = []
        for offset in range(5, attributes, 3):
            x, y = transform.source_xy(float(row[offset]), float(row[offset + 1]))
            keypoints.append(Keypoint(x, y, min(max(float(row[offset + 2]), 0.0), 1.0)))
        candidates.append(PoseDetection((x1, y1, x2, y2), confidence, tuple(keypoints)))
    selected: list[PoseDetection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(_iou(candidate.box_xyxy, kept.box_xyxy) <= config.iou_threshold for kept in selected):
            selected.append(candidate)
    return tuple(selected)


class OnnxPoseEstimator:
    """Validate and run a raw YOLO11-pose ONNX graph on CPU."""

    def __init__(self, config: PoseConfig, *, session: Any | None = None) -> None:
        self.config = config
        if session is None:
            if not config.model_path.is_file():
                raise PoseError(f"pose model not found: {config.model_path}; supply an authorized YOLO11-pose ONNX export")
            runtime = _load_module("onnxruntime", "model")
            try:
                session = runtime.InferenceSession(str(config.model_path), providers=["CPUExecutionProvider"])
            except Exception as error:
                raise PoseError(f"could not load ONNX pose model: {error}") from error
        self.session = session
        self.input_name, self.output_name = self._validate_contract()

    def _validate_contract(self) -> tuple[str, str]:
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PoseError("supported pose model must have exactly one input and one output")
        input_meta, output_meta = inputs[0], outputs[0]
        if input_meta.type != "tensor(float)" or len(input_meta.shape) != 4:
            raise PoseError("pose input must be a rank-4 float tensor")
        batch, channels, height, width = input_meta.shape
        if isinstance(batch, int) and batch != 1 or channels != 3:
            raise PoseError(f"pose input must have shape [1, 3, H, W]; received {input_meta.shape}")
        if isinstance(height, int) and height != self.config.input_height:
            raise PoseError(f"model input height is {height}, configured height is {self.config.input_height}")
        if isinstance(width, int) and width != self.config.input_width:
            raise PoseError(f"model input width is {width}, configured width is {self.config.input_width}")
        if output_meta.type != "tensor(float)" or len(output_meta.shape) != 3:
            raise PoseError("pose output must be a rank-3 float tensor")
        output_batch = output_meta.shape[0]
        if isinstance(output_batch, int) and output_batch != 1:
            raise PoseError("pose output batch must be 1")
        attributes = 5 + self.config.keypoint_count * 3
        concrete = [dimension for dimension in output_meta.shape[1:] if isinstance(dimension, int)]
        if len(concrete) == 2 and attributes not in concrete:
            raise PoseError(f"pose output metadata must contain {attributes} attributes; received {output_meta.shape}")
        return input_meta.name, output_meta.name

    def describe(self) -> dict[str, object]:
        return {
            "model_path": str(self.config.model_path),
            "provider": "CPUExecutionProvider",
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": [1, 3, self.config.input_height, self.config.input_width],
            "output_contract": f"raw [1,{5 + self.config.keypoint_count * 3},N] or [1,N,{5 + self.config.keypoint_count * 3}]",
            "keypoint_count": self.config.keypoint_count,
        }

    def estimate(self, frame: FramePacket) -> tuple[PoseDetection, ...]:
        tensor, transform = preprocess_frame(frame, self.config)
        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
        except Exception as error:
            raise PoseError(f"ONNX pose inference failed: {error}") from error
        if len(outputs) != 1:
            raise PoseError("ONNX Runtime returned an unexpected output count")
        return decode_output(outputs[0], transform, self.config)


def detections_as_records(detections: tuple[PoseDetection, ...]) -> list[dict[str, object]]:
    return [asdict(detection) for detection in detections]
