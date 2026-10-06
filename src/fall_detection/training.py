"""Dataset, split, and ONNX export contracts for a future temporal model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Mapping, Protocol, Sequence

from .contracts import FallState
from .features import PoseFeatureSample, TemporalFeatureWindow


DATASET_SCHEMA_VERSION = 1
MODEL_CONTRACT_VERSION = 1
TRAINABLE_STATES = (FallState.NORMAL, FallState.FALLING, FallState.FALLEN)
CLASS_NAMES = tuple(state.value for state in TRAINABLE_STATES)
_RECORD_KEYS = {
    "schema_version", "sequence_id", "subject_id", "session_id",
    "authorization_id", "label", "features",
}


class TrainingError(ValueError):
    """Raised when training inputs or an exported artifact violate their contract."""


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible data and model-interface settings; no optimizer is implied."""

    dataset_path: Path = Path("data/temporal_sequences.jsonl")
    model_path: Path = Path("models/temporal_state.onnx")
    sequence_length: int = 30
    keypoint_count: int = 17
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: str = "fall-detection-v1"
    max_sequences: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_path, Path) or not isinstance(self.model_path, Path):
            raise TrainingError("dataset_path and model_path must be paths")
        if type(self.sequence_length) is not int or not 2 <= self.sequence_length <= 10_000:
            raise TrainingError("sequence_length must be an integer between 2 and 10000")
        if type(self.keypoint_count) is not int or not 1 <= self.keypoint_count <= 100:
            raise TrainingError("keypoint_count must be an integer between 1 and 100")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(type(value) not in (int, float) or not isfinite(value) or value <= 0
               for value in fractions):
            raise TrainingError("split fractions must be finite numbers greater than zero")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise TrainingError("train, validation, and test fractions must sum to 1")
        if not isinstance(self.split_seed, str) or not self.split_seed.strip():
            raise TrainingError("split_seed must be a nonempty string")
        if type(self.max_sequences) is not int or not 1 <= self.max_sequences <= 1_000_000:
            raise TrainingError("max_sequences must be an integer between 1 and 1000000")

    @property
    def feature_width(self) -> int:
        # time offset, observed, detection score, box center/size/aspect, velocity,
        # two coordinates per keypoint, and one availability mask per keypoint.
        return 10 + 3 * self.keypoint_count


def load_training_config(path: Path | None = None, **overrides: object) -> TrainingConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"training"} or not isinstance(document["training"], dict):
            raise TrainingError("training configuration must contain only a [training] table")
        settings.update(document["training"])
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise TrainingError("unknown training settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    for name in ("dataset_path", "model_path"):
        if name in settings:
            if not isinstance(settings[name], (str, Path)):
                raise TrainingError(f"{name} must be a path string")
            settings[name] = Path(settings[name])
    return TrainingConfig(**settings)


def feature_names(keypoint_count: int) -> tuple[str, ...]:
    if type(keypoint_count) is not int or not 1 <= keypoint_count <= 100:
        raise TrainingError("keypoint_count must be an integer between 1 and 100")
    names = [
        "time_offset_seconds", "observed", "detection_confidence",
        "box_center_x", "box_center_y", "box_width", "box_height",
        "box_aspect_ratio", "center_velocity_x_per_second",
        "center_velocity_y_per_second",
    ]
    for index in range(keypoint_count):
        names.extend((f"keypoint_{index:02d}_x", f"keypoint_{index:02d}_y"))
    names.extend(f"keypoint_{index:02d}_available" for index in range(keypoint_count))
    return tuple(names)


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise TrainingError(f"{name} must be a finite number")
    return float(value)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise TrainingError(f"{name} must be a nonempty string of at most 128 characters")
    if any(ord(character) < 32 for character in value):
        raise TrainingError(f"{name} cannot contain control characters")
    return value


def _validate_feature_matrix(
    matrix: object, config: TrainingConfig
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(matrix, list) or len(matrix) != config.sequence_length:
        raise TrainingError(
            f"features must contain exactly {config.sequence_length} time steps"
        )
    rows = []
    previous_offset = -1.0
    keypoint_start = 10
    mask_start = keypoint_start + 2 * config.keypoint_count
    for row_index, raw_row in enumerate(matrix):
        if not isinstance(raw_row, list) or len(raw_row) != config.feature_width:
            raise TrainingError(
                f"feature row {row_index} must contain exactly {config.feature_width} values"
            )
        row = tuple(_finite_number(value, f"feature[{row_index}]") for value in raw_row)
        offset, observed, confidence = row[:3]
        if row_index == 0 and offset != 0:
            raise TrainingError("the first time offset must be zero")
        if offset <= previous_offset:
            raise TrainingError("time offsets must increase strictly")
        previous_offset = offset
        if observed not in (0.0, 1.0):
            raise TrainingError("observed values must be 0 or 1")
        masks = row[mask_start:]
        if any(value not in (0.0, 1.0) for value in masks):
            raise TrainingError("keypoint availability values must be 0 or 1")
        if observed == 0:
            if any(value != 0 for value in row[2:]):
                raise TrainingError("missing time steps must be zero-filled after observed")
        else:
            if not 0 <= confidence <= 1:
                raise TrainingError("detection confidence must be between 0 and 1")
            center_x, center_y, width, height, aspect = row[3:8]
            if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
                raise TrainingError("observed box centers must be normalized to [0, 1]")
            if width <= 0 or height <= 0 or aspect <= 0:
                raise TrainingError("observed box width, height, and aspect must be positive")
            for keypoint_index, available in enumerate(masks):
                point = row[
                    keypoint_start + 2 * keypoint_index:
                    keypoint_start + 2 * keypoint_index + 2
                ]
                if available == 0 and point != (0.0, 0.0):
                    raise TrainingError("masked keypoint coordinates must be zero-filled")
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class TemporalSequence:
    sequence_id: str
    subject_id: str
    session_id: str
    authorization_id: str
    label: FallState
    features: tuple[tuple[float, ...], ...]

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "sequence_id": self.sequence_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "authorization_id": self.authorization_id,
            "label": self.label.value,
            "features": [list(row) for row in self.features],
        }


def parse_sequence_record(record: object, config: TrainingConfig) -> TemporalSequence:
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise TrainingError(
            "each sequence must contain exactly: " + ", ".join(sorted(_RECORD_KEYS))
        )
    if type(record["schema_version"]) is not int or record["schema_version"] != DATASET_SCHEMA_VERSION:
        raise TrainingError(f"schema_version must be {DATASET_SCHEMA_VERSION}")
    try:
        label = FallState(record["label"])
    except (TypeError, ValueError) as error:
        raise TrainingError("label must be normal, falling, or fallen") from error
    if label not in TRAINABLE_STATES:
        raise TrainingError("unknown is not a supervised training label")
    return TemporalSequence(
        sequence_id=_identifier(record["sequence_id"], "sequence_id"),
        subject_id=_identifier(record["subject_id"], "subject_id"),
        session_id=_identifier(record["session_id"], "session_id"),
        authorization_id=_identifier(record["authorization_id"], "authorization_id"),
        label=label,
        features=_validate_feature_matrix(record["features"], config),
    )


def load_sequence_dataset(path: Path, config: TrainingConfig) -> tuple[TemporalSequence, ...]:
    if not isinstance(path, Path):
        raise TrainingError("dataset path must be a Path")
    if not path.is_file():
        raise TrainingError(f"dataset file not found: {path}")
    records = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise TrainingError(f"dataset line {line_number} is blank")
            if len(line.encode("utf-8")) > 10_000_000:
                raise TrainingError(f"dataset line {line_number} exceeds 10 MB")
            try:
                raw = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                    TrainingError(f"non-finite JSON number {value}")
                ))
            except (json.JSONDecodeError, TrainingError) as error:
                raise TrainingError(f"invalid JSON on dataset line {line_number}: {error}") from error
            sequence = parse_sequence_record(raw, config)
            if sequence.sequence_id in seen_ids:
                raise TrainingError(f"duplicate sequence_id: {sequence.sequence_id}")
            seen_ids.add(sequence.sequence_id)
            records.append(sequence)
            if len(records) > config.max_sequences:
                raise TrainingError(f"dataset exceeds max_sequences={config.max_sequences}")
    if not records:
        raise TrainingError("dataset must contain at least one sequence")
    return tuple(records)


def _encode_sample(sample: PoseFeatureSample, first_timestamp: float, keypoint_count: int) -> tuple[float, ...]:
    if not isinstance(sample, PoseFeatureSample):
        raise TrainingError("feature windows must contain PoseFeatureSample values")
    if len(sample.keypoints_xy) != keypoint_count or len(sample.keypoint_mask) != keypoint_count:
        raise TrainingError(f"expected {keypoint_count} keypoints in every sample")
    offset = _finite_number(sample.timestamp_seconds - first_timestamp, "time offset")
    if not sample.observed:
        if any(value is not None for value in (
                sample.detection_confidence, sample.box_center_xy, sample.box_size_wh,
                sample.box_aspect_ratio, sample.center_velocity_xy_per_second)):
            raise TrainingError("missing samples cannot contain detection features")
        return (offset,) + (0.0,) * (9 + 3 * keypoint_count)
    required = (
        sample.detection_confidence, sample.box_center_xy,
        sample.box_size_wh, sample.box_aspect_ratio,
    )
    if any(value is None for value in required):
        raise TrainingError("observed samples require detection and box features")
    center = sample.box_center_xy
    size = sample.box_size_wh
    velocity = sample.center_velocity_xy_per_second or (0.0, 0.0)
    assert center is not None and size is not None
    row = [
        offset, 1.0, float(sample.detection_confidence),
        float(center[0]), float(center[1]), float(size[0]), float(size[1]),
        float(sample.box_aspect_ratio), float(velocity[0]), float(velocity[1]),
    ]
    masks = []
    for point, available in zip(sample.keypoints_xy, sample.keypoint_mask):
        if type(available) is not bool:
            raise TrainingError("keypoint masks must contain booleans")
        if available:
            if (not isinstance(point, tuple) or len(point) != 2
                    or any(type(value) not in (int, float) or not isfinite(value)
                           for value in point)):
                raise TrainingError("available keypoints must contain finite x/y coordinates")
            row.extend((float(point[0]), float(point[1])))
            masks.append(1.0)
        else:
            if point != (None, None):
                raise TrainingError("unavailable keypoints must use (None, None)")
            row.extend((0.0, 0.0))
            masks.append(0.0)
    row.extend(masks)
    return tuple(row)


def encode_feature_window(window: TemporalFeatureWindow, config: TrainingConfig) -> tuple[tuple[float, ...], ...]:
    """Convert one exact-length runtime window to the versioned numeric model layout."""
    if not isinstance(window, TemporalFeatureWindow):
        raise TrainingError("window must be a TemporalFeatureWindow")
    if not isinstance(window.samples, tuple) or len(window.samples) != config.sequence_length:
        raise TrainingError(f"window must contain exactly {config.sequence_length} samples")
    first_timestamp = window.samples[0].timestamp_seconds
    rows = tuple(_encode_sample(sample, first_timestamp, config.keypoint_count)
                 for sample in window.samples)
    # Reuse the serialized-input validator so runtime encoding and dataset loading
    # cannot silently drift apart.
    return _validate_feature_matrix([list(row) for row in rows], config)


@dataclass(frozen=True)
class SplitAssignment:
    sequence_id: str
    subject_id: str
    session_id: str
    label: str
    split: str

    def as_record(self) -> dict[str, str]:
        return {
            "sequence_id": self.sequence_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "label": self.label,
            "split": self.split,
        }


@dataclass(frozen=True)
class SplitManifest:
    dataset_sha256: str
    assignments: tuple[SplitAssignment, ...]
    sequence_counts: Mapping[str, int]
    subject_counts: Mapping[str, int]

    def as_record(self, config: TrainingConfig) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_sha256": self.dataset_sha256,
            "split_seed": config.split_seed,
            "sequence_length": config.sequence_length,
            "keypoint_count": config.keypoint_count,
            "feature_width": config.feature_width,
            "feature_names": list(feature_names(config.keypoint_count)),
            "class_names": list(CLASS_NAMES),
            "sequence_counts": dict(self.sequence_counts),
            "subject_counts": dict(self.subject_counts),
            "assignments": [assignment.as_record() for assignment in self.assignments],
        }


def _dataset_digest(records: Sequence[TemporalSequence]) -> str:
    digest = sha256()
    for sequence in sorted(records, key=lambda item: item.sequence_id):
        encoded = json.dumps(
            sequence.as_record(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _subject_counts(total: int, config: TrainingConfig) -> tuple[int, int, int]:
    fractions = (config.train_fraction, config.validation_fraction, config.test_fraction)
    counts = [1, 1, 1]
    remaining = total - 3
    if remaining < 0:
        raise TrainingError("at least three subjects are required for separated splits")
    raw = [remaining * value for value in fractions]
    additions = [int(value) for value in raw]
    for index, value in enumerate(additions):
        counts[index] += value
    leftovers = remaining - sum(additions)
    order = sorted(range(3), key=lambda index: (-(raw[index] - additions[index]), index))
    for index in order[:leftovers]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def build_subject_split(
    records: Sequence[TemporalSequence], config: TrainingConfig
) -> SplitManifest:
    if not isinstance(records, (tuple, list)) or not records:
        raise TrainingError("records must be a nonempty tuple or list")
    if any(not isinstance(record, TemporalSequence) for record in records):
        raise TrainingError("records must contain TemporalSequence values")
    sequence_ids = [record.sequence_id for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise TrainingError("sequence IDs must be unique")
    subjects = sorted(
        {record.subject_id for record in records},
        key=lambda value: (sha256(f"{config.split_seed}\0{value}".encode()).hexdigest(), value),
    )
    train_count, validation_count, _ = _subject_counts(len(subjects), config)
    subject_split = {}
    for index, subject in enumerate(subjects):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        subject_split[subject] = split
    assignments = tuple(
        SplitAssignment(record.sequence_id, record.subject_id, record.session_id,
                        record.label.value, subject_split[record.subject_id])
        for record in sorted(records, key=lambda item: item.sequence_id)
    )
    split_names = ("train", "validation", "test")
    sequence_counts = {
        name: sum(assignment.split == name for assignment in assignments)
        for name in split_names
    }
    subject_counts = {
        name: sum(value == name for value in subject_split.values())
        for name in split_names
    }
    if any(count == 0 for count in sequence_counts.values()):
        raise TrainingError("train, validation, and test splits must all be nonempty")
    return SplitManifest(_dataset_digest(records), assignments, sequence_counts, subject_counts)


def write_split_manifest(path: Path, manifest: SplitManifest, config: TrainingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest.as_record(config), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class TrainingPlan:
    config: TrainingConfig
    dataset_path: Path
    records: tuple[TemporalSequence, ...] = ()
    manifest: SplitManifest | None = None
    blockers: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        if not self.records or self.manifest is None or self.blockers:
            return False
        records_by_id = {record.sequence_id: record for record in self.records}
        if len(records_by_id) != len(self.records):
            return False
        assignments_by_id = {
            assignment.sequence_id: assignment for assignment in self.manifest.assignments
        }
        if (len(assignments_by_id) != len(self.manifest.assignments)
                or set(records_by_id) != set(assignments_by_id)
                or self.manifest.dataset_sha256 != _dataset_digest(self.records)):
            return False
        split_by_subject: dict[str, set[str]] = {}
        training_labels = set()
        for sequence_id, sequence in records_by_id.items():
            assignment = assignments_by_id[sequence_id]
            if (assignment.subject_id != sequence.subject_id
                    or assignment.session_id != sequence.session_id
                    or assignment.label != sequence.label.value
                    or assignment.split not in {"train", "validation", "test"}):
                return False
            split_by_subject.setdefault(assignment.subject_id, set()).add(assignment.split)
            if assignment.split == "train":
                training_labels.add(assignment.label)
        return (all(len(splits) == 1 for splits in split_by_subject.values())
                and training_labels == set(CLASS_NAMES))

    def as_record(self) -> dict[str, object]:
        return {
            "event": "temporal_training_check",
            "status": "ready_for_training_backend" if self.ready else "blocked",
            "dataset_path": str(self.dataset_path),
            "sequence_count": len(self.records),
            "subject_count": len({record.subject_id for record in self.records}),
            "split_sequence_counts": (
                dict(self.manifest.sequence_counts) if self.manifest else None
            ),
            "dataset_sha256": self.manifest.dataset_sha256 if self.manifest else None,
            "blockers": list(self.blockers),
            "training_performed": False,
            "model_artifact_available": False,
            "fall_detection_available": False,
        }


def prepare_training_plan(
    config: TrainingConfig, dataset_path: Path | None = None
) -> TrainingPlan:
    path = dataset_path or config.dataset_path
    if not path.is_file():
        return TrainingPlan(config, path, blockers=("authorized_labeled_dataset_missing",))
    records = load_sequence_dataset(path, config)
    manifest = build_subject_split(records, config)
    train_labels = {
        assignment.label for assignment in manifest.assignments if assignment.split == "train"
    }
    missing = tuple(name for name in CLASS_NAMES if name not in train_labels)
    blockers = (("training_split_missing_labels:" + ",".join(missing),) if missing else ())
    return TrainingPlan(config, path, records, manifest, blockers)


class TemporalModelExporter(Protocol):
    """A future trainer must write one ONNX graph and return no claimed metrics."""

    def train_and_export(self, plan: TrainingPlan, output_path: Path) -> None: ...


@dataclass(frozen=True)
class TemporalModelContract:
    model_path: Path
    input_shape: tuple[object, ...]
    output_shape: tuple[object, ...]
    class_names: tuple[str, ...]
    sha256: str

    def as_record(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "input_name": "features",
            "input_shape": list(self.input_shape),
            "output_name": "logits",
            "output_shape": list(self.output_shape),
            "class_names": list(self.class_names),
            "sha256": self.sha256,
            "provider": "CPUExecutionProvider",
        }


def _batch_dimension_valid(value: object) -> bool:
    return value in (None, 1, "batch") or isinstance(value, str)


def verify_temporal_onnx_contract(
    path: Path, config: TrainingConfig, *, smoke_run: bool = True
) -> TemporalModelContract:
    if not path.is_file():
        raise TrainingError(f"temporal model file not found: {path}")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        raise TrainingError(
            "temporal ONNX verification requires the model extra: pip install -e '.[model]'"
        ) from error
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise TrainingError(f"could not load temporal ONNX model: {error}") from error
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "features" or inputs[0].type != "tensor(float)":
        raise TrainingError("temporal model must have one tensor(float) input named features")
    if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].type != "tensor(float)":
        raise TrainingError("temporal model must have one tensor(float) output named logits")
    input_shape, output_shape = tuple(inputs[0].shape), tuple(outputs[0].shape)
    if (len(input_shape) != 3 or not _batch_dimension_valid(input_shape[0])
            or input_shape[1:] != (config.sequence_length, config.feature_width)):
        raise TrainingError(
            "temporal input shape must be [batch, sequence_length, feature_width]"
        )
    if (len(output_shape) != 2 or not _batch_dimension_valid(output_shape[0])
            or output_shape[1] != len(CLASS_NAMES)):
        raise TrainingError("temporal output shape must be [batch, 3]")
    metadata = session.get_modelmeta().custom_metadata_map
    expected_metadata = {
        "model_contract_version": str(MODEL_CONTRACT_VERSION),
        "dataset_schema_version": str(DATASET_SCHEMA_VERSION),
        "class_names": ",".join(CLASS_NAMES),
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise TrainingError(f"temporal model metadata {name} must equal {expected}")
    if smoke_run:
        tensor = np.zeros((1, config.sequence_length, config.feature_width), dtype=np.float32)
        try:
            result = session.run(["logits"], {"features": tensor})[0]
        except Exception as error:
            raise TrainingError(f"temporal ONNX smoke inference failed: {error}") from error
        if result.shape != (1, len(CLASS_NAMES)) or not np.isfinite(result).all():
            raise TrainingError("temporal ONNX smoke output must be finite with shape [1, 3]")
    digest = sha256(path.read_bytes()).hexdigest()
    return TemporalModelContract(path, input_shape, output_shape, CLASS_NAMES, digest)


def export_temporal_model(
    plan: TrainingPlan,
    exporter: TemporalModelExporter,
    output_path: Path | None = None,
) -> TemporalModelContract:
    """Run an injected backend, validate its graph, and publish it without overwrite."""
    if not isinstance(plan, TrainingPlan) or not plan.ready:
        raise TrainingError("training plan is blocked and cannot produce a model")
    target = output_path or plan.config.model_path
    if target.exists():
        raise TrainingError(f"model output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        exporter.train_and_export(plan, temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise TrainingError("training backend did not produce a nonempty ONNX file")
        contract = verify_temporal_onnx_contract(temporary, plan.config)
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise TrainingError(f"model output already exists: {target}") from error
        return TemporalModelContract(
            target, contract.input_shape, contract.output_shape,
            contract.class_names, contract.sha256,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _smoke_matrix(config: TrainingConfig, *, offset: float) -> list[list[float]]:
    matrix = []
    for index in range(config.sequence_length):
        row = [
            index * 0.1, 1.0, 0.9, 0.5, 0.3 + offset,
            0.2, 0.4, 0.5, 0.0, 0.0,
        ]
        row.extend([0.0, 0.0] * config.keypoint_count)
        row.extend([1.0] * config.keypoint_count)
        matrix.append(row)
    return matrix


def run_training_smoke(config: TrainingConfig | None = None) -> dict[str, object]:
    """Exercise schema and split code without training or producing an artifact."""
    actual = config or TrainingConfig(sequence_length=3, keypoint_count=2)
    labels = CLASS_NAMES * 3
    records = []
    for index, label in enumerate(labels):
        records.append(parse_sequence_record({
            "schema_version": DATASET_SCHEMA_VERSION,
            "sequence_id": f"synthetic-sequence-{index}",
            "subject_id": f"synthetic-subject-{index}",
            "session_id": f"synthetic-session-{index}",
            "authorization_id": "synthetic-contract-fixture",
            "label": label,
            "features": _smoke_matrix(actual, offset=index * 0.01),
        }, actual))
    manifest = build_subject_split(records, actual)
    split_by_subject: dict[str, set[str]] = {}
    for assignment in manifest.assignments:
        split_by_subject.setdefault(assignment.subject_id, set()).add(assignment.split)
    return {
        "event": "training_contract_smoke",
        "synthetic": True,
        "records_validated": len(records),
        "subjects": len(split_by_subject),
        "feature_width": actual.feature_width,
        "split_sequence_counts": dict(manifest.sequence_counts),
        "subject_leakage": any(len(splits) != 1 for splits in split_by_subject.values()),
        "training_performed": False,
        "model_artifact_available": False,
        "artifact_blocker": "authorized_labeled_dataset_missing",
        "fall_detection_available": False,
    }
