"""Reproducible held-out evaluation and CPU temporal-model latency reporting."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import platform
import statistics
import time
import tomllib
from typing import Callable, Mapping, Protocol, Sequence

from .contracts import FallState
from .training import (
    CLASS_NAMES,
    DATASET_SCHEMA_VERSION,
    SplitAssignment,
    SplitManifest,
    TemporalModelContract,
    TemporalSequence,
    TrainingConfig,
    dataset_digest,
    feature_names,
    load_sequence_dataset,
    verify_temporal_onnx_contract,
)


EVALUATION_REPORT_SCHEMA_VERSION = 1
PREDICTION_NAMES = (*CLASS_NAMES, FallState.UNKNOWN.value)
_SPLIT_NAMES = ("train", "validation", "test")
_MANIFEST_KEYS = {
    "schema_version", "dataset_sha256", "split_seed", "sequence_length",
    "keypoint_count", "feature_width", "feature_names", "class_names",
    "sequence_counts", "subject_counts", "assignments",
}
_ASSIGNMENT_KEYS = {"sequence_id", "subject_id", "session_id", "label", "split"}


class EvaluationError(ValueError):
    """Raised when evaluation inputs, execution, or reports violate their contract."""


def _identifier(value: object, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvaluationError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise EvaluationError(f"{name} cannot contain control characters")
    return value


def _nonnegative_integer(value: object, name: str, *, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise EvaluationError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _sha256_text(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise EvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings that affect evaluation membership, timing, and report size."""

    training_config_path: Path = Path("configs/training.toml")
    split_manifest_path: Path = Path("outputs/split-manifest.json")
    warmup_runs: int = 3
    timed_runs: int = 5
    failure_case_limit: int = 100

    def __post_init__(self) -> None:
        if (not isinstance(self.training_config_path, Path)
                or not isinstance(self.split_manifest_path, Path)):
            raise EvaluationError("training_config_path and split_manifest_path must be paths")
        _nonnegative_integer(self.warmup_runs, "warmup_runs", maximum=10_000)
        if type(self.timed_runs) is not int or not 1 <= self.timed_runs <= 10_000:
            raise EvaluationError("timed_runs must be an integer between 1 and 10000")
        if type(self.failure_case_limit) is not int or not 0 <= self.failure_case_limit <= 10_000:
            raise EvaluationError("failure_case_limit must be an integer between 0 and 10000")


def load_evaluation_config(path: Path | None = None, **overrides: object) -> EvaluationConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"evaluation"} or not isinstance(document["evaluation"], dict):
            raise EvaluationError("evaluation configuration must contain only an [evaluation] table")
        settings.update(document["evaluation"])
    allowed = {field.name for field in fields(EvaluationConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise EvaluationError("unknown evaluation settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    for name in ("training_config_path", "split_manifest_path"):
        if name in settings:
            if not isinstance(settings[name], (str, Path)):
                raise EvaluationError(f"{name} must be a path string")
            settings[name] = Path(settings[name])
    return EvaluationConfig(**settings)


def _exact_count_map(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_SPLIT_NAMES):
        raise EvaluationError(f"{name} must contain train, validation, and test")
    result = {}
    for split in _SPLIT_NAMES:
        result[split] = _nonnegative_integer(value[split], f"{name}.{split}")
    return result


def load_split_manifest(path: Path, config: TrainingConfig) -> SplitManifest:
    """Load the immutable split contract written during training preparation."""
    if not isinstance(path, Path):
        raise EvaluationError("split manifest path must be a Path")
    if not path.is_file():
        raise EvaluationError(f"split manifest not found: {path}")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvaluationError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, EvaluationError) as error:
        raise EvaluationError(f"invalid split manifest JSON: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise EvaluationError("split manifest has unknown or missing fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != DATASET_SCHEMA_VERSION:
        raise EvaluationError(f"split manifest schema_version must be {DATASET_SCHEMA_VERSION}")
    _sha256_text(raw["dataset_sha256"], "dataset_sha256")
    if raw["split_seed"] != config.split_seed:
        raise EvaluationError("split manifest split_seed does not match training configuration")
    expected_scalars = {
        "sequence_length": config.sequence_length,
        "keypoint_count": config.keypoint_count,
        "feature_width": config.feature_width,
    }
    for name, expected in expected_scalars.items():
        if type(raw[name]) is not int or raw[name] != expected:
            raise EvaluationError(f"split manifest {name} does not match training configuration")
    if raw["feature_names"] != list(feature_names(config.keypoint_count)):
        raise EvaluationError("split manifest feature_names do not match the runtime contract")
    if raw["class_names"] != list(CLASS_NAMES):
        raise EvaluationError("split manifest class_names do not match the runtime contract")
    sequence_counts = _exact_count_map(raw["sequence_counts"], "sequence_counts")
    subject_counts = _exact_count_map(raw["subject_counts"], "subject_counts")
    raw_assignments = raw["assignments"]
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise EvaluationError("split manifest assignments must be a nonempty list")
    assignments = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_assignments):
        if not isinstance(item, dict) or set(item) != _ASSIGNMENT_KEYS:
            raise EvaluationError(f"split assignment {index} has unknown or missing fields")
        sequence_id = _identifier(item["sequence_id"], "sequence_id")
        if sequence_id in seen_ids:
            raise EvaluationError(f"duplicate split assignment: {sequence_id}")
        seen_ids.add(sequence_id)
        subject_id = _identifier(item["subject_id"], "subject_id")
        session_id = _identifier(item["session_id"], "session_id")
        if item["label"] not in CLASS_NAMES:
            raise EvaluationError(f"split assignment {index} has an invalid label")
        if item["split"] not in _SPLIT_NAMES:
            raise EvaluationError(f"split assignment {index} has an invalid split")
        assignments.append(SplitAssignment(
            sequence_id, subject_id, session_id, item["label"], item["split"],
        ))
    actual_sequence_counts = {
        split: sum(item.split == split for item in assignments) for split in _SPLIT_NAMES
    }
    actual_subject_counts = {
        split: len({item.subject_id for item in assignments if item.split == split})
        for split in _SPLIT_NAMES
    }
    if sequence_counts != actual_sequence_counts or subject_counts != actual_subject_counts:
        raise EvaluationError("split manifest count summaries do not match assignments")
    by_subject: dict[str, set[str]] = {}
    for item in assignments:
        by_subject.setdefault(item.subject_id, set()).add(item.split)
    if any(len(splits) != 1 for splits in by_subject.values()):
        raise EvaluationError("split manifest leaks a subject across partitions")
    return SplitManifest(
        raw["dataset_sha256"], tuple(assignments), sequence_counts, subject_counts,
    )


def validate_manifest_dataset(
    manifest: SplitManifest,
    records: Sequence[TemporalSequence],
) -> tuple[TemporalSequence, ...]:
    """Return the held-out test records after validating manifest/data identity."""
    if not isinstance(manifest, SplitManifest):
        raise EvaluationError("manifest must be a SplitManifest")
    if not isinstance(records, (tuple, list)) or not records:
        raise EvaluationError("evaluation records must be a nonempty tuple or list")
    if any(not isinstance(record, TemporalSequence) for record in records):
        raise EvaluationError("evaluation records must contain TemporalSequence values")
    records_by_id = {record.sequence_id: record for record in records}
    if len(records_by_id) != len(records):
        raise EvaluationError("evaluation sequence IDs must be unique")
    assignments_by_id = {item.sequence_id: item for item in manifest.assignments}
    if len(assignments_by_id) != len(manifest.assignments):
        raise EvaluationError("split assignment IDs must be unique")
    if set(records_by_id) != set(assignments_by_id):
        raise EvaluationError("split manifest assignments do not cover the dataset exactly")
    if manifest.dataset_sha256 != dataset_digest(records):
        raise EvaluationError("split manifest dataset fingerprint does not match the dataset")
    for sequence_id, record in records_by_id.items():
        assignment = assignments_by_id[sequence_id]
        if (assignment.subject_id != record.subject_id
                or assignment.session_id != record.session_id
                or assignment.label != record.label.value):
            raise EvaluationError(f"split assignment metadata mismatch for {sequence_id}")
    test_records = tuple(
        records_by_id[item.sequence_id]
        for item in sorted(manifest.assignments, key=lambda value: value.sequence_id)
        if item.split == "test"
    )
    if not test_records:
        raise EvaluationError("split manifest has no held-out test sequences")
    return test_records


@dataclass(frozen=True)
class EvaluationPlan:
    config: EvaluationConfig
    training_config: TrainingConfig
    dataset_path: Path
    manifest_path: Path
    model_path: Path
    records: tuple[TemporalSequence, ...] = ()
    test_records: tuple[TemporalSequence, ...] = ()
    manifest: SplitManifest | None = None
    model_contract: TemporalModelContract | None = None
    dataset_file_sha256: str | None = None
    manifest_file_sha256: str | None = None
    blockers: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        if (self.blockers or not self.records or not self.test_records
                or self.manifest is None or self.model_contract is None):
            return False
        if self.model_contract.model_path != self.model_path:
            return False
        try:
            validated_test = validate_manifest_dataset(self.manifest, self.records)
        except EvaluationError:
            return False
        return validated_test == self.test_records

    def as_record(self) -> dict[str, object]:
        support = {
            name: sum(record.label.value == name for record in self.test_records)
            for name in CLASS_NAMES
        }
        return {
            "event": "replay_evaluation_check",
            "status": "ready_for_authorized_replay_evaluation" if self.ready else "blocked",
            "dataset_path": str(self.dataset_path),
            "split_manifest_path": str(self.manifest_path),
            "model_path": str(self.model_path),
            "dataset_sequence_count": len(self.records),
            "test_sequence_count": len(self.test_records),
            "test_class_support": support,
            "dataset_sha256": self.manifest.dataset_sha256 if self.manifest else None,
            "dataset_file_sha256": self.dataset_file_sha256,
            "split_manifest_file_sha256": self.manifest_file_sha256,
            "model_sha256": self.model_contract.sha256 if self.model_contract else None,
            "model_contract_validated": self.model_contract is not None,
            "blockers": list(self.blockers),
            "evaluation_performed": False,
            "quality_metrics_available": False,
            "real_world_quality_measured": False,
            "fall_detection_available": False,
        }


ModelVerifier = Callable[[Path, TrainingConfig], TemporalModelContract]


def _default_model_verifier(path: Path, config: TrainingConfig) -> TemporalModelContract:
    return verify_temporal_onnx_contract(path, config, smoke_run=False)


def prepare_evaluation_plan(
    config: EvaluationConfig,
    training_config: TrainingConfig,
    *,
    model_verifier: ModelVerifier = _default_model_verifier,
) -> EvaluationPlan:
    """Validate all actual-input prerequisites without producing any metrics."""
    if not isinstance(config, EvaluationConfig):
        raise EvaluationError("config must be an EvaluationConfig")
    if not isinstance(training_config, TrainingConfig):
        raise EvaluationError("training_config must be a TrainingConfig")
    dataset_path = training_config.dataset_path
    manifest_path = config.split_manifest_path
    model_path = training_config.model_path
    blockers = []
    if not dataset_path.is_file():
        blockers.append("authorized_labeled_replay_sequences_missing")
    if not manifest_path.is_file():
        blockers.append("held_out_split_manifest_missing")
    if not model_path.is_file():
        blockers.append("trained_temporal_model_missing")
    if blockers:
        return EvaluationPlan(
            config, training_config, dataset_path, manifest_path, model_path,
            blockers=tuple(blockers),
        )
    records = load_sequence_dataset(dataset_path, training_config)
    manifest = load_split_manifest(manifest_path, training_config)
    test_records = validate_manifest_dataset(manifest, records)
    missing_classes = tuple(
        name for name in CLASS_NAMES
        if not any(record.label.value == name for record in test_records)
    )
    if missing_classes:
        blockers.append("test_split_missing_labels:" + ",".join(missing_classes))
    model_contract = model_verifier(model_path, training_config)
    return EvaluationPlan(
        config=config,
        training_config=training_config,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        model_path=model_path,
        records=records,
        test_records=test_records,
        manifest=manifest,
        model_contract=model_contract,
        dataset_file_sha256=sha256(dataset_path.read_bytes()).hexdigest(),
        manifest_file_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        blockers=tuple(blockers),
    )


class TemporalSequencePredictor(Protocol):
    def predict(self, sequence: TemporalSequence) -> FallState: ...


class OnnxTemporalSequencePredictor:
    """CPU-only predictor for the validated temporal ONNX contract."""

    def __init__(self, path: Path, config: TrainingConfig) -> None:
        verify_temporal_onnx_contract(path, config, smoke_run=False)
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise EvaluationError(
                "temporal evaluation requires the model extra: pip install -e '.[model]'"
            ) from error
        try:
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise EvaluationError(f"could not load temporal ONNX model: {error}") from error
        self._np = np
        self.runtime_metadata = {
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
        }
        self._config = config

    def predict(self, sequence: TemporalSequence) -> FallState:
        if not isinstance(sequence, TemporalSequence):
            raise EvaluationError("temporal predictor requires a TemporalSequence")
        array = self._np.asarray(sequence.features, dtype=self._np.float32)[None, :, :]
        expected = (1, self._config.sequence_length, self._config.feature_width)
        if array.shape != expected or not self._np.isfinite(array).all():
            raise EvaluationError("evaluation tensor violates the temporal input contract")
        try:
            logits = self._session.run(["logits"], {"features": array})[0]
        except Exception as error:
            raise EvaluationError(f"temporal ONNX evaluation failed: {error}") from error
        if logits.shape != (1, len(CLASS_NAMES)) or not self._np.isfinite(logits).all():
            raise EvaluationError("temporal ONNX evaluation output must be finite with shape [1, 3]")
        return FallState(CLASS_NAMES[int(self._np.argmax(logits[0]))])


@dataclass(frozen=True)
class EvaluationOutcome:
    sequence_id: str
    expected: FallState
    predicted: FallState
    median_latency_ms: float

    def __post_init__(self) -> None:
        _identifier(self.sequence_id, "sequence_id")
        if self.expected not in tuple(FallState(name) for name in CLASS_NAMES):
            raise EvaluationError("expected state must be a supervised class")
        if not isinstance(self.predicted, FallState):
            raise EvaluationError("predicted state must be a FallState")
        if (type(self.median_latency_ms) not in (int, float)
                or not isfinite(self.median_latency_ms) or self.median_latency_ms < 0):
            raise EvaluationError("median_latency_ms must be finite and nonnegative")


@dataclass(frozen=True)
class EvaluationResults:
    outcomes: tuple[EvaluationOutcome, ...]
    latency_samples_ms: tuple[float, ...]


def evaluate_sequences(
    records: Sequence[TemporalSequence],
    predictor: TemporalSequencePredictor,
    config: EvaluationConfig,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationResults:
    """Run deterministic sequence inference and retain every timed sample."""
    if not isinstance(records, (tuple, list)) or not records:
        raise EvaluationError("evaluation requires at least one held-out sequence")
    if any(not isinstance(record, TemporalSequence) for record in records):
        raise EvaluationError("evaluation inputs must contain TemporalSequence values")
    sequence_ids = [record.sequence_id for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise EvaluationError("evaluation sequence IDs must be unique")
    if not isinstance(config, EvaluationConfig):
        raise EvaluationError("config must be an EvaluationConfig")
    ordered = tuple(sorted(records, key=lambda record: record.sequence_id))
    for _ in range(config.warmup_runs):
        warmup = predictor.predict(ordered[0])
        if not isinstance(warmup, FallState):
            raise EvaluationError("predictor must return a FallState")
    outcomes = []
    all_latencies = []
    for record in ordered:
        predictions = []
        latencies = []
        for _ in range(config.timed_runs):
            started = clock()
            prediction = predictor.predict(record)
            finished = clock()
            if not isinstance(prediction, FallState):
                raise EvaluationError("predictor must return a FallState")
            elapsed_ms = (finished - started) * 1_000.0
            if not isfinite(elapsed_ms) or elapsed_ms < 0:
                raise EvaluationError("evaluation clock must produce finite, ordered values")
            predictions.append(prediction)
            latencies.append(elapsed_ms)
        if len(set(predictions)) != 1:
            raise EvaluationError(f"nondeterministic predictions for {record.sequence_id}")
        outcomes.append(EvaluationOutcome(
            record.sequence_id, record.label, predictions[0], statistics.median(latencies),
        ))
        all_latencies.extend(latencies)
    return EvaluationResults(tuple(outcomes), tuple(all_latencies))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise EvaluationError("latency summary requires at least one sample")
    if any(type(value) not in (int, float) or not isfinite(value) or value < 0
           for value in values):
        raise EvaluationError("latency samples must be finite and nonnegative")
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "maximum_ms": ordered[-1],
    }


def summarize_results(results: EvaluationResults, failure_case_limit: int) -> dict[str, object]:
    if not isinstance(results, EvaluationResults) or not results.outcomes:
        raise EvaluationError("results must contain evaluation outcomes")
    if type(failure_case_limit) is not int or not 0 <= failure_case_limit <= 10_000:
        raise EvaluationError("failure_case_limit must be an integer between 0 and 10000")
    matrix = {
        expected: {predicted: 0 for predicted in PREDICTION_NAMES}
        for expected in CLASS_NAMES
    }
    seen = set()
    failures = []
    for outcome in results.outcomes:
        if not isinstance(outcome, EvaluationOutcome):
            raise EvaluationError("results contain an invalid outcome")
        if outcome.sequence_id in seen:
            raise EvaluationError("result sequence IDs must be unique")
        seen.add(outcome.sequence_id)
        matrix[outcome.expected.value][outcome.predicted.value] += 1
        if outcome.expected is not outcome.predicted:
            failures.append(outcome.sequence_id)
    per_class = {}
    f1_values = []
    for name in CLASS_NAMES:
        true_positive = matrix[name][name]
        support = sum(matrix[name].values())
        predicted_count = sum(matrix[expected][name] for expected in CLASS_NAMES)
        precision = _ratio(true_positive, predicted_count)
        recall = _ratio(true_positive, support)
        if support == 0:
            f1 = None
        elif not precision or not recall:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if f1 is not None:
            f1_values.append(f1)
    total = len(results.outcomes)
    correct = sum(matrix[name][name] for name in CLASS_NAMES)
    unknown = sum(matrix[name][FallState.UNKNOWN.value] for name in CLASS_NAMES)
    return {
        "sequence_count": total,
        "correct_count": correct,
        "unknown_prediction_count": unknown,
        "accuracy": correct / total,
        "coverage": (total - unknown) / total,
        "macro_f1": statistics.fmean(f1_values) if f1_values else None,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "regression": {
            "case_count": total,
            "pass_count": total - len(failures),
            "failure_count": len(failures),
            "failure_sequence_ids": failures[:failure_case_limit],
            "failure_sequence_ids_truncated": len(failures) > failure_case_limit,
        },
        "temporal_model_latency": {
            "scope": "CPU temporal-model inference only; excludes decode, pose, tracking, and features",
            "warmup_runs_per_first_sequence": None,
            **_latency_summary(results.latency_samples_ms),
        },
    }


def build_evaluation_report(
    plan: EvaluationPlan,
    results: EvaluationResults,
    *,
    runtime_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(plan, EvaluationPlan) or not plan.ready:
        raise EvaluationError("evaluation plan is blocked and cannot produce a quality report")
    if not isinstance(results, EvaluationResults):
        raise EvaluationError("results must be EvaluationResults")
    if {item.sequence_id for item in results.outcomes} != {
            item.sequence_id for item in plan.test_records}:
        raise EvaluationError("results do not cover the held-out test split exactly")
    summary = summarize_results(results, plan.config.failure_case_limit)
    summary["temporal_model_latency"]["warmup_runs_per_first_sequence"] = plan.config.warmup_runs
    summary["temporal_model_latency"]["timed_runs_per_sequence"] = plan.config.timed_runs
    metadata = dict(runtime_metadata or {"predictor": "injected"})
    if (not metadata or any(not isinstance(key, str) or not key.strip() for key in metadata)
            or any(value is None for value in metadata.values())):
        raise EvaluationError("runtime metadata must contain nonempty string keys and values")
    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "event": "held_out_replay_evaluation_report",
        "measurement_scope": "provided_authorized_labeled_feature_sequences",
        "synthetic": False,
        "training_split_evaluated": False,
        "held_out_split": "test",
        "dataset_sha256": plan.manifest.dataset_sha256,
        "dataset_file_sha256": plan.dataset_file_sha256,
        "split_manifest_file_sha256": plan.manifest_file_sha256,
        "model_sha256": plan.model_contract.sha256,
        "authorization_reference_count": len({
            record.authorization_id for record in plan.test_records
        }),
        "authorization_verified_by_software": False,
        "model_training_provenance_verified_by_software": False,
        "quality_metrics_available": True,
        "real_world_quality_measured": True,
        "clinical_validation": False,
        "external_benchmarks_used": False,
        "fall_detection_available": False,
        "runtime_environment": metadata,
        **summary,
    }
    json.dumps(report, allow_nan=False)
    return report


def run_evaluation(
    plan: EvaluationPlan,
    *,
    predictor: TemporalSequencePredictor | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    if not isinstance(plan, EvaluationPlan) or not plan.ready:
        raise EvaluationError("evaluation plan is blocked")
    actual_predictor = predictor or OnnxTemporalSequencePredictor(
        plan.model_path, plan.training_config,
    )
    results = evaluate_sequences(
        plan.test_records, actual_predictor, plan.config, clock=clock,
    )
    metadata = getattr(actual_predictor, "runtime_metadata", {"predictor": "injected"})
    return build_evaluation_report(plan, results, runtime_metadata=metadata)


def write_evaluation_report(path: Path, report: Mapping[str, object]) -> None:
    if not isinstance(path, Path):
        raise EvaluationError("report path must be a Path")
    if not isinstance(report, Mapping) or report.get("event") != "held_out_replay_evaluation_report":
        raise EvaluationError("only a held-out evaluation report can be written")
    if report.get("synthetic") is not False or report.get("quality_metrics_available") is not True:
        raise EvaluationError("synthetic or unavailable metrics cannot be written as a quality report")
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _synthetic_sequence(index: int, label: FallState, config: TrainingConfig) -> TemporalSequence:
    rows = []
    for step in range(config.sequence_length):
        row = [
            step * 0.1, 1.0, 0.9, 0.5, 0.3 + index * 0.01,
            0.2, 0.4, 0.5, 0.0, 0.0,
        ]
        row.extend([0.0, 0.0] * config.keypoint_count)
        row.extend([1.0] * config.keypoint_count)
        rows.append(tuple(row))
    return TemporalSequence(
        f"synthetic-evaluation-{index}", f"synthetic-subject-{index}",
        f"synthetic-session-{index}", "synthetic-contract-fixture", label, tuple(rows),
    )


def run_evaluation_smoke() -> dict[str, object]:
    """Exercise accounting with synthetic records without publishing quality metrics."""
    training_config = TrainingConfig(sequence_length=3, keypoint_count=2)
    evaluation_config = EvaluationConfig(warmup_runs=1, timed_runs=2)
    labels = tuple(FallState(name) for name in CLASS_NAMES) * 2
    records = tuple(
        _synthetic_sequence(index, label, training_config)
        for index, label in enumerate(labels)
    )
    predictions = {
        records[0].sequence_id: FallState.NORMAL,
        records[1].sequence_id: FallState.FALLING,
        records[2].sequence_id: FallState.FALLEN,
        records[3].sequence_id: FallState.UNKNOWN,
        records[4].sequence_id: FallState.NORMAL,
        records[5].sequence_id: FallState.FALLEN,
    }

    class Predictor:
        def predict(self, sequence: TemporalSequence) -> FallState:
            return predictions[sequence.sequence_id]

    class Clock:
        value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 0.001
            return current

    results = evaluate_sequences(records, Predictor(), evaluation_config, clock=Clock())
    summary = summarize_results(results, evaluation_config.failure_case_limit)
    return {
        "event": "evaluation_contract_smoke",
        "synthetic": True,
        "cases_validated": summary["sequence_count"],
        "regression_pass_count": summary["regression"]["pass_count"],
        "regression_failure_count": summary["regression"]["failure_count"],
        "unknown_predictions_exercised": summary["unknown_prediction_count"],
        "latency_samples_validated": summary["temporal_model_latency"]["count"],
        "quality_metrics_available": False,
        "real_world_quality_measured": False,
        "report_written": False,
        "fall_detection_available": False,
    }
