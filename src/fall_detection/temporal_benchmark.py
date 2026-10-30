"""CPU temporal ONNX inference microbenchmark; never an accuracy evaluation.

Measures validated feature encoding plus ORT inference on synthetic pose windows.
It does NOT measure camera, YOLO, tracking, end-to-end latency or safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable

from .features import PoseFeatureSample, TemporalFeatureWindow
from .temporal_inference import OnnxTemporalClassifier
from .training import (
    CLASS_NAMES, DATASET_SCHEMA_VERSION, MODEL_CONTRACT_VERSION,
    TrainingConfig, TrainingError, load_training_config,
)


def synthetic_window(config: TrainingConfig) -> TemporalFeatureWindow:
    """Construct valid, entirely synthetic upright pose evidence."""
    keypoints = tuple((0.5, 0.5) for _ in range(config.keypoint_count))
    mask = (True,) * config.keypoint_count
    samples = tuple(
        PoseFeatureSample(
            track_id=1, frame_index=index, timestamp_seconds=index / 30.0,
            observed=True, detection_confidence=0.95,
            box_center_xy=(0.5, 0.5), box_size_wh=(0.2, 0.4),
            box_aspect_ratio=0.5, center_velocity_xy_per_second=(0.0, 0.0),
            keypoints_xy=keypoints, keypoint_mask=mask,
        )
        for index in range(config.sequence_length)
    )
    return TemporalFeatureWindow(1, samples, len(samples), True)


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    """Return an observed percentile (nearest-rank, not interpolation)."""
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires nonempty samples and fraction in (0, 1]")
    return sorted(values)[ceil(fraction * len(values)) - 1]


def benchmark_classifier(
    classifier: OnnxTemporalClassifier,
    window: TemporalFeatureWindow,
    *,
    iterations: int = 100,
    warmup: int = 10,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Time classify() with a ready synthetic window; warmups excluded."""
    if type(iterations) is not int or not 1 <= iterations <= 100_000:
        raise ValueError("iterations must be an integer from 1 to 100000")
    if type(warmup) is not int or not 0 <= warmup <= 10_000:
        raise ValueError("warmup must be an integer from 0 to 10000")
    for _ in range(warmup):
        classifier.classify(window)
    durations: list[float] = []
    counts: Counter[str] = Counter()
    for _ in range(iterations):
        start = clock()
        decision = classifier.classify(window)
        duration = (clock() - start) * 1000.0
        if not isfinite(duration) or duration < 0:
            raise ValueError("benchmark clock must return increasing finite values")
        durations.append(duration)
        counts[decision.state.value] += 1
    return {
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms": {
            "min": round(min(durations), 6),
            "p50": round(percentile_nearest_rank(durations, 0.50), 6),
            "p95": round(percentile_nearest_rank(durations, 0.95), 6),
            "p99": round(percentile_nearest_rank(durations, 0.99), 6),
            "max": round(max(durations), 6),
        },
        "state_counts": dict(sorted(counts.items())),
        "timed_inference_seconds": round(sum(durations) / 1000.0, 6),
    }


def create_constant_fixture(path: Path, config: TrainingConfig) -> None:
    """Create a contract-valid, UNTRAINED constant-logit ONNX graph for CI."""
    try:
        import onnx
    except ImportError as error:
        raise TrainingError("fixture creation requires pip install -e '.[test]'") from error
    helper, tensor = onnx.helper, onnx.TensorProto
    constant = helper.make_node(
        "Constant", inputs=[], outputs=["logits"],
        value=helper.make_tensor("scores", tensor.FLOAT, [1, 3], [0.0, 0.0, 0.0]),
    )
    graph = helper.make_graph(
        [constant], "synthetic-constant-temporal-benchmark",
        [helper.make_tensor_value_info(
            "features", tensor.FLOAT,
            [1, config.sequence_length, config.feature_width],
        )],
        [helper.make_tensor_value_info("logits", tensor.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    helper.set_model_props(model, {
        "model_contract_version": str(MODEL_CONTRACT_VERSION),
        "dataset_schema_version": str(DATASET_SCHEMA_VERSION),
        "class_names": ",".join(CLASS_NAMES),
    })
    onnx.save(model, str(path))


def benchmark_model(
    model_path: Path,
    config: TrainingConfig,
    *,
    iterations: int = 100,
    warmup: int = 10,
    synthetic_fixture: bool = False,
) -> dict[str, object]:
    """Verify the real ORT contract and benchmark its CPU inference path."""
    classifier = OnnxTemporalClassifier(model_path, config)
    report = benchmark_classifier(
        classifier, synthetic_window(config), iterations=iterations, warmup=warmup
    )
    digest = sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "event": "temporal_onnx_microbenchmark",
        "synthetic_input": True,
        "model_kind": ("untrained_constant_fixture" if synthetic_fixture
                       else "user_supplied_unverified_training"),
        "model_sha256": digest.hexdigest(),
        "input_shape": [1, config.sequence_length, config.feature_width],
        "provider": "CPUExecutionProvider",
        "scope": "python_feature_encoding_and_onnxruntime_only",
        "camera_pose_tracking_included": False,
        "model_accuracy_evaluated": False,
        "fall_detection_available": False,
        **report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Temporal ONNX CPU microbenchmark, NOT real-video latency or accuracy."
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--model", type=Path, help="Compatible ONNX temporal model")
    sources.add_argument("--synthetic-fixture", action="store_true",
                         help="Generate an untrained constant-output ONNX graph")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--report", type=Path, help="Write JSON, refusing overwrite")
    args = parser.parse_args(argv)
    try:
        if args.max_p95_ms is not None and (
            not isfinite(args.max_p95_ms) or args.max_p95_ms <= 0
        ):
            raise ValueError("--max-p95-ms must be finite and positive")
        config = load_training_config(args.training_config)
        if args.synthetic_fixture:
            with TemporaryDirectory() as directory:
                model_path = Path(directory) / "synthetic_constant.onnx"
                create_constant_fixture(model_path, config)
                report = benchmark_model(
                    model_path, config, iterations=args.iterations,
                    warmup=args.warmup, synthetic_fixture=True,
                )
        else:
            report = benchmark_model(
                args.model, config, iterations=args.iterations, warmup=args.warmup
            )
        if args.max_p95_ms is not None:
            report["p95_within_limit"] = report["latency_ms"]["p95"] <= args.max_p95_ms
            report["max_p95_ms"] = args.max_p95_ms
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if report.get("p95_within_limit", True) else 1
    except (OSError, ValueError, TrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
