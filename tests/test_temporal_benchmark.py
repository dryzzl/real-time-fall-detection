"""Temporal microbenchmark regression tests; no trained model or video data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.contracts import FallState
from fall_detection.temporal_benchmark import (
    benchmark_classifier, benchmark_model, create_constant_fixture, main,
    percentile_nearest_rank, synthetic_window,
)
from fall_detection.temporal_inference import OnnxTemporalClassifier
from fall_detection.training import TrainingConfig, encode_feature_window


class FakeClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, window):
        self.calls += 1
        class Decision:
            state = FallState.NORMAL
        return Decision()


class TemporalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(sequence_length=3, keypoint_count=2)

    def test_synthetic_input_respects_versioned_feature_contract(self):
        window = synthetic_window(self.config)
        self.assertEqual(window.observed_count, 3)
        self.assertTrue(window.ready)
        matrix = encode_feature_window(window, self.config)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix[0]), 16)

    def test_nearest_rank_and_empty_input(self):
        self.assertEqual(percentile_nearest_rank([3, 1, 2], 0.5), 2)
        self.assertEqual(percentile_nearest_rank([3, 1, 2], 0.95), 3)
        with self.assertRaises(ValueError):
            percentile_nearest_rank([], 0.5)

    def test_deterministic_latency_and_warmup_exclusion(self):
        classifier = FakeClassifier()
        timestamps = iter([0.0, 0.001, 0.001, 0.003, 0.003, 0.006])
        result = benchmark_classifier(
            classifier, synthetic_window(self.config),
            iterations=3, warmup=2, clock=lambda: next(timestamps),
        )
        self.assertEqual(classifier.calls, 5)
        self.assertEqual(result["latency_ms"]["p50"], 2.0)
        self.assertEqual(result["latency_ms"]["p95"], 3.0)
        self.assertEqual(result["latency_ms"]["p99"], 3.0)
        self.assertEqual(result["state_counts"], {"normal": 3})

    def test_rejects_invalid_iterations_and_warmup(self):
        classifier = FakeClassifier()
        window = synthetic_window(self.config)
        for iterations in (0, True, 100001, 1.5):
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                benchmark_classifier(classifier, window, iterations=iterations)
        for warmup in (-1, True, 10001):
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                benchmark_classifier(classifier, window, warmup=warmup)

    def test_rejects_invalid_clock(self):
        classifier = FakeClassifier()
        with self.assertRaisesRegex(ValueError, "clock"):
            benchmark_classifier(
                classifier, synthetic_window(self.config), iterations=1,
                warmup=0, clock=iter([2.0, 1.0]).__next__,
            )

    def test_real_onnxruntime_fixture_is_explicitly_untrained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.onnx"
            create_constant_fixture(path, self.config)
            report = benchmark_model(
                path, self.config, iterations=3, warmup=1,
                synthetic_fixture=True,
            )
            self.assertEqual(report["model_kind"], "untrained_constant_fixture")
            self.assertEqual(report["input_shape"], [1, 3, 16])
            self.assertEqual(report["state_counts"], {"normal": 3})
            self.assertFalse(report["model_accuracy_evaluated"])
            self.assertFalse(report["fall_detection_available"])
            self.assertEqual(len(report["model_sha256"]), 64)

    def test_cli_writes_report_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "bench.json"
            output = io.StringIO()
            args = [
                "--synthetic-fixture", "--iterations", "3", "--warmup", "1",
                "--report", str(report_path),
            ]
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["model_kind"], "untrained_constant_fixture")
            self.assertEqual(json.loads(report_path.read_text()), result)
            before = report_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertEqual(report_path.read_bytes(), before)

    def test_cli_rejects_invalid_threshold(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(["--synthetic-fixture", "--max-p95-ms", "nan"]), 2
            )


if __name__ == "__main__":
    unittest.main()
