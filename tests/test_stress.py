"""Synthetic pipeline stress regression tests; no ML performance claims."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.config import RuntimeConfig
from fall_detection.stress import main, run_stress


class StressTests(unittest.TestCase):
    def test_bounded_stream_and_honest_predictions(self):
        report = run_stress(RuntimeConfig(frames=40, width=16, height=8))
        self.assertTrue(report["passed"])
        self.assertEqual(report["frames_processed"], 40)
        self.assertEqual(report["bytes_streamed"], 40 * 16 * 8 * 3)
        self.assertEqual(report["unexpected_predictions"], 0)
        self.assertEqual(report["malformed_records"], 0)
        self.assertFalse(report["fall_detection_available"])
        self.assertTrue(report["synthetic"])
        self.assertGreater(report["peak_traced_python_memory_mib"], 0)

    def test_resource_threshold_failure_is_reported_not_hidden(self):
        report = run_stress(
            RuntimeConfig(frames=3, width=8, height=8),
            max_peak_memory_mib=0.000001,
            min_throughput_fps=1e12,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["peak_python_memory_within_limit"])
        self.assertFalse(report["checks"]["throughput_meets_minimum"])

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_stress(RuntimeConfig(frames=1, width=2, height=2),
                           max_peak_memory_mib=value)

    def test_cli_report_is_json_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "stress.json"
            args = ["--frames", "5", "--width", "8", "--height", "4",
                    "--report", str(target)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["frames_processed"], 5)
            original = target.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
