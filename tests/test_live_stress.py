"""Synthetic live-stream stress regression checks; not fall accuracy tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.live_stress import main, run_live_stress


class LiveStressTests(unittest.TestCase):
    def test_multi_person_tracking_survives_short_occlusion(self):
        report = run_live_stress(frames=45, people=4, width=160, height=120)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["frames_processed"], 45)
        self.assertGreater(report["missing_observations"], 0)
        self.assertEqual(report["max_window_samples"], 30)
        self.assertTrue(report["checks"]["stable_person_ids"])
        self.assertTrue(report["checks"]["no_unverified_fall_predictions"])
        self.assertFalse(report["fall_detection_available"])

    def test_latency_profile_covers_every_synthetic_frame(self):
        report = run_live_stress(frames=12, people=2, width=96, height=72, profile_latency=True)
        self.assertTrue(report["passed"], report["checks"])
        profile = report["latency_profile"]
        self.assertEqual(profile["frames_seen"], 12)
        self.assertEqual(profile["frames_sampled"], 12)
        self.assertIsNotNone(profile["latency_ms"]["total_ms"])
        self.assertFalse(profile["camera_to_alert_latency_measured"])

    def test_single_frame_is_valid_without_occlusion(self):
        report = run_live_stress(frames=1, people=1, width=48, height=48)
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["missing_observations"], 0)
        self.assertEqual(report["max_window_samples"], 1)

    def test_memory_threshold_failure_is_explicit(self):
        report = run_live_stress(
            frames=3, people=1, width=48, height=48,
            max_peak_memory_mib=0.000000001,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["peak_python_memory_within_limit"])

    def test_invalid_parameters_fail_closed(self):
        for kwargs in (
            {"frames": 0}, {"people": 0}, {"people": 17},
            {"people": 4, "width": 24},
            {"fps": float("inf")}, {"fps": 0},
            {"max_peak_memory_mib": -1}, {"profile_latency": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_live_stress(**kwargs)

    def test_cli_writes_exclusive_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live-stress.json"
            args = ["--frames", "12", "--people", "2",
                    "--width", "96", "--height", "72", "--profile-latency", "--report", str(path)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["latency_profile"]["frames_seen"], 12)
            original = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
