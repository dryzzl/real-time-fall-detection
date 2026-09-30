import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.config import RuntimeConfig, load_config
from fall_detection.contracts import FallState, FramePacket, Prediction
from fall_detection.pipeline import run_pipeline
from fall_detection.sources import synthetic_frames


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_runtime_settings(self):
        for overrides in ({"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")},
                          {"fps": True}, {"frames": -1}, {"width": True},
                          {"width": 1921}, {"height": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                RuntimeConfig(**overrides)

    def test_file_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text("[runtime]\nwidth=64\nframes=9\n", encoding="utf-8")
            result = load_config(path, frames=3, fps=None)
            self.assertEqual((result.width, result.frames, result.fps), (64, 3, 30.0))

    def test_unknown_or_malformed_table_is_rejected(self):
        for text in ("[runtime]\nfsp=30\n", "[run]\nfps=30\n", "runtime=30\n",
                     "[runtime]\nfps=30\n[other]\nx=1\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.toml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)


class ContractTests(unittest.TestCase):
    def test_rejects_truncated_image_and_nonfinite_timestamp(self):
        with self.assertRaises(ValueError):
            FramePacket(0, 0.0, 2, 2, b"\0" * 11)
        with self.assertRaises(ValueError):
            FramePacket(0, float("nan"), 1, 1, b"\0" * 3)

    def test_rejects_invalid_prediction_confidence(self):
        for score in (-0.1, 1.1, float("nan"), True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                Prediction(confidence=score)
        with self.assertRaises(ValueError):
            Prediction(state="normal")


class StreamTests(unittest.TestCase):
    def test_synthetic_frames_have_correct_shape_timing_and_motion(self):
        frames = list(synthetic_frames(RuntimeConfig(width=4, height=2, frames=3, fps=10)))
        self.assertEqual([f.timestamp_seconds for f in frames], [0.0, 0.1, 0.2])
        self.assertTrue(all(len(f.bgr) == 24 for f in frames))
        self.assertNotEqual(frames[0].bgr, frames[1].bgr)
        self.assertEqual(frames[1].bgr[3:6], b"\x50\xb0\xff")

    def test_no_model_never_reports_normal_or_a_fall(self):
        records = []
        summary = run_pipeline(synthetic_frames(RuntimeConfig(frames=5)), records.append)
        self.assertEqual(summary.frames_processed, 5)
        self.assertTrue(all(r["state"] == "unknown" for r in records))
        self.assertTrue(all(r["confidence"] is None for r in records))
        self.assertTrue(all(r["reason"] == "detection_not_implemented" for r in records))
        self.assertTrue(all("bgr" not in r for r in records))

    def test_stream_is_consumed_incrementally(self):
        records = []
        def frames():
            for index, frame in enumerate(synthetic_frames(RuntimeConfig(frames=4))):
                self.assertEqual(len(records), index)
                yield frame
        run_pipeline(frames(), records.append)

    def test_out_of_order_or_duplicate_frames_fail(self):
        first = next(synthetic_frames(RuntimeConfig(frames=1)))
        for second in (first, replace(first, index=1), replace(first, timestamp_seconds=1.0)):
            with self.subTest(second=second), self.assertRaises(ValueError):
                run_pipeline([first, second], lambda record: None)

    def test_realtime_pacing_uses_source_time_without_sleeping_in_test(self):
        now = [10.0]
        sleeps = []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        frames = synthetic_frames(RuntimeConfig(frames=3, fps=2))
        result = run_pipeline(frames, lambda record: None, realtime=True,
                              clock=lambda: now[0], sleep=sleep)
        self.assertEqual(sleeps, [0.5, 0.5])
        self.assertEqual(result.source_duration_seconds, 1.0)
        self.assertEqual(result.elapsed_seconds, 1.0)

    def test_empty_stream_has_finite_zero_summary(self):
        result = run_pipeline([], lambda record: None, clock=lambda: 1.0)
        self.assertEqual(result.frames_processed, 0)
        self.assertEqual(result.throughput_fps, 0.0)
        self.assertEqual(result.source_duration_seconds, 0.0)

    def test_predictor_injection_preserves_contract(self):
        class TestPredictor:
            def predict(self, frame):
                return Prediction(FallState.UNKNOWN, reason="test_only")
        records = []
        run_pipeline(synthetic_frames(RuntimeConfig(frames=1)), records.append,
                     predictor=TestPredictor())
        self.assertEqual(records[0]["reason"], "test_only")


class CliTests(unittest.TestCase):
    def test_demo_stdout_is_valid_jsonl_with_summary(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["demo", "--frames", "3", "--width", "8", "--height", "4"])
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual([r["frame_index"] for r in records[:-1]], [0, 1, 2])
        self.assertEqual(records[-1]["frames_processed"], 3)

    def test_output_file_is_written_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results" / "run.jsonl"
            args = ["demo", "--frames", "2", "--output", str(target)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            original = target.read_bytes()
            self.assertEqual(len(original.splitlines()), 3)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertEqual(target.read_bytes(), original)

    def test_invalid_config_returns_error_without_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad.jsonl"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["demo", "--fps", "nan", "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertFalse(target.exists())

    def test_status_states_detection_is_unavailable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["status"]), 0)
        self.assertIs(json.loads(output.getvalue())["fall_detection_available"], False)


if __name__ == "__main__":
    unittest.main()
