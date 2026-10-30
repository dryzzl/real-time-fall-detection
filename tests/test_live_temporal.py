"""Synthetic end-to-end pose-to-temporal adapter integration checks."""
import unittest
from pathlib import Path

from fall_detection.contracts import FramePacket
from fall_detection.features import FeatureConfig
from fall_detection.live import run_live_pipeline
from fall_detection.pose import Keypoint, PoseDetection
from fall_detection.temporal_inference import OnnxTemporalClassifier
from fall_detection.training import TrainingConfig


class FakePoseEstimator:
    def estimate(self, frame):
        return (PoseDetection(
            (20.0, 5.0, 35.0, 65.0), 0.95,
            tuple(Keypoint(27.0, 20.0 + i, 0.95) for i in range(17)),
        ),)


class FakeTemporalSession:
    def __init__(self):
        self.calls = 0

    def run(self, output_names, inputs):
        self.calls += 1
        assert inputs["features"].shape == (1, 5, 61)
        return [[[0.0, 0.0, 5.0]]]


class LiveTemporalIntegrationTests(unittest.TestCase):
    def test_model_only_runs_when_window_ready_and_preserves_global_unknown(self):
        session = FakeTemporalSession()
        model = OnnxTemporalClassifier(
            Path("synthetic-not-a-real-model.onnx"),
            TrainingConfig(sequence_length=5, keypoint_count=17),
            session=session,
        )
        frames = [
            FramePacket(i, i * 0.2, 80, 80, bytes(80 * 80 * 3), "synthetic")
            for i in range(6)
        ]
        events = []
        summary = run_live_pipeline(
            frames, events.append, estimator=FakePoseEstimator(),
            feature_config=FeatureConfig(window_size=5, min_observed_samples=3),
            temporal_classifier=model,
        )
        self.assertEqual(summary.frames_processed, 6)
        self.assertEqual(session.calls, 2)
        self.assertEqual(events[0]["people"][0]["state"], "unknown")
        self.assertEqual(events[-1]["people"][0]["state"], "fallen")
        self.assertGreater(events[-1]["people"][0]["confidence"], 0.98)
        self.assertEqual(events[-1]["state_source"], "onnx_temporal_unvalidated")
        self.assertTrue(all(e["state"] == "unknown" for e in events))
        self.assertTrue(all(not e["fall_detection_available"] for e in events))


if __name__ == "__main__":
    unittest.main()
