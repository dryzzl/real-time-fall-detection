"""Contract tests for the optional ONNX temporal inference adapter.

All sessions and features here are synthetic; no trained weights are tested.
"""
import unittest

import numpy as np

from fall_detection.contracts import FallState
from fall_detection.features import PoseFeatureSample, TemporalFeatureWindow
from fall_detection.temporal_inference import OnnxTemporalClassifier
from fall_detection.training import TrainingConfig, TrainingError


class FakeSession:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def run(self, outputs, inputs):
        self.calls.append((outputs, inputs))
        return [self.logits]


def window(track_id=1, observed=(True, True, True), ready=True):
    samples = []
    for index, seen in enumerate(observed):
        samples.append(PoseFeatureSample(
            track_id=track_id, frame_index=index, timestamp_seconds=index * 0.1,
            observed=seen,
            detection_confidence=0.9 if seen else None,
            box_center_xy=(0.5, 0.4) if seen else None,
            box_size_wh=(0.2, 0.4) if seen else None,
            box_aspect_ratio=0.5 if seen else None,
            center_velocity_xy_per_second=(0.0, 0.0) if seen else None,
            keypoints_xy=((0.4, 0.5), (0.6, 0.5)) if seen else ((None, None),) * 2,
            keypoint_mask=(True, True) if seen else (False, False),
        ))
    return TemporalFeatureWindow(track_id, tuple(samples), sum(observed), ready)


class TemporalOnnxAdapterTests(unittest.TestCase):
    def make_classifier(self, logits, **kwargs):
        session = FakeSession(logits)
        model = OnnxTemporalClassifier(
            __import__("pathlib").Path("synthetic-only.onnx"),
            TrainingConfig(sequence_length=3, keypoint_count=2),
            session=session, **kwargs,
        )
        return model, session

    def test_predicts_from_versioned_feature_contract(self):
        model, session = self.make_classifier([[0.0, 10.0, 0.0]])
        decision = model.classify(window())
        self.assertEqual(decision.state, FallState.FALLING)
        self.assertGreater(decision.confidence, 0.99)
        self.assertEqual(decision.reason, "onnx_temporal_unvalidated")
        self.assertEqual(session.calls[0][0], ["logits"])
        self.assertEqual(session.calls[0][1]["features"].shape, (1, 3, 16))
        self.assertEqual(session.calls[0][1]["features"].dtype, np.float32)

    def test_fail_closed_on_missing_current_observation(self):
        model, session = self.make_classifier([[0.0, 0.0, 1.0]])
        result = model.classify(window(observed=(True, True, False)))
        self.assertEqual(result.state, FallState.UNKNOWN)
        self.assertIsNone(result.confidence)
        self.assertEqual(session.calls, [])

    def test_fail_closed_on_insufficient_evidence(self):
        model, session = self.make_classifier([[0.0, 0.0, 1.0]])
        for item in (window(observed=(False, True, True)), window(ready=False),
                     window(observed=(True, True))):
            self.assertEqual(model.classify(item).state, FallState.UNKNOWN)
        self.assertEqual(session.calls, [])

    def test_rejects_invalid_model_output(self):
        for output in ([[float("nan"), 0.0, 0.0]], [[1.0, 2.0]], [[1, 2, 3], [4, 5, 6]]):
            with self.subTest(output=output):
                model, _ = self.make_classifier(output)
                with self.assertRaises(TrainingError):
                    model.classify(window())

    def test_large_logits_are_stable(self):
        model, _ = self.make_classifier([[10000.0, 10001.0, 9999.0]])
        decision = model.classify(window())
        self.assertEqual(decision.state, FallState.FALLING)
        self.assertTrue(0 <= decision.confidence <= 1)

    def test_independent_tracks(self):
        model, session = self.make_classifier([[0.0, 0.0, 4.0]])
        decisions = model.classify_many((window(1), window(2)))
        self.assertEqual([d.track_id for d in decisions], [1, 2])
        self.assertEqual([d.state for d in decisions], [FallState.FALLEN] * 2)
        self.assertEqual(len(session.calls), 2)

    def test_rejects_bad_threshold(self):
        for threshold in (0, 1.1, float("nan"), True):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    self.make_classifier([[0, 0, 0]], min_observed_fraction=threshold)


if __name__ == "__main__":
    unittest.main()
