"""Deterministic integration checks; synthetic pose fixtures are NOT accuracy data."""
import unittest

from fall_detection.contracts import FramePacket
from fall_detection.features import FeatureConfig
from fall_detection.live import run_live_pipeline
from fall_detection.pose import Keypoint, PoseDetection
from fall_detection.state import StateConfig


def frame(index):
    return FramePacket(
        index=index, timestamp_seconds=index * 0.2,
        width=80, height=80, bgr=b"\\0" * (80 * 80 * 3),
        source="synthetic_fixture",
    )


def detection(x=20.0):
    return PoseDetection(
        (x, 5.0, x + 15.0, 65.0), 0.95,
        tuple(Keypoint(x + 7.0, 20.0 + i, 0.95) for i in range(17)),
    )


class FakeEstimator:
    def __init__(self, missing=(), two_people=False):
        self.missing = set(missing)
        self.two_people = two_people

    def estimate(self, packet):
        if packet.index in self.missing:
            return ()
        return (detection(20), detection(55)) if self.two_people else (detection(20),)


class LiveIntegrationTests(unittest.TestCase):
    def run_fixture(self, count, estimator):
        events = []
        summary = run_live_pipeline(
            (frame(i) for i in range(count)), events.append,
            estimator=estimator,
            feature_config=FeatureConfig(window_size=5, min_observed_samples=3),
            state_config=StateConfig(min_contiguous_observations=3),
        )
        return events, summary

    def test_pose_to_stable_track_to_normal_baseline(self):
        events, summary = self.run_fixture(6, FakeEstimator())
        self.assertEqual(summary.frames_processed, 6)
        self.assertEqual([e["tracked_person_count"] for e in events], [1] * 6)
        self.assertEqual({e["people"][0]["track_id"] for e in events}, {1})
        self.assertEqual(events[0]["people"][0]["state"], "unknown")
        self.assertEqual(events[-1]["people"][0]["state"], "normal")
        self.assertEqual(events[-1]["people"][0]["window_samples"], 5)
        self.assertTrue(all(e["state"] == "unknown" for e in events))
        self.assertTrue(all(e["confidence"] is None for e in events))
        self.assertTrue(all(e["fall_detection_available"] is False for e in events))
        self.assertTrue(all("bgr" not in e and "keypoints" not in str(e) for e in events))

    def test_two_people_and_short_occlusion(self):
        events, _ = self.run_fixture(6, FakeEstimator(missing=(3,), two_people=True))
        self.assertEqual(events[0]["tracked_person_count"], 2)
        self.assertEqual([p["track_id"] for p in events[2]["people"]], [1, 2])
        self.assertTrue(all(not p["observed"] for p in events[3]["people"]))
        self.assertTrue(all(p["state"] == "unknown" for p in events[3]["people"]))
        self.assertEqual([p["track_id"] for p in events[4]["people"]], [1, 2])

    def test_no_detections_does_not_invent_people(self):
        events, _ = self.run_fixture(4, FakeEstimator(missing=range(4)))
        self.assertTrue(all(e["tracked_person_count"] == 0 and e["people"] == [] for e in events))

    def test_rejects_non_monotonic_frames_before_inference(self):
        records = []
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            run_live_pipeline([frame(0), frame(0)], records.append, estimator=FakeEstimator())
        self.assertEqual(len(records), 1)

    def test_empty_input_and_early_stop(self):
        events = []
        summary = run_live_pipeline([], events.append, estimator=FakeEstimator())
        self.assertEqual(summary.frames_processed, 0)
        self.assertEqual(events, [])
        summary = run_live_pipeline(
            (frame(i) for i in range(10)), events.append,
            estimator=FakeEstimator(),
            on_frame=lambda packet, decisions: packet.index < 2,
        )
        self.assertEqual(summary.frames_processed, 3)
        self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
