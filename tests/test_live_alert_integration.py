"""Synthetic end-to-end research alert propagation through the live pipeline.

These tests do not validate fall-detection accuracy or emergency delivery.
"""
import unittest

from fall_detection.alerts import AlertConfig, AlertManager
from fall_detection.contracts import FallState, FramePacket
from fall_detection.live import run_live_pipeline
from fall_detection.pose import Keypoint, PoseDetection
from fall_detection.state import StateDecision, StateEvidence


def frames(count):
    for index in range(count):
        yield FramePacket(
            index=index, timestamp_seconds=index * 0.25,
            width=120, height=80, bgr=bytes(120 * 80 * 3),
            source="synthetic_alert_fixture",
        )


class FakeEstimator:
    def estimate(self, frame):
        return (PoseDetection(
            (20.0, 5.0, 35.0, 65.0), 0.95,
            tuple(Keypoint(27.0, 20.0 + i, 0.95) for i in range(17)),
        ),)


class ScriptedClassifier:
    def __init__(self, states):
        self.states = states

    def classify_many(self, windows):
        return tuple(
            StateDecision(
                window.track_id, window.latest.frame_index,
                self.states[window.latest.frame_index],
                "synthetic_unvalidated_state",
                StateEvidence(3, 1.0, 1.5, 0.0),
            )
            for window in windows
        )


class LiveResearchAlertTests(unittest.TestCase):
    def test_persistence_and_latched_recovery_are_visible_in_frame_metadata(self):
        manager = AlertManager(AlertConfig(fallen_persistence_seconds=0.5))
        events = []
        states = (FallState.FALLEN, FallState.FALLEN, FallState.FALLEN,
                  FallState.NORMAL, FallState.NORMAL)
        summary = run_live_pipeline(
            frames(len(states)), events.append, estimator=FakeEstimator(),
            temporal_classifier=ScriptedClassifier(states), alert_manager=manager,
        )
        self.assertEqual(summary.frames_processed, len(states))
        self.assertEqual([e["alerts"][0]["status"] for e in events],
                         ["pending", "pending", "active", "active", "active"])
        self.assertEqual(events[3]["alerts"][0]["observed_state"], "normal")
        self.assertEqual(events[2]["alerts"][0]["alert_id"],
                         events[4]["alerts"][0]["alert_id"])
        self.assertTrue(all(e["alerts_research_only"] for e in events))
        self.assertTrue(all(not e["fall_detection_available"] for e in events))
        self.assertNotIn("bgr", str(events))
        self.assertEqual([e.transition for e in manager.events],
                         ["pending_started", "alert_opened"])

    def test_default_live_path_does_not_invent_alerts(self):
        events = []
        run_live_pipeline(
            frames(2), events.append, estimator=FakeEstimator(),
            temporal_classifier=ScriptedClassifier((FallState.FALLEN,) * 2),
        )
        self.assertTrue(all("alerts" not in e for e in events))

    def test_rejects_invalid_alert_manager_before_processing(self):
        events = []
        with self.assertRaisesRegex(ValueError, "alert_manager"):
            run_live_pipeline(frames(1), events.append, estimator=FakeEstimator(),
                              alert_manager=object())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
