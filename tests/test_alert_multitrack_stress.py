"""Sustained synthetic multi-track alert lifecycle regression tests.

These tests exercise state-machine invariants, not fall classification accuracy.
"""
import unittest

from fall_detection.alerts import AlertConfig, AlertManager, AlertStatus
from fall_detection.contracts import FallState
from fall_detection.state import StateDecision, StateEvidence


def decision(track, frame, state):
    return StateDecision(
        track, frame, state, "synthetic_alert_stress",
        StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


class MultiTrackAlertStressTests(unittest.TestCase):
    def test_sixteen_alerts_remain_latched_through_prolonged_track_loss(self):
        manager = AlertManager(AlertConfig(
            fallen_persistence_seconds=1.0,
            cooldown_seconds=2.0,
            max_tracks=16,
            fsync_events=False,
        ))
        for frame in range(80):
            timestamp = float(frame)
            for track in range(1, 17):
                state = (FallState.FALLEN if frame < 2 else
                         FallState.UNKNOWN if frame < 60 else FallState.NORMAL)
                snapshot = manager.update(decision(track, frame, state), timestamp)
                self.assertEqual(
                    snapshot.status,
                    AlertStatus.PENDING if frame == 0 else AlertStatus.ACTIVE,
                    (frame, track),
                )
                if frame >= 1:
                    self.assertTrue(snapshot.overlay.visible)
        snapshots = manager.snapshots()
        self.assertEqual(len(snapshots), 16)
        self.assertEqual(len({item.alert_id for item in snapshots}), 16)
        self.assertEqual(
            [event.transition for event in manager.events].count("alert_opened"), 16
        )
        self.assertEqual(len(manager.events), 32)
        self.assertEqual(
            [event.event_id for event in manager.events], list(range(1, 33))
        )

    def test_short_fall_bursts_and_missing_observations_never_open_alert(self):
        manager = AlertManager(AlertConfig(
            fallen_persistence_seconds=2.0, max_tracks=16, fsync_events=False,
        ))
        for frame in range(120):
            for track in range(1, 17):
                # Alternating FALLEN/UNKNOWN must cancel pending on every track.
                state = FallState.FALLEN if frame % 2 == 0 else FallState.UNKNOWN
                snapshot = manager.update(decision(track, frame, state), float(frame))
                self.assertNotEqual(snapshot.status, AlertStatus.ACTIVE)
        self.assertEqual(
            [event.transition for event in manager.events].count("alert_opened"), 0
        )
        self.assertEqual(
            [event.transition for event in manager.events].count("pending_started"),
            16 * 60,
        )
        self.assertEqual(
            [event.transition for event in manager.events].count("pending_cancelled"),
            16 * 60,
        )


if __name__ == "__main__":
    unittest.main()
