"""Synthetic multi-track acknowledgment/reset/cooldown invariants.

This is state-machine testing only, not real-world fall-detection validation.
"""
import unittest

from fall_detection.alerts import AlertConfig, AlertError, AlertManager, AlertStatus
from fall_detection.contracts import FallState
from fall_detection.state import StateDecision, StateEvidence


def decision(track, frame, state):
    return StateDecision(
        track, frame, state, "synthetic_recovery_stress",
        StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


class MultiTrackRecoveryStressTests(unittest.TestCase):
    def test_independent_acknowledgment_reset_and_rearm_across_sixteen_tracks(self):
        manager = AlertManager(AlertConfig(
            fallen_persistence_seconds=1.0, cooldown_seconds=3.0,
            max_tracks=16, fsync_events=False,
        ))
        for track in range(1, 17):
            manager.update(decision(track, 0, FallState.FALLEN), 0.0)
            self.assertEqual(
                manager.update(decision(track, 1, FallState.FALLEN), 1.0).status,
                AlertStatus.ACTIVE,
            )
        original_ids = {item.track_id: item.alert_id for item in manager.snapshots()}
        for track in range(1, 17, 2):
            self.assertEqual(manager.acknowledge(track, 1.1).status, AlertStatus.ACKNOWLEDGED)
            # An unknown observation cannot authorize reset.
            manager.update(decision(track, 2, FallState.UNKNOWN), 1.2)
            with self.assertRaises(AlertError):
                manager.reset(track, 1.3)
            manager.update(decision(track, 3, FallState.NORMAL), 1.4)
            self.assertEqual(manager.reset(track, 1.5).status, AlertStatus.COOLDOWN)
            self.assertEqual(
                manager.update(decision(track, 4, FallState.FALLEN), 4.6).status,
                AlertStatus.COOLDOWN,
            )
            self.assertEqual(
                manager.update(decision(track, 5, FallState.NORMAL), 4.7).status,
                AlertStatus.IDLE,
            )
        for track in range(2, 17, 2):
            snapshot = manager.snapshot(track)
            self.assertEqual(snapshot.status, AlertStatus.ACTIVE)
            self.assertEqual(snapshot.alert_id, original_ids[track])
        self.assertEqual(
            [event.transition for event in manager.events].count("alert_opened"), 16
        )
        self.assertEqual(
            [event.transition for event in manager.events].count("alert_rearmed"), 8
        )
        for track in range(1, 17, 2):
            self.assertEqual(manager.update(
                decision(track, 6, FallState.FALLEN), 4.8
            ).status, AlertStatus.PENDING)
            opened = manager.update(decision(track, 7, FallState.FALLEN), 5.8)
            self.assertEqual(opened.status, AlertStatus.ACTIVE)
            self.assertNotEqual(opened.alert_id, original_ids[track])
        self.assertEqual(len({item.alert_id for item in manager.snapshots()}), 16)
        self.assertEqual(
            [event.event_id for event in manager.events],
            list(range(1, len(manager.events) + 1)),
        )


if __name__ == "__main__":
    unittest.main()
