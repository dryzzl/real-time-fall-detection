"""Temporal robustness regressions against the production alert lifecycle.

These synthetic state sequences test alert behavior, not detector accuracy.
Run with: python -m unittest discover -s tests -v
"""

import unittest

from fall_detection.alerts import AlertConfig, AlertError, AlertManager, AlertStatus
from fall_detection.contracts import FallState
from fall_detection.state import StateDecision, StateEvidence


def decision(track_id: int, frame: int, state: FallState) -> StateDecision:
    return StateDecision(
        track_id=track_id,
        frame_index=frame,
        state=state,
        reason=f"synthetic_{state.value}",
        evidence=StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


class TemporalAlertRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AlertManager(AlertConfig(
            fallen_persistence_seconds=1.0,
            cooldown_seconds=2.0,
            fsync_events=False,
        ))

    def update(self, frame: int, state: FallState, time: float,
               track_id: int = 1) -> AlertStatus:
        return self.manager.update(decision(track_id, frame, state), time).status

    def test_single_fallen_observation_cannot_open_alert(self):
        self.assertEqual(self.update(0, FallState.NORMAL, 0.0), AlertStatus.IDLE)
        self.assertEqual(self.update(1, FallState.FALLEN, 0.1), AlertStatus.PENDING)
        self.assertEqual(self.update(2, FallState.NORMAL, 0.2), AlertStatus.IDLE)
        self.assertEqual(
            [event.transition for event in self.manager.events],
            ["pending_started", "pending_cancelled"],
        )

    def test_falling_state_alone_never_opens_alert(self):
        for frame in range(10):
            self.assertEqual(
                self.update(frame, FallState.FALLING, frame * 0.2),
                AlertStatus.IDLE,
            )
        self.assertEqual(self.manager.events, ())

    def test_unknown_cancels_pending_but_not_active(self):
        self.assertEqual(self.update(0, FallState.FALLEN, 0.0), AlertStatus.PENDING)
        self.assertEqual(self.update(1, FallState.UNKNOWN, 0.5), AlertStatus.IDLE)
        self.assertEqual(self.update(2, FallState.FALLEN, 1.0), AlertStatus.PENDING)
        self.assertEqual(self.update(3, FallState.FALLEN, 2.0), AlertStatus.ACTIVE)
        for frame, time in ((4, 2.2), (5, 2.4), (6, 3.0)):
            self.assertEqual(self.update(frame, FallState.UNKNOWN, time), AlertStatus.ACTIVE)
        self.assertEqual(
            [event.transition for event in self.manager.events].count("alert_opened"), 1
        )

    def test_active_alert_requires_acknowledgment_and_guarded_reset(self):
        self.update(0, FallState.FALLEN, 0.0)
        self.assertEqual(self.update(1, FallState.FALLEN, 1.0), AlertStatus.ACTIVE)
        self.assertEqual(self.update(2, FallState.NORMAL, 1.5), AlertStatus.ACTIVE)
        with self.assertRaises(AlertError):
            self.manager.reset(1, 1.6)
        self.assertEqual(self.manager.acknowledge(1, 1.7).status, AlertStatus.ACKNOWLEDGED)
        self.assertEqual(self.manager.reset(1, 1.8).status, AlertStatus.COOLDOWN)
        self.assertEqual(self.update(3, FallState.FALLEN, 2.0), AlertStatus.COOLDOWN)
        self.assertEqual(self.update(4, FallState.NORMAL, 3.9), AlertStatus.IDLE)

    def test_independent_tracks_do_not_share_alert_state(self):
        self.assertEqual(self.update(0, FallState.FALLEN, 0.0, 1), AlertStatus.PENDING)
        self.assertEqual(self.update(0, FallState.NORMAL, 0.0, 2), AlertStatus.IDLE)
        self.assertEqual(self.update(1, FallState.FALLEN, 1.0, 1), AlertStatus.ACTIVE)
        self.assertEqual(self.update(1, FallState.FALLING, 1.0, 2), AlertStatus.IDLE)
        self.assertEqual(self.manager.snapshot(1).status, AlertStatus.ACTIVE)
        self.assertEqual(self.manager.snapshot(2).status, AlertStatus.IDLE)

    def test_rejected_out_of_order_frames_leave_state_unchanged(self):
        self.update(0, FallState.FALLEN, 0.0)
        before = self.manager.snapshot(1)
        events = self.manager.events
        for frame, time in ((0, 0.5), (1, 0.0), (1, float("nan"))):
            with self.subTest(frame=frame, time=time), self.assertRaises(AlertError):
                self.update(frame, FallState.FALLEN, time)
            self.assertEqual(self.manager.snapshot(1), before)
            self.assertEqual(self.manager.events, events)


if __name__ == "__main__":
    unittest.main()
