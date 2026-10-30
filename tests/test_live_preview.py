"""Synthetic overlay regression tests; not a fall-accuracy evaluation."""
import unittest
from unittest.mock import Mock
from fall_detection.contracts import FallState, FramePacket
from fall_detection.features import PoseFeatureSample, TemporalFeatureWindow
from fall_detection.live_preview import render_overlay
from fall_detection.state import StateDecision, StateEvidence


def fixture(observed=True, masked=False):
    mask = tuple(not masked and i in (5, 6) for i in range(17))
    sample = PoseFeatureSample(
        1, 0, 0.0, observed, 0.9 if observed else None,
        (0.5, 0.5) if observed else None,
        (0.25, 0.5) if observed else None,
        1.0 if observed else None, None,
        tuple((0.0, 0.0) if observed and valid else (None, None) for valid in mask), mask,
    )
    return TemporalFeatureWindow(1, (sample,), int(observed), observed)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.cv = Mock()
        self.cv.FONT_HERSHEY_SIMPLEX = 0
        self.frame = FramePacket(0, 0.0, 200, 100, bytes(60000), "synthetic_test")

    def draw(self, observed=True, masked=False, track_id=1):
        decision = StateDecision(track_id, 0, FallState.UNKNOWN, "synthetic",
                                 StateEvidence(1, 0.0, None, None))
        render_overlay(self.cv, object(), self.frame, (fixture(observed, masked),), (decision,))

    def test_observed_box_and_masked_skeleton(self):
        self.draw()
        self.cv.rectangle.assert_called_once()
        self.assertEqual(self.cv.rectangle.call_args.args[1:3], ((75, 25), (125, 75)))
        self.assertEqual(self.cv.circle.call_count, 2)
        self.assertEqual(self.cv.line.call_count, 1)
        self.assertIn("UNVALIDATED", self.cv.putText.call_args_list[0].args[1])

    def test_masked_points_not_drawn(self):
        self.draw(masked=True)
        self.cv.circle.assert_not_called()
        self.cv.line.assert_not_called()

    def test_missing_person_has_no_stale_box(self):
        self.draw(observed=False)
        self.cv.rectangle.assert_not_called()
        self.cv.circle.assert_not_called()

    def test_mismatched_track_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "track IDs"):
            self.draw(track_id=2)
        self.cv.putText.assert_not_called()


if __name__ == "__main__":
    unittest.main()
