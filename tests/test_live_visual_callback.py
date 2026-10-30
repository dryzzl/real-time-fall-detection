"""Synthetic callback integration test; no detector accuracy is claimed."""
import unittest
from fall_detection.contracts import FramePacket
from fall_detection.live import run_live_pipeline
from fall_detection.pose import Keypoint, PoseDetection


class Estimator:
    def estimate(self, frame):
        return (PoseDetection((5.0, 5.0, 30.0, 65.0), 0.9,
                              tuple(Keypoint(12.0, 20.0, 0.9) for _ in range(17))),)


class LiveCallbackTests(unittest.TestCase):
    def test_visual_callback_can_stop_and_receives_same_track(self):
        records, callbacks = [], []
        frames = (FramePacket(i, i * 0.2, 80, 80, bytes(19200)) for i in range(5))
        def show(frame, windows, decisions):
            callbacks.append((frame.index, windows[0].track_id, decisions[0].track_id))
            return False
        summary = run_live_pipeline(frames, records.append, estimator=Estimator(), on_visual_frame=show)
        self.assertEqual(summary.frames_processed, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(callbacks, [(0, 1, 1)])
        self.assertFalse(records[0]["fall_detection_available"])


if __name__ == "__main__":
    unittest.main()
