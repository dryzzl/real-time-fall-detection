"""Exercise the real OpenCV video decoder without camera hardware."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fall_detection.source_probe import probe_source, SourceProbeError

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


@unittest.skipIf(cv2 is None, "OpenCV video extras not installed")
class VideoProbeIntegrationTests(unittest.TestCase):
    def test_generated_video_decodes_and_reports_end_of_stream(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            try:
                if not writer.isOpened():
                    self.skipTest("MJPG encoder unavailable")
                for index in range(4):
                    writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
            finally:
                writer.release()
            result = probe_source(path, frames=4)
            self.assertEqual((result.width, result.height, result.channels), (64, 48, 3))
            self.assertEqual(result.frames_decoded, 4)
            with self.assertRaisesRegex(SourceProbeError, "frame 5/5"):
                probe_source(path, frames=5)
