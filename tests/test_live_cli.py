"""CLI failures must not claim fall detection or create misleading output."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from fall_detection.live_cli import main


class LiveCliTests(unittest.TestCase):
    def test_missing_model_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "missing-pose.onnx"
            output = root / "live.jsonl"
            with redirect_stderr(io.StringIO()) as errors:
                result = main(["--camera", "0", "--model", str(model), "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertIn("pose model not found", errors.getvalue())
            self.assertFalse(output.exists())

    def test_rejects_negative_camera_before_capture(self):
        with redirect_stderr(io.StringIO()) as errors:
            result = main(["--camera", "-1"])
        self.assertEqual(result, 2)
        self.assertIn("camera index", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
