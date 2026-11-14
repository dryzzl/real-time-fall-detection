"""Browser monitor exposes capture summary throughput without claiming inference accuracy."""
import unittest
from fall_detection.dashboard import HTML, _sanitize_record


class BrowserSummaryTests(unittest.TestCase):
    def test_capture_summary_is_sanitized_for_browser(self):
        record = _sanitize_record({"event": "summary", "frames_processed": 120,
            "throughput_fps": 24.25, "raw_video": "SECRET_FRAME"})
        self.assertEqual(record["throughput_fps"], 24.25)
        self.assertNotIn("raw_video", record)

    def test_bootstrap_and_poll_render_summary_and_reset_throughput(self):
        self.assertIn('id="throughput"', HTML)
        self.assertIn('id="processed"', HTML)
        self.assertIn('id="elapsed"', HTML)
        self.assertIn("function addSummary(event)", HTML)
        self.assertEqual(HTML.count("else if(event.event==='summary')addSummary(event)"), 2)
        self.assertIn("$('throughput').textContent='—'", HTML)
        self.assertIn("Capture summary received", HTML)
        self.assertIn("Number.isSafeInteger(processed)", HTML)
        self.assertIn("Number.isFinite(elapsed)", HTML)
        self.assertIn("processed').textContent='—'", HTML)
        self.assertIn("elapsed').textContent='—'", HTML)
