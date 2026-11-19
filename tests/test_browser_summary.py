"""Browser monitor exposes capture summary throughput without claiming inference accuracy."""
import unittest
from fall_detection.dashboard import HTML, _sanitize_record


class BrowserSummaryTests(unittest.TestCase):
    def test_capture_summary_is_sanitized_for_browser(self):
        record = _sanitize_record({"event": "summary", "frames_processed": 120,
            "throughput_fps": 24.25, "raw_video": "SECRET_FRAME"})
        self.assertEqual(record["throughput_fps"], 24.25)
        self.assertNotIn("raw_video", record)

    def test_browser_shows_completed_capture_and_latest_frame_processing_latency(self):
        self.assertIn("status==='completed'?'Finished'", HTML)
        self.assertIn("status==='completed'?'Capture finished'", HTML)
        self.assertIn('id="processing"', HTML)
        self.assertIn("Number.isFinite(event.processing_ms)", HTML)
        self.assertIn("event.processing_ms>=0", HTML)
        self.assertIn("$('processing').textContent=event.processing_ms.toFixed(1)+' ms'", HTML)
        self.assertIn("$('processing').textContent='—'", HTML)
        self.assertEqual(_sanitize_record({"event": "frame", "processing_ms": 13.5})["processing_ms"], 13.5)
        self.assertNotIn("processing_ms", _sanitize_record({"event": "frame", "processing_ms": float("inf")}))

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
