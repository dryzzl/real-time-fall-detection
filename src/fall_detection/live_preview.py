"""Local-only visualization for the unvalidated pose/tracking baseline."""
from __future__ import annotations

from .capture import CaptureError
from .contracts import FallState, FramePacket
from .preview import Preview

# COCO pose connections. Only confident, observed points are rendered.
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16), (0, 5), (0, 6))
COLORS = {FallState.UNKNOWN: (180, 180, 180),
          FallState.NORMAL: (80, 200, 80),
          FallState.FALLING: (0, 185, 255),
          FallState.FALLEN: (0, 145, 255)}


def clamp(value, bound):
    return max(0, min(bound - 1, round(value)))


def render_overlay(cv, pixels, frame, windows, decisions):
    """Draw observed poses without serializing raw pixels or invented detections."""
    if tuple(w.track_id for w in windows) != tuple(d.track_id for d in decisions):
        raise ValueError("preview track IDs must match state decision IDs")
    cv.putText(pixels, "UNVALIDATED HEURISTIC POSE DEMO - NOT A FALL ALERT",
               (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
    missing = sum(not w.latest.observed for w in windows)
    cv.putText(pixels, f"Frame {frame.index} | Tracks {len(windows)} | Missing {missing}",
               (8, 42), cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    for window, decision in zip(windows, decisions):
        sample = window.latest
        color = COLORS[decision.state]
        if not sample.observed:
            cv.putText(pixels, f"ID {window.track_id}: missing (unknown)",
                       (8, min(frame.height - 1, 62 + 16 * (window.track_id % 8))),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[FallState.UNKNOWN], 1)
            continue
        if sample.box_center_xy is None or sample.box_size_wh is None:
            raise ValueError("observed sample missing bounding box")
        cx, cy = sample.box_center_xy[0] * frame.width, sample.box_center_xy[1] * frame.height
        bw, bh = sample.box_size_wh[0] * frame.width, sample.box_size_wh[1] * frame.height
        left, top = clamp(cx - bw / 2, frame.width), clamp(cy - bh / 2, frame.height)
        right, bottom = clamp(cx + bw / 2, frame.width), clamp(cy + bh / 2, frame.height)
        cv.rectangle(pixels, (left, top), (right, bottom), color, 2)
        cv.putText(pixels, f"ID {window.track_id} {decision.state.value} (heuristic)",
                   (left, max(58, top - 7)), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        points = {}
        for i, (coords, valid) in enumerate(zip(sample.keypoints_xy, sample.keypoint_mask)):
            if valid and coords[0] is not None and coords[1] is not None:
                points[i] = (clamp(cx + coords[0] * bw, frame.width),
                             clamp(cy + coords[1] * bh, frame.height))
        for a, b in EDGES:
            if a in points and b in points:
                cv.line(pixels, points[a], points[b], color, 2)
        for point in points.values():
            cv.circle(pixels, point, 3, color, -1)


class LivePreview(Preview):
    title = "Fall Detection | Unvalidated Pose Demo | Q/Esc quits"

    def show_live(self, frame, windows, decisions):
        pixels = self.np.frombuffer(frame.bgr, dtype=self.np.uint8).reshape(
            frame.height, frame.width, 3).copy()
        try:
            render_overlay(self.cv, pixels, frame, windows, decisions)
            self.cv.imshow(self.title, pixels)
            key = self.cv.waitKey(1) & 0xFF
        except self.cv.error as error:
            raise CaptureError(f"live preview failed: {error}") from error
        if key in (ord("q"), ord("Q"), 27):
            return False
        try:
            return self.cv.getWindowProperty(self.title, self.cv.WND_PROP_VISIBLE) >= 1
        except self.cv.error:
            return False
