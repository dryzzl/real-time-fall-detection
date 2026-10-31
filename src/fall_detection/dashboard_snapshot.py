"""Bounded, tail-first snapshot reader for the local metadata dashboard.

Only complete JSONL records are returned. The caller supplies a metadata
sanitizer; no raw video frames or arbitrary log keys are exposed by this module.
The returned cursor points immediately after the last complete line so a
subsequent incremental read cannot miss a partially written record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_LINE_BYTES = 1_000_000
MAX_EVENTS = 200


def read_recent_events(
    path: Path,
    sanitize: Callable[[dict], dict | None],
    *,
    limit: int = 100,
    max_bytes: int = MAX_SNAPSHOT_BYTES,
) -> dict:
    """Read the newest physical JSONL lines with bounded memory and I/O.

    A clipped first line is discarded, malformed/oversized records are skipped,
    and an incomplete final line is retained for the next incremental poll.
    This is a recent-window view, not a whole-file aggregate.
    """
    if not isinstance(path, Path):
        raise ValueError("path must be a Path")
    if not callable(sanitize):
        raise ValueError("sanitize must be callable")
    if type(limit) is not int or not 1 <= limit <= MAX_EVENTS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_EVENTS}")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 8 * 1024 * 1024:
        raise ValueError("max_bytes must be between 1 and 8388608")

    missing = {
        "events": [], "next_cursor": "0", "source_exists": False,
        "window_truncated": False, "partial_trailing_line": False,
        "bytes_scanned": 0,
    }
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return missing

    with stream:
        stat = os.fstat(stream.fileno())
        if not os.path.isfile(path):
            raise ValueError("log path must refer to a regular file")
        size = stat.st_size
        start = max(0, size - max_bytes)
        preceding = b"\n"
        if start:
            stream.seek(start - 1)
            preceding = stream.read(1)
        stream.seek(start)
        data = stream.read(size - start)

    identity = f"{stat.st_dev}-{stat.st_ino}"
    last_newline = data.rfind(b"\n")
    complete = data[:last_newline + 1] if last_newline >= 0 else b""
    lines = complete.split(b"\n")[:-1]
    if start and preceding != b"\n" and lines:
        lines = lines[1:]  # The first physical line began before our window.

    events: list[dict] = []
    for line in lines[-limit:]:
        if not line or len(line) > MAX_LINE_BYTES:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            safe = sanitize(record)
            if safe is not None:
                events.append(safe)

    # If no complete newline was found, retain the starting offset for retry.
    next_offset = start + last_newline + 1 if last_newline >= 0 else start
    return {
        "events": events,
        "next_cursor": f"{identity}:{next_offset}",
        "source_exists": True,
        "window_truncated": start > 0,
        "partial_trailing_line": bool(data) and not data.endswith(b"\n"),
        "bytes_scanned": len(data) + int(start > 0),
    }
