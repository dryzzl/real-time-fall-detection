"""Local, read-only web dashboard for fall-detection JSONL metadata.

No model, webcam, or raw video is required. Intended for the existing
``fall-detection demo/capture --output ...`` records. This dashboard does not
turn unknown/synthetic predictions into verified fall-detection results.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MAX_EVENTS = 200
MAX_LINE_BYTES = 1_000_000
FRAME_FIELDS = (
    "event", "source", "frame_index", "timestamp_seconds", "timestamp_basis",
    "width", "height", "state", "confidence", "reason", "mode",
    "trained_temporal_model_available", "fall_detection_available", "processing_ms",
    "state_source", "tracked_person_count", "synthetic",
)
SUMMARY_FIELDS = (
    "event", "frames_processed", "source_duration_seconds", "elapsed_seconds",
    "throughput_fps", "mode", "fall_detection_available",
)
PERSON_FIELDS = (
    "track_id", "state", "confidence", "reason", "frame_index",
    "observed", "window_ready", "window_samples",
)
ALERT_FIELDS = ("track_id", "status", "observed_state", "reason", "alert_id")


def _safe_scalars(record: dict, fields: tuple[str, ...]) -> dict:
    """Whitelist scalar metadata, never returning image/video blobs."""
    return {
        key: record[key]
        for key in fields
        if key in record and (record[key] is None or type(record[key]) in (str, int, float, bool))
    }


def _sanitize_record(record: dict) -> dict | None:
    kind = record.get("event")
    if kind not in ("frame", "summary"):
        return None
    result = _safe_scalars(record, FRAME_FIELDS if kind == "frame" else SUMMARY_FIELDS)
    if kind == "frame":
        for name, fields in (("people", PERSON_FIELDS), ("alerts", ALERT_FIELDS)):
            if isinstance(record.get(name), list):
                result[name] = [
                    _safe_scalars(item, fields)
                    for item in record[name][:100]
                    if isinstance(item, dict)
                ]
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError):
        return None
    return result


class DashboardError(ValueError):
    """Invalid cursor, query, or JSONL reader input."""


def read_events(path: Path, cursor: str = "0", limit: int = 100) -> dict:
    """Incrementally read complete JSONL lines without loading the whole file.

    The cursor contains a local file identity and byte offset, allowing log
    replacement/truncation to reset the stream. Incomplete trailing lines are
    retried on the next poll rather than silently discarded.
    """
    if not isinstance(path, Path):
        raise DashboardError("path must be a Path")
    if type(limit) is not int or not 1 <= limit <= MAX_EVENTS:
        raise DashboardError(f"limit must be an integer from 1 to {MAX_EVENTS}")
    if not isinstance(cursor, str) or len(cursor) > 100:
        raise DashboardError("invalid cursor")
    if not path.exists():
        return {"events": [], "next_cursor": "0", "source_exists": False, "reset": False}
    if not path.is_file():
        raise DashboardError("log path must refer to a regular file")

    stat = path.stat()
    identity = f"{stat.st_dev}-{stat.st_ino}"
    reset = False
    if cursor == "0":
        offset = 0
    else:
        try:
            prefix, raw_offset = cursor.rsplit(":", 1)
            offset = int(raw_offset)
        except (ValueError, TypeError) as exc:
            raise DashboardError("invalid cursor") from exc
        if offset < 0:
            raise DashboardError("invalid cursor")
        if prefix != identity or offset > stat.st_size:
            offset, reset = 0, True

    events: list[dict] = []
    with path.open("rb") as stream:
        stream.seek(offset)
        # Limit physical lines as well as emitted events to bound per-poll work.
        for _ in range(limit):
            start = stream.tell()
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                # Oversized lines are never parsed or returned. If the line is
                # incomplete, retry it after the writer completes the line.
                while line and not line.endswith(b"\n"):
                    line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    stream.seek(start)
                    break
                continue
            if not line.endswith(b"\n"):
                stream.seek(start)
                break
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                sanitized = _sanitize_record(record)
                if sanitized is not None:
                    events.append(sanitized)
        next_offset = stream.tell()
    return {
        "events": events,
        "next_cursor": f"{identity}:{next_offset}",
        "source_exists": True,
        "reset": reset,
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fall Detection | Local Research Monitor</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#101522;color:#eef2fa}
*{box-sizing:border-box}body{margin:0;max-width:1200px;padding:32px 24px 70px;margin-inline:auto}
header{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap}
h1{font-size:clamp(1.5rem,3vw,2.4rem);letter-spacing:-.04em;margin:4px 0}
p{color:#9eabc3;line-height:1.55}small{color:#9eabc3}.badge{background:#193a33;color:#91f0c9;border:1px solid #286c58;border-radius:99px;padding:7px 14px}
.notice{border:1px solid #4b3d20;background:#292418;border-radius:14px;padding:14px 18px;color:#f5d98e;margin:24px 0}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}
.card{border:1px solid #303b50;border-radius:16px;background:#192236;padding:20px;min-width:0}
.card .value{font-size:2rem;font-weight:750;margin-top:12px;overflow-wrap:anywhere}
.card .label{font-size:.85rem;color:#a7b6ce}
section{border:1px solid #303b50;border-radius:16px;background:#192236;overflow:hidden}
section h2{font-size:1.15rem;margin:20px}section+section{margin-top:18px}.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px 20px;border-top:1px solid #303b50;font-size:.9rem;white-space:nowrap}
th{color:#9eabc3;font-weight:500}td{font-variant-numeric:tabular-nums}
.state{border-radius:6px;padding:4px 8px;background:#29344c}.fallen{color:#ffb2b2;background:#4b2731}.falling{color:#ffdf99;background:#493f27}.normal{color:#9de4c0;background:#244439}
#connection{font-weight:650}footer{margin-top:20px}
@media(max-width:720px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}body{padding:20px 12px}}
</style>
</head>
<body>
<header><div><small>LOCAL / RESEARCH PROTOTYPE</small><h1>Fall Detection Monitor</h1><p>Read-only, live metadata from the existing JSONL capture pipeline.</p></div><div class="badge" id="connection" aria-live="polite">Connecting…</div></header>
<div class="notice">Experimental software — a state of <b>unknown</b> means fall detection has not been established. Synthetic or heuristic outputs are not medical or emergency alerts.</div>
<div class="grid">
<div class="card"><div class="label">Frames observed</div><div class="value" id="frames">0</div></div>
<div class="card"><div class="label">Latest reported state</div><div class="value" id="state">—</div></div>
<div class="card"><div class="label">Tracked people (latest frame)</div><div class="value" id="people">—</div></div>
<div class="card"><div class="label">Visible alerts (latest frame)</div><div class="value" id="alerts">—</div></div>
</div>
<section><h2>Tracked people — latest frame (unvalidated)</h2><div class="table-wrap"><table><thead><tr><th>Track</th><th>Heuristic state</th><th>Observed</th><th>Window ready</th><th>Reason</th></tr></thead><tbody id="person-rows"></tbody></table></div></section>
<section><h2>Recent events</h2><div class="table-wrap"><table><thead><tr><th>Frame</th><th>Source time</th><th>State</th><th>Reason</th><th>People</th><th>Processing</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<footer><small>Runs on 127.0.0.1 only. Video frames and model weights are not served. Refreshing resets this browser's counters; no data is deleted.</small></footer>
<script>
'use strict';
let cursor='0',frames=0,busy=false;
const $=id=>document.getElementById(id);
function cell(row,value){const td=document.createElement('td');td.textContent=String(value);row.appendChild(td);return td;}
function addFrame(event){
  frames+=1;$('frames').textContent=frames.toLocaleString();
  const state=typeof event.state==='string'?event.state:'unknown';$('state').textContent=state;
  $('people').textContent=Number.isInteger(event.tracked_person_count)?event.tracked_person_count:(Array.isArray(event.people)?event.people.length:'—');
  const personRows=$('person-rows');personRows.replaceChildren();
  for(const person of (Array.isArray(event.people)?event.people:[])){
    const pr=document.createElement('tr');
    cell(pr,person.track_id??'—');
    cell(pr,typeof person.state==='string'?person.state+' (heuristic)':'unknown');
    cell(pr,person.observed===true?'yes':person.observed===false?'no':'—');
    cell(pr,person.window_ready===true?'yes':person.window_ready===false?'no':'—');
    cell(pr,person.reason??'—');personRows.appendChild(pr);
  }
  $('alerts').textContent=Array.isArray(event.alerts)?event.alerts.length:'—';
  const row=document.createElement('tr');
  cell(row,event.frame_index??'—');
  cell(row,typeof event.timestamp_seconds==='number'?event.timestamp_seconds.toFixed(2)+'s':'—');
  const badge=cell(row,state);badge.className='state '+(state==='fallen'||state==='falling'||state==='normal'?state:'');
  cell(row,event.reason??'—');
  cell(row,Array.isArray(event.people)?event.people.length:'—');
  cell(row,typeof event.processing_ms==='number'?event.processing_ms.toFixed(1)+' ms':'—');
  $('rows').prepend(row);while($('rows').children.length>30){$('rows').lastChild.remove();}
}
async function poll(){
  if(busy)return;busy=true;
  try{
    const res=await fetch('/api/events?cursor='+encodeURIComponent(cursor)+'&limit=200',{cache:'no-store'});
    if(!res.ok)throw new Error('HTTP '+res.status);
    const payload=await res.json();
    if(payload.reset){frames=0;$('frames').textContent='0';$('state').textContent='—';$('people').textContent='—';$('alerts').textContent='—';$('rows').replaceChildren();$('person-rows').replaceChildren();}
    cursor=payload.next_cursor;
    for(const event of payload.events){if(event.event==='frame')addFrame(event);}
    $('connection').textContent=payload.source_exists?'Reading local log':'Waiting for log';
  }catch(err){$('connection').textContent='Connection unavailable';}
  finally{busy=false;}
}
poll();setInterval(poll,1000);
</script>
</body></html>'''


def create_server(log_path: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Create a loopback-only HTTP server; caller controls its lifetime."""
    if not isinstance(log_path, Path):
        raise DashboardError("log_path must be a Path")
    if type(port) is not int or not 0 <= port <= 65535:
        raise DashboardError("port must be an integer from 0 to 65535")

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status: int, body: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, payload: dict) -> None:
            self.send_body(status, json.dumps(payload, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            # Reject DNS-rebinding Host headers; only local browser origins allowed.
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                self.send_json(403, {"error": "local host required"})
                return
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self.send_body(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self.send_json(200, {"ok": True, "log_exists": log_path.is_file(), "mode": "read_only_metadata"})
                return
            if parsed.path == "/api/events":
                try:
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    if set(params) - {"cursor", "limit"} or any(len(v) != 1 for v in params.values()):
                        raise DashboardError("unknown or repeated query parameter")
                    cursor = params.get("cursor", ["0"])[0]
                    limit = int(params.get("limit", ["100"])[0])
                    self.send_json(200, read_events(log_path, cursor, limit))
                except (DashboardError, ValueError, OSError) as exc:
                    self.send_json(400, {"error": str(exc)})
                return
            self.send_json(404, {"error": "not found"})

        def log_message(self, fmt: str, *args: object) -> None:
            return  # Avoid logging arbitrary query strings and local file metadata.

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local read-only fall-detection metadata dashboard")
    parser.add_argument("--log", type=Path, default=Path("outputs/demo.jsonl"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        with create_server(args.log, args.port) as server:
            print(f"Dashboard: http://127.0.0.1:{server.server_port} (reading {args.log})", flush=True)
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    except (DashboardError, OSError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
