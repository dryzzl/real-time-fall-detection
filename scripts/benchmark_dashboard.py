"""Synthetic-only throughput/memory benchmark for the JSONL dashboard reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import tracemalloc

from fall_detection.dashboard import read_events


def benchmark(events: int, batch_size: int) -> dict:
    if type(events) is not int or events <= 0:
        raise ValueError("events must be a positive integer")
    if type(batch_size) is not int or not 1 <= batch_size <= 200:
        raise ValueError("batch size must be 1..200")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for index in range(events):
                output.write(json.dumps({
                    "event": "frame", "frame_index": index, "timestamp_seconds": index / 30,
                    "state": "unknown", "reason": "synthetic_benchmark_only",
                }) + "\n")
        size_bytes = path.stat().st_size
        cursor = "0"
        count = 0
        batches = 0
        tracemalloc.start()
        started = time.perf_counter()
        try:
            while count < events:
                result = read_events(path, cursor, batch_size)
                if not result["events"]:
                    raise RuntimeError("reader stopped before all synthetic events were consumed")
                if result["reset"]:
                    raise RuntimeError("unexpected log reset")
                count += len(result["events"])
                cursor = result["next_cursor"]
                batches += 1
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return {
        "benchmark": "synthetic_jsonl_reader_only",
        "synthetic": True,
        "events_processed": count,
        "batches": batches,
        "input_bytes": size_bytes,
        "elapsed_seconds": round(elapsed, 6),
        "events_per_second": round(count / elapsed, 2) if elapsed else None,
        "peak_traced_bytes": peak,
        "fall_detection_accuracy_measured": False,
        "camera_or_model_inference_measured": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.events, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
