# Synthetic streaming stress harness

Run a repeatable **software-pipeline-only** stress test without installing OpenCV,
PyTorch, ONNX Runtime, or model weights:

\`\`\`bash
python -m fall_detection.stress --frames 2000 --width 160 --height 90
python -m fall_detection.stress --frames 10000 --width 320 --height 180 --report outputs/stress.json
\`\`\`

The command streams frames incrementally through the existing pipeline,
validates sequential JSON frame records, checks that no unsupported fall
predictions are emitted, and measures throughput and peak Python allocation
memory. It does not store the frames or all output records.

Optional acceptance thresholds:

\`\`\`bash
python -m fall_detection.stress --frames 5000 \
  --max-peak-memory-mib 8 --min-throughput-fps 100
\`\`\`

Exit codes: 0 = checks pass; 1 = one or more acceptance thresholds fail;
2 = invalid arguments or report-file I/O failure. Existing report files are
never overwritten.

**Scope and limitations:** The input is synthetic colored-bar frames, not
human motion or falls. The predictor intentionally returns \`unknown\`. This
benchmark measures synthetic frame generation and pipeline record handling,
**not** pose inference, BoT-SORT, LSTM accuracy, alert latency, or real camera
performance. Memory uses \`tracemalloc\` and excludes native library, GPU, and
OS-level RSS allocations. Throughput varies with CPU, Python version, image
dimensions, and tracing overhead. Results must not be represented as detection
accuracy or production latency. The next benchmark should cover real-video
capture through a supplied ONNX model and persistent alert decisions.
