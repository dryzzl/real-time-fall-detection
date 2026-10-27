# Synthetic live-pipeline stress benchmark

The `fall_detection.live_stress` module exercises the **actual live pipeline**
from fake pose observations through local multi-person tracking, bounded
temporal feature windows, and heuristic state decisions. The fixture uses
stationary upright people with repeated two-frame occlusions, so it checks
track continuity, missing observations, ordered records, memory bounds, and
the safety contract that no validated fall prediction is emitted.

Run after installing the package:

```bash
python -m pip install -e ".[headless,model,test]"
python -m unittest tests.test_live_stress -v
python -m fall_detection.live_stress --frames 1000 --people 4 --width 160 --height 120
python -m fall_detection.live_stress --frames 5000 --people 8 --width 320 --height 240 --report outputs/live-stress.json
```

A report includes frames processed, synthetic missing observations, peak
temporal window size, throughput, and peak traced Python memory. Reports
are created exclusively and cannot overwrite existing files. A memory
threshold can be enforced with `--max-peak-memory-mib N`; a failed
threshold produces exit status 1, and invalid arguments/file I/O produce 2.

**Interpretation:** This is a synthetic software integration benchmark,
not a real camera, ONNX pose inference, BoT-SORT, LSTM, medical alert,
or model-accuracy benchmark. Throughput is not camera-to-alert latency.
`tracemalloc` excludes native memory, GPU buffers, and operating-system
RSS. No trained model or fall footage is required, and the global state
remains unknown with `fall_detection_available=false`.

Next: replay authorized video through an actual ONNX pose model and
profile model inference, tracker, temporal classifier and alert latency
separately. Only report real-world metrics when measured from real data.
