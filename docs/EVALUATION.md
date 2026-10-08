# Held-out replay evaluation

The milestone 9 evaluation harness measures a trained temporal ONNX model against the immutable held-out `test` split created by the training-data contract. It reports class outcomes, regression failures, and CPU model-inference latency while preserving the exact dataset, manifest, and model fingerprints used.

The repository currently has no authorized labeled replay sequences, split manifest, or trained temporal model. The harness is implemented, but the milestone's actual-input acceptance gate remains blocked and no real quality or latency result is claimed.

## Required inputs

A real evaluation requires all three local artifacts:

1. The authorized labeled sequence JSONL named by `dataset_path` in `configs/training.toml`. Each sequence must include pseudonymous subject/session IDs and an authorization reference.
2. The subject-separated split manifest generated from that exact dataset by `training-check --manifest-output`. The held-out test split must contain `normal`, `falling`, and `fallen` labels.
3. A trained temporal ONNX model named by `model_path` in `configs/training.toml` and accepted by the temporal model contract.

The evaluator rejects a manifest if its canonical dataset SHA-256, schema, feature layout, class order, assignments, counts, or subject separation differs from the loaded dataset. The model is checked for the expected input/output shapes and metadata before evaluation.

Authorization references are auditable claims supplied with the dataset. Software cannot verify consent, licensing, retention rules, label quality, or permitted use; the user must verify them before evaluation. The current ONNX metadata contract also does not prove which dataset trained a model, so the report records that model-training provenance was not verified by software.

## Readiness and execution

Check current readiness without running inference or producing metrics:

```bash
fall-detection evaluation-check --config configs/evaluation.toml
```

The current repository reports these blockers:

- `authorized_labeled_replay_sequences_missing`
- `held_out_split_manifest_missing`
- `trained_temporal_model_missing`

After supplying verified local artifacts, write a new report:

```bash
fall-detection evaluation-check \
  --config configs/evaluation.toml \
  --report-output outputs/held-out-evaluation.json
```

`--dataset`, `--manifest`, `--model`, and `--training-config` can select other local artifacts. Report output uses exclusive creation and never overwrites an existing file.

## Report contract

The report is tied to:

- the canonical dataset digest and exact dataset-file digest;
- the exact split-manifest file digest;
- the validated ONNX model digest;
- the held-out split name and class support;
- Python, platform, NumPy, ONNX Runtime, and CPU execution-provider metadata.

Reported outcome data includes a `normal`/`falling`/`fallen` by `normal`/`falling`/`fallen`/`unknown` confusion matrix, overall accuracy, non-unknown coverage, macro F1, per-class precision/recall/F1, and bounded pseudonymous failure-case IDs. An `unknown` prediction counts as incorrect for accuracy and as abstention for coverage; it is never silently converted to `normal`.

Every test sequence is a regression case. A pass means its predicted class equals its labeled class. Failure IDs are bounded by `failure_case_limit` so reports cannot grow without limit.

## Latency method

The evaluator uses `CPUExecutionProvider`, performs the configured warmup count on the first held-out sequence, and then performs `timed_runs` repeated calls for every test sequence. A nondeterministic class result across repeated runs fails closed. The report includes minimum, mean, median, nearest-rank p95, and maximum latency plus the runtime environment.

Latency covers the temporal ONNX call only. It explicitly excludes video decoding, pose inference, tracking, temporal feature construction, preview, and alerts. It is not end-to-end camera latency and cannot be compared across machines without considering the recorded environment.

## Synthetic smoke separation

```bash
fall-detection evaluation-check --smoke
```

The dependency-free smoke check exercises deterministic ordering, incorrect and `unknown` outcomes, regression counts, and latency-sample validation. Its output is marked `synthetic`, omits accuracy and F1, sets `quality_metrics_available` and `real_world_quality_measured` to false, and cannot write a quality report. Synthetic unit fixtures and constant ONNX graphs validate code contracts only.

## Limitations

- The evaluator consumes labeled temporal feature sequences, not raw video. End-to-end replay through decode, pose, tracking, features, state, and alerts remains an integration task.
- Dataset and model hashes make a report traceable, but they do not prove that labels are correct or representative.
- Contract-compatible model metadata does not prove that the model was trained from the accompanying training split; training provenance must be verified outside this evaluator.
- Held-out performance on one dataset is not clinical validation, monitored-emergency-service certification, or evidence that the system is safe as a sole protective measure.
- No footage, feature dataset, model, credentials, or report is committed by this milestone; `data/`, `models/`, and `outputs/` remain ignored.
