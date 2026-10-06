# Temporal-model data, split, and export contract

Milestone 7 defines the reproducible boundary for a future learned temporal classifier. It does not bundle a dataset, training framework, trained weights, or detection-quality results. The checked-in state remains blocked at the artifact step until suitable labeled sequences are supplied under documented authorization.

## Dataset contract

The loader accepts UTF-8 JSON Lines: one complete sequence object per line. Blank lines, duplicate sequence IDs, unknown fields, non-finite numbers, and partially valid files are rejected. Every object contains exactly:

```json
{
  "schema_version": 1,
  "sequence_id": "sequence-001",
  "subject_id": "pseudonymous-subject-001",
  "session_id": "session-001",
  "authorization_id": "consent-or-license-record-001",
  "label": "falling",
  "features": [[0.0, 1.0, 0.9]]
}
```

The short `features` value above illustrates the field only; it is not valid by itself. A real record must contain exactly `sequence_length` rows, and each row must have `10 + 3 × keypoint_count` finite numeric values. Defaults are 30 steps, 17 keypoints, and 61 values per step.

`authorization_id` is a required provenance reference, not proof that consent or a license exists. The operator remains responsible for verifying authorization, privacy, retention, and permitted use before placing data under `data/`. Raw footage and generated dataset files are Git-ignored.

### Feature order

Each step contains:

1. source-relative time offset;
2. observed flag;
3. detection confidence;
4. normalized box center x/y, width/height, and aspect ratio;
5. normalized center velocity x/y per source second;
6. box-relative x/y coordinates for each keypoint;
7. one availability mask per keypoint.

Time offsets start at zero and increase strictly. A missing step has `observed = 0` and every later value in that row is zero. An unavailable keypoint has mask zero and zero-filled coordinates. These masks keep missing data distinct from a real coordinate of zero. `encode_feature_window` applies the same layout to runtime `TemporalFeatureWindow` values and revalidates the result against the serialized contract.

The only supervised labels are `normal`, `falling`, and `fallen`. `unknown` represents insufficient inference evidence and is not silently converted into a training target.

## Evaluation split design

`build_subject_split` assigns complete pseudonymous subjects—not individual sequences—to train, validation, or test. All sessions and sequences for one subject therefore remain in one split. Assignment is deterministic from `split_seed` and the subject ID, independent of input file order.

At least three subjects are required so every split is nonempty. The default target fractions are 70% train, 15% validation, and 15% test, apportioned by subject count with a guaranteed subject in each split. The split is not claimed to be label-stratified. A training plan remains blocked if its training partition lacks any of the three supervised labels.

The manifest records the dataset SHA-256, schema and feature layout, class order, split seed, per-split counts, and every sequence assignment. It is written with exclusive creation and never overwrites an existing file:

```bash
fall-detection training-check \
  --config configs/training.toml \
  --dataset data/temporal_sequences.jsonl \
  --manifest-output outputs/temporal-split.json
```

With no dataset present, `training-check` reports `authorized_labeled_dataset_missing`, performs no training, and leaves model availability false.

## Training and export boundary

`TemporalModelExporter` is an injected interface for a future local training backend. `export_temporal_model` accepts only a ready plan, sends the validated records and immutable split manifest to that backend, verifies the resulting ONNX graph, and publishes it without overwriting an existing artifact. No optimizer, framework, or pretrained temporal weights are selected in this milestone.

An accepted temporal ONNX graph has:

- one float input named `features`, shaped `[batch, sequence_length, feature_width]`;
- one float output named `logits`, shaped `[batch, 3]`;
- class order `normal,falling,fallen`;
- metadata `model_contract_version=1`, `dataset_schema_version=1`, and `class_names=normal,falling,fallen`;
- a finite CPU smoke output for an all-zero contract tensor.

Verify a future artifact with:

```bash
fall-detection temporal-model-check \
  --config configs/training.toml \
  --model models/temporal_state.onnx
```

Loading and executing the graph only verifies compatibility. It does not establish accuracy, calibration, latency, or safety.

## Dependency-free smoke check

```bash
fall-detection training-check --config configs/training.toml --smoke
```

The command parses nine in-memory synthetic contract records and constructs separated splits. It does not train, evaluate, or export a model; it always reports `model_artifact_available: false`. Synthetic fixture behavior must never be reported as real-world model performance.

## Remaining requirements

- Obtain authorized, representative labeled sequences with pseudonymous subject and session groupings.
- Define and document the labeling protocol and adjudication process.
- Select a CPU-compatible training implementation and record exact dependencies and seeds.
- Train only on the training partition; use validation for model selection and preserve test for final evaluation.
- Measure class-specific precision/recall, macro metrics, confusion cases, and latency on actual inputs in milestone 9. No inherited or synthetic score is a project result.
