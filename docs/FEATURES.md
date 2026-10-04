# Temporal feature contract

Milestone 5 converts each active `TrackSnapshot` into a timestamped, bounded feature window. This stage prepares deterministic input for the explainable state baseline planned in milestone 6; it does not classify a person or report that a scene is safe.

## Configuration

`configs/features.toml` defines:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `window_size` | 30 | Maximum samples retained per active track |
| `keypoint_count` | 17 | Exact pose layout expected from the configured estimator |
| `min_keypoint_confidence` | 0.30 | Inclusive confidence threshold for a usable keypoint |
| `min_observed_samples` | 8 | Observed samples required before a current window is marked ready |
| `max_motion_gap_seconds` | 1.0 | Longest elapsed source-time interval used for center velocity |

Unknown settings, malformed tables, non-finite thresholds, and incompatible bounds fail explicitly. The keypoint count must match the pose adapter configuration.

## Sample representation

Each `PoseFeatureSample` includes:

- the track ID, frame index, and source timestamp;
- an explicit `observed` flag and detector confidence;
- box center and size normalized to frame width and height;
- box width-to-height ratio;
- center velocity in normalized-frame units per source second when it is valid; and
- keypoint coordinates normalized around the detected box center, plus a separate confidence mask.

Translation and pixel scale are separated from within-box pose geometry: box values retain scene position and size, while keypoint coordinates describe relative pose. Coordinates are not silently clipped, which preserves detector output outside the box for later validation.

A keypoint below the confidence threshold becomes `(null, null)` with a false mask value. A missed track becomes a complete missing sample: its detection, box, motion, and keypoints are all unavailable. The implementation does not repeat stale poses, zero-fill missing joints, or turn absent evidence into a normal state.

Center velocity is computed only from the immediately preceding sample for the same track. It remains unavailable for a new track, after a missing sample, when timestamps do not advance, or across an interval longer than `max_motion_gap_seconds`. This prevents false high-confidence motion continuity across gaps.

## Window lifecycle

`TemporalFeatureBank.update(frame, snapshots)` expects the complete active tracker snapshot set for one frame. It validates strictly increasing indices and source timestamps, matching snapshot indices, unique positive track IDs, finite pose values, and the configured keypoint count.

Each track owns an independent `deque` capped at `window_size`. Missing observations occupy one slot so downstream logic can reason about data quality. A window is `ready` only when the current sample is observed and the bounded history contains at least `min_observed_samples` observed samples. A tracker-expired ID disappears from the input set and its history is deleted, bounding storage to active tracks rather than every ID ever seen.

`PoseTrackingFeatureStage` composes an existing tracking stage with the feature bank. It returns windows only and deliberately has no `state` field. The next milestone will consume this contract to produce explainable `unknown`, `normal`, `falling`, or `fallen` states.

## Synthetic smoke check

```bash
fall-detection feature-smoke --config configs/features.toml
```

The command generates six metadata-only frames, maintains two independent tracks, masks one low-confidence joint, and records one missed observation before recovery. It validates code behavior, memory bounds, and JSON serialization only. It is not real footage, a fall classifier, a latency benchmark, or evidence of accuracy.
