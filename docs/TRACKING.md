# Person tracking baseline

Milestone 4 assigns process-local IDs to pose detections so later stages can keep independent temporal histories. It does not determine whether a person is standing, falling, or fallen.

## Association

For each new frame, `PersonTracker`:

1. Sorts detections by box coordinates and keypoints so results do not depend on detector output order.
2. Predicts each active box and confident keypoint using a smoothed constant-velocity estimate.
3. Scores gated track/detection pairs using normalized center distance, predicted-box IoU, and mean confident-keypoint distance.
4. Greedily selects the lowest-cost pairs with deterministic track-ID and detection-order tie breaks.
5. Creates monotonically increasing IDs for unmatched detections and emits explicit unobserved snapshots for temporarily missed tracks.
6. Expires a track after more than `max_missed_frames` absent frames. An observation after expiry receives a new ID.

Distances are normalized by the larger candidate box diagonal so the defaults are resolution-independent. Configuration is in `configs/tracker.toml`. Frame indices, not wall time, define velocity, age, and expiry. Skipped frame indices count as missed frames.

`PoseTrackingStage` composes any estimator with the tracker. The output is a tuple of `TrackSnapshot` values ordered by track ID. An observed snapshot contains a `PoseDetection`; a missed snapshot contains `detection=None`, its missed-frame count, and a motion-predicted box. That distinction lets later feature extraction represent missing observations rather than silently reusing stale keypoints.

## Deterministic smoke check

```bash
fall-detection track-smoke --config configs/tracker.toml
```

The check uses two synthetic people moving through a clean crossing, deliberately reverses input detection order, omits one person for one frame, and verifies that IDs recover. It is a functional regression check, not real video, accuracy evidence, or proof that IDs will remain stable in crowded scenes.

## Baseline limitations

- This is a small greedy association baseline, not BoT-SORT. It has no appearance embedding, camera-motion compensation, Kalman covariance, global assignment, or cross-session re-identification.
- Constant-velocity prediction works for smooth short motion but can fail on sudden direction changes, long occlusion, close interactions, severe detector jitter, or people with similar boxes and poses.
- Greedy matching can be globally suboptimal when several candidates have similar costs. Deterministic output does not guarantee correct identity.
- IDs start at 1 and reset with the tracker process. They are not identities of real people and must not be persisted as biometric identifiers.
- Low-confidence or missing keypoints reduce matching to box motion and overlap. False detections can create short-lived tracks; missed detections can expire a real track.
- Synthetic crossing, missed-frame, and expiry tests validate code behavior only. No tracking quality metric or real-world performance claim is available.

These limitations are intentional for an explainable first baseline. A stronger tracker may replace the association strategy behind the same tracked-pose boundary after it is evaluated on authorized recordings.
