# Roadmap

Only **Phase 1** is implemented. Phases 2–7 are planned work and do not exist in the code. Scope and order may change as evaluation results come in.

| Phase | Goal | Status |
| --- | --- | --- |
| 1 | Video → decode → person detection → annotated video and JSON artifacts; web UI; CLI | **Implemented** |
| 2 | Multi-object tracking | Planned |
| 3 | Team classification and 2D pitch mapping | Planned |
| 4 | Ball intelligence and event understanding | Planned |
| 5 | Interactive 3D reconstruction | Planned |
| 6 | Research and evaluation | Planned |
| 7 | Production architecture | Planned |

## Phase 1: person detection (implemented)

MP4 upload with media validation, RF-DETR (COCO `person`) detection with configurable confidence, stride, resolution and device, a persistent subprocess job system, versioned artifacts, a Next.js UI, a CLI, tests and CI. See the [README](../README.md).

## Phase 2: multi-object tracking

- **Algorithms to evaluate:** Kalman filtering with Hungarian assignment, ByteTrack, BoT-SORT, and appearance-based re-identification.
- **Output:** `TrackObservation` with `frame_index`, timestamp, `track_id`, bbox, confidence, observed or predicted, and tracking method ([DATA_CONTRACTS.md](DATA_CONTRACTS.md#2-planned-not-implemented)).
- **Challenges to handle:** camera motion, and players entering and leaving the frame.
- **Evaluation:** IDF1, HOTA and identity switches on annotated data.
- **Not planned:** persistent real-world identities, until there is sufficient evidence to support them.

## Phase 3: team classification and 2D pitch mapping

**Teams**
- Extract player crops and build jersey and appearance embeddings (for example SigLIP).
- Cluster them (for example K-means) with temporal smoothing, keeping an explicit *unknown team* state.

**Pitch geometry**
- Detect pitch keypoints and lines, estimate a homography with RANSAC, and report calibration confidence.
- Recalibrate as the camera moves.
- Convert image coordinates to pitch coordinates, using player ground-contact points rather than box centres.
- **Coordinates:** meters, with X along the length, Y along the width and Z as height. Pitch size is configurable (default 105 × 68 m).
- **Tactical replay:** a top-down view from calibrated coordinates. If calibration is invalid, projected positions are omitted or clearly flagged.

## Phase 4: ball intelligence and events

- **Ball tracking:** a dedicated ball detector with small-object inference, optical flow, Kalman or EKF filtering, motion gating and temporal interpolation.
- **Event candidates:** possession inference, pass and shot candidates, and trajectory analysis.
- Confirmed events are distinguished from inferred ones.
- Ball height or 3D flight is not claimed from monocular video without sufficient constraints.

## Phase 5: interactive 3D reconstruction

- **Viewer:** Three.js / React Three Fiber, with accurate pitch geometry, player and ball entities, and time-synced animation.
- **Cameras:** top-down, sideline and free, plus trajectory trails.
- The first version renders 2D ground-plane positions inside a 3D scene. Generic markers are never presented as reconstructed bodies.
- **Later and experimental:** MMPose, 3D pose lifting, camera extrinsics, depth estimation (VGGT, Video Depth Anything) and temporal pose smoothing, with failure modes reported.

## Phase 6: research and evaluation

- **Dataset:** a small, legally usable annotated evaluation set. Gated datasets such as SoccerNet are never downloaded automatically, and their license terms are respected.
- **Metrics by area:**

  | Area | Metrics |
  | --- | --- |
  | Detection | mAP, precision, recall |
  | Tracking | HOTA, IDF1, identity switches |
  | Calibration | Reprojection error, ground-coordinate error |
  | Ball | Localization error, miss rate, fragmentation |
  | Events | Precision, recall, F1, timestamp error |
  | Performance | FPS, time per minute of video, peak RAM and GPU memory |

- **Ablations** over model and algorithm choices, with configurations, splits, seeds and model versions preserved.
- Results go in [MODEL_EVALUATION.md](MODEL_EVALUATION.md). No benchmark numbers are invented.

## Phase 7: production architecture

Candidates include PostgreSQL with SQLAlchemy and Alembic, Redis and Celery workers, S3 or MinIO object storage, Docker Compose, cloud GPU processing, stage-level artifact caching, OpenTelemetry, Prometheus and Grafana, CI/CD deployments, and infrastructure as code.

These are adopted only when a real requirement justifies them. Phase 1 deliberately uses SQLite, local disk and subprocess workers.
