# Data contracts

The source of truth is the Pydantic models in [`packages/soccervision/schemas/`](../packages/soccervision/schemas). This document explains them. The TypeScript mirrors live in [`apps/web/src/types/api.ts`](../apps/web/src/types/api.ts).

Section 1 describes what Phase 1 **writes today**. Section 2 describes **planned** schemas, which are not implemented, so their field names may still change.

The offline evaluation module (milestone 2A.1) writes no artifacts. Its in-memory models (`EvaluationDataset`, `DatasetMetadata`, `Prediction`, `DetectionMetrics`) live in `packages/soccervision/evaluation/` and are documented in [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

## Versioning

- Every JSON artifact carries `schema_version` (currently **`1.0.0`**, `ARTIFACT_SCHEMA_VERSION`) and `artifact_type`.
- Additive, backward-compatible fields bump the **minor** version. Renames, removals or changed semantics bump the **major** version.
- `pipeline.version` (currently `1.0.0`) is versioned separately and changes when processing behaviour changes, such as post-processing rules, even if the schema does not.

## 1. Implemented (Phase 1)

### Run directory

```text
data/artifacts/{video_id}/{run_id}/     # run_id == job_id for API runs
├── annotated.mp4      H.264 (libx264), yuv420p, +faststart; same frame count and timestamps as input
├── detections.json    DetectionsArtifact
├── metadata.json      RunMetadata (written last; present for failed runs too)
└── worker.log         API runs only; not served over HTTP
```

### Conventions

- **Image coordinates** are pixels in the decoded frame (`width × height`), with the origin at the top-left. Boxes are `[x1, y1, x2, y2]` with `x2 > x1` and `y2 > y1`, clipped to the frame bounds.
- **Timestamps** are seconds relative to the first frame of the video stream: `(pts − stream.start_time) × time_base`. `timestamp_source` is `"pts"`, or `"estimated"` when a frame has no PTS, in which case the time is the previous frame's time plus one nominal frame interval. Estimated frames are counted in `stats.frames_with_estimated_timestamps`.
- **Frame indices** are 0-based decode order.
- `confidence` is the model score in [0, 1].

### `detections.json` (`DetectionsArtifact`)

```jsonc
{
  "schema_version": "1.0.0",
  "artifact_type": "detections",
  "run_id": "e5005427…",
  "video_id": "352fd630…",
  "model": { /* ModelInfo, see below */ },
  "confidence_threshold": 0.5,
  "target_classes": ["person"],
  "inference_stride": 2,
  "coordinate_system": {
    "type": "image_pixels", "origin": "top_left", "bbox_format": "xyxy", "units": "pixels",
    "frame_width": 1280, "frame_height": 720
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "timestamp_source": "pts",
      "inference_status": "analyzed",
      "detections": [
        {
          "video_id": "352fd630…",
          "frame_index": 0,
          "timestamp_seconds": 0.0,
          "detection_id": "f0-d0",
          "class_name": "person",
          "confidence": 0.822,
          "bbox_xyxy": [365.4, 571.3, 406.8, 702.4],
          "model_id": "rfdetr-nano-coco",
          "source": "model_detection"
        }
      ]
    },
    {
      "frame_index": 1,
      "timestamp_seconds": 0.0333,
      "timestamp_source": "pts",
      "inference_status": "skipped",
      "detections": null
    }
  ]
}
```

- `frames` contains **every decoded frame**, in order.
- `inference_status: "analyzed"` means the detector ran, and `detections` is a list that may be empty.
- `inference_status: "skipped"` means the detector did not run because of the stride, and `detections` is `null`. **Consumers must not treat skipped frames as zero detections.**
- `detection_id` (`f{frame}-d{k}`) is unique within a run only. It is **not** a track or person identity.
- `source` is always `model_detection` in Phase 1. The other `ObservationSource` values (`tracker_prediction`, `geometric_projection`, `interpolation`) are reserved for later phases so that model outputs can always be told apart from derived data.

### `ModelInfo`

| Field | Example | Notes |
| --- | --- | --- |
| `model_id` | `rfdetr-nano-coco` | Stable identifier recorded on every detection |
| `family` / `variant` | `rf-detr` / `rfdetr-nano` | |
| `label_space` | `coco-80` | |
| `license` | `Apache-2.0` | License of the weights |
| `weights_file`, `weights_md5` | `rf-detr-nano.pth`, `fb6504cc…` | MD5 computed from the file actually loaded |
| `input_resolution` | `384` | Square model input. Must be a multiple of 32 for N/S/M. |
| `device` | `mps` | The device actually used, after `auto` resolution |
| `framework_versions` | `{"rfdetr": "1.11.2", "torch": "2.15.0", …}` | |
| `is_synthetic` | `false` | `true` only for test fakes; such runs also carry a warning |

### `metadata.json` (`RunMetadata`)

| Field | Content |
| --- | --- |
| `status` | `completed` or `failed` |
| `pipeline` | `name` (`person-detection`), `version`, `package_version`, `stages` |
| `configuration` | The full `PipelineConfig` used |
| `model` | `ModelInfo`, or `null` if the model never loaded |
| `input_file`, `input_sha256`, `input_video` | Input name, content hash, and the `VideoProbe` (container, codec, size, fps, duration, frame count, time base, start time, frame-rate mode, audio presence, bytes) |
| `output_video` | `file`, `codec`, `pixel_format`, `width`, `height`, `browser_playable`, `padded_to_even_dimensions`; `null` on failure |
| `stats` | `decoded_frames`, `analyzed_frames`, `skipped_frames`, `total_detections`, `mean_detections_per_analyzed_frame`, `max_detections_in_frame`, `frames_with_estimated_timestamps`, `clipped_boxes`, `discarded_degenerate_boxes` |
| `timing` | `started_at`, `completed_at`, `total_seconds`, `model_load_seconds`, `inference_seconds`, `mean_inference_ms`, `processing_fps` |
| `environment` | Python version, platform, package versions |
| `artifacts` | Relative file names |
| `warnings` | Human-readable caveats (for example a synthetic detector, or the MPEG-4 fallback codec) |
| `error` | `{code, message, frame_index}` for failed runs, otherwise `null` |

`total_detections` is a sum over analyzed frames. It is **not** a count of distinct people.

### Video metadata (`VideoMetadata`, returned by the API)

`video_id`, `filename` (sanitized original name), `duration_seconds`, `width`, `height`, `fps`, `frame_count`, `created_at`, `processing_status` (`uploaded | queued | processing | completed | failed`, derived from the latest job), `container_format`, `video_codec`, `size_bytes`, `frame_rate_mode` (`constant | variable | unknown`), `has_audio`.

### Jobs (`PipelineJob`)

`job_id`, `video_id`, `status`, `progress` (0–1, real decoded-frame progress), `created_at`, `started_at`, `completed_at`, `error_code`, `error_message`, `pipeline_version`, `configuration`, `artifact_paths`.

```text
PENDING ──▶ PROCESSING ──▶ COMPLETED
   │            │
   └────────────┴────────▶ FAILED
```

Any other transition raises `InvalidJobTransitionError`, and terminal states are final. Common `error_code` values are `invalid_video`, `video_limit_exceeded`, `model_unavailable`, `device_unavailable`, `inference_failed`, `video_encoding_failed`, `internal_error`, `interrupted`, `worker_crashed`, `worker_launch_failed` and `video_missing`. A worker that finds its job already reconciled writes `job_abandoned` to its run's `metadata.json`.

## 2. Planned (not implemented)

These sketches exist so that Phase 1 choices, such as `ObservationSource`, the stable `video_id` and `frame_index` keys, and explicit coordinate systems, don't paint later phases into a corner. **None of these are produced by the current code.**

### World coordinate convention (planned)

- Units: **meters**.
- Origin: pitch centre on the ground plane, unless a calibration says otherwise.
- **X** runs along the pitch **length** (touchline direction), **Y** along the **width** (goal-line direction), and **Z** is **height** above the ground (up is positive).
- Pitch dimensions are configurable per calibration. The default is **105 × 68 m** (the FIFA-recommended size); real pitches vary.

### Sketches

| Schema | Key fields | [Roadmap](ROADMAP.md) phase |
| --- | --- | --- |
| `TrackObservation` | `video_id`, `frame_index`, `timestamp_seconds`, `track_id`, `bbox_xyxy`, `confidence`, `source` (`model_detection` when observed, `tracker_prediction` when predicted), `tracking_method`, `detection_id?` | 2B |
| `TeamClassification` | `track_id`, `team` (`team_a`, `team_b`, `referee`, `goalkeeper` or `unknown`), `confidence`, `method` | 3 |
| `PitchCalibration` | `video_id`, `frame_index` (or a range), `homography` (3×3, image to pitch plane), `pitch_length_m`, `pitch_width_m`, `reprojection_error_px`, `confidence`, `valid` | 3 |
| `WorldPosition` | `track_id`, `frame_index`, `x_m`, `y_m`, `z_m?`, `source` (`geometric_projection` or `interpolation`), `calibration_id`, `uncertainty_m?` (ground-contact point, not box centre) | 3 |
| `BallObservation` | `frame_index`, `bbox_xyxy?`, `image_xy?`, `world_xyz?` (only with sufficient constraints), `confidence`, `visible`, `source` | 5 |
| `SoccerEvent` | `event_id`, `type` (`pass`, `shot` and so on), `start_frame`, `end_frame`, `actors` (track IDs), `confidence`, `status` (`confirmed` or `inferred`) | 6 |
| `PoseObservation` | `track_id`, `frame_index`, `keypoints` (name, x, y, confidence), `skeleton` (format ID); experimental | 7 (research milestones 7.14–7.15) |
| `SceneFrame` | `frame_index`, `timestamp_seconds`, `players[]` (WorldPosition and team), `ball?`, `calibration_id` | 7 |

Derived values (anything not directly output by a model on that frame) must carry a non-`model_detection` `source` so that consumers can always tell measurement from inference.
