# Architecture

This document describes the system as implemented: the Phase 1 detection platform, plus the offline evaluation module from milestones 2A.1 and 2A.2. Planned work (tracking, pitch geometry, 3D replay, distributed infrastructure) is described in [ROADMAP.md](ROADMAP.md) only.

## Components

```text
┌──────────────┐  HTTP/JSON   ┌───────────────────────────────────────────────┐
│  apps/web    │ ───────────▶ │ apps/api  (uvicorn, one process)              │
│  Next.js 16  │ ◀─────────── │  ├─ FastAPI routers  /api/v1/...              │
└──────────────┘  video Range │  ├─ BodySizeLimitMiddleware, CORS             │
                              │  └─ JobRunner dispatcher thread               │
                              │        │ claims PENDING → PROCESSING          │
                              │        │ spawns ≤ max_concurrent_jobs         │
                              └────────┼──────────────────────────────────────┘
                                       ▼
                         python -m soccervision_api.worker <job_id>   (subprocess)
                                       │
                         packages/soccervision.pipeline.run_detection_pipeline
                         decode (PyAV) ▶ detect (RF-DETR) ▶ annotate (Pillow) ▶ export (libx264 + JSON)
                                       │
           data/soccervision.db (SQLite, WAL)    data/artifacts/{video_id}/{run_id}/
```

### `packages/soccervision`: the core library

This package has no web dependencies. The API, the CLI and the tests all call the same `run_detection_pipeline` function.

| Module | Responsibility |
| --- | --- |
| `config.py` | `DetectorConfig` and `PipelineConfig` (frozen Pydantic models, `extra="forbid"`) |
| `devices.py` | Device discovery and resolution. `auto` picks CUDA, then MPS, then CPU. An explicitly requested device that is unavailable is an error, never a silent fallback. |
| `schemas/` | Versioned data contracts for videos, detections, jobs and artifacts ([DATA_CONTRACTS.md](DATA_CONTRACTS.md)) |
| `vision/video/` | `probe_video` (container and stream validation, metadata), `VideoDecoder` (RGB frames with PTS-derived timestamps), `AnnotatedVideoWriter` (H.264/yuv420p/faststart, falling back to MPEG-4 Part 2 if libx264 is missing) |
| `errors.py` | `SoccerVisionError` hierarchy, each with a stable `code` (for example `invalid_video`, `model_unavailable`, `invalid_dataset`) |
| `vision/detection/` | `ObjectDetector` protocol, `RFDETRDetector` (lazy import, pinned weight MD5, warm-up, input resolution divisible by 32) and the factory |
| `vision/annotation.py` | Draws boxes and the HUD. Skipped frames are explicitly labelled **NOT ANALYZED**. |
| `pipeline/runner.py` | Orchestration, post-processing, statistics, failure handling and atomic artifact writes |
| `evaluation/` | Offline detection evaluation: COCO ground-truth loading and validation, COCO predictions, prediction export (any `ObjectDetector` over a dataset's images, with reproducibility metadata), and metrics via `pycocotools` (optional `eval` extra). Not used by the API or the analysis pipeline. See [MODEL_EVALUATION.md](MODEL_EVALUATION.md). |
| `cli.py` | `soccervision analyze` / `soccervision devices` |

### `apps/api/soccervision_api`: the service

| Module | Responsibility |
| --- | --- |
| `app.py` / `main.py` | `create_app()` factory (used by tests). `main.app` is the uvicorn entrypoint. |
| `settings.py` | `SOCCERVISION_*` settings, serialized into worker subprocess environments |
| `persistence/database.py` | SQLite `videos` and `jobs` tables. Every status change is a conditional `UPDATE`. |
| `services/uploads.py` | Streamed, size-bounded upload, followed by media validation and an atomic rename |
| `services/jobs.py` | `JobRunner`: dispatcher thread, subprocess launching, reaping, reconciliation, shutdown |
| `services/results.py` | Artifact whitelist and path containment, and the per-frame count summary |
| `worker.py` | Executes one claimed job in its own process |
| `api/` | Routers: system, videos, jobs |

### `apps/web`: the frontend

Next.js 16 (App Router, Cache Components) with Tailwind CSS 4. Pages are static shells, and all data is fetched client-side from the API (`src/lib/api.ts`, `src/hooks/usePolling.ts`). The analysis page polls while a job is active and stops once it finishes.

## Key decisions

### Jobs: a SQLite queue with worker subprocesses

The requirements were that work never runs inside a request, concurrency is bounded, state survives restarts, and there is no Redis, Celery or other broker.

- The `jobs` table *is* the queue. `POST /analyze` inserts a `PENDING` row inside `BEGIN IMMEDIATE`, rejecting a second active job for the same video with 409, and wakes the dispatcher.
- A single dispatcher thread in the API process claims `PENDING → PROCESSING` with a conditional UPDATE and launches `python -m soccervision_api.worker <job_id>`. It never runs more than `max_concurrent_jobs` workers.
- **Why subprocesses rather than threads:** PyTorch inference holds significant memory and is CPU-heavy. A separate process isolates crashes (including native segfaults), gives each job a clean model and allocator state, and can be terminated. Model load costs about 6 s per job. That is acceptable at this scale, and a persistent worker pool is a possible later optimization.
- The worker records progress as the fraction of decoded frames over the expected frame count, throttled to twice a second. The final `COMPLETED` transition is conditional, so a worker can never overwrite a job that has been reconciled to `FAILED`.

**Failure handling, all exercised by tests or live checks:**

| Situation | Result |
| --- | --- |
| Pipeline error (bad media, model unavailable, inference error) | `FAILED` with a specific `error_code`; `metadata.json` describes the failure |
| Worker exits non-zero without recording a result | Dispatcher marks `FAILED` / `worker_crashed` |
| Worker cannot be launched | `FAILED` / `worker_launch_failed` |
| API shutdown (SIGINT/SIGTERM) | Workers receive SIGTERM, which raises `KeyboardInterrupt`, and the job is marked `FAILED` / `interrupted` |
| API killed hard (SIGKILL) and restarted | On startup every `PROCESSING` job becomes `FAILED` / `interrupted`, and its leftover `.partial` files are removed; `PENDING` jobs resume |
| Video deleted (`DELETE /api/v1/videos/{id}`) | Refused with `409` while a job is `PENDING` or `PROCESSING`; otherwise the rows (jobs cascade), upload directory and artifact directory are removed |
| Orphaned worker still running after such a restart | Its next progress update sees that the job is no longer `PROCESSING`, so it stops (`job_abandoned` in `metadata.json`) and cleans up |

### Honest outputs

- **Skipped frames are not zero.** With `inference_stride > 1`, unanalyzed frames have `inference_status: "skipped"` and `detections: null`. Statistics such as the mean are computed over analyzed frames only, and the per-frame counts API omits skipped frames.
- **Total detections are not unique people.** The UI labels the total as "summed over frames".
- **Synthetic detectors are flagged.** `ModelInfo.is_synthetic` is part of the schema, and the pipeline adds a warning whenever it is true. Only the test suite uses one.
- **Run metadata records provenance:** model ID, weights MD5, device, input resolution, framework and package versions, the input file's SHA-256, configuration, and timings.

### Video I/O: PyAV without OpenCV

PyAV gives direct access to PTS and time bases, so timestamps come from the stream rather than from `index / fps`. Its wheels also ship FFmpeg with libx264, so there's no system FFmpeg to install.

OpenCV is deliberately absent. On macOS its bundled FFmpeg conflicts with PyAV's (duplicate Objective-C classes from two `libavdevice` copies). RF-DETR and supervision work without it, and annotation is drawn with Pillow.

Output frames keep the input's relative PTS, so the annotated video stays in sync with the original even for variable-frame-rate sources. Odd frame dimensions are padded to even values, as `yuv420p` requires, and this is recorded in the metadata.

### Offline evaluation, separate from the product pipeline

`soccervision.evaluation` is a library used offline; the API, workers and analysis pipeline never import it.

- **Reference implementation for metrics.** mAP, precision and recall all come from one `pycocotools` `COCOeval` run rather than a hand-written matcher, so the numbers are comparable with published COCO results. `pycocotools` is an optional `eval` extra, and a missing install raises `evaluation_unavailable`.
- **Strict, complete validation.** Malformed annotation or prediction files raise `invalid_dataset` or `invalid_predictions`, listing every issue found instead of only the first.
- **Traceability.** Every dataset carries its annotation file's SHA-256 and a provenance block, and its `synthetic` flag is copied into every metrics result.

Details are in [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

### Upload validation

1. The extension must be `.mp4` (otherwise 415).
2. The streamed copy is capped at `max_upload_bytes`. This is enforced both by middleware (Content-Length and streamed byte counting) and by the copy loop (413).
3. `probe_video` checks the ISO-BMFF signature, opens the demuxer, requires a video stream with a valid rate, decodes the first frame, and extracts metadata (422 `invalid_video`).
4. The duration limit is checked (422 `video_limit_exceeded`).
5. The file is atomically renamed to `uploads/{video_id}/source.mp4`.

Any failure removes the upload directory. Video IDs are random UUIDs, and the client-supplied filename is sanitized and stored for display only.

### Artifact serving

Artifacts are served only by whitelisted name, and only if the job's recorded `artifact_paths` contain them. Resolved paths must stay under the artifact root. Video responses support HTTP Range requests, which browser seeking needs. JSON artifacts are written to a temporary file and then renamed, so readers never see partial files.

## Data flow for one analysis

1. Probe the input and check limits, then compute the SHA-256 of the input.
2. Load the model with the factory, timing it, and run a warm-up prediction. With downloads disabled, the checkpoint's MD5 is checked against the pinned value before rfdetr loads it, because rfdetr would otherwise re-download a mismatched file. In every case, the loaded file is re-verified after loading.
3. For each decoded frame:
   - Stop with `video_limit_exceeded` if decoded timestamps pass the duration limit, in case the container's duration is wrong.
   - If `index % stride == 0`, run the detector, filter to the target classes at or above the threshold, drop non-finite boxes, clip to the frame (counted), and discard degenerate boxes (counted).
   - Draw the annotation and HUD, and encode to `annotated.mp4.partial`.
4. Write `detections.json`, rename the video to `annotated.mp4`, then write `metadata.json` last as the commit marker.
5. On any exception, delete the partial outputs and write `metadata.json` with `status: "failed"` and the error.

## Not implemented

There is no authentication or multi-tenancy, no horizontal scaling, no object storage, and no GPU scheduling across hosts. The single-host SQLite design is intentional for a local, open-source tool. Tracking, pitch geometry, ball and event inference, 3D replay, and Redis, PostgreSQL or S3 are all roadmap items ([Phase 10](ROADMAP.md#phase-10-production-ml-infrastructure-planned) covers infrastructure).
