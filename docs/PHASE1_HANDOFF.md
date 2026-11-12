# SoccerVision 3D: Phase 1 handoff for review

This document is for a reviewer (human or AI) checking the Phase 1 implementation and planning the next steps. It summarizes what was built against the original "SoccerVision 3D master prompt", how it was verified, where it deviates from the prompt, and the open questions.

**State of the repository:** all work is uncommitted in the working tree. Nothing has been committed, pushed or deployed. The local Git repo and its remote are still named `real-time-facial-recognition-app` (`github.com/leowu555/real-time-facial-recognition-app`), left over from an earlier, unrelated project. Its old C++ README was replaced.

---

## 1. Scope delivered

Phase 1 only: **Video → Decode → Detect (pretrained RF-DETR, COCO `person`) → Annotate → Export**, with an API, background jobs, a web UI, a CLI, tests, docs, CI and a Dockerfile. Phases 2–7 (tracking, teams and pitch, ball and events, 3D, research, production) are documented in `docs/ROADMAP.md` and **not implemented**.

### Backend (Python, `packages/soccervision` + `apps/api/soccervision_api`)

**Upload:** `POST /api/v1/videos`
- Accepts `.mp4` only (otherwise 415).
- The upload is streamed to disk with a size cap, enforced both by middleware and by the copy loop (413).
- Validation is done on the real media:
  - ISO-BMFF signature check.
  - Demuxer opens the file.
  - A video stream with a valid frame rate exists.
  - The first frame decodes.
  - Failures return 422 `invalid_video`.
- Duration limit returns 422 `video_limit_exceeded`.
- The file is then atomically renamed into place. IDs are random UUIDs, and filenames are sanitized.
- Metadata comes from PyAV (an FFmpeg binding): container, codec, width and height, fps, duration, frame count, time base, start time, constant or variable frame rate, audio presence and size.

**Detection** (`vision/detection/rfdetr.py`)
- RF-DETR Nano, Small or Medium (rfdetr 1.11.2, Apache-2.0 weights), loaded lazily. MD5 of the weights is recorded, and a warm-up inference runs at load.
- Configurable confidence, frame stride, input resolution (multiple of 32) and device (`auto`/`cpu`/`mps`/`cuda`).
- `auto` picks CUDA, then MPS, then CPU. An explicitly requested device that is unavailable is an error, not a silent fallback.

**Pipeline** (`pipeline/runner.py`)
- Probes the input, checks limits, and computes the input's SHA-256.
- Frame loop:
  - Every Nth frame runs inference. Other frames are recorded as `inference_status: "skipped"` with `detections: null`, never zero.
  - Post-processing filters by class and threshold, drops non-finite boxes, clips boxes to the frame (counted), and drops degenerate boxes (counted).
  - Pillow draws boxes and a HUD, which shows "NOT ANALYZED" on skipped frames.
  - Frames are encoded with libx264 (`yuv420p`, `+faststart`), keeping the original relative PTS. Odd dimensions are padded.
- Artifacts are written atomically, with `metadata.json` last.
- On any failure, partial outputs are removed and a failed `metadata.json` with an error code is written.

**Artifacts:** `data/artifacts/{video_id}/{run_id}/annotated.mp4`, `detections.json` and `metadata.json` (schema 1.0.0, documented in `docs/DATA_CONTRACTS.md`). The metadata records model, weights MD5, device, framework versions, configuration, statistics, timings and environment.

**Jobs**
- The SQLite (WAL) `jobs` table is the queue, with states `PENDING → PROCESSING → COMPLETED | FAILED`. Every transition is a conditional UPDATE.
- `POST /videos/{id}/analyze` returns 202, or 409 if a job is already active for that video.
- A dispatcher thread in the API claims jobs and spawns `python -m soccervision_api.worker <job_id>` subprocesses, at most `max_concurrent_jobs` (default 1) at a time.
- **Progress** is real: decoded frames over expected frames.
- **Startup:** `PROCESSING` jobs become `FAILED/interrupted` and `PENDING` jobs resume.
- **Shutdown:** workers get SIGTERM and are marked `interrupted`.
- **Worker exits with a non-zero code:** `worker_crashed`.
- **Orphaned worker** (still running after its job was reconciled): it detects this on its next progress update and stops itself, recording `job_abandoned`.

**Results:** `GET /videos/{id}/results` returns the latest completed run's metadata plus per-analyzed-frame counts. `GET /jobs/{id}/artifacts/{name}` serves only whitelisted names that are recorded on the job, with path-containment checks and HTTP Range support.

**CLI:** `python -m soccervision.cli analyze --input … --output …` uses the same pipeline function as the worker, with exit codes 0/1/2. `soccervision devices` lists available devices.

### Frontend (`apps/web`, Next.js 16.4, React 19.3, Tailwind 4)

- **Dashboard:** backend and model status, recent analyses with status, and lists of "available now" and "roadmap (not implemented)" features.
- **Upload:**
  - Drag-and-drop with client-side validation and a local preview.
  - Upload progress over XHR, then the server's validated metadata.
  - Analysis settings (detector, confidence, stride, device), then **Analyze**.
- **Analysis:**
  - Job status with polling and a real progress bar, and failure messages with hints.
  - Synced original and annotated players.
  - Statistics from the run's artifacts: analyzed vs decoded frames, total detections explicitly labelled "summed over frames, not unique people", mean and peak per analyzed frame, and timing.
  - An SVG detection timeline that can be clicked to seek.
  - Source metadata, plus run details (model, MD5, configuration, artifact links).
  - When the latest job failed, a notice and the results of the previous completed run.

### Tooling
- **CI** (`.github/workflows/ci.yml`):
  - Backend on Python 3.12 and 3.13 without the `ml` extra: ruff, ruff format, mypy (strict), pytest.
  - Frontend on Node 24: lint, typecheck, vitest, build.
- **`Dockerfile`:** CPU API image with CPU-only torch wheels.
- **Config files:** `.env.example` (all `SOCCERVISION_*` settings), `.gitignore` (data, weights, media, env), and an Apache-2.0 `LICENSE`.
- **Docs:** `README.md`, plus `docs/ARCHITECTURE.md`, `ROADMAP.md`, `DEVELOPMENT.md`, `DATA_CONTRACTS.md` and `MODEL_EVALUATION.md`.

---

## 2. Verification evidence (all actually executed)

| Check | Result |
| --- | --- |
| `ruff check .`, `ruff format --check .` | Clean |
| `mypy` (strict, 42 source files) | No issues |
| `pytest` (fake detectors; no GPU or network) | 99 passed, 1 skipped (the opt-in integration test). Same result in a fresh venv **without torch**, and on Python 3.12. |
| Real-inference integration test (`SOCCERVISION_RUN_INTEGRATION=1`, real weights) | 2 passed |
| Frontend `npm run lint` / `typecheck` / `test` (25 vitest) / `build` | All pass |

**Live end-to-end run** (uvicorn plus the production Next.js server, on macOS arm64 with MPS):
- **Uploads:** a fake `.mp4` was rejected with `invalid_video`. The real clip was accepted with correct metadata (1280×720, 30 fps, 419 frames, 13.97 s, h264).
- **Analyze and results:**
  - `analyze` returned 202, and a second request returned 409 `job_already_active`.
  - While the job was running, `/results` returned 404 `results_not_found`.
  - Progress went 0 → 0.095 → … → 1.0, and the job completed.
- **Artifacts:**
  - The annotated video is libx264, browser-playable, and loads and decodes in Chromium.
  - Range requests return 206.
  - `worker.log` and path traversal return 404.
- **Graceful shutdown during a job:** the worker was terminated, the job became `FAILED/interrupted`, and the partial video was removed.
- **SIGKILL of the API, then restart:** the job was reconciled to `FAILED/interrupted`, and the orphaned worker stopped itself with `job_abandoned`.
- **UI flow:** drop a file, upload, server validation, choose stride 2, analyze, live progress, results. The analysis page shows 210 of 419 frames analyzed, and skipped frames show "NOT ANALYZED" in the video.

**Measured results** (RF-DETR Nano, 384 px, confidence 0.5, CC BY 3.0 corner-kick clip):

| Device | Stride | Analyzed | Total detections | Mean per analyzed frame | Peak | Inference | End-to-end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MPS | 1 | 419 | 3,502 | 8.36 | 12 | 38.6 ms | 25.7 s |
| CPU | 1 | 419 | 3,502 | 8.36 | 12 | 58 ms | 32.3 s |
| MPS | 2 | 210 | 1,754 | 8.35 | 12 | 35.6 ms | 16.3 s |

No accuracy metrics (mAP, precision or recall) exist yet. Qualitatively, Nano at 384 px misses several small, distant players near the far goal, and it detects officials as `person`.

**Not verified:**
- Docker build: Docker isn't installed.
- CUDA: no NVIDIA GPU.
- GitHub Actions: never run, because that needs a push.
- The native file-picker dialog: browser automation can't use it, so uploads were tested with a scripted drag-and-drop.

---

## 3. Deviations from the original prompt and decisions to review

1. **Python ≥ 3.12, not 3.11.** PyAV 19 has no 3.11 wheels, which a real install on 3.11 confirmed.
2. **No OpenCV.** On macOS, opencv-python's bundled FFmpeg conflicts with PyAV's (duplicate libavdevice ObjC classes). Decoding and encoding use PyAV, and drawing uses Pillow. RF-DETR and supervision work without OpenCV.
3. **Subprocess-per-job workers**, not a long-lived worker pool. This gives isolation and the ability to kill a job, at the cost of about 6 s of model load per job.
4. **Frontend fetches all data client-side.** Pages are static shells under Next 16's `cacheComponents`.
5. **Licensing:**
   - PyAV is BSD, but its wheels bundle FFmpeg linked with libx264 and libx265, which are GPL. This matters if Docker images or binaries are redistributed, and it's documented.
   - Ultralytics (AGPL) is not used.
6. **Demo media:** small derived media (a 2.1 MB GIF, one JPEG comparison and four UI screenshots) is committed under `docs/media/` for the README, at the user's request. It is derived from the CC BY 3.0 clip, with attribution. No source footage, weights or full annotated videos are committed.

## 4. Known issues and open questions (please check)

Fixed during finalization (each with tests):

- **Corrupt weights were silently replaced.** With a truncated `rf-detr-nano.pth` and `--no-download`, rfdetr logged an MD5 mismatch, re-downloaded and overwrote the file, and the run succeeded. Pinned MD5s now live in `RFDETR_VARIANTS` (a test checks them against the installed rfdetr registry). With downloads disabled, a mismatch raises `model_unavailable` before rfdetr is imported, and the file is left untouched. The loaded file is re-verified after load, and warm-up failures now raise `model_unavailable` instead of an unclassified error.
- **The duration limit trusted the container.** The pipeline now also stops with `video_limit_exceeded` once decoded timestamps pass `max_duration_seconds`. No real file with a wrong duration was found (fragmented MP4s reported correctly), so this is defence in depth.
- **Hard-killed workers left `.annotated.mp4.partial` files.** Startup reconciliation now removes them for the jobs it marks `interrupted`.
- **The original player could drift from the annotated one.** While "Sync" is on, the original no longer shows its own controls, and re-enabling sync realigns time, rate and play state immediately.

- **Analysis-page loading delay (fixed in finalization).** Measured in headless desktop Chrome, detection statistics appeared 1.65–1.86 s after navigation. The cause was `usePolling`: when the video loaded, the results key changed, but the new key's first fetch was skipped because the old key's request was still in flight. It then waited a full 1.5 s poll interval, and the old response was discarded as stale. In-flight tracking is now per key, and statistics appear in 0.34–0.43 s. The remaining ~300 ms is React's intentional batching of Suspense reveals. Cursor's embedded browser throttles timers and animation frames, which stretched the same bug into multi-second stalls there.
- **One unexplained analyze request with confidence 0.65 (not reproduced).** During UI automation, one job (`a0237ca9…`) was submitted with `confidence_threshold: 0.65` although the slider displayed 0.50 just before. Every other job in the database used 0.50, the server default is 0.50, and replaying the steps submitted 0.50. One real mechanism for silently changed settings was found and fixed: if `/health` loaded after the form rendered, the settings panel remounted and reset the user's edits. It now adopts server defaults only if the user hasn't touched the form. It's still unproven whether that caused the 0.65 job.
- **Encoding sensitivity.** A CLI run on the original Ogg/Theora file gave 3,520 detections, against 3,502 on the H.264 conversion. Borderline detections flip with lossy re-encoding.
- **No retention policy.** `DELETE /api/v1/videos/{id}` and a Delete button now remove a video's rows, upload and artifacts (refused while a job is active), but nothing expires automatically.
- **No authentication.** It's a single-host design and not hardened for public exposure.
- **Starlette deprecation warning.** Starlette's TestClient warns that `httpx` should become `httpx2` in the future (only a warning).

## 5. Where to look (review guide)

| Concern | Files |
| --- | --- |
| Pipeline correctness, skipped-frame semantics, statistics | `packages/soccervision/pipeline/runner.py`, `schemas/detection.py`, `schemas/artifacts.py` |
| Timestamps, PTS, encoding | `packages/soccervision/vision/video/{probe,decoder,encoder}.py` |
| Model wrapper | `packages/soccervision/vision/detection/rfdetr.py` |
| Job state machine and concurrency | `apps/api/soccervision_api/persistence/database.py`, `services/jobs.py`, `worker.py`, `soccervision/schemas/job.py` |
| Upload security and limits | `apps/api/soccervision_api/services/uploads.py`, `middleware.py` |
| Artifact serving and path safety | `apps/api/soccervision_api/services/results.py`, `api/jobs.py` |
| Frontend honesty (labels, skipped frames) | `apps/web/src/features/analysis/{DetectionStats,DetectionTimeline,RunDetails}.tsx` |
| Tests | `tests/unit`, `tests/api`, `tests/integration`, `apps/web/src/lib/*.test.ts` |

## 6. Acceptance criteria (from the master prompt)

| # | Criterion | Status |
| --- | --- | --- |
| 1 | Backend starts | ✅ |
| 2 | Frontend starts | ✅ (dev and production) |
| 3 | Valid MP4 upload | ✅ |
| 4 | Metadata extracted correctly | ✅ |
| 5 | Analysis job can be submitted | ✅ |
| 6 | Actual pretrained inference | ✅ (RF-DETR Nano, MPS and CPU) |
| 7 | People shown with bounding boxes in output | ✅ |
| 8 | Confidence and coordinates persisted | ✅ (`detections.json`) |
| 9 | Job status queryable | ✅ |
| 10 | Failures surfaced meaningfully | ✅ (error codes, messages, UI hints) |
| 11 | Output video plays in the browser | ✅ (H.264; plays, seeks and stays in sync with the original in headless Chrome) |
| 12 | CLI analyzes a clip on its own | ✅ |
| 13 | Unit tests pass without GPU or model downloads | ✅ |
| 14 | Reproducible setup instructions | ✅ (Docker and CUDA paths untested) |
| 15 | README reflects what is implemented | ✅ |

## 7. Suggested next steps

1. **Commit Phase 1** (and rename or recreate the GitHub repo), then confirm that GitHub Actions passes.
2. **Spot-check in a headed desktop browser.** Reveal timing and playback sync were verified in headless Chrome, not by hand.
3. **Build the evaluation harness** (roadmap Phase 6, pulled forward). Measure detection mAP and recall by box size for N/S/M × input resolution on a small, legally usable labelled soccer set, before choosing the default model and starting tracking.
4. **Phase 2: multi-object tracking.** Use ByteTrack or BoT-SORT, producing `TrackObservation` with observed or predicted flags, evaluated with HOTA, IDF1 and ID switches. The `ObservationSource` enum and the per-frame `analyzed`/`skipped` records were designed to support this.
5. **Smaller improvements:**
   - A persistent model-loaded worker, to avoid the ~6 s load per job.
   - An automatic retention policy (manual deletion exists).
   - Optional tiled inference for small players.
