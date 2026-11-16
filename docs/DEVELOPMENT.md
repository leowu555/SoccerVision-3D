# Development

## Environment

| Tool | Version used | Notes |
| --- | --- | --- |
| Python | 3.13.12 (CI: 3.12, 3.13) | PyAV 19 has no Python 3.11 wheels, so `requires-python >= 3.12` |
| Node.js | 24.x (requires ≥ 22.12) | npm 11 |
| FFmpeg | Bundled in PyAV wheels | No system install needed |

Dependency ranges are pinned in `pyproject.toml` (Python) and `apps/web/package-lock.json` (frontend). Versions used during development include rfdetr 1.11.2, torch 2.15.0, torchvision 0.29.2, av 19.0.2, fastapi 0.162.0, pydantic 2.14.3, next 16.4.5 and react 19.3.0.

### Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,api,eval,dev]"  # drop `ml` to work on API/frontend without PyTorch
cd apps/web && npm ci
```

`ml` is the only heavy extra. Without it, all unit and API tests still pass, because they use fake detectors, and `/health` reports `ml_dependencies_installed: false`.

The editable install maps both `packages/` and `apps/api/` onto `sys.path`. If you add a new top-level package directory, re-run `pip install -e .`.

### Running locally

```bash
uvicorn soccervision_api.main:app --reload --port 8000     # from repo root
cd apps/web && npm run dev                                  # http://localhost:3000
```

`--reload` restarts the API on code changes. Any job that is running at that moment is marked `FAILED / interrupted`, by design.

### Device notes

- `device=auto` resolves to CUDA, then MPS, then CPU. Asking for an unavailable device fails the job instead of silently falling back.
- **MPS:** the first inference is slow (about 2 s), so the detector does a warm-up prediction while loading. On the development machine (Apple Silicon), RF-DETR Nano took about 35–40 ms per 720p frame on MPS and about 58–80 ms on CPU.
- **CUDA** has not been exercised in development, because no NVIDIA GPU was available. The code path is the standard PyTorch device string.
- Do not install `opencv-python` into the same environment on macOS. Its bundled FFmpeg clashes with PyAV's (duplicate `libavdevice` Objective-C classes). Nothing in this project needs it.

## Checks

```bash
ruff check . && ruff format --check .
mypy                                   # strict; covers packages/, apps/api/, scripts/
pytest                                 # ~100 tests, a few seconds, no network/GPU

cd apps/web
npm run lint && npm run typecheck && npm test && npm run build
```

CI (`.github/workflows/ci.yml`) runs the same checks on Ubuntu, without the `ml` extra and without a GPU.

### Real-inference integration test (opt-in)

```bash
SOCCERVISION_RUN_INTEGRATION=1 \
SOCCERVISION_INTEGRATION_VIDEO=data/samples/fc_utrecht_corner.mp4 \
SOCCERVISION_WEIGHTS_DIR=models \
pytest tests/integration -v
# Set SOCCERVISION_INTEGRATION_ALLOW_DOWNLOAD=1 to let it fetch weights.
```

This test asserts structural properties of real model output (non-synthetic model info, the person class only, valid boxes, consistent statistics). It does not compare against hard-coded detections.

## Test layout

| Directory | What it covers |
| --- | --- |
| `tests/unit/` | Schemas and state machine, video probe, decoder and encoder, pipeline (with a fake detector), devices, CLI |
| `tests/api/` | Uploads (validation, limits), jobs (dispatch, bounded concurrency, crashes, reconciliation, shutdown), results and artifacts |
| `tests/integration/` | Real RF-DETR inference (skipped by default) |
| `apps/web/src/lib/*.test.ts` | API client error handling, formatting, validation, timeline helpers |

`tests/support.py` generates tiny synthetic MP4s with a moving white rectangle and provides `FakeDetector`. Fake outputs are flagged `is_synthetic: true` and must never be presented as model results.

## Conventions

- Keep the core (`packages/soccervision`) free of web dependencies. The API and CLI are thin layers over `run_detection_pipeline`.
- Every user-facing failure is a `SoccerVisionError` subclass or an `ApiError`, with a stable `code`.
- Any change to artifact JSON bumps `ARTIFACT_SCHEMA_VERSION` and updates [DATA_CONTRACTS.md](DATA_CONTRACTS.md) and `apps/web/src/types/api.ts`.
- Footage, weights and generated artifacts stay out of Git, which `.gitignore` enforces. Only use footage you have the right to use, and never commit match broadcasts.
- Adding a dependency means checking its license first. **Ultralytics is AGPL-3.0**, and adopting it requires documenting the implications and keeping it optional.

## Frontend notes

- Next.js 16 with `cacheComponents` enabled. Dynamic route params are awaited inside a `<Suspense>` boundary, and all API data is fetched client-side.
- `apps/web/AGENTS.md`: Next.js 16 APIs differ from older versions, so consult `node_modules/next/dist/docs/` when in doubt.
- `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`) is inlined at build time.
