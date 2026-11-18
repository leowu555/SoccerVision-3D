# Development

## Environment

| Tool | Version used | Notes |
| --- | --- | --- |
| Python | 3.13.12 (CI: 3.12, 3.13) | PyAV 19 has no Python 3.11 wheels, so `requires-python >= 3.12` |
| Node.js | 24.x (requires ≥ 22.12) | npm 11 |
| FFmpeg | Bundled in PyAV wheels | No system install needed |

`pyproject.toml` declares compatible version **ranges** for Python dependencies (for example `av>=19.0.2,<20`). They are not exact pins, so a fresh install can resolve newer versions within those ranges. The frontend's exact versions are locked in `apps/web/package-lock.json`, and `npm ci` installs exactly those. As of November 2026, development used rfdetr 1.11.2, torch 2.15.0, torchvision 0.29.2, av 19.0.2, fastapi 0.162.0, pydantic 2.14.3, pycocotools 2.0.11, next 16.4.5 and react 19.3.0.

### Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,api,eval,dev]"  # drop `ml` to work on API/frontend without PyTorch
cd apps/web && npm ci
```

| Extra | Provides | Needed for |
| --- | --- | --- |
| `ml` | torch, torchvision, rfdetr | Real inference (API jobs, CLI, integration tests). The only heavy extra. |
| `api` | FastAPI, pydantic-settings, uvicorn | Running the API; API tests are skipped without it |
| `eval` | pycocotools | Detection metrics; metric tests are skipped without it |
| `dev` | pytest, ruff, mypy, httpx | Checks and tests |

Without `ml`, all unit and API tests still pass, because they use fake detectors, and `/health` reports `ml_dependencies_installed: false`.

The editable install maps both `packages/` and `apps/api/` onto `sys.path` through absolute paths. If you add a new top-level package directory, re-run `pip install -e .`. If you move or rename the repository, see [Troubleshooting](#troubleshooting).

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
mypy                                   # strict; the soccervision and soccervision_api packages
pytest                                 # unit, API and evaluation tests; a few seconds, no network/GPU

cd apps/web
npm run lint && npm run typecheck && npm test && npm run build
```

`mypy` checks only the packages listed in `pyproject.toml`; `scripts/` and `tests/` are not type-checked. At the time of writing (November 2026), `pytest` reports 162 passed and 1 skipped (the opt-in integration module), and Vitest runs 26 tests. Treat these counts as a snapshot, not a contract.

CI (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

- **Backend:** Ubuntu with Python 3.12 and 3.13. It installs `.[api,eval,dev]` (no `ml` extra, no GPU) and runs `ruff check .`, `ruff format --check .`, `mypy` and `pytest`.
- **Frontend:** Node 24. It runs `npm ci`, then `lint`, `typecheck`, `test` and `build`.

`ruff format --check .` also formats Python code blocks inside Markdown files, so documentation examples must be Ruff-formatted too.

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
| `tests/unit/` | Schemas and state machine, video probe, decoder and encoder, pipeline (with a fake detector), devices, CLI, RF-DETR weight integrity, COCO dataset validation and detection metrics |
| `tests/api/` | Uploads (validation, limits), jobs (dispatch, bounded concurrency, crashes, reconciliation, shutdown), results and artifacts |
| `tests/integration/` | Real RF-DETR inference (skipped by default) |
| `tests/fixtures/evaluation/` | A hand-built synthetic COCO dataset, labelled `SYNTHETIC` |
| `apps/web/src/lib/*.test.ts` | API client error handling, formatting, validation, timeline helpers |

`tests/support.py` generates tiny synthetic MP4s with a moving white rectangle, provides `FakeDetector`, and builds synthetic COCO documents. Fake outputs are flagged `is_synthetic: true`, and synthetic datasets carry `synthetic: true` provenance. Neither may be presented as model results or accuracy.

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

## Troubleshooting

### `.venv` breaks after moving or renaming the repository

A virtual environment stores absolute paths: in its `activate` script, in the shebang lines of its console scripts (`pytest`, `ruff`, `mypy`, …), and in the editable install's `.pth` file. If the repository directory is moved or renamed, these still point to the old location. Symptoms include:

- `pytest` fails while loading `tests/conftest.py` with `ModuleNotFoundError: No module named 'soccervision'`.
- After `source .venv/bin/activate`, `ruff`, `mypy` or `pytest` are not found, or run from the wrong environment.

To check, run `.venv/bin/python -c "import soccervision; print(soccervision.__file__)"`. The import should succeed and print a path inside the current repository.

The reliable fix is to recreate the environment from the repository root:

```bash
rm -rf .venv
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,api,eval,dev]"
```

Re-running only `pip install -e .` fixes the import path, but not the `activate` script or the console-script shebangs.
