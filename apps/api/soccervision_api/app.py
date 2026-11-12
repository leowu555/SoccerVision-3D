"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from soccervision import __version__
from soccervision_api.api import router
from soccervision_api.errors import install_error_handlers
from soccervision_api.middleware import BodySizeLimitMiddleware
from soccervision_api.persistence import Database
from soccervision_api.schemas import API_PREFIX
from soccervision_api.services.jobs import JobRunner
from soccervision_api.settings import Settings

logger = logging.getLogger(__name__)

# Allowance for multipart boundaries and headers on top of the file itself.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


def create_app(
    settings: Settings | None = None,
    *,
    runner: JobRunner | None = None,
    start_runner: bool = True,
) -> FastAPI:
    settings = (settings or Settings()).resolved()
    settings.ensure_directories()
    db = Database(settings.database_path)
    db.initialize()
    job_runner = runner or JobRunner(db, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if start_runner:
            job_runner.start()
        logger.info("SoccerVision API ready (data_dir=%s)", settings.data_dir)
        try:
            yield
        finally:
            if start_runner:
                job_runner.stop()

    app = FastAPI(
        title="SoccerVision 3D API",
        version=__version__,
        description="Phase 1: upload soccer video, run pretrained person detection, retrieve artifacts.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.runner = job_runner

    install_error_handlers(app)
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
        path_prefixes=(f"{API_PREFIX}/videos",),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    app.include_router(router)
    return app
