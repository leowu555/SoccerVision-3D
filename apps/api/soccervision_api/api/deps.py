from typing import Annotated

from fastapi import Depends, Request

from soccervision.schemas.job import PipelineJob
from soccervision.schemas.video import VideoMetadata
from soccervision_api.errors import ApiError
from soccervision_api.persistence import Database
from soccervision_api.services.jobs import JobRunner
from soccervision_api.settings import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def get_runner(request: Request) -> JobRunner:
    runner: JobRunner = request.app.state.runner
    return runner


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Database, Depends(get_db)]
RunnerDep = Annotated[JobRunner, Depends(get_runner)]


def require_video(db: Database, video_id: str) -> VideoMetadata:
    video = db.get_video(video_id)
    if video is None:
        raise ApiError(404, "video_not_found", f"Video {video_id} does not exist.")
    return video


def require_job(db: Database, job_id: str) -> PipelineJob:
    job = db.get_job(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"Job {job_id} does not exist.")
    return job
