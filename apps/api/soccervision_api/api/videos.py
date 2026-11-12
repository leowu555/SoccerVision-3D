from typing import Annotated

from fastapi import APIRouter, Body, File, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from soccervision.config import DetectorConfig, PipelineConfig
from soccervision.schemas.job import JobStatus
from soccervision_api.api.deps import DbDep, RunnerDep, SettingsDep, require_video
from soccervision_api.errors import ApiError
from soccervision_api.persistence import JobConflictError
from soccervision_api.schemas import (
    AnalyzeRequest,
    JobResponse,
    ResultsResponse,
    UploadResponse,
    VideoListResponse,
    VideoResponse,
)
from soccervision_api.services import results as results_service
from soccervision_api.services.jobs import submit_job
from soccervision_api.services.uploads import delete_video, source_path, store_upload
from soccervision_api.settings import Settings

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", status_code=201, response_model=UploadResponse)
def upload_video(
    file: Annotated[UploadFile, File(description="MP4 video")], settings: SettingsDep, db: DbDep
) -> UploadResponse:
    video = store_upload(stream=file.file, filename=file.filename, settings=settings, db=db)
    return UploadResponse(
        video_id=video.video_id,
        original_filename=video.filename,
        metadata=VideoResponse.from_video(video),
    )


@router.get("", response_model=VideoListResponse)
def list_videos(db: DbDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> VideoListResponse:
    return VideoListResponse(
        videos=[
            VideoResponse.from_video(v, db.latest_job(v.video_id)) for v in db.list_videos(limit)
        ]
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: DbDep) -> VideoResponse:
    video = require_video(db, video_id)
    return VideoResponse.from_video(video, db.latest_job(video_id))


@router.delete("/{video_id}", status_code=204)
def remove_video(video_id: str, db: DbDep, settings: SettingsDep) -> Response:
    require_video(db, video_id)
    try:
        if not delete_video(video_id=video_id, settings=settings, db=db):
            raise ApiError(404, "video_not_found", f"Video {video_id} does not exist.")
    except JobConflictError as exc:
        raise ApiError(
            409,
            "job_already_active",
            "This video has a queued or running analysis job; wait for it to finish.",
            {"job_id": exc.active_job_id},
        ) from exc
    return Response(status_code=204)


@router.get("/{video_id}/source", response_class=FileResponse)
def get_video_source(video_id: str, db: DbDep, settings: SettingsDep) -> FileResponse:
    video = require_video(db, video_id)
    path = source_path(settings, video)
    if not path.is_file():
        raise ApiError(410, "source_missing", "The uploaded source file no longer exists on disk.")
    return FileResponse(path, media_type="video/mp4")


def _pipeline_config(settings: Settings, request: AnalyzeRequest) -> PipelineConfig:
    try:
        return PipelineConfig(
            detector=DetectorConfig(
                name=request.detector or settings.default_detector,
                confidence_threshold=(
                    request.confidence_threshold
                    if request.confidence_threshold is not None
                    else settings.default_confidence_threshold
                ),
                inference_resolution=request.inference_resolution,
                device=request.device or settings.default_device,
            ),
            inference_stride=request.inference_stride or settings.default_inference_stride,
            max_duration_seconds=settings.max_video_duration_seconds,
        )
    except ValidationError as exc:
        raise ApiError(
            422, "invalid_configuration", "Invalid analysis configuration.", exc.errors()
        ) from exc


@router.post("/{video_id}/analyze", status_code=202, response_model=JobResponse)
def analyze_video(
    video_id: str,
    db: DbDep,
    settings: SettingsDep,
    runner: RunnerDep,
    request: Annotated[AnalyzeRequest | None, Body()] = None,
) -> JobResponse:
    require_video(db, video_id)
    config = _pipeline_config(settings, request or AnalyzeRequest())
    try:
        job = submit_job(db, video_id, config)
    except JobConflictError as exc:
        raise ApiError(
            409,
            "job_already_active",
            "An analysis job is already queued or running for this video.",
            {"job_id": exc.active_job_id},
        ) from exc
    runner.wake()
    return JobResponse.from_job(job)


@router.get("/{video_id}/results", response_model=ResultsResponse)
def get_results(
    video_id: str,
    db: DbDep,
    settings: SettingsDep,
    job_id: Annotated[
        str | None, Query(description="Specific run; defaults to the latest completed one.")
    ] = None,
) -> ResultsResponse:
    video = require_video(db, video_id)
    if job_id is not None:
        job = db.get_job(job_id)
        if job is None or job.video_id != video_id:
            raise ApiError(404, "job_not_found", f"Job {job_id} does not exist for this video.")
        if job.status is not JobStatus.COMPLETED:
            raise ApiError(
                409,
                "results_not_ready",
                f"Job {job_id} is {job.status.value}; results exist only for completed jobs.",
            )
    else:
        job = db.latest_job(video_id, JobStatus.COMPLETED)
        if job is None:
            latest = db.latest_job(video_id)
            detail = (
                f" Latest job is {latest.status.value}."
                if latest
                else " No analysis has been requested."
            )
            raise ApiError(
                404, "results_not_found", f"No completed analysis for this video.{detail}"
            )

    metadata = results_service.run_metadata(
        results_service.resolve_artifact(settings, job, "metadata")
    )
    counts = results_service.frame_counts(
        results_service.resolve_artifact(settings, job, "detections")
    )
    return ResultsResponse(
        video=VideoResponse.from_video(video, db.latest_job(video_id)),
        job=JobResponse.from_job(job),
        run=metadata,
        frame_counts=counts,
    )
