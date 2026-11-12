"""HTTP request/response models. Internal storage paths are never exposed; URLs are relative."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from soccervision.config import DetectorName, DeviceName
from soccervision.schemas.artifacts import RunMetadata
from soccervision.schemas.job import JobStatus, PipelineJob
from soccervision.schemas.video import FrameRateMode, VideoMetadata, VideoProcessingStatus

API_PREFIX = "/api/v1"
ARTIFACT_NAMES = {
    "annotated_video": "annotated.mp4",
    "detections": "detections.json",
    "metadata": "metadata.json",
}


class JobResponse(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus
    progress: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None
    pipeline_version: str
    configuration: dict[str, object]
    artifact_urls: dict[str, str]

    @classmethod
    def from_job(cls, job: PipelineJob) -> "JobResponse":
        urls = {
            name: f"{API_PREFIX}/jobs/{job.job_id}/artifacts/{ARTIFACT_NAMES[name]}"
            for name in job.artifact_paths
            if name in ARTIFACT_NAMES
        }
        return cls(
            job_id=job.job_id,
            video_id=job.video_id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_code=job.error_code,
            error_message=job.error_message,
            pipeline_version=job.pipeline_version,
            configuration=job.configuration,
            artifact_urls=urls,
        )


class VideoResponse(BaseModel):
    video_id: str
    filename: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    created_at: datetime
    processing_status: VideoProcessingStatus
    container_format: str
    video_codec: str
    size_bytes: int
    frame_rate_mode: FrameRateMode
    has_audio: bool
    source_url: str
    latest_job: JobResponse | None = None

    @classmethod
    def from_video(
        cls, video: VideoMetadata, latest_job: PipelineJob | None = None
    ) -> "VideoResponse":
        return cls(
            **video.model_dump(exclude={"source_path"}),
            source_url=f"{API_PREFIX}/videos/{video.video_id}/source",
            latest_job=JobResponse.from_job(latest_job) if latest_job else None,
        )


class UploadResponse(BaseModel):
    video_id: str
    original_filename: str
    upload_status: Literal["stored"] = "stored"
    metadata: VideoResponse


class VideoListResponse(BaseModel):
    videos: list[VideoResponse]


class AnalyzeRequest(BaseModel):
    """Optional per-job overrides of the server defaults."""

    model_config = ConfigDict(extra="forbid")

    detector: DetectorName | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    inference_stride: int | None = Field(default=None, ge=1, le=300)
    inference_resolution: int | None = Field(default=None, ge=64, le=2048)
    device: DeviceName | None = None


class FrameCount(BaseModel):
    frame_index: int
    timestamp_seconds: float
    detections: int


class ResultsResponse(BaseModel):
    video: VideoResponse
    job: JobResponse
    run: RunMetadata
    frame_counts: list[FrameCount] = Field(
        description="Detections per *analyzed* frame. Skipped frames are omitted (unknown, not zero)."
    )


class DeviceStatus(BaseModel):
    torch_installed: bool
    cuda: bool
    mps: bool
    cpu: bool


class ModelStatus(BaseModel):
    detector: DetectorName
    ml_dependencies_installed: bool
    weights_present: bool
    weights_download_allowed: bool


class Limits(BaseModel):
    max_upload_bytes: int
    max_video_duration_seconds: float
    max_concurrent_jobs: int


class AnalysisDefaults(BaseModel):
    detector: DetectorName
    device: DeviceName
    confidence_threshold: float
    inference_stride: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    pipeline_name: str
    pipeline_version: str
    devices: DeviceStatus
    model: ModelStatus
    limits: Limits
    defaults: AnalysisDefaults
