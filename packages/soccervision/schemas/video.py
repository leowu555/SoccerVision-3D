from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FrameRateMode = Literal["constant", "variable", "unknown"]


class VideoProcessingStatus(StrEnum):
    """Status of a video, derived from its most recent analysis job."""

    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoProbe(BaseModel):
    """Facts read from the media container and its primary video stream."""

    model_config = ConfigDict(frozen=True)

    container_format: str = Field(
        description="FFmpeg demuxer name, e.g. 'mov,mp4,m4a,3gp,3g2,mj2'."
    )
    video_codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0, description="Average frame rate reported by the container.")
    duration_seconds: float = Field(ge=0)
    frame_count: int | None = Field(
        default=None, ge=1, description="Frame count from the container index; None when unknown."
    )
    time_base: str = Field(description="Stream time base as 'num/den'.")
    start_time_seconds: float = Field(
        default=0.0,
        description="Presentation time of the first packet; subtracted from frame timestamps.",
    )
    frame_rate_mode: FrameRateMode = "unknown"
    has_audio: bool = False
    size_bytes: int = Field(ge=0)


class VideoMetadata(BaseModel):
    """A stored, validated input video."""

    video_id: str
    filename: str = Field(description="Sanitized original filename, for display only.")
    duration_seconds: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int | None = Field(default=None, ge=1)
    created_at: datetime
    source_path: str = Field(description="Internal storage path. Never derived from user input.")
    processing_status: VideoProcessingStatus = VideoProcessingStatus.UPLOADED
    container_format: str
    video_codec: str
    size_bytes: int = Field(ge=0)
    frame_rate_mode: FrameRateMode = "unknown"
    has_audio: bool = False
