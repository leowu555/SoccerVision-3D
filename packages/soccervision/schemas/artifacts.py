"""On-disk artifact schemas (``detections.json`` and ``metadata.json``).

Bump ``ARTIFACT_SCHEMA_VERSION`` on any change: minor for additive fields, major for breaking ones.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from soccervision.schemas.detection import FrameRecord, ModelInfo
from soccervision.schemas.video import VideoProbe

ARTIFACT_SCHEMA_VERSION = "1.0.0"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class CoordinateSystem(BaseModel):
    type: Literal["image_pixels"] = "image_pixels"
    origin: Literal["top_left"] = "top_left"
    bbox_format: Literal["xyxy"] = "xyxy"
    units: Literal["pixels"] = "pixels"
    frame_width: int
    frame_height: int


class PipelineInfo(BaseModel):
    name: str
    version: str
    package_version: str
    stages: list[str]


class RunStats(BaseModel):
    decoded_frames: int = Field(description="Frames decoded from the source.")
    analyzed_frames: int = Field(description="Frames the detector actually ran on.")
    skipped_frames: int = Field(description="Decoded frames not analyzed due to inference_stride.")
    total_detections: int = Field(
        description="Sum of detections over analyzed frames. NOT a count of unique people."
    )
    mean_detections_per_analyzed_frame: float | None
    max_detections_in_frame: int
    frames_with_estimated_timestamps: int
    clipped_boxes: int = Field(
        description="Boxes partly outside the frame that were clipped to it."
    )
    discarded_degenerate_boxes: int


class RunTiming(BaseModel):
    started_at: datetime
    completed_at: datetime
    total_seconds: float
    model_load_seconds: float | None = None
    inference_seconds: float = Field(description="Wall time inside detector calls.")
    mean_inference_ms: float | None = None
    processing_fps: float | None = Field(
        default=None, description="Decoded frames per wall-clock second over the whole run."
    )


class OutputVideoInfo(BaseModel):
    file: str
    codec: str
    pixel_format: str
    width: int
    height: int
    browser_playable: bool
    padded_to_even_dimensions: bool = False


class EnvironmentInfo(BaseModel):
    python: str
    platform: str
    packages: dict[str, str]


class ErrorInfo(BaseModel):
    code: str
    message: str
    frame_index: int | None = None


class DetectionsArtifact(BaseModel):
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    artifact_type: Literal["detections"] = "detections"
    run_id: str
    video_id: str
    model: ModelInfo
    confidence_threshold: float
    target_classes: list[str]
    inference_stride: int
    coordinate_system: CoordinateSystem
    frames: list[FrameRecord]


class RunMetadata(BaseModel):
    """Processing metadata and execution summary for one run (``metadata.json``)."""

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    artifact_type: Literal["run_metadata"] = "run_metadata"
    run_id: str
    video_id: str
    status: RunStatus
    pipeline: PipelineInfo
    configuration: dict[str, Any]
    model: ModelInfo | None
    input_file: str
    input_sha256: str | None
    input_video: VideoProbe | None
    output_video: OutputVideoInfo | None
    stats: RunStats | None
    timing: RunTiming
    environment: EnvironmentInfo
    artifacts: dict[str, str] = Field(
        description="Artifact name -> file name within the run directory."
    )
    warnings: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None
