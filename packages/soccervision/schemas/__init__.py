"""Versioned data contracts shared by the pipeline, CLI, API, and artifacts.

See docs/DATA_CONTRACTS.md for the human-readable specification.
"""

from soccervision.schemas.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    CoordinateSystem,
    DetectionsArtifact,
    EnvironmentInfo,
    ErrorInfo,
    OutputVideoInfo,
    PipelineInfo,
    RunMetadata,
    RunStats,
    RunStatus,
    RunTiming,
)
from soccervision.schemas.detection import (
    BBoxXYXY,
    Detection,
    FrameRecord,
    InferenceStatus,
    ModelInfo,
    ObservationSource,
    TimestampSource,
)
from soccervision.schemas.job import (
    InvalidJobTransitionError,
    JobStatus,
    PipelineJob,
    ensure_transition,
    is_terminal,
)
from soccervision.schemas.video import (
    FrameRateMode,
    VideoMetadata,
    VideoProbe,
    VideoProcessingStatus,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BBoxXYXY",
    "CoordinateSystem",
    "Detection",
    "DetectionsArtifact",
    "EnvironmentInfo",
    "ErrorInfo",
    "FrameRateMode",
    "FrameRecord",
    "InferenceStatus",
    "InvalidJobTransitionError",
    "JobStatus",
    "ModelInfo",
    "ObservationSource",
    "OutputVideoInfo",
    "PipelineInfo",
    "PipelineJob",
    "RunMetadata",
    "RunStats",
    "RunStatus",
    "RunTiming",
    "TimestampSource",
    "VideoMetadata",
    "VideoProbe",
    "VideoProcessingStatus",
    "ensure_transition",
    "is_terminal",
]
