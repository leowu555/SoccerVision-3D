"""Pipeline orchestration. Phase 1: decode -> detect -> annotate -> export."""

from soccervision.pipeline.artifacts import (
    ANNOTATED_VIDEO_FILE,
    DETECTIONS_FILE,
    METADATA_FILE,
    load_detections,
    load_run_metadata,
)
from soccervision.pipeline.runner import (
    PIPELINE_NAME,
    PIPELINE_STAGES,
    PIPELINE_VERSION,
    PipelineResult,
    PipelineRunError,
    ProgressCallback,
    ProgressUpdate,
    run_detection_pipeline,
)

__all__ = [
    "ANNOTATED_VIDEO_FILE",
    "DETECTIONS_FILE",
    "METADATA_FILE",
    "PIPELINE_NAME",
    "PIPELINE_STAGES",
    "PIPELINE_VERSION",
    "PipelineResult",
    "PipelineRunError",
    "ProgressCallback",
    "ProgressUpdate",
    "load_detections",
    "load_run_metadata",
    "run_detection_pipeline",
]
