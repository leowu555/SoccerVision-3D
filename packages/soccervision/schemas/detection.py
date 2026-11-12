"""Detection-stage schemas.

Coordinate convention (all image-space schemas): pixel coordinates in the *decoded source frame*,
origin at the top-left corner, x increasing rightwards, y increasing downwards. Boxes are
``(x_min, y_min, x_max, y_max)`` with continuous (float) edges, so a box covering the whole frame
is ``(0, 0, width, height)``.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

TimestampSource = Literal["pts", "estimated"]


class InferenceStatus(StrEnum):
    ANALYZED = "analyzed"
    """The detector ran on this frame. An empty detection list means "nothing above threshold"."""

    SKIPPED = "skipped"
    """The detector did not run (frame sampling). Detections are unknown, not absent."""


class ObservationSource(StrEnum):
    """Provenance of an observation. Only MODEL_DETECTION is produced in Phase 1."""

    MODEL_DETECTION = "model_detection"
    TRACKER_PREDICTION = "tracker_prediction"
    GEOMETRIC_PROJECTION = "geometric_projection"
    INTERPOLATION = "interpolation"


def _validate_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if min(box) < 0:
        raise ValueError(f"bbox coordinates must be non-negative, got {box}")
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"bbox must satisfy x_max > x_min and y_max > y_min, got {box}")
    return box


BBoxXYXY = Annotated[tuple[float, float, float, float], AfterValidator(_validate_xyxy)]


class ModelInfo(BaseModel):
    """Identity of the model that produced detections, recorded for reproducibility."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Stable identifier, e.g. 'rfdetr-nano-coco'.")
    family: str
    variant: str
    label_space: str = Field(
        description="Class vocabulary the model was trained on, e.g. 'coco-80'."
    )
    license: str
    weights_file: str | None = None
    weights_md5: str | None = None
    input_resolution: int | None = None
    device: str
    framework_versions: dict[str, str] = Field(default_factory=dict)
    is_synthetic: bool = Field(
        default=False,
        description="True only for test doubles. Synthetic outputs must never be presented as model results.",
    )


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    frame_index: int = Field(ge=0, description="0-based index in decode (presentation) order.")
    timestamp_seconds: float = Field(
        ge=0, description="Presentation time relative to the first frame."
    )
    detection_id: str = Field(description="Unique within a run: 'f{frame_index}-d{k}'.")
    class_name: str = Field(description="Model label, e.g. 'person'. Not a soccer role.")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: BBoxXYXY
    model_id: str
    source: ObservationSource = ObservationSource.MODEL_DETECTION


class FrameRecord(BaseModel):
    """Per-frame outcome. Every decoded frame gets one record, analyzed or not."""

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    timestamp_source: TimestampSource = "pts"
    inference_status: InferenceStatus
    detections: list[Detection] | None = Field(
        description="Detections for analyzed frames; null for skipped frames (unknown, not empty)."
    )

    @model_validator(mode="after")
    def _status_matches_detections(self) -> Self:
        if self.inference_status is InferenceStatus.ANALYZED and self.detections is None:
            raise ValueError("analyzed frames must carry a detections list (possibly empty)")
        if self.inference_status is InferenceStatus.SKIPPED and self.detections is not None:
            raise ValueError("skipped frames must have detections=null")
        for det in self.detections or []:
            if det.frame_index != self.frame_index:
                raise ValueError("detection frame_index does not match its frame record")
        return self
