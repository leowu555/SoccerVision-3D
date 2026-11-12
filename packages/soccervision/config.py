"""Processing configuration.

Everything that can change the numerical output of a run lives here and is recorded verbatim in
the run metadata. Environment concerns (where weights are stored, whether downloads are allowed)
are deliberately kept out of this model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DeviceName = Literal["auto", "cpu", "mps", "cuda"]
DetectorName = Literal["rfdetr-nano", "rfdetr-small", "rfdetr-medium"]


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: DetectorName = Field(
        default="rfdetr-nano",
        description="Pretrained detector. All listed RF-DETR sizes are Apache-2.0 licensed.",
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Detections below this score are discarded."
    )
    target_classes: tuple[str, ...] = Field(
        default=("person",),
        min_length=1,
        description="Model class names to keep. Phase 1 keeps only COCO 'person'.",
    )
    inference_resolution: int | None = Field(
        default=None,
        ge=64,
        le=2048,
        description=(
            "Square model input size in pixels. None uses the model's native resolution. "
            "Must be divisible by the model's patch_size * num_windows (32 for RF-DETR N/S/M)."
        ),
    )
    device: DeviceName = Field(
        default="auto", description="Compute device; 'auto' picks CUDA > MPS > CPU."
    )


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    inference_stride: int = Field(
        default=1,
        ge=1,
        le=300,
        description="Run inference on every Nth decoded frame. Other frames are marked 'skipped'.",
    )
    max_duration_seconds: float | None = Field(
        default=600.0,
        gt=0,
        description="Reject inputs longer than this. None disables the check.",
    )
    draw_hud: bool = Field(default=True, description="Draw the frame/status overlay on the output.")
