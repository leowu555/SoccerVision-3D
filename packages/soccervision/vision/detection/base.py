from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from soccervision.config import DetectorConfig
from soccervision.schemas.detection import ModelInfo


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A detector output before pipeline post-processing (class filter, clipping, IDs)."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    """Source-frame pixel coordinates; may extend slightly beyond the frame."""


class ObjectDetector(Protocol):
    """Single-image detector. Implementations must accept RGB uint8 arrays of shape (H, W, 3)."""

    @property
    def info(self) -> ModelInfo: ...

    def detect(self, image_rgb: npt.NDArray[np.uint8]) -> Sequence[RawDetection]: ...


DetectorFactory = Callable[[DetectorConfig], ObjectDetector]
