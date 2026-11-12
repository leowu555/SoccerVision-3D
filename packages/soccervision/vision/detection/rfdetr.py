"""RF-DETR adapter (Roboflow, Apache-2.0; COCO-pretrained Nano/Small/Medium weights are Apache-2.0).

``rfdetr`` and ``torch`` are imported lazily so the rest of the system works without them.
"""

import hashlib
import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from soccervision.config import DetectorConfig, DetectorName
from soccervision.errors import InferenceError, ModelUnavailableError
from soccervision.schemas.detection import ModelInfo
from soccervision.vision.detection.base import RawDetection

logger = logging.getLogger(__name__)

# name -> (rfdetr class, weights filename the rfdetr registry knows how to download)
RFDETR_VARIANTS: dict[DetectorName, tuple[str, str]] = {
    "rfdetr-nano": ("RFDETRNano", "rf-detr-nano.pth"),
    "rfdetr-small": ("RFDETRSmall", "rf-detr-small.pth"),
    "rfdetr-medium": ("RFDETRMedium", "rf-detr-medium.pth"),
}


def _package_versions(*names: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RFDETRDetector:
    def __init__(
        self,
        config: DetectorConfig,
        *,
        device: str,
        weights_dir: Path,
        allow_download: bool,
    ) -> None:
        class_name, weights_filename = RFDETR_VARIANTS[config.name]
        weights_path = weights_dir / weights_filename
        if not weights_path.is_file() and not allow_download:
            raise ModelUnavailableError(
                f"Weights for {config.name} not found at {weights_path} and downloads are disabled. "
                "Run `python scripts/download_weights.py` or enable downloads."
            )

        try:
            rfdetr = importlib.import_module("rfdetr")
        except ImportError as exc:
            raise ModelUnavailableError(
                "RF-DETR is not installed. Install ML extras: pip install -e '.[ml]'"
            ) from exc

        weights_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model: Any = getattr(rfdetr, class_name)(
                pretrain_weights=str(weights_path), device=device
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load {config.name} on {device}: {exc}") from exc

        model_config = self._model.model_config
        self._shape: tuple[int, int] | None = None
        resolution = int(model_config.resolution)
        if config.inference_resolution is not None:
            divisor = int(model_config.patch_size) * int(model_config.num_windows)
            if config.inference_resolution % divisor:
                raise ModelUnavailableError(
                    f"inference_resolution={config.inference_resolution} must be divisible by {divisor} "
                    f"for {config.name}."
                )
            resolution = config.inference_resolution
            self._shape = (resolution, resolution)

        self._threshold = config.confidence_threshold
        self._coco_classes = self._load_coco_classes()
        self._info = ModelInfo(
            model_id=f"{config.name}-coco",
            family="rf-detr",
            variant=config.name,
            label_space="coco-80",
            license="Apache-2.0",
            weights_file=weights_filename,
            weights_md5=_md5(weights_path),
            input_resolution=resolution,
            device=device,
            framework_versions=_package_versions("rfdetr", "torch", "torchvision", "supervision"),
        )
        # First inference compiles kernels (notably on MPS); keep it out of per-frame timings.
        self._predict(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    @property
    def info(self) -> ModelInfo:
        return self._info

    @staticmethod
    def _load_coco_classes() -> dict[int, str]:
        try:
            module = importlib.import_module("rfdetr.assets.coco_classes")
            return dict(module.COCO_CLASSES)
        except (ImportError, AttributeError):
            return {}

    def _predict(self, image_rgb: npt.NDArray[np.uint8]) -> Any:
        kwargs: dict[str, Any] = {"threshold": self._threshold, "include_source_image": False}
        if self._shape is not None:
            kwargs["shape"] = self._shape
        return self._model.predict(image_rgb, **kwargs)

    def detect(self, image_rgb: npt.NDArray[np.uint8]) -> Sequence[RawDetection]:
        try:
            result = self._predict(image_rgb)
        except Exception as exc:
            raise InferenceError(f"RF-DETR inference failed: {exc}") from exc

        xyxy = np.asarray(result.xyxy, dtype=float).reshape(-1, 4)
        confidence = np.asarray(result.confidence, dtype=float).reshape(-1)
        class_ids = np.asarray(result.class_id).reshape(-1)
        names = result.data.get("class_name") if getattr(result, "data", None) else None

        detections: list[RawDetection] = []
        for i in range(len(xyxy)):
            label = (
                str(names[i]) if names is not None else self._coco_classes.get(int(class_ids[i]))
            )
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            detections.append(RawDetection(label, float(confidence[i]), (x1, y1, x2, y2)))
        return detections
