"""RF-DETR adapter (Roboflow, Apache-2.0; COCO-pretrained Nano/Small/Medium weights are Apache-2.0).

``rfdetr`` and ``torch`` are imported lazily so the rest of the system works without them.
"""

import hashlib
import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from soccervision.config import DetectorConfig, DetectorName
from soccervision.errors import InferenceError, ModelUnavailableError
from soccervision.schemas.detection import ModelInfo
from soccervision.vision.detection.base import RawDetection

logger = logging.getLogger(__name__)


class RFDETRVariant(NamedTuple):
    class_name: str
    weights_filename: str
    """Filename the rfdetr registry knows how to download."""
    weights_md5: str
    """Published MD5 from ``rfdetr.assets.model_weights.ModelWeights`` (rfdetr 1.11)."""


# rfdetr silently re-downloads and overwrites a checkpoint whose MD5 does not match its registry,
# whatever we pass for downloads, so integrity is checked here before rfdetr ever sees the file.
RFDETR_VARIANTS: dict[DetectorName, RFDETRVariant] = {
    "rfdetr-nano": RFDETRVariant(
        "RFDETRNano", "rf-detr-nano.pth", "fb6504cce7fbdc783f7a46991f07639f"
    ),
    "rfdetr-small": RFDETRVariant(
        "RFDETRSmall", "rf-detr-small.pth", "fb37061c1af7bace359c91b723a8d5c1"
    ),
    "rfdetr-medium": RFDETRVariant(
        "RFDETRMedium", "rf-detr-medium.pth", "7223f764a87b863f02eb8d52bf0ce2ee"
    ),
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


def verify_weights(weights_path: Path, variant: RFDETRVariant, name: str) -> str:
    """Return the checkpoint's MD5, raising ModelUnavailableError if it is missing or corrupt."""
    if not weights_path.is_file():
        raise ModelUnavailableError(
            f"Weights for {name} not found at {weights_path} and downloads are disabled. "
            "Run `python scripts/download_weights.py` or enable downloads."
        )
    actual = _md5(weights_path)
    if actual != variant.weights_md5:
        raise ModelUnavailableError(
            f"Weights at {weights_path} failed the integrity check (md5 {actual}, expected "
            f"{variant.weights_md5}). Delete the file and run `python scripts/download_weights.py`."
        )
    return actual


class RFDETRDetector:
    def __init__(
        self,
        config: DetectorConfig,
        *,
        device: str,
        weights_dir: Path,
        allow_download: bool,
    ) -> None:
        variant = RFDETR_VARIANTS[config.name]
        weights_filename = variant.weights_filename
        weights_path = weights_dir / weights_filename
        if not allow_download:
            verify_weights(weights_path, variant, config.name)

        try:
            rfdetr = importlib.import_module("rfdetr")
        except ImportError as exc:
            raise ModelUnavailableError(
                "RF-DETR is not installed. Install ML extras: pip install -e '.[ml]'"
            ) from exc

        weights_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model: Any = getattr(rfdetr, variant.class_name)(
                pretrain_weights=str(weights_path), device=device
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load {config.name} on {device}: {exc}") from exc
        weights_md5 = verify_weights(weights_path, variant, config.name)

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
            weights_md5=weights_md5,
            input_resolution=resolution,
            device=device,
            framework_versions=_package_versions("rfdetr", "torch", "torchvision", "supervision"),
        )
        # First inference compiles kernels (notably on MPS); keep it out of per-frame timings.
        try:
            self._predict(np.zeros((resolution, resolution, 3), dtype=np.uint8))
        except Exception as exc:
            raise ModelUnavailableError(
                f"Warm-up inference failed for {config.name} on {device}: {exc}"
            ) from exc

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
