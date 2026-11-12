from pathlib import Path

from soccervision.config import DetectorConfig, DetectorName
from soccervision.devices import resolve_device
from soccervision.vision.detection.base import ObjectDetector
from soccervision.vision.detection.rfdetr import RFDETR_VARIANTS, RFDETRDetector


def weights_path_for(name: DetectorName, weights_dir: Path) -> Path:
    return weights_dir / RFDETR_VARIANTS[name].weights_filename


def create_detector(
    config: DetectorConfig, *, weights_dir: Path, allow_download: bool
) -> ObjectDetector:
    """Build the configured pretrained detector. Raises ModelUnavailableError / DeviceUnavailableError."""
    device = resolve_device(config.device)
    return RFDETRDetector(
        config, device=device, weights_dir=weights_dir, allow_download=allow_download
    )
