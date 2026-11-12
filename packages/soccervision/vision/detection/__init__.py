"""Object detection interfaces and model adapters."""

from soccervision.vision.detection.base import DetectorFactory, ObjectDetector, RawDetection
from soccervision.vision.detection.factory import create_detector, weights_path_for

__all__ = [
    "DetectorFactory",
    "ObjectDetector",
    "RawDetection",
    "create_detector",
    "weights_path_for",
]
