"""Detection evaluation against COCO-format ground truth (no datasets are downloaded)."""

from soccervision.evaluation.dataset import (
    DEFAULT_CATEGORY_MAP,
    SUPPORTED_CLASSES,
    DatasetMetadata,
    DatasetProvenance,
    EvaluationDataset,
    EvaluationImage,
    GroundTruthBox,
    load_coco_dataset,
    parse_coco_dataset,
)
from soccervision.evaluation.metrics import (
    AREA_RANGES,
    IOU_THRESHOLDS,
    DetectionMetrics,
    SizeRecall,
    evaluate_detections,
)
from soccervision.evaluation.predictions import (
    Prediction,
    load_coco_predictions,
    parse_coco_predictions,
)

__all__ = [
    "AREA_RANGES",
    "DEFAULT_CATEGORY_MAP",
    "IOU_THRESHOLDS",
    "SUPPORTED_CLASSES",
    "DatasetMetadata",
    "DatasetProvenance",
    "DetectionMetrics",
    "EvaluationDataset",
    "EvaluationImage",
    "GroundTruthBox",
    "Prediction",
    "SizeRecall",
    "evaluate_detections",
    "load_coco_dataset",
    "load_coco_predictions",
    "parse_coco_dataset",
    "parse_coco_predictions",
]
