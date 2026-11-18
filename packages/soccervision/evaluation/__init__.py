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
from soccervision.evaluation.export import (
    DEFAULT_EXPORT_CONFIDENCE,
    CocoPrediction,
    PredictionExport,
    PredictionExportMetadata,
    export_predictions,
    resolve_category_ids,
    serialize_predictions,
    write_prediction_export,
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
    "DEFAULT_EXPORT_CONFIDENCE",
    "IOU_THRESHOLDS",
    "SUPPORTED_CLASSES",
    "CocoPrediction",
    "DatasetMetadata",
    "DatasetProvenance",
    "DetectionMetrics",
    "EvaluationDataset",
    "EvaluationImage",
    "GroundTruthBox",
    "Prediction",
    "PredictionExport",
    "PredictionExportMetadata",
    "SizeRecall",
    "evaluate_detections",
    "export_predictions",
    "load_coco_dataset",
    "load_coco_predictions",
    "parse_coco_dataset",
    "parse_coco_predictions",
    "resolve_category_ids",
    "serialize_predictions",
    "write_prediction_export",
]
