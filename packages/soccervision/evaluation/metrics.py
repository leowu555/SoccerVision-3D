"""Detection metrics computed with the reference COCO evaluator (``pycocotools``, BSD-2-Clause).

All numbers come from one ``COCOeval`` run, so mAP and the thresholded counts share the same
matching: per image, predictions are taken in descending score order (capped at
``max_detections_per_image``) and each is greedily matched to the unmatched ground-truth box with
the highest IoU at or above the threshold. A second prediction on an already matched box is a
false positive. Matches to crowd regions are neither true nor false positives.

Because matching is greedy by score, applying the confidence threshold after matching gives the
same result as matching only the predictions at or above the threshold.
"""

import contextlib
import importlib.metadata
import io
import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soccervision.errors import EvaluationUnavailableError
from soccervision.evaluation.dataset import EvaluationDataset
from soccervision.evaluation.predictions import Prediction, validate_predictions

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
"""COCO IoU thresholds 0.50, 0.55, ..., 0.95."""

SizeBucket = Literal["small", "medium", "large"]
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}
"""COCO area ranges in square pixels. Ground truth is bucketed by its ``area``."""

_CATEGORY_ID = 1


class SizeRecall(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_objects: int
    matched: int
    recall: float | None = Field(description="null when the bucket has no objects.")


class DetectionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_name: str
    confidence_threshold: float = Field(description="Applies to precision, recall and counts.")
    iou_threshold: float = Field(description="IoU for precision, recall and counts.")
    max_detections_per_image: int

    map_50: float | None = Field(
        description="AP at IoU 0.50 over all predictions; null if no objects."
    )
    map_50_95: float | None = Field(description="AP averaged over IoU 0.50:0.05:0.95.")

    precision: float | None = Field(description="TP / (TP + FP); null if nothing is predicted.")
    recall: float | None = Field(description="TP / objects; null if there are no objects.")
    true_positives: int
    false_positives: int
    false_negatives: int
    recall_by_size: dict[SizeBucket, SizeRecall]

    num_images: int
    num_objects: int = Field(description="Non-crowd ground-truth boxes of this class.")
    num_predictions: int = Field(description="Predictions of this class, before any filtering.")
    num_predictions_above_threshold: int = Field(
        description="At or above the confidence threshold, after the per-image cap."
    )

    dataset_synthetic: bool = Field(
        description="Copied from provenance; synthetic is not accuracy."
    )
    evaluator: dict[str, str]


def evaluate_detections(
    dataset: EvaluationDataset,
    predictions: list[Prediction],
    *,
    class_name: str = "person",
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    max_detections_per_image: int = 100,
) -> DetectionMetrics:
    if class_name not in dataset.metadata.classes:
        raise ValueError(f"class {class_name!r} is not in the dataset's classes")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    iou_index = next(
        (i for i, t in enumerate(IOU_THRESHOLDS) if math.isclose(t, iou_threshold)), None
    )
    if iou_index is None:
        raise ValueError(f"iou_threshold must be one of {IOU_THRESHOLDS}")
    if max_detections_per_image < 1:
        raise ValueError("max_detections_per_image must be at least 1")
    validate_predictions(predictions, dataset)

    coco_cls, cocoeval_cls = _import_pycocotools()
    class_predictions = [p for p in predictions if p.class_name == class_name]
    coco_gt = _coco_index(coco_cls, dataset, _ground_truth_annotations(dataset, class_name))
    coco_dt = _coco_index(coco_cls, dataset, _prediction_annotations(class_predictions))

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = cocoeval_cls(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [max_detections_per_image]
        evaluator.params.areaRng = [list(r) for r in AREA_RANGES.values()]
        evaluator.params.areaRngLbl = list(AREA_RANGES)
        evaluator.evaluate()
        evaluator.accumulate()

    precision_table = evaluator.eval["precision"][:, :, 0, 0, 0]  # [IoU, recall thresholds]
    counts = _threshold_counts(evaluator.evalImgs, iou_index, confidence_threshold)
    tp, fp, num_objects, above = counts["all"]
    return DetectionMetrics(
        class_name=class_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=IOU_THRESHOLDS[iou_index],
        max_detections_per_image=max_detections_per_image,
        map_50=_mean_valid(precision_table[iou_index]),
        map_50_95=_mean_valid(precision_table),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / num_objects if num_objects else None,
        true_positives=tp,
        false_positives=fp,
        false_negatives=num_objects - tp,
        recall_by_size={
            bucket: SizeRecall(
                num_objects=counts[bucket][2],
                matched=counts[bucket][0],
                recall=counts[bucket][0] / counts[bucket][2] if counts[bucket][2] else None,
            )
            for bucket in ("small", "medium", "large")
        },
        num_images=dataset.metadata.num_images,
        num_objects=num_objects,
        num_predictions=len(class_predictions),
        num_predictions_above_threshold=above,
        dataset_synthetic=dataset.metadata.provenance.synthetic,
        evaluator={"pycocotools": importlib.metadata.version("pycocotools")},
    )


def _import_pycocotools() -> tuple[Any, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise EvaluationUnavailableError(
            "pycocotools is not installed. Install the evaluation extra: pip install -e '.[eval]'"
        ) from exc
    return COCO, COCOeval


def _xywh(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _ground_truth_annotations(dataset: EvaluationDataset, class_name: str) -> list[dict[str, Any]]:
    # Ids are renumbered from 1 because COCOeval uses 0 to mean "unmatched".
    boxes = [b for b in dataset.annotations if b.class_name == class_name]
    return [
        {
            "id": index,
            "image_id": box.image_id,
            "category_id": _CATEGORY_ID,
            "bbox": _xywh(box.bbox_xyxy),
            "area": box.area,
            "iscrowd": int(box.is_crowd),
        }
        for index, box in enumerate(boxes, start=1)
    ]


def _prediction_annotations(predictions: list[Prediction]) -> list[dict[str, Any]]:
    annotations = []
    for index, prediction in enumerate(predictions, start=1):
        bbox = _xywh(prediction.bbox_xyxy)
        annotations.append(
            {
                "id": index,
                "image_id": prediction.image_id,
                "category_id": _CATEGORY_ID,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "score": prediction.score,
            }
        )
    return annotations


def _coco_index(
    coco_cls: Any, dataset: EvaluationDataset, annotations: list[dict[str, Any]]
) -> Any:
    coco = coco_cls()
    coco.dataset = {
        "images": [
            {"id": i.image_id, "width": i.width, "height": i.height} for i in dataset.images
        ],
        "categories": [{"id": _CATEGORY_ID, "name": "target"}],
        "annotations": annotations,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _threshold_counts(
    eval_images: list[dict[str, Any] | None], iou_index: int, threshold: float
) -> dict[str, tuple[int, int, int, int]]:
    """Per area range: (true positives, false positives, objects, predictions kept)."""
    labels = {tuple(r): label for label, r in AREA_RANGES.items()}
    totals = dict.fromkeys(AREA_RANGES, (0, 0, 0, 0))
    for entry in eval_images:
        if entry is None:
            continue
        label = labels[tuple(entry["aRng"])]
        kept = np.asarray(entry["dtScores"], dtype=float) >= threshold
        matched = np.asarray(entry["dtMatches"])[iou_index] > 0
        ignored = np.asarray(entry["dtIgnore"])[iou_index].astype(bool)
        objects = int(np.count_nonzero(np.asarray(entry["gtIgnore"]) == 0))
        tp = int(np.count_nonzero(kept & matched & ~ignored))
        fp = int(np.count_nonzero(kept & ~matched & ~ignored))
        old = totals[label]
        totals[label] = (old[0] + tp, old[1] + fp, old[2] + objects, old[3] + int(kept.sum()))
    return totals


def _mean_valid(values: Any) -> float | None:
    """Mean over entries COCOeval did not mark undefined (-1)."""
    array = np.asarray(values, dtype=float)
    valid = array[array > -1]
    return float(valid.mean()) if valid.size else None
