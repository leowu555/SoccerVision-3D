"""Detector predictions to evaluate against an ``EvaluationDataset``.

The on-disk format is the standard COCO results list:
``[{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]``,
where ``category_id`` refers to the *dataset's* categories.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from soccervision.errors import InvalidPredictionsError
from soccervision.evaluation.dataset import EvaluationDataset
from soccervision.schemas.detection import BBoxXYXY


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: int
    class_name: str
    bbox_xyxy: BBoxXYXY
    score: float = Field(ge=0.0, le=1.0)


class _RawResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    image_id: int
    category_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float = Field(ge=0.0, le=1.0)


def load_coco_predictions(path: Path, dataset: EvaluationDataset) -> list[Prediction]:
    """Load COCO results. Predictions in categories the dataset does not map are dropped."""
    try:
        data = json.loads(path.read_bytes())
    except OSError as exc:
        raise InvalidPredictionsError(f"Cannot read {path}", [str(exc)]) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidPredictionsError(f"{path} is not valid JSON", [str(exc)]) from exc
    return parse_coco_predictions(data, dataset)


def parse_coco_predictions(data: object, dataset: EvaluationDataset) -> list[Prediction]:
    if not isinstance(data, list):
        raise InvalidPredictionsError("COCO results must be a JSON list", ["top level"])
    categories = dataset.metadata.dataset_categories
    mapping = dataset.metadata.category_map
    image_ids = dataset.image_ids()
    issues: list[str] = []
    predictions: list[Prediction] = []
    for index, item in enumerate(data):
        raw = _parse_result(item, index, issues)
        if raw is None:
            continue
        where = f"results[{index}]"
        if raw.image_id not in image_ids:
            issues.append(f"{where}: image_id {raw.image_id} is not in the dataset")
            continue
        if raw.category_id not in categories:
            issues.append(f"{where}: category_id {raw.category_id} is not a dataset category")
            continue
        x, y, w, h = raw.bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            issues.append(f"{where}: bbox {raw.bbox} needs x, y >= 0 and positive width/height")
            continue
        target = mapping.get(categories[raw.category_id])
        if target is not None:
            predictions.append(
                Prediction(
                    image_id=raw.image_id,
                    class_name=target,
                    bbox_xyxy=(x, y, x + w, y + h),
                    score=raw.score,
                )
            )
    if issues:
        raise InvalidPredictionsError("Invalid COCO predictions", issues)
    return predictions


def validate_predictions(predictions: list[Prediction], dataset: EvaluationDataset) -> None:
    """Reject predictions for images that are not in the dataset."""
    image_ids = dataset.image_ids()
    issues = [
        f"predictions[{index}]: image_id {p.image_id} is not in the dataset"
        for index, p in enumerate(predictions)
        if p.image_id not in image_ids
    ]
    if issues:
        raise InvalidPredictionsError("Predictions do not match the dataset", issues)


def _parse_result(item: Any, index: int, issues: list[str]) -> _RawResult | None:
    try:
        return _RawResult.model_validate(item)
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            issues.append(f"results[{index}].{field}: {error['msg']}")
        return None
