"""COCO-format ground truth for detection evaluation: loading, validation and provenance.

Only the subset of COCO used for box detection is read: ``images``, ``categories``,
``annotations`` (``bbox``, ``iscrowd``, optional ``area``) and the optional ``info`` and
``licenses`` blocks. Unknown keys are ignored. COCO boxes are ``[x, y, width, height]`` in pixels;
they are converted to this project's ``(x_min, y_min, x_max, y_max)`` convention on load.
"""

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from soccervision.errors import InvalidDatasetError
from soccervision.pipeline.artifacts import sha256_file
from soccervision.schemas.detection import BBoxXYXY

SUPPORTED_CLASSES = ("person",)
"""Evaluation classes. Dataset categories are mapped onto these; others are ignored."""

DEFAULT_CATEGORY_MAP: Mapping[str, str] = {"person": "person"}

_BOUNDS_TOLERANCE_PX = 1e-6


class DatasetProvenance(BaseModel):
    """Where a dataset came from. Recorded with every evaluation so results are traceable."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str | None = None
    license: str | None = None
    source_url: str | None = None
    description: str | None = None
    synthetic: bool = Field(
        default=False,
        description="True for hand-built test fixtures. Their metrics are not model accuracy.",
    )


class EvaluationImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: int
    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class GroundTruthBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    annotation_id: int
    image_id: int
    class_name: str
    bbox_xyxy: BBoxXYXY
    area: float = Field(
        gt=0, description="COCO 'area' if given, else box width x height. Drives size buckets."
    )
    is_crowd: bool = Field(
        description="COCO crowd region: never counted as missed, and matches to it are ignored."
    )


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    provenance: DatasetProvenance
    annotation_file: str
    annotation_sha256: str
    classes: list[str]
    category_map: dict[str, str] = Field(description="Dataset category name -> evaluation class.")
    dataset_categories: dict[int, str]
    num_images: int
    num_images_without_objects: int = Field(
        description="Images with no evaluated (mapped, non-crowd) objects; still evaluated."
    )
    num_objects: int = Field(description="Mapped, non-crowd ground-truth boxes.")
    num_crowd_regions: int
    num_ignored_annotations: int = Field(description="Annotations in unmapped categories.")


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: DatasetMetadata
    images: list[EvaluationImage]
    annotations: list[GroundTruthBox] = Field(description="Mapped categories only.")

    def image_ids(self) -> set[int]:
        return {image.image_id for image in self.images}


class _RawImage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_name: str = Field(min_length=1)


class _RawCategory(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    id: int
    name: str = Field(min_length=1)


class _RawAnnotation(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    id: int
    image_id: int
    category_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    iscrowd: int = Field(default=0, ge=0, le=1)
    area: float | None = Field(default=None, gt=0)


def load_coco_dataset(
    path: Path,
    *,
    category_map: Mapping[str, str] | None = None,
    provenance: DatasetProvenance | None = None,
) -> EvaluationDataset:
    """Load and validate a COCO annotation file. Never downloads anything.

    Raises InvalidDatasetError listing every problem found.
    """
    try:
        data = json.loads(path.read_bytes())
    except OSError as exc:
        raise InvalidDatasetError(f"Cannot read {path}", [str(exc)]) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidDatasetError(f"{path} is not valid JSON", [str(exc)]) from exc
    return parse_coco_dataset(
        data,
        annotation_file=path.name,
        annotation_sha256=sha256_file(path),
        category_map=category_map,
        provenance=provenance,
    )


def parse_coco_dataset(
    data: object,
    *,
    annotation_file: str,
    annotation_sha256: str,
    category_map: Mapping[str, str] | None = None,
    provenance: DatasetProvenance | None = None,
) -> EvaluationDataset:
    """Validate an already-decoded COCO document. See ``load_coco_dataset``."""
    mapping = dict(DEFAULT_CATEGORY_MAP if category_map is None else category_map)
    issues: list[str] = []

    if not isinstance(data, dict):
        raise InvalidDatasetError("COCO annotations must be a JSON object", ["top level"])
    sections: dict[str, list[Any]] = {}
    for key in ("images", "annotations", "categories"):
        value = data.get(key)
        if not isinstance(value, list):
            issues.append(f"'{key}' must be a list")
        else:
            sections[key] = value
    if issues:
        raise InvalidDatasetError("Missing or malformed COCO sections", issues)

    images = _parse_items(sections["images"], _RawImage, "images", issues)
    categories = _parse_items(sections["categories"], _RawCategory, "categories", issues)
    annotations = _parse_items(sections["annotations"], _RawAnnotation, "annotations", issues)

    if not sections["images"]:
        issues.append("dataset contains no images")
    _report_duplicates((i.id for i in images), "image id", issues)
    _report_duplicates((i.file_name for i in images), "image file_name", issues)
    _report_duplicates((c.id for c in categories), "category id", issues)
    _report_duplicates((c.name for c in categories), "category name", issues)
    _report_duplicates((a.id for a in annotations), "annotation id", issues)

    category_names = {c.id: c.name for c in categories}
    for name, evaluation_class in mapping.items():
        if name not in category_names.values():
            issues.append(f"category_map key {name!r} is not a dataset category")
        if evaluation_class not in SUPPORTED_CLASSES:
            issues.append(
                f"category_map value {evaluation_class!r} is not a supported class "
                f"{SUPPORTED_CLASSES}"
            )

    images_by_id = {i.id: i for i in images}
    boxes: list[GroundTruthBox] = []
    ignored = 0
    for index, ann in enumerate(annotations):
        where = f"annotations[{index}] (id {ann.id})"
        image = images_by_id.get(ann.image_id)
        if image is None:
            issues.append(f"{where}: image_id {ann.image_id} does not exist")
        if ann.category_id not in category_names:
            issues.append(f"{where}: category_id {ann.category_id} does not exist")
            continue
        box_issue = _bbox_issue(ann.bbox, image)
        if box_issue:
            issues.append(f"{where}: {box_issue}")
            continue
        target = mapping.get(category_names[ann.category_id])
        if target is None:
            ignored += 1
            continue
        if image is None:
            continue
        x, y, w, h = ann.bbox
        boxes.append(
            GroundTruthBox(
                annotation_id=ann.id,
                image_id=ann.image_id,
                class_name=target,
                bbox_xyxy=(x, y, x + w, y + h),
                area=ann.area if ann.area is not None else w * h,
                is_crowd=bool(ann.iscrowd),
            )
        )

    if issues:
        raise InvalidDatasetError("Invalid COCO dataset", issues)

    eval_images = [
        EvaluationImage(image_id=i.id, file_name=i.file_name, width=i.width, height=i.height)
        for i in images
    ]
    with_objects = {b.image_id for b in boxes if not b.is_crowd}
    crowd = sum(b.is_crowd for b in boxes)
    metadata = DatasetMetadata(
        provenance=provenance or _provenance_from_info(data, annotation_file),
        annotation_file=annotation_file,
        annotation_sha256=annotation_sha256,
        classes=sorted(set(mapping.values())),
        category_map=mapping,
        dataset_categories=category_names,
        num_images=len(eval_images),
        num_images_without_objects=len(eval_images) - len(with_objects),
        num_objects=len(boxes) - crowd,
        num_crowd_regions=crowd,
        num_ignored_annotations=ignored,
    )
    return EvaluationDataset(metadata=metadata, images=eval_images, annotations=boxes)


def _parse_items[M: BaseModel](
    items: list[Any], model: type[M], section: str, issues: list[str]
) -> list[M]:
    parsed: list[M] = []
    for index, item in enumerate(items):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"])
                location = f"{section}[{index}].{field}" if field else f"{section}[{index}]"
                issues.append(f"{location}: {error['msg']}")
    return parsed


def _report_duplicates(values: Any, label: str, issues: list[str]) -> None:
    for value, count in Counter(values).items():
        if count > 1:
            issues.append(f"duplicate {label} {value!r} ({count} times)")


def _bbox_issue(bbox: list[float], image: _RawImage | None) -> str | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return f"bbox {bbox} must have positive width and height"
    if x < 0 or y < 0:
        return f"bbox {bbox} has negative coordinates"
    if image is not None and (
        x + w > image.width + _BOUNDS_TOLERANCE_PX or y + h > image.height + _BOUNDS_TOLERANCE_PX
    ):
        return f"bbox {bbox} extends outside the {image.width}x{image.height} image"
    return None


def _provenance_from_info(data: dict[str, Any], annotation_file: str) -> DatasetProvenance:
    raw_info = data.get("info")
    info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}

    def text(value: object) -> str | None:
        return str(value) if isinstance(value, str | int | float) and str(value) else None

    licenses = data.get("licenses")
    license_names = sorted(
        {str(item["name"]) for item in licenses if isinstance(item, dict) and item.get("name")}
        if isinstance(licenses, list)
        else set()
    )
    return DatasetProvenance(
        name=text(info.get("description")) or Path(annotation_file).stem,
        version=text(info.get("version")),
        license="; ".join(license_names) or None,
        source_url=text(info.get("url")),
    )
