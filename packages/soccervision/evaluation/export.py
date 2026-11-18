"""Run a detector over a COCO evaluation dataset and export standard COCO results.

The predictions file is a plain COCO results list,
``[{"image_id", "category_id", "bbox": [x, y, w, h], "score"}, ...]``, readable by
``load_coco_predictions`` and any COCO tool. That format has no room for provenance, so everything
needed to reproduce a file (dataset fingerprint, model, checkpoint hash, threshold, timings) goes
into a separate metadata file, written last as the commit marker.

Images are read with Pillow in their stored pixel orientation (EXIF rotation is not applied), and
their decoded size must equal the dataset's ``width`` x ``height``, so that predicted and
ground-truth boxes refer to the same pixels.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from soccervision import __version__
from soccervision.config import DetectorConfig
from soccervision.errors import (
    InferenceError,
    InvalidPredictionsError,
    ModelUnavailableError,
    PredictionExportError,
    SoccerVisionError,
)
from soccervision.evaluation.dataset import EvaluationDataset, EvaluationImage
from soccervision.evaluation.predictions import parse_coco_predictions
from soccervision.schemas.artifacts import EnvironmentInfo
from soccervision.schemas.detection import ModelInfo
from soccervision.vision.detection.base import DetectorFactory, ObjectDetector, RawDetection

DEFAULT_EXPORT_CONFIDENCE = 0.01
"""mAP is defined over all predictions, so export far below the 0.5 display threshold."""

EXPORT_METADATA_SCHEMA_VERSION = "1.0.0"

_HIGH_THRESHOLD_WARNING = 0.05
_IMAGE_ERRORS = (OSError, ValueError, Image.DecompressionBombError)


class CocoPrediction(BaseModel):
    """One COCO result. ``bbox`` is ``[x, y, width, height]`` in pixels of the stored image."""

    model_config = ConfigDict(frozen=True)

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float = Field(ge=0.0, le=1.0)


class ExportDatasetInfo(BaseModel):
    name: str
    version: str | None
    annotation_file: str
    annotation_sha256: str
    synthetic: bool
    image_root: str


class ExportStats(BaseModel):
    num_images: int = Field(description="Every dataset image; the detector ran on all of them.")
    images_without_predictions: int = Field(
        description="Images where nothing was kept. They are still evaluated (as all-missed)."
    )
    num_predictions: int
    discarded_other_classes: int = Field(description="Detections of classes not exported.")
    discarded_below_threshold: int
    clipped_boxes: int = Field(description="Boxes partly outside the image, clipped to it.")
    discarded_degenerate_boxes: int = Field(description="Boxes with no area after clipping.")


class ExportTiming(BaseModel):
    started_at: datetime
    completed_at: datetime
    total_seconds: float
    model_load_seconds: float
    inference_seconds: float = Field(description="Wall time inside detector calls.")
    mean_inference_ms: float | None


class PredictionExportMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = EXPORT_METADATA_SCHEMA_VERSION
    artifact_type: Literal["coco_predictions_metadata"] = "coco_predictions_metadata"
    predictions_sha256: str = Field(description="SHA-256 of the predictions file's exact bytes.")
    dataset: ExportDatasetInfo
    model: ModelInfo
    detector_config: DetectorConfig
    confidence_threshold: float = Field(description="Minimum exported score.")
    class_to_category: dict[str, int] = Field(description="Detector class -> dataset category_id.")
    stats: ExportStats
    timing: ExportTiming
    environment: EnvironmentInfo
    warnings: list[str]


@dataclass(frozen=True)
class PredictionExport:
    predictions: list[CocoPrediction]
    metadata: PredictionExportMetadata

    def predictions_json(self) -> bytes:
        return serialize_predictions(self.predictions)


@dataclass
class _Counts:
    empty_images: int = 0
    other_classes: int = 0
    below_threshold: int = 0
    clipped: int = 0
    degenerate: int = 0


def serialize_predictions(predictions: Sequence[CocoPrediction]) -> bytes:
    """The exact bytes of a COCO results file; ``predictions_sha256`` is computed over them."""
    return json.dumps([p.model_dump() for p in predictions], separators=(",", ":")).encode()


def resolve_category_ids(
    dataset: EvaluationDataset, detector_classes: Sequence[str]
) -> dict[str, int]:
    """Map each detector class to the dataset category its predictions are written under.

    A detector class must be one of the dataset's evaluation classes. When several dataset
    categories map to it (for example player, referee and goalkeeper all map to person), the
    category with the same name is used, otherwise the lowest id. Evaluation treats them alike.
    """
    meta = dataset.metadata
    issues: list[str] = []
    mapping: dict[str, int] = {}
    for detector_class in detector_classes:
        if detector_class not in meta.classes:
            issues.append(
                f"detector class {detector_class!r} is not an evaluation class of this dataset "
                f"{meta.classes}"
            )
            continue
        candidates = sorted(
            category_id
            for category_id, name in meta.dataset_categories.items()
            if meta.category_map.get(name) == detector_class
        )
        same_name = [c for c in candidates if meta.dataset_categories[c] == detector_class]
        mapping[detector_class] = (same_name or candidates)[0]
    if issues:
        raise PredictionExportError("Unsupported class mapping", issues)
    return mapping


def export_predictions(
    dataset: EvaluationDataset,
    *,
    image_root: Path,
    detector_factory: DetectorFactory,
    detector_config: DetectorConfig | None = None,
) -> PredictionExport:
    """Run the detector on every dataset image and return COCO results plus their metadata.

    ``detector_config`` defaults to the default detector at ``DEFAULT_EXPORT_CONFIDENCE``; its
    ``target_classes`` are the classes exported. Images and the class mapping are checked before
    the model loads, and all problems are reported together. Any failure after that aborts the
    export, so no image is ever skipped.
    """
    config = detector_config or DetectorConfig(confidence_threshold=DEFAULT_EXPORT_CONFIDENCE)
    started_at = datetime.now(UTC)
    t_start = perf_counter()
    class_to_category = resolve_category_ids(dataset, config.target_classes)
    paths = _check_images(dataset, image_root)

    t_load = perf_counter()
    detector = _create_detector(detector_factory, config)
    model_load_seconds = perf_counter() - t_load
    model = detector.info

    counts = _Counts()
    predictions: list[CocoPrediction] = []
    inference_seconds = 0.0
    for image in dataset.images:
        pixels = _load_rgb(paths[image.image_id], image)
        t_infer = perf_counter()
        raw = _detect(detector, pixels, image)
        inference_seconds += perf_counter() - t_infer
        kept = _to_coco(raw, image, config.confidence_threshold, class_to_category, counts)
        if not kept:
            counts.empty_images += 1
        predictions.extend(kept)

    payload = serialize_predictions(predictions)
    try:
        parse_coco_predictions(json.loads(payload), dataset)
    except InvalidPredictionsError as exc:
        raise PredictionExportError(
            "Exported predictions are not valid COCO results", exc.issues
        ) from exc

    num_images = len(dataset.images)
    provenance = dataset.metadata.provenance
    metadata = PredictionExportMetadata(
        predictions_sha256=hashlib.sha256(payload).hexdigest(),
        dataset=ExportDatasetInfo(
            name=provenance.name,
            version=provenance.version,
            annotation_file=dataset.metadata.annotation_file,
            annotation_sha256=dataset.metadata.annotation_sha256,
            synthetic=provenance.synthetic,
            image_root=str(image_root),
        ),
        model=model,
        detector_config=config,
        confidence_threshold=config.confidence_threshold,
        class_to_category=class_to_category,
        stats=ExportStats(
            num_images=num_images,
            images_without_predictions=counts.empty_images,
            num_predictions=len(predictions),
            discarded_other_classes=counts.other_classes,
            discarded_below_threshold=counts.below_threshold,
            clipped_boxes=counts.clipped,
            discarded_degenerate_boxes=counts.degenerate,
        ),
        timing=ExportTiming(
            started_at=started_at,
            completed_at=datetime.now(UTC),
            total_seconds=perf_counter() - t_start,
            model_load_seconds=model_load_seconds,
            inference_seconds=inference_seconds,
            mean_inference_ms=1000 * inference_seconds / num_images if num_images else None,
        ),
        environment=_environment(model),
        warnings=_warnings(model, dataset, config.confidence_threshold),
    )
    return PredictionExport(predictions=predictions, metadata=metadata)


def write_prediction_export(
    export: PredictionExport, *, predictions_path: Path, metadata_path: Path
) -> None:
    """Write the COCO results file, then the metadata file, each atomically.

    The metadata file is written last: if it exists, its predictions file is complete.
    """
    if predictions_path.resolve() == metadata_path.resolve():
        raise ValueError("predictions_path and metadata_path must be different files")
    try:
        _write_atomic(predictions_path, export.predictions_json())
    except OSError as exc:
        raise PredictionExportError(f"Cannot write {predictions_path}", [str(exc)]) from exc
    try:
        _write_atomic(metadata_path, export.metadata.model_dump_json(indent=2).encode())
    except OSError as exc:
        with contextlib.suppress(OSError):
            predictions_path.unlink()
        raise PredictionExportError(f"Cannot write {metadata_path}", [str(exc)]) from exc


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file + rename so readers never see a partial file; clean up on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _check_images(dataset: EvaluationDataset, image_root: Path) -> dict[int, Path]:
    """Resolve every image inside ``image_root`` and check it opens with the expected size."""
    if not image_root.is_dir():
        raise PredictionExportError("Image root is not a directory", [str(image_root)])
    root = image_root.resolve()
    issues: list[str] = []
    paths: dict[int, Path] = {}
    for image in dataset.images:
        where = _where(image)
        relative = Path(image.file_name)
        if relative.is_absolute():
            issues.append(f"{where}: file_name must be relative to the image root")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            issues.append(f"{where}: resolves outside the image root")
            continue
        if not path.is_file():
            issues.append(f"{where}: file not found under {root}")
            continue
        try:
            with Image.open(path) as handle:
                size = handle.size
        except _IMAGE_ERRORS as exc:
            issues.append(f"{where}: unreadable image ({exc})")
            continue
        if size != (image.width, image.height):
            issues.append(
                f"{where}: image is {size[0]}x{size[1]}, dataset says {image.width}x{image.height}"
            )
            continue
        paths[image.image_id] = path
    if issues:
        raise PredictionExportError("Dataset images failed validation", issues)
    return paths


def _load_rgb(path: Path, image: EvaluationImage) -> npt.NDArray[np.uint8]:
    try:
        with Image.open(path) as handle:
            pixels = np.array(handle.convert("RGB"), dtype=np.uint8)
    except _IMAGE_ERRORS as exc:
        raise PredictionExportError(
            "Image could not be decoded", [f"{_where(image)}: {exc}"]
        ) from exc
    if pixels.shape != (image.height, image.width, 3):
        raise PredictionExportError(
            "Decoded image has unexpected dimensions",
            [f"{_where(image)}: decoded shape {pixels.shape}"],
        )
    return pixels


def _create_detector(factory: DetectorFactory, config: DetectorConfig) -> ObjectDetector:
    try:
        return factory(config)
    except SoccerVisionError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"Detector could not be created: {exc}") from exc


def _detect(
    detector: ObjectDetector, pixels: npt.NDArray[np.uint8], image: EvaluationImage
) -> list[RawDetection]:
    try:
        raw: object = detector.detect(pixels)
    except Exception as exc:
        detail = (
            exc.message if isinstance(exc, SoccerVisionError) else f"{type(exc).__name__}: {exc}"
        )
        raise InferenceError(f"Inference failed on {_where(image)}: {detail}") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise PredictionExportError(
            "Detector returned invalid output",
            [f"{_where(image)}: expected a sequence of detections, got {type(raw).__name__}"],
        )
    return list(raw)


def _to_coco(
    raw: list[RawDetection],
    image: EvaluationImage,
    threshold: float,
    class_to_category: dict[str, int],
    counts: _Counts,
) -> list[CocoPrediction]:
    """Validate, filter and clip one image's detections, as the Phase 1 pipeline does."""
    issues = [
        f"{_where(image)}, detection {index}: {problem}"
        for index, det in enumerate(raw)
        if (problem := _output_problem(det)) is not None
    ]
    if issues:
        raise PredictionExportError("Detector returned invalid output", issues)

    out: list[CocoPrediction] = []
    for det in raw:
        category_id = class_to_category.get(det.class_name)
        if category_id is None:
            counts.other_classes += 1
            continue
        if det.confidence < threshold:
            counts.below_threshold += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in det.bbox_xyxy)
        cx1, cy1 = min(max(x1, 0.0), image.width), min(max(y1, 0.0), image.height)
        cx2, cy2 = min(max(x2, 0.0), image.width), min(max(y2, 0.0), image.height)
        if cx2 <= cx1 or cy2 <= cy1:
            counts.degenerate += 1
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            counts.clipped += 1
        out.append(
            CocoPrediction(
                image_id=image.image_id,
                category_id=category_id,
                bbox=(cx1, cy1, cx2 - cx1, cy2 - cy1),
                score=float(det.confidence),
            )
        )
    return out


def _output_problem(det: object) -> str | None:
    if not isinstance(det, RawDetection):
        return f"expected RawDetection, got {type(det).__name__}"
    if not isinstance(det.class_name, str):
        return f"class_name {det.class_name!r} is not a string"
    if not _finite(det.confidence) or not 0.0 <= det.confidence <= 1.0:
        return f"confidence {det.confidence!r} is not a number in [0, 1]"
    box: object = det.bbox_xyxy
    if not isinstance(box, tuple | list) or len(box) != 4 or not all(_finite(v) for v in box):
        return f"bbox_xyxy {box!r} is not four finite numbers"
    return None


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _where(image: EvaluationImage) -> str:
    return f"image {image.image_id} ({image.file_name!r})"


def _warnings(model: ModelInfo, dataset: EvaluationDataset, threshold: float) -> list[str]:
    warnings: list[str] = []
    if model.is_synthetic:
        warnings.append("Predictions come from a synthetic test detector, not a real model.")
    if dataset.metadata.provenance.synthetic:
        warnings.append("The dataset is synthetic; its metrics are not model accuracy.")
    if threshold > _HIGH_THRESHOLD_WARNING:
        warnings.append(
            f"confidence_threshold {threshold} drops low-scoring predictions, which truncates "
            "the precision-recall curve and can understate mAP."
        )
    return warnings


def _environment(model: ModelInfo) -> EnvironmentInfo:
    packages: dict[str, str] = {"soccervision": __version__}
    for name in ("numpy", "pillow", "pydantic"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    packages.update(model.framework_versions)
    return EnvironmentInfo(
        python=sys.version.split()[0], platform=platform.platform(), packages=packages
    )
