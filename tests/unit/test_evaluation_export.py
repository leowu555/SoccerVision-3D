"""Prediction export with a scripted, synthetic detector. No model weights, no GPU.

Each test image is filled with a grey level equal to its image id, which is how the scripted
detector knows which image it is looking at.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from soccervision.config import DetectorConfig
from soccervision.errors import InferenceError, ModelUnavailableError, PredictionExportError
from soccervision.evaluation import (
    DEFAULT_EXPORT_CONFIDENCE,
    EvaluationDataset,
    PredictionExportMetadata,
    export_predictions,
    load_coco_predictions,
    parse_coco_dataset,
    resolve_category_ids,
    write_prediction_export,
)
from soccervision.schemas.detection import ModelInfo
from soccervision.vision.detection import RawDetection
from support import FAKE_MODEL_INFO, SYNTHETIC_PROVENANCE

W, H = 64, 48
PLAYER, BALL = 5, 9


class ScriptedDetector:
    """Returns the detections scripted for the image whose pixels equal its image id."""

    def __init__(self, script: dict[int, Any]) -> None:
        self.script = script
        self.calls = 0

    @property
    def info(self) -> ModelInfo:
        return FAKE_MODEL_INFO

    def detect(self, image_rgb: npt.NDArray[np.uint8]) -> Any:
        self.calls += 1
        assert image_rgb.dtype == np.uint8 and image_rgb.shape == (H, W, 3)
        return self.script[int(image_rgb[0, 0, 0])]


class Factory:
    def __init__(self, detector: Any = None, error: Exception | None = None) -> None:
        self.detector = detector
        self.error = error
        self.configs: list[DetectorConfig] = []

    def __call__(self, config: DetectorConfig) -> Any:
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        return self.detector


def person(confidence: float, box: Sequence[float]) -> RawDetection:
    x1, y1, x2, y2 = box
    return RawDetection("person", confidence, (x1, y1, x2, y2))


def make_dataset(
    image_ids: Sequence[int],
    *,
    categories: Sequence[tuple[int, str]] = ((PLAYER, "player"), (BALL, "ball")),
    category_map: dict[str, str] | None = None,
    file_names: dict[int, str] | None = None,
    size: tuple[int, int] = (W, H),
) -> EvaluationDataset:
    names = file_names or {}
    document = {
        "images": [
            {
                "id": image_id,
                "width": size[0],
                "height": size[1],
                "file_name": names.get(image_id, f"match_a/img_{image_id}.png"),
            }
            for image_id in image_ids
        ],
        "categories": [{"id": cid, "name": name} for cid, name in categories],
        "annotations": [],
    }
    return parse_coco_dataset(
        document,
        annotation_file="synthetic.json",
        annotation_sha256="a" * 64,
        category_map=category_map or {"player": "person"},
        provenance=SYNTHETIC_PROVENANCE,
    )


def write_images(root: Path, image_ids: Sequence[int], size: tuple[int, int] = (W, H)) -> Path:
    for image_id in image_ids:
        path = root / f"match_a/img_{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (image_id, image_id, image_id)).save(path)
    return root


def run(
    tmp_path: Path, script: dict[int, Any], **kwargs: Any
) -> tuple[Any, ScriptedDetector, Factory]:
    dataset = kwargs.pop("dataset", None) or make_dataset(list(script))
    write_images(tmp_path / "images", list(script))
    detector = ScriptedDetector(script)
    factory = Factory(detector)
    export = export_predictions(
        dataset, image_root=tmp_path / "images", detector_factory=factory, **kwargs
    )
    return export, detector, factory


# --- conversion ----------------------------------------------------------------------------------


def test_preserves_image_ids_category_ids_scores_and_converts_to_xywh(tmp_path: Path) -> None:
    script = {
        7: [person(0.9, (10, 20, 30, 44))],
        3: [person(0.4213, (1.5, 2.25, 11.5, 12.75)), person(0.05, (0, 0, 8, 8))],
    }

    export, detector, _ = run(tmp_path, script)

    results = [p.model_dump() for p in export.predictions]
    assert results == [
        {"image_id": 7, "category_id": PLAYER, "bbox": (10.0, 20.0, 20.0, 24.0), "score": 0.9},
        {"image_id": 3, "category_id": PLAYER, "bbox": (1.5, 2.25, 10.0, 10.5), "score": 0.4213},
        {"image_id": 3, "category_id": PLAYER, "bbox": (0.0, 0.0, 8.0, 8.0), "score": 0.05},
    ]
    assert detector.calls == 2
    assert export.metadata.class_to_category == {"person": PLAYER}


def test_images_without_detections_produce_no_results_but_are_counted(tmp_path: Path) -> None:
    export, detector, _ = run(tmp_path, {1: [], 2: [person(0.8, (0, 0, 10, 10))]})

    assert [p.image_id for p in export.predictions] == [2]
    assert detector.calls == 2
    stats = export.metadata.stats
    assert (stats.num_images, stats.images_without_predictions, stats.num_predictions) == (2, 1, 1)


def test_filters_other_classes_and_low_scores(tmp_path: Path) -> None:
    script = {
        1: [
            RawDetection("sports ball", 0.99, (1, 1, 5, 5)),
            person(0.009, (0, 0, 10, 10)),
            person(0.01, (0, 0, 12, 12)),
        ]
    }

    export, _, _ = run(tmp_path, script)

    assert [p.score for p in export.predictions] == [0.01]
    stats = export.metadata.stats
    assert (stats.discarded_other_classes, stats.discarded_below_threshold) == (1, 1)


def test_out_of_bounds_boxes_are_clipped_to_the_image(tmp_path: Path) -> None:
    script = {1: [person(0.9, (-5, -3, 20, 20)), person(0.8, (60, 40, 80, 60))]}

    export, _, _ = run(tmp_path, script)

    assert [p.bbox for p in export.predictions] == [(0, 0, 20, 20), (60, 40, W - 60, H - 40)]
    assert export.metadata.stats.clipped_boxes == 2


@pytest.mark.parametrize(
    "box",
    [
        (70, 10, 90, 20),  # entirely right of the image
        (10, -20, 20, -5),  # entirely above
        (20, 20, 10, 30),  # inverted x
        (10, 10, 10, 30),  # zero width
    ],
)
def test_degenerate_boxes_are_discarded(tmp_path: Path, box: tuple[float, ...]) -> None:
    export, _, _ = run(tmp_path, {1: [person(0.9, box), person(0.7, (1, 1, 9, 9))]})

    assert [p.score for p in export.predictions] == [0.7]
    assert export.metadata.stats.discarded_degenerate_boxes == 1


def test_numpy_scalars_from_a_real_model_are_accepted(tmp_path: Path) -> None:
    det = RawDetection("person", np.float32(0.5), tuple(np.float32(v) for v in (1, 2, 3, 4)))  # type: ignore[arg-type]

    export, _, _ = run(tmp_path, {1: [det]})

    assert export.predictions[0].bbox == (1.0, 2.0, 2.0, 2.0)
    assert type(export.predictions[0].score) is float


# --- category mapping ----------------------------------------------------------------------------


def test_category_with_the_class_name_is_preferred() -> None:
    dataset = make_dataset(
        [1],
        categories=[(1, "player"), (2, "person"), (3, "referee")],
        category_map={"player": "person", "person": "person", "referee": "person"},
    )
    assert resolve_category_ids(dataset, ["person"]) == {"person": 2}


def test_lowest_mapped_category_id_is_used_otherwise() -> None:
    dataset = make_dataset(
        [1],
        categories=[(8, "referee"), (4, "goalkeeper"), (6, "player")],
        category_map={"referee": "person", "goalkeeper": "person", "player": "person"},
    )
    assert resolve_category_ids(dataset, ["person"]) == {"person": 4}


def test_unsupported_class_mapping_fails_before_loading_the_model(tmp_path: Path) -> None:
    dataset = make_dataset([1])
    factory = Factory(ScriptedDetector({1: []}))

    with pytest.raises(PredictionExportError) as excinfo:
        export_predictions(
            dataset,
            image_root=write_images(tmp_path, [1]),
            detector_factory=factory,
            detector_config=DetectorConfig(target_classes=("car",)),
        )

    assert excinfo.value.code == "prediction_export_failed"
    assert "'car'" in excinfo.value.issues[0]
    assert factory.configs == []


# --- image problems (all reported before the model loads) ----------------------------------------


def test_missing_images_are_all_reported_and_nothing_is_skipped(tmp_path: Path) -> None:
    dataset = make_dataset([1, 2, 3])
    write_images(tmp_path, [2])
    factory = Factory(ScriptedDetector({}))

    with pytest.raises(PredictionExportError) as excinfo:
        export_predictions(dataset, image_root=tmp_path, detector_factory=factory)

    issues = excinfo.value.issues
    assert len(issues) == 2 and all("file not found" in i for i in issues)
    assert "image 1 " in issues[0] and "image 3 " in issues[1]
    assert factory.configs == []


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    write_images(tmp_path, [1], size=(W - 1, H))

    with pytest.raises(PredictionExportError, match="failed validation") as excinfo:
        export_predictions(make_dataset([1]), image_root=tmp_path, detector_factory=Factory())

    assert f"image is {W - 1}x{H}, dataset says {W}x{H}" in excinfo.value.issues[0]


def test_unreadable_image_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "match_a/img_1.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an image")

    with pytest.raises(PredictionExportError) as excinfo:
        export_predictions(make_dataset([1]), image_root=tmp_path, detector_factory=Factory())

    assert "unreadable image" in excinfo.value.issues[0]


def test_truncated_image_fails_when_decoded(tmp_path: Path) -> None:
    write_images(tmp_path, [1])
    path = tmp_path / "match_a/img_1.png"
    path.write_bytes(path.read_bytes()[:60])

    with pytest.raises(PredictionExportError, match="could not be decoded"):
        export_predictions(
            make_dataset([1]),
            image_root=tmp_path,
            detector_factory=Factory(ScriptedDetector({1: []})),
        )


@pytest.mark.parametrize(
    ("file_name", "message"),
    [
        ("../outside.png", "outside the image root"),
        ("match_a/../../outside.png", "outside the image root"),
        ("/etc/hosts", "must be relative"),
    ],
)
def test_file_names_cannot_escape_the_image_root(
    tmp_path: Path, file_name: str, message: str
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (W, H), (1, 1, 1)).save(tmp_path / "outside.png")
    dataset = make_dataset([1], file_names={1: file_name})

    with pytest.raises(PredictionExportError) as excinfo:
        export_predictions(dataset, image_root=root, detector_factory=Factory())

    assert message in excinfo.value.issues[0]


def test_symlinks_cannot_escape_the_image_root(tmp_path: Path) -> None:
    root = tmp_path / "images"
    (root / "match_a").mkdir(parents=True)
    Image.new("RGB", (W, H), (1, 1, 1)).save(tmp_path / "outside.png")
    (root / "match_a/img_1.png").symlink_to(tmp_path / "outside.png")

    with pytest.raises(PredictionExportError) as excinfo:
        export_predictions(make_dataset([1]), image_root=root, detector_factory=Factory())

    assert "outside the image root" in excinfo.value.issues[0]


def test_image_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(PredictionExportError, match="not a directory"):
        export_predictions(
            make_dataset([1]), image_root=tmp_path / "missing", detector_factory=Factory()
        )


# --- detector problems ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (None, "expected a sequence"),
        ([("person", 0.9, (0, 0, 1, 1))], "expected RawDetection"),
        ([RawDetection("person", math.nan, (0, 0, 1, 1))], "confidence"),
        ([RawDetection("person", 1.5, (0, 0, 1, 1))], "confidence"),
        ([RawDetection("person", 0.9, (0, 0, math.inf, 1))], "four finite numbers"),
        ([RawDetection("person", 0.9, (0, 0, 1))], "four finite numbers"),  # type: ignore[arg-type]
        ([RawDetection(None, 0.9, (0, 0, 1, 1))], "class_name"),  # type: ignore[arg-type]
    ],
)
def test_invalid_detector_output_is_rejected(tmp_path: Path, output: Any, message: str) -> None:
    with pytest.raises(PredictionExportError, match="invalid output") as excinfo:
        run(tmp_path, {1: output})

    assert message in excinfo.value.issues[0]
    assert "image 1 " in excinfo.value.issues[0]


def test_detector_crash_names_the_image(tmp_path: Path) -> None:
    class Crashing(ScriptedDetector):
        def detect(self, image_rgb: npt.NDArray[np.uint8]) -> Any:
            raise RuntimeError("simulated crash")

    write_images(tmp_path, [4])

    with pytest.raises(InferenceError, match=r"image 4 .*RuntimeError: simulated crash"):
        export_predictions(
            make_dataset([4]), image_root=tmp_path, detector_factory=Factory(Crashing({}))
        )


def test_model_initialization_failures(tmp_path: Path) -> None:
    write_images(tmp_path, [1])
    dataset = make_dataset([1])

    with pytest.raises(ModelUnavailableError, match="could not be created: boom"):
        export_predictions(
            dataset, image_root=tmp_path, detector_factory=Factory(error=RuntimeError("boom"))
        )

    original = ModelUnavailableError("weights missing")
    with pytest.raises(ModelUnavailableError) as excinfo:
        export_predictions(dataset, image_root=tmp_path, detector_factory=Factory(error=original))
    assert excinfo.value is original


# --- metadata and serialization ------------------------------------------------------------------


def test_default_threshold_is_low_and_recorded(tmp_path: Path) -> None:
    export, _, factory = run(tmp_path, {1: []})

    assert factory.configs[0].confidence_threshold == DEFAULT_EXPORT_CONFIDENCE == 0.01
    assert export.metadata.confidence_threshold == 0.01
    assert export.metadata.detector_config == factory.configs[0]


def test_high_threshold_is_respected_and_warned_about(tmp_path: Path) -> None:
    script = {1: [person(0.49, (0, 0, 9, 9)), person(0.5, (0, 0, 8, 8))]}

    export, _, _ = run(tmp_path, script, detector_config=DetectorConfig(confidence_threshold=0.5))

    assert [p.score for p in export.predictions] == [0.5]
    assert export.metadata.confidence_threshold == 0.5
    assert any("understate mAP" in w for w in export.metadata.warnings)


def test_reproducibility_metadata(tmp_path: Path) -> None:
    export, _, _ = run(tmp_path, {1: [person(0.9, (0, 0, 9, 9))], 2: []})
    meta = export.metadata

    assert meta.artifact_type == "coco_predictions_metadata"
    assert meta.dataset.name == "synthetic-test-fixture"
    assert meta.dataset.annotation_file == "synthetic.json"
    assert meta.dataset.annotation_sha256 == "a" * 64
    assert meta.dataset.synthetic is True
    assert meta.dataset.image_root == str(tmp_path / "images")
    assert meta.model == FAKE_MODEL_INFO
    assert (meta.stats.num_images, meta.stats.num_predictions) == (2, 1)
    timing = meta.timing
    assert timing.completed_at >= timing.started_at
    assert timing.total_seconds >= timing.model_load_seconds + timing.inference_seconds >= 0
    assert timing.mean_inference_ms is not None and timing.mean_inference_ms >= 0
    assert meta.environment.packages["soccervision"]
    assert any("synthetic test detector" in w for w in meta.warnings)
    assert any("dataset is synthetic" in w for w in meta.warnings)


def test_written_files_round_trip(tmp_path: Path) -> None:
    script = {3: [person(0.9, (10, 20, 30, 40))], 4: []}
    export, _, _ = run(tmp_path, script)
    out = tmp_path / "out"

    write_prediction_export(
        export,
        predictions_path=out / "predictions.json",
        metadata_path=out / "predictions.meta.json",
    )

    raw = (out / "predictions.json").read_bytes()
    assert json.loads(raw) == [
        {"image_id": 3, "category_id": PLAYER, "bbox": [10.0, 20.0, 20.0, 20.0], "score": 0.9}
    ]
    meta = PredictionExportMetadata.model_validate_json(
        (out / "predictions.meta.json").read_bytes()
    )
    assert meta == export.metadata
    assert hashlib.sha256(raw).hexdigest() == meta.predictions_sha256
    loaded = load_coco_predictions(out / "predictions.json", make_dataset([3, 4]))
    assert [(p.image_id, p.class_name, p.bbox_xyxy, p.score) for p in loaded] == [
        (3, "person", (10.0, 20.0, 30.0, 40.0), 0.9)
    ]
    assert sorted(p.name for p in out.iterdir()) == ["predictions.json", "predictions.meta.json"]


def test_empty_export_is_an_empty_list(tmp_path: Path) -> None:
    export, _, _ = run(tmp_path, {1: []})
    path = tmp_path / "predictions.json"

    write_prediction_export(export, predictions_path=path, metadata_path=tmp_path / "meta.json")

    assert json.loads(path.read_bytes()) == []


def test_exported_predictions_score_with_the_metrics_module(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    from soccervision.evaluation import evaluate_detections, parse_coco_predictions

    document = {
        "images": [{"id": 1, "width": W, "height": H, "file_name": "match_a/img_1.png"}],
        "categories": [{"id": PLAYER, "name": "player"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": PLAYER, "bbox": [4, 4, 20, 30]}],
    }
    dataset = parse_coco_dataset(
        document,
        annotation_file="synthetic.json",
        annotation_sha256="a" * 64,
        category_map={"player": "person"},
        provenance=SYNTHETIC_PROVENANCE,
    )
    export, _, _ = run(tmp_path, {1: [person(0.8, (4, 4, 24, 34))]}, dataset=dataset)

    predictions = parse_coco_predictions(json.loads(export.predictions_json()), dataset)
    metrics = evaluate_detections(dataset, predictions)

    assert metrics.map_50 == pytest.approx(1.0)
    assert (metrics.true_positives, metrics.false_positives) == (1, 0)


def test_write_failures_are_reported_and_leave_no_partial_files(tmp_path: Path) -> None:
    export, _, _ = run(tmp_path, {1: []})
    out = tmp_path / "out"
    (out / "predictions.json").mkdir(parents=True)

    with pytest.raises(PredictionExportError, match="Cannot write"):
        write_prediction_export(
            export, predictions_path=out / "predictions.json", metadata_path=out / "meta.json"
        )
    assert sorted(p.name for p in out.iterdir()) == ["predictions.json"]

    (out / "meta.json").mkdir()
    with pytest.raises(PredictionExportError, match=r"Cannot write .*meta\.json"):
        write_prediction_export(
            export, predictions_path=out / "other.json", metadata_path=out / "meta.json"
        )
    assert not (out / "other.json").exists()
    assert not list(out.glob(".*.tmp"))


def test_predictions_and_metadata_paths_must_differ(tmp_path: Path) -> None:
    export, _, _ = run(tmp_path, {1: []})

    with pytest.raises(ValueError, match="different files"):
        write_prediction_export(
            export, predictions_path=tmp_path / "x.json", metadata_path=tmp_path / "x.json"
        )
