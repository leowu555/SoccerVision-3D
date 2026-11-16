import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from soccervision.errors import (
    EvaluationUnavailableError,
    InvalidDatasetError,
    InvalidPredictionsError,
)
from soccervision.evaluation import (
    DatasetProvenance,
    evaluate_detections,
    load_coco_dataset,
    load_coco_predictions,
    parse_coco_dataset,
    parse_coco_predictions,
)
from support import synthetic_coco

FIXTURE = Path(__file__).parent.parent / "fixtures" / "evaluation" / "synthetic_soccer_coco.json"
SOCCER_MAP = {"player": "person", "referee": "person"}


def _parse(document: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return parse_coco_dataset(
        document, annotation_file="test.json", annotation_sha256="0" * 64, **kwargs
    )


def _issues(document: Any, **kwargs: Any) -> list[str]:
    with pytest.raises(InvalidDatasetError) as info:
        _parse(document, **kwargs)
    assert info.value.code == "invalid_dataset"
    return info.value.issues


class TestLoadValidDataset:
    def test_loads_fixture_with_category_mapping(self) -> None:
        dataset = load_coco_dataset(FIXTURE, category_map=SOCCER_MAP)
        meta = dataset.metadata

        assert meta.num_images == 3
        assert meta.num_objects == 3
        assert meta.num_crowd_regions == 1
        assert meta.num_ignored_annotations == 2
        assert meta.num_images_without_objects == 2
        assert meta.classes == ["person"]
        assert meta.dataset_categories == {1: "player", 2: "referee", 3: "ball"}
        assert {b.class_name for b in dataset.annotations} == {"person"}

    def test_records_provenance_and_checksum(self) -> None:
        meta = load_coco_dataset(FIXTURE, category_map=SOCCER_MAP).metadata

        assert meta.annotation_file == FIXTURE.name
        assert meta.annotation_sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        assert meta.provenance.name.startswith("SYNTHETIC")
        assert meta.provenance.version == "1.0"
        assert meta.provenance.license == "CC0-1.0"
        assert meta.provenance.source_url == "https://example.invalid/synthetic"

    def test_explicit_provenance_overrides_info(self) -> None:
        provenance = DatasetProvenance(name="custom", license="CC-BY-4.0", synthetic=True)
        meta = load_coco_dataset(FIXTURE, category_map=SOCCER_MAP, provenance=provenance).metadata
        assert meta.provenance == provenance

    def test_converts_xywh_to_xyxy_and_defaults_area(self) -> None:
        dataset = load_coco_dataset(FIXTURE, category_map=SOCCER_MAP)
        by_id = {b.annotation_id: b for b in dataset.annotations}

        assert by_id[100].bbox_xyxy == (100, 200, 140, 290)
        assert by_id[100].area == 40 * 90
        assert by_id[101].bbox_xyxy == (600.5, 300.25, 612.5, 330.25)
        assert by_id[101].area == 300
        assert by_id[104].is_crowd

    def test_images_without_annotations_are_kept(self) -> None:
        dataset = _parse(synthetic_coco({1: [[10, 10, 20, 40]], 2: []}))
        assert [i.image_id for i in dataset.images] == [1, 2]
        assert dataset.metadata.num_images_without_objects == 1

    def test_default_map_requires_person_category(self) -> None:
        document = synthetic_coco({1: []})
        document["categories"] = [{"id": 1, "name": "player"}]
        assert any("'person' is not a dataset category" in i for i in _issues(document))

    def test_box_touching_image_edge_is_valid(self) -> None:
        dataset = _parse(synthetic_coco({1: [[600, 400, 40, 80]]}, size=(640, 480)))
        assert dataset.annotations[0].bbox_xyxy == (600, 400, 640, 480)


class TestInvalidDataset:
    def test_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(InvalidDatasetError, match="not valid JSON"):
            load_coco_dataset(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidDatasetError, match="Cannot read"):
            load_coco_dataset(tmp_path / "missing.json")

    def test_top_level_must_be_object(self) -> None:
        with pytest.raises(InvalidDatasetError, match="JSON object"):
            _parse([1, 2])

    def test_missing_sections(self) -> None:
        issues = _issues({"images": []})
        assert issues == ["'annotations' must be a list", "'categories' must be a list"]

    def test_empty_dataset(self) -> None:
        assert "dataset contains no images" in _issues(synthetic_coco({}))

    def test_invalid_image_fields(self) -> None:
        document = synthetic_coco({1: [], 2: []})
        document["images"][0]["width"] = 0
        document["images"][1]["height"] = 720.5
        del document["images"][1]["file_name"]
        issues = _issues(document)
        assert any(i.startswith("images[0].width") for i in issues)
        assert any(i.startswith("images[1].height") for i in issues)
        assert any(i.startswith("images[1].file_name") for i in issues)

    def test_duplicate_ids_and_names(self) -> None:
        document = synthetic_coco({1: [[0, 0, 10, 10], [5, 5, 10, 10]], 2: []})
        document["images"][1]["id"] = 1
        document["annotations"][1]["id"] = 1
        document["categories"].append({"id": 1, "name": "person"})
        issues = _issues(document)
        assert "duplicate image id 1 (2 times)" in issues
        assert "duplicate annotation id 1 (2 times)" in issues
        assert "duplicate category id 1 (2 times)" in issues
        assert "duplicate category name 'person' (2 times)" in issues

    def test_dangling_references(self) -> None:
        document = synthetic_coco({1: [[0, 0, 10, 10], [5, 5, 10, 10]]})
        document["annotations"][0]["image_id"] = 99
        document["annotations"][1]["category_id"] = 7
        issues = _issues(document)
        assert any("image_id 99 does not exist" in i for i in issues)
        assert any("category_id 7 does not exist" in i for i in issues)

    def test_invalid_category_map(self) -> None:
        issues = _issues(
            synthetic_coco({1: []}), category_map={"person": "ball", "coach": "person"}
        )
        assert any("'ball' is not a supported class" in i for i in issues)
        assert any("'coach' is not a dataset category" in i for i in issues)

    def test_reports_every_issue(self) -> None:
        document = synthetic_coco({1: [[0, 0, 0, 10], [-1, 0, 10, 10], [0, 0, 10, 10]]})
        document["annotations"][2]["iscrowd"] = 2
        with pytest.raises(InvalidDatasetError) as info:
            _parse(document)
        assert len(info.value.issues) == 3
        assert "3 issue(s)" in info.value.message


class TestBoundingBoxValidation:
    @pytest.mark.parametrize(
        ("bbox", "expected"),
        [
            ([10, 10, 0, 20], "positive width and height"),
            ([10, 10, 20, -5], "positive width and height"),
            ([-0.5, 10, 20, 20], "negative coordinates"),
            ([10, -3, 20, 20], "negative coordinates"),
            ([630, 10, 20, 20], "outside the 640x480 image"),
            ([10, 470, 20, 20], "outside the 640x480 image"),
        ],
    )
    def test_rejects_invalid_geometry(self, bbox: list[float], expected: str) -> None:
        issues = _issues(synthetic_coco({1: [bbox]}))
        assert len(issues) == 1 and expected in issues[0]

    @pytest.mark.parametrize(
        "bbox", [[10, 10, 20], [10, 10, 20, 20, 1], ["10", 10, 20, 20], [float("nan"), 10, 20, 20]]
    )
    def test_rejects_malformed_bbox(self, bbox: list[Any]) -> None:
        issues = _issues(synthetic_coco({1: [bbox]}))
        assert issues and all(i.startswith("annotations[0].bbox") for i in issues)

    def test_rejects_non_positive_area(self) -> None:
        document = synthetic_coco({1: [[10, 10, 20, 20]]})
        document["annotations"][0]["area"] = 0
        assert any(i.startswith("annotations[0].area") for i in _issues(document))


def test_metrics_report_missing_pycocotools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pycocotools.coco", None)
    with pytest.raises(EvaluationUnavailableError, match=r"\.\[eval\]"):
        evaluate_detections(_parse(synthetic_coco({1: []})), [])


class TestPredictionsLoading:
    def _dataset(self):  # type: ignore[no-untyped-def]
        return load_coco_dataset(FIXTURE, category_map=SOCCER_MAP)

    def test_loads_and_maps_categories(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {"image_id": 10, "category_id": 1, "bbox": [100, 200, 40, 90], "score": 0.9},
                    {"image_id": 10, "category_id": 2, "bbox": [900, 250, 35, 85], "score": 0.4},
                    {"image_id": 12, "category_id": 3, "bbox": [5, 5, 4, 4], "score": 0.7},
                ]
            )
        )
        predictions = load_coco_predictions(path, self._dataset())

        assert len(predictions) == 2  # the 'ball' prediction is outside the mapped classes
        assert predictions[0].bbox_xyxy == (100, 200, 140, 290)
        assert {p.class_name for p in predictions} == {"person"}

    def test_rejects_mismatched_predictions(self) -> None:
        good = {"image_id": 10, "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}
        bad = [
            {**good, "image_id": 99},
            {**good, "category_id": 42},
            {**good, "score": 1.5},
            {**good, "bbox": [1, 1, 0, 5]},
            {k: v for k, v in good.items() if k != "score"},
        ]
        with pytest.raises(InvalidPredictionsError) as info:
            parse_coco_predictions(copy.deepcopy(bad), self._dataset())
        assert info.value.code == "invalid_predictions"
        assert len(info.value.issues) == 5

    def test_top_level_must_be_list(self) -> None:
        with pytest.raises(InvalidPredictionsError, match="JSON list"):
            parse_coco_predictions({"results": []}, self._dataset())
