"""Metric tests on hand-built (synthetic) boxes. Expected values are derived by hand from the COCO
definitions: AP is the mean of interpolated precision at 101 recall points (0.00, 0.01, ..., 1.00).
"""

import pytest

pytest.importorskip("pycocotools")

from soccervision.errors import InvalidPredictionsError
from soccervision.evaluation import evaluate_detections
from support import prediction, synthetic_dataset

A = [10, 10, 100, 200]
B = [300, 50, 80, 160]
C = [500, 300, 60, 120]
HALF_RECALL_AP = 51 / 101  # precision 1.0 for recall 0.00..0.50, 0 beyond


def test_perfect_predictions() -> None:
    dataset = synthetic_dataset({1: [A, B], 2: [C]})
    preds = [prediction(1, A, 0.9), prediction(1, B, 0.8), prediction(2, C, 0.7)]

    m = evaluate_detections(dataset, preds)

    assert m.map_50 == pytest.approx(1.0)
    assert m.map_50_95 == pytest.approx(1.0)
    assert (m.precision, m.recall) == (1.0, 1.0)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (3, 0, 0)
    assert m.num_objects == 3 and m.num_predictions_above_threshold == 3
    assert m.dataset_synthetic is True
    assert "pycocotools" in m.evaluator


def test_missed_object() -> None:
    dataset = synthetic_dataset({1: [A, B]})

    m = evaluate_detections(dataset, [prediction(1, A, 0.9)])

    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 1)
    assert (m.precision, m.recall) == (1.0, 0.5)
    assert m.map_50 == pytest.approx(HALF_RECALL_AP)


def test_false_positive_ranked_below_true_positive() -> None:
    dataset = synthetic_dataset({1: [A]})
    preds = [prediction(1, A, 0.9), prediction(1, C, 0.8)]

    m = evaluate_detections(dataset, preds)

    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)
    assert (m.precision, m.recall) == (0.5, 1.0)
    assert m.map_50 == pytest.approx(1.0)


def test_false_positive_ranked_above_true_positive_lowers_ap() -> None:
    dataset = synthetic_dataset({1: [A]})
    preds = [prediction(1, C, 0.95), prediction(1, A, 0.9)]

    m = evaluate_detections(dataset, preds)

    assert (m.precision, m.recall) == (0.5, 1.0)
    assert m.map_50 == pytest.approx(0.5)


def test_duplicate_predictions_count_once() -> None:
    dataset = synthetic_dataset({1: [A]})
    shifted = [A[0] + 2, A[1] + 2, A[2], A[3]]
    preds = [prediction(1, A, 0.9), prediction(1, shifted, 0.8)]

    m = evaluate_detections(dataset, preds)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)
    assert m.map_50 == pytest.approx(1.0)

    above_duplicate = evaluate_detections(dataset, preds, confidence_threshold=0.85)
    assert (above_duplicate.true_positives, above_duplicate.false_positives) == (1, 0)


def test_image_without_objects_turns_predictions_into_false_positives() -> None:
    dataset = synthetic_dataset({1: [A], 2: []})
    preds = [prediction(1, A, 0.9), prediction(2, B, 0.8)]

    m = evaluate_detections(dataset, preds)

    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)
    assert m.num_images == 2


def test_empty_image_without_predictions_changes_nothing() -> None:
    preds = [prediction(1, A, 0.9), prediction(1, C, 0.4)]
    with_empty = evaluate_detections(synthetic_dataset({1: [A, B], 2: []}), preds)
    without = evaluate_detections(synthetic_dataset({1: [A, B]}), preds)

    assert with_empty.model_dump(exclude={"num_images"}) == without.model_dump(
        exclude={"num_images"}
    )


def test_images_without_predictions() -> None:
    dataset = synthetic_dataset({1: [A], 2: [B, C]})

    m = evaluate_detections(dataset, [prediction(1, A, 0.9)])
    assert (m.true_positives, m.false_negatives) == (1, 2)
    assert m.recall == pytest.approx(1 / 3)

    nothing = evaluate_detections(dataset, [])
    assert nothing.precision is None
    assert nothing.recall == 0.0
    assert (nothing.map_50, nothing.map_50_95) == (0.0, 0.0)
    assert nothing.false_negatives == 3


def test_dataset_without_objects_has_undefined_ap_and_recall() -> None:
    dataset = synthetic_dataset({1: [], 2: []})

    m = evaluate_detections(dataset, [prediction(1, A, 0.9)])

    assert m.map_50 is None and m.map_50_95 is None and m.recall is None
    assert (m.precision, m.false_positives) == (0.0, 1)


def test_confidence_threshold_filters_counts_but_not_ap() -> None:
    dataset = synthetic_dataset({1: [A, B]})
    preds = [prediction(1, A, 0.9), prediction(1, B, 0.3)]

    m = evaluate_detections(dataset, preds, confidence_threshold=0.5)

    assert (m.true_positives, m.false_negatives, m.recall) == (1, 1, 0.5)
    assert m.num_predictions == 2 and m.num_predictions_above_threshold == 1
    assert m.map_50 == pytest.approx(1.0)
    inclusive = evaluate_detections(dataset, preds, confidence_threshold=0.3)
    assert inclusive.true_positives == 2


def test_iou_threshold_controls_matching() -> None:
    dataset = synthetic_dataset({1: [[0, 0, 100, 100]]})
    preds = [prediction(1, [0, 0, 100, 160], 0.9)]  # IoU = 10000 / 16000 = 0.625

    at_50 = evaluate_detections(dataset, preds, iou_threshold=0.5)
    at_75 = evaluate_detections(dataset, preds, iou_threshold=0.75)

    assert (at_50.true_positives, at_75.true_positives) == (1, 0)
    assert at_75.false_positives == 1
    assert at_50.map_50 == pytest.approx(1.0)
    assert at_50.map_50_95 == pytest.approx(0.3)  # matched at 0.50, 0.55, 0.60 of 10 thresholds


def test_recall_by_object_size() -> None:
    small, medium, large = [5, 5, 20, 20], [100, 100, 50, 50], [300, 100, 120, 120]
    dataset = synthetic_dataset({1: [small, medium, large]})
    preds = [prediction(1, medium, 0.9), prediction(1, large, 0.8)]

    sizes = evaluate_detections(dataset, preds).recall_by_size

    small_bucket = sizes["small"]
    assert (small_bucket.num_objects, small_bucket.matched, small_bucket.recall) == (1, 0, 0.0)
    assert sizes["medium"].recall == 1.0
    assert sizes["large"].recall == 1.0


def test_empty_size_bucket_is_undefined() -> None:
    m = evaluate_detections(synthetic_dataset({1: [A]}), [prediction(1, A, 0.9)])
    assert m.recall_by_size["small"].recall is None
    assert m.recall_by_size["large"].recall == 1.0


def test_crowd_regions_are_neither_missed_nor_false_positives() -> None:
    crowd_box = [200, 0, 400, 100]
    dataset = synthetic_dataset({1: [A]}, crowd={1: [crowd_box]})
    inside_crowd = [250, 10, 40, 80]
    preds = [prediction(1, A, 0.9), prediction(1, inside_crowd, 0.8)]

    m = evaluate_detections(dataset, preds)

    assert m.num_objects == 1
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)


def test_max_detections_per_image_caps_lowest_scores() -> None:
    dataset = synthetic_dataset({1: [A, B, C]})
    preds = [prediction(1, A, 0.9), prediction(1, B, 0.8), prediction(1, C, 0.7)]

    m = evaluate_detections(dataset, preds, max_detections_per_image=2)

    assert (m.true_positives, m.false_negatives) == (2, 1)
    assert m.num_predictions_above_threshold == 2


def test_rejects_predictions_for_unknown_images() -> None:
    with pytest.raises(InvalidPredictionsError):
        evaluate_detections(synthetic_dataset({1: [A]}), [prediction(9, A, 0.9)])


def test_accepts_any_coco_iou_threshold() -> None:
    m = evaluate_detections(synthetic_dataset({1: [A]}), [], iou_threshold=0.6000000001)
    assert m.iou_threshold == 0.6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iou_threshold": 0.42},
        {"confidence_threshold": 1.5},
        {"max_detections_per_image": 0},
        {"class_name": "ball"},
    ],
)
def test_rejects_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        evaluate_detections(synthetic_dataset({1: [A]}), [], **kwargs)  # type: ignore[arg-type]


def test_map_matches_reference_cocoeval_summary() -> None:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_boxes = {1: [A, B, C], 2: [[20, 30, 15, 25], [400, 200, 90, 180]], 3: []}
    preds = [
        prediction(1, [12, 14, 98, 190], 0.92),
        prediction(1, [300, 50, 70, 150], 0.81),
        prediction(1, [305, 55, 80, 160], 0.77),
        prediction(1, [100, 300, 60, 60], 0.55),
        prediction(2, [22, 28, 14, 26], 0.66),
        prediction(2, [380, 210, 100, 170], 0.48),
        prediction(3, [10, 10, 30, 60], 0.35),
    ]
    dataset = synthetic_dataset(gt_boxes)
    ours = evaluate_detections(dataset, preds)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": i} for i in gt_boxes],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [
                {"id": k, "image_id": b.image_id, "category_id": 1, "iscrowd": 0, "area": b.area,
                 "bbox": [b.bbox_xyxy[0], b.bbox_xyxy[1], b.bbox_xyxy[2] - b.bbox_xyxy[0],
                          b.bbox_xyxy[3] - b.bbox_xyxy[1]]}
                for k, b in enumerate(dataset.annotations, start=1)
            ],
        }  # fmt: skip
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(
            [
                {"image_id": p.image_id, "category_id": 1, "score": p.score,
                 "bbox": [p.bbox_xyxy[0], p.bbox_xyxy[1], p.bbox_xyxy[2] - p.bbox_xyxy[0],
                          p.bbox_xyxy[3] - p.bbox_xyxy[1]]}
                for p in preds
            ]
        )  # fmt: skip
        reference = COCOeval(coco_gt, coco_dt, "bbox")
        reference.evaluate()
        reference.accumulate()
        reference.summarize()

    assert ours.map_50_95 == pytest.approx(reference.stats[0])
    assert ours.map_50 == pytest.approx(reference.stats[1])
    assert 0 < ours.map_50_95 < ours.map_50 < 1  # type: ignore[operator]


def test_results_are_deterministic() -> None:
    dataset = synthetic_dataset({1: [A, B], 2: [C]})
    preds = [prediction(1, A, 0.9), prediction(2, B, 0.6), prediction(2, C, 0.6)]
    assert evaluate_detections(dataset, preds) == evaluate_detections(dataset, preds)
