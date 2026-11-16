"""Test doubles and fixture generators. Nothing here is used outside the test suite."""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import numpy.typing as npt

from soccervision.config import DetectorConfig
from soccervision.evaluation import (
    DatasetProvenance,
    EvaluationDataset,
    Prediction,
    parse_coco_dataset,
)
from soccervision.schemas.detection import ModelInfo
from soccervision.vision.detection import RawDetection

FAKE_MODEL_INFO = ModelInfo(
    model_id="fake-bright-region-detector",
    family="test",
    variant="fake",
    label_space="test",
    license="n/a",
    device="cpu",
    is_synthetic=True,
)


def make_test_video(
    path: Path,
    *,
    frames: int = 12,
    width: int = 96,
    height: int = 64,
    fps: int = 12,
    codec: str = "mpeg4",
    container_format: str = "mp4",
) -> Path:
    """Write a synthetic clip: a white 16x24 rectangle moving right over a dark background."""
    with av.open(str(path), mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(frames):
            image = np.full((height, width, 3), 20, dtype=np.uint8)
            x = min(4 + 4 * i, width - 20)
            image[20:44, x : x + 16] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = i
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path


class FakeDetector:
    """Deterministic, explicitly synthetic detector.

    Finds the bright rectangle and reports it as a 'person', plus distractors that the pipeline
    must filter or fix: a non-target class, a sub-threshold person, and an out-of-bounds person.
    """

    def __init__(self, config: DetectorConfig, *, fail_on_call: int | None = None) -> None:
        self.config = config
        self.calls = 0
        self.fail_on_call = fail_on_call

    @property
    def info(self) -> ModelInfo:
        return FAKE_MODEL_INFO

    def detect(self, image_rgb: npt.NDArray[np.uint8]) -> Sequence[RawDetection]:
        self.calls += 1
        if self.fail_on_call is not None and self.calls >= self.fail_on_call:
            raise RuntimeError("simulated detector crash")
        height, width = image_rgb.shape[:2]
        ys, xs = np.nonzero(image_rgb[..., 0] > 128)
        out = [
            RawDetection("sports ball", 0.95, (1.0, 1.0, 5.0, 5.0)),
            RawDetection("person", 0.10, (0.0, 0.0, 10.0, 10.0)),
            RawDetection("person", 0.80, (width - 8.0, height - 8.0, width + 12.0, height + 5.0)),
        ]
        if len(xs):
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            out.append(RawDetection("person", 0.90, box))
        return out


class FakeDetectorFactory:
    def __init__(self, **kwargs: int | None) -> None:
        self.kwargs = kwargs
        self.instances: list[FakeDetector] = []

    def __call__(self, config: DetectorConfig) -> FakeDetector:
        detector = FakeDetector(config, **self.kwargs)
        self.instances.append(detector)
        return detector


SYNTHETIC_PROVENANCE = DatasetProvenance(
    name="synthetic-test-fixture",
    description="Hand-built boxes for unit tests. Not footage, not model accuracy.",
    synthetic=True,
)


def synthetic_coco(
    boxes: dict[int, Sequence[Sequence[float]]],
    *,
    size: tuple[int, int] = (640, 480),
    crowd: dict[int, Sequence[Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """A COCO document with one 'person' category: {image_id: [[x, y, w, h], ...]}."""
    width, height = size
    annotations: list[dict[str, Any]] = []
    for is_crowd, source in ((0, boxes), (1, crowd or {})):
        for image_id, image_boxes in source.items():
            for bbox in image_boxes:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": list(bbox),
                        "iscrowd": is_crowd,
                    }
                )
    return {
        "info": {"description": "SYNTHETIC test fixture"},
        "images": [
            {"id": image_id, "width": width, "height": height, "file_name": f"frame_{image_id}.jpg"}
            for image_id in sorted(boxes)
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": annotations,
    }


def synthetic_dataset(
    boxes: dict[int, Sequence[Sequence[float]]], **kwargs: Any
) -> EvaluationDataset:
    return parse_coco_dataset(
        synthetic_coco(boxes, **kwargs),
        annotation_file="synthetic.json",
        annotation_sha256="0" * 64,
        provenance=SYNTHETIC_PROVENANCE,
    )


def prediction(image_id: int, bbox_xywh: Sequence[float], score: float) -> Prediction:
    x, y, w, h = bbox_xywh
    return Prediction(
        image_id=image_id, class_name="person", bbox_xyxy=(x, y, x + w, y + h), score=score
    )
