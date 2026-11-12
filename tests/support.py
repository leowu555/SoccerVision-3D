"""Test doubles and fixture generators. Nothing here is used outside the test suite."""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

from soccervision.config import DetectorConfig
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
