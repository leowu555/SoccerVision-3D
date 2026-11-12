"""Real RF-DETR inference. Opt-in; never substitutes fake detections.

Enable with:
    SOCCERVISION_RUN_INTEGRATION=1 \
    SOCCERVISION_INTEGRATION_VIDEO=path/to/clip_with_people.mp4 \
    pytest -m integration

Weights are read from SOCCERVISION_WEIGHTS_DIR (default ./models) and are not downloaded unless
SOCCERVISION_INTEGRATION_ALLOW_DOWNLOAD=1.
"""

import os
from pathlib import Path

import pytest

from soccervision.config import DetectorConfig, PipelineConfig
from soccervision.pipeline import load_detections, run_detection_pipeline
from soccervision.schemas import InferenceStatus
from soccervision.vision.detection import ObjectDetector, create_detector, weights_path_for

pytestmark = pytest.mark.integration

if os.environ.get("SOCCERVISION_RUN_INTEGRATION") != "1":
    pytest.skip(
        "set SOCCERVISION_RUN_INTEGRATION=1 to run real-inference tests", allow_module_level=True
    )

pytest.importorskip("rfdetr", reason="ML extras not installed: pip install -e '.[ml]'")

WEIGHTS_DIR = Path(os.environ.get("SOCCERVISION_WEIGHTS_DIR", "models"))
ALLOW_DOWNLOAD = os.environ.get("SOCCERVISION_INTEGRATION_ALLOW_DOWNLOAD") == "1"
VIDEO = os.environ.get("SOCCERVISION_INTEGRATION_VIDEO")


@pytest.fixture(scope="module")
def detector_config() -> DetectorConfig:
    if not ALLOW_DOWNLOAD and not weights_path_for("rfdetr-nano", WEIGHTS_DIR).is_file():
        pytest.skip(f"RF-DETR Nano weights not found in {WEIGHTS_DIR}")
    return DetectorConfig(name="rfdetr-nano", confidence_threshold=0.4, device="auto")


@pytest.fixture(scope="module")
def detector(detector_config: DetectorConfig) -> ObjectDetector:
    return create_detector(detector_config, weights_dir=WEIGHTS_DIR, allow_download=ALLOW_DOWNLOAD)


def test_model_identity_is_real(detector: ObjectDetector) -> None:
    info = detector.info
    assert info.is_synthetic is False
    assert info.model_id == "rfdetr-nano-coco"
    assert info.weights_md5 and len(info.weights_md5) == 32
    assert "rfdetr" in info.framework_versions


def test_pipeline_detects_people_in_real_footage(
    detector: ObjectDetector, detector_config: DetectorConfig, tmp_path: Path
) -> None:
    if not VIDEO:
        pytest.skip("set SOCCERVISION_INTEGRATION_VIDEO to a clip that contains people")
    result = run_detection_pipeline(
        input_path=Path(VIDEO),
        run_dir=tmp_path / "run",
        config=PipelineConfig(detector=detector_config, inference_stride=10),
        detector_factory=lambda _config: detector,
        video_id="integration",
        run_id="integration",
    )
    stats = result.metadata.stats
    assert stats is not None and stats.analyzed_frames > 0
    assert stats.total_detections > 0, "expected at least one person detection in real footage"
    artifact = load_detections(result.detections_path)
    for frame in artifact.frames:
        if frame.inference_status is InferenceStatus.ANALYZED:
            for det in frame.detections or []:
                assert det.class_name == "person"
                assert det.confidence >= detector_config.confidence_threshold
    assert (
        result.metadata.output_video is not None and result.metadata.output_video.browser_playable
    )
