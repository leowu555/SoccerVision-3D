from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from soccervision.config import DetectorConfig, PipelineConfig
from soccervision.schemas import (
    Detection,
    FrameRecord,
    InferenceStatus,
    InvalidJobTransitionError,
    JobStatus,
    VideoMetadata,
    ensure_transition,
    is_terminal,
)


def _detection(**overrides: object) -> Detection:
    fields: dict[str, object] = {
        "video_id": "v1",
        "frame_index": 3,
        "timestamp_seconds": 0.1,
        "detection_id": "f3-d0",
        "class_name": "person",
        "confidence": 0.9,
        "bbox_xyxy": (10.0, 20.0, 30.0, 60.0),
        "model_id": "m",
    }
    fields.update(overrides)
    return Detection.model_validate(fields)


class TestDetection:
    def test_valid_detection_defaults_to_model_provenance(self) -> None:
        det = _detection()
        assert det.source == "model_detection"
        assert det.bbox_xyxy == (10.0, 20.0, 30.0, 60.0)

    @pytest.mark.parametrize(
        "bbox",
        [
            (30.0, 20.0, 10.0, 60.0),  # x_max < x_min
            (10.0, 60.0, 30.0, 20.0),  # y_max < y_min
            (10.0, 20.0, 10.0, 60.0),  # zero width
            (-1.0, 20.0, 30.0, 60.0),  # negative
        ],
    )
    def test_rejects_invalid_boxes(self, bbox: tuple[float, ...]) -> None:
        with pytest.raises(ValidationError):
            _detection(bbox_xyxy=bbox)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_rejects_out_of_range_confidence(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            _detection(confidence=confidence)

    def test_rejects_negative_frame_index_and_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            _detection(frame_index=-1)
        with pytest.raises(ValidationError):
            _detection(timestamp_seconds=-0.5)


class TestFrameRecord:
    def test_skipped_frame_has_unknown_detections(self) -> None:
        record = FrameRecord(
            frame_index=1,
            timestamp_seconds=0.0,
            inference_status=InferenceStatus.SKIPPED,
            detections=None,
        )
        assert record.detections is None

    def test_skipped_frame_cannot_carry_detections(self) -> None:
        with pytest.raises(ValidationError):
            FrameRecord(
                frame_index=1,
                timestamp_seconds=0.0,
                inference_status=InferenceStatus.SKIPPED,
                detections=[],
            )

    def test_analyzed_frame_requires_list(self) -> None:
        with pytest.raises(ValidationError):
            FrameRecord(
                frame_index=1,
                timestamp_seconds=0.0,
                inference_status=InferenceStatus.ANALYZED,
                detections=None,
            )

    def test_detection_frame_index_must_match(self) -> None:
        with pytest.raises(ValidationError):
            FrameRecord(
                frame_index=4,
                timestamp_seconds=0.1,
                inference_status=InferenceStatus.ANALYZED,
                detections=[_detection(frame_index=3)],
            )


class TestVideoMetadata:
    def _fields(self) -> dict[str, object]:
        return {
            "video_id": "v",
            "filename": "a.mp4",
            "duration_seconds": 1.0,
            "width": 1280,
            "height": 720,
            "fps": 30.0,
            "frame_count": 30,
            "created_at": datetime.now(UTC),
            "source_path": "v/source.mp4",
            "container_format": "mov,mp4",
            "video_codec": "h264",
            "size_bytes": 10,
        }

    def test_valid(self) -> None:
        assert VideoMetadata.model_validate(self._fields()).processing_status == "uploaded"

    @pytest.mark.parametrize(
        ("field", "value"), [("width", 0), ("height", -1), ("fps", 0.0), ("frame_count", 0)]
    )
    def test_rejects_non_positive(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            VideoMetadata.model_validate({**self._fields(), field: value})


class TestJobStateMachine:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (JobStatus.PENDING, JobStatus.PROCESSING),
            (JobStatus.PROCESSING, JobStatus.COMPLETED),
            (JobStatus.PROCESSING, JobStatus.FAILED),
            (JobStatus.PENDING, JobStatus.FAILED),
        ],
    )
    def test_allowed(self, current: JobStatus, target: JobStatus) -> None:
        ensure_transition(current, target)

    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (JobStatus.PENDING, JobStatus.COMPLETED),
            (JobStatus.COMPLETED, JobStatus.PROCESSING),
            (JobStatus.COMPLETED, JobStatus.FAILED),
            (JobStatus.FAILED, JobStatus.PROCESSING),
            (JobStatus.PROCESSING, JobStatus.PENDING),
        ],
    )
    def test_forbidden(self, current: JobStatus, target: JobStatus) -> None:
        with pytest.raises(InvalidJobTransitionError):
            ensure_transition(current, target)

    def test_terminal_states(self) -> None:
        assert is_terminal(JobStatus.COMPLETED)
        assert is_terminal(JobStatus.FAILED)
        assert not is_terminal(JobStatus.PENDING)
        assert not is_terminal(JobStatus.PROCESSING)


class TestConfig:
    def test_defaults_target_person_only(self) -> None:
        assert PipelineConfig().detector.target_classes == ("person",)

    def test_rejects_unknown_fields_and_bad_values(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"inference_stride": 0})
        with pytest.raises(ValidationError):
            DetectorConfig.model_validate({"confidence_threshold": 1.5})
        with pytest.raises(ValidationError):
            DetectorConfig.model_validate({"name": "yolov8"})
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"unexpected": True})

    def test_round_trips_through_json(self) -> None:
        config = PipelineConfig(
            detector=DetectorConfig(confidence_threshold=0.3), inference_stride=2
        )
        assert PipelineConfig.model_validate(config.model_dump(mode="json")) == config
