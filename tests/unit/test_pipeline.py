import json
from pathlib import Path

import pytest

from soccervision.config import DetectorConfig, PipelineConfig
from soccervision.errors import ModelUnavailableError
from soccervision.pipeline import (
    ANNOTATED_VIDEO_FILE,
    DETECTIONS_FILE,
    METADATA_FILE,
    PipelineRunError,
    ProgressUpdate,
    load_detections,
    load_run_metadata,
    run_detection_pipeline,
    runner,
)
from soccervision.schemas import ARTIFACT_SCHEMA_VERSION, InferenceStatus, RunStatus
from soccervision.schemas.video import VideoProbe
from soccervision.vision.video import VideoDecoder, probe_video
from support import FakeDetectorFactory, make_test_video


def _run(sample: Path, run_dir: Path, factory: FakeDetectorFactory, **config: object):  # type: ignore[no-untyped-def]
    return run_detection_pipeline(
        input_path=sample,
        run_dir=run_dir,
        config=PipelineConfig.model_validate(config),
        detector_factory=factory,
        video_id="vid",
        run_id="run",
    )


class TestSuccessfulRun:
    def test_writes_all_artifacts(self, sample_mp4: Path, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        result = _run(sample_mp4, run_dir, FakeDetectorFactory())
        assert {p.name for p in run_dir.iterdir()} == {
            ANNOTATED_VIDEO_FILE,
            DETECTIONS_FILE,
            METADATA_FILE,
        }
        assert result.metadata.status is RunStatus.COMPLETED
        assert result.metadata.schema_version == ARTIFACT_SCHEMA_VERSION

    def test_annotated_video_matches_source_frames_and_timing(
        self, sample_mp4: Path, tmp_path: Path
    ) -> None:
        result = _run(sample_mp4, tmp_path / "run", FakeDetectorFactory())
        source = probe_video(sample_mp4)
        output = probe_video(result.annotated_video_path)
        assert (output.width, output.height) == (source.width, source.height)
        assert output.frame_count == source.frame_count
        with VideoDecoder(result.annotated_video_path) as decoder:
            out_ts = [f.timestamp_seconds for f in decoder.frames()]
        frames = load_detections(result.detections_path).frames
        assert out_ts == pytest.approx([f.timestamp_seconds for f in frames])

    def test_filters_classes_threshold_and_clips_boxes(
        self, sample_mp4: Path, tmp_path: Path
    ) -> None:
        result = _run(sample_mp4, tmp_path / "run", FakeDetectorFactory())
        artifact = load_detections(result.detections_path)
        width, height = (
            artifact.coordinate_system.frame_width,
            artifact.coordinate_system.frame_height,
        )
        for frame in artifact.frames:
            assert frame.detections is not None
            # Fake emits: sports ball (dropped), 0.10 person (dropped), out-of-bounds person (clipped),
            # bright-rectangle person (kept).
            assert len(frame.detections) == 2
            for det in frame.detections:
                assert det.class_name == "person"
                assert det.confidence >= 0.5
                x1, y1, x2, y2 = det.bbox_xyxy
                assert 0 <= x1 < x2 <= width
                assert 0 <= y1 < y2 <= height
                assert det.frame_index == frame.frame_index
                assert det.model_id == "fake-bright-region-detector"
        stats = result.metadata.stats
        assert stats is not None
        assert stats.clipped_boxes == 12
        assert stats.total_detections == 24
        assert stats.mean_detections_per_analyzed_frame == pytest.approx(2.0)

    def test_detection_tracks_moving_object(self, sample_mp4: Path, tmp_path: Path) -> None:
        result = _run(sample_mp4, tmp_path / "run", FakeDetectorFactory())
        frames = load_detections(result.detections_path).frames
        tracked = [max(f.detections or [], key=lambda d: d.confidence).bbox_xyxy for f in frames]
        assert tracked[0][0] < tracked[-1][0]
        # mpeg4 compression blurs edges by a pixel or so; the rectangle is 16x24.
        assert tracked[0][2] - tracked[0][0] == pytest.approx(16, abs=2)

    def test_stride_marks_unanalyzed_frames_as_skipped(
        self, sample_mp4: Path, tmp_path: Path
    ) -> None:
        factory = FakeDetectorFactory()
        result = _run(sample_mp4, tmp_path / "run", factory, inference_stride=3)
        frames = load_detections(result.detections_path).frames
        assert len(frames) == 12
        analyzed = [f.frame_index for f in frames if f.inference_status is InferenceStatus.ANALYZED]
        assert analyzed == [0, 3, 6, 9]
        assert all(
            f.detections is None for f in frames if f.inference_status is InferenceStatus.SKIPPED
        )
        assert factory.instances[0].calls == 4
        stats = result.metadata.stats
        assert stats is not None
        assert (stats.decoded_frames, stats.analyzed_frames, stats.skipped_frames) == (12, 4, 8)
        assert stats.mean_detections_per_analyzed_frame == pytest.approx(2.0)

    def test_records_reproducibility_metadata(self, sample_mp4: Path, tmp_path: Path) -> None:
        result = _run(sample_mp4, tmp_path / "run", FakeDetectorFactory(), inference_stride=2)
        meta = load_run_metadata(result.metadata_path)
        assert meta.configuration["inference_stride"] == 2
        assert meta.configuration["detector"]["confidence_threshold"] == 0.5
        assert meta.model is not None and meta.model.is_synthetic
        assert any("SYNTHETIC" in w for w in meta.warnings)
        assert meta.input_sha256 and len(meta.input_sha256) == 64
        assert meta.input_video is not None and meta.input_video.frame_count == 12
        assert meta.pipeline.stages == ["decode", "detect", "annotate", "export"]
        assert meta.timing.total_seconds > 0
        assert "av" in meta.environment.packages

    def test_reports_real_progress(self, sample_mp4: Path, tmp_path: Path) -> None:
        updates: list[ProgressUpdate] = []
        run_detection_pipeline(
            input_path=sample_mp4,
            run_dir=tmp_path / "run",
            config=PipelineConfig(),
            detector_factory=FakeDetectorFactory(),
            video_id="v",
            run_id="r",
            progress=updates.append,
        )
        processed = [u.processed_frames for u in updates]
        assert processed == sorted(processed)
        assert updates[-1].fraction == 1.0
        assert updates[-1].processed_frames == 12

    def test_artifact_json_round_trips(self, sample_mp4: Path, tmp_path: Path) -> None:
        result = _run(sample_mp4, tmp_path / "run", FakeDetectorFactory())
        raw = json.loads(result.detections_path.read_text())
        assert raw["schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert raw["coordinate_system"] == {
            "type": "image_pixels",
            "origin": "top_left",
            "bbox_format": "xyxy",
            "units": "pixels",
            "frame_width": 96,
            "frame_height": 64,
        }
        reloaded = load_detections(result.detections_path)
        assert reloaded.model_dump(mode="json") == raw


class TestFailures:
    def test_detector_crash_writes_failed_metadata_and_removes_partials(
        self, sample_mp4: Path, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        with pytest.raises(PipelineRunError) as info:
            _run(sample_mp4, run_dir, FakeDetectorFactory(fail_on_call=5))
        assert info.value.code == "internal_error"
        assert "simulated detector crash" in info.value.message
        assert [p.name for p in run_dir.iterdir()] == [METADATA_FILE]
        meta = load_run_metadata(run_dir / METADATA_FILE)
        assert meta.status is RunStatus.FAILED
        assert meta.error is not None and meta.error.frame_index == 4
        assert meta.output_video is None

    def test_model_unavailable_is_reported(self, sample_mp4: Path, tmp_path: Path) -> None:
        def unavailable(config: DetectorConfig) -> FakeDetectorFactory:
            raise ModelUnavailableError("weights missing")

        with pytest.raises(PipelineRunError) as info:
            _run(sample_mp4, tmp_path / "run", unavailable)  # type: ignore[arg-type]
        assert info.value.code == "model_unavailable"
        meta = load_run_metadata(tmp_path / "run" / METADATA_FILE)
        assert meta.model is None and meta.error is not None

    def test_invalid_input(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"\x00" * 1000)
        with pytest.raises(PipelineRunError) as info:
            _run(bogus, tmp_path / "run", FakeDetectorFactory())
        assert info.value.code == "invalid_video"

    def test_duration_limit(self, tmp_path: Path) -> None:
        clip = make_test_video(tmp_path / "long.mp4", frames=36, fps=12)
        factory = FakeDetectorFactory()
        with pytest.raises(PipelineRunError) as info:
            _run(clip, tmp_path / "run", factory, max_duration_seconds=1.0)
        assert info.value.code == "video_limit_exceeded"
        assert factory.instances == []

    def test_duration_limit_enforced_on_decoded_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = make_test_video(tmp_path / "long.mp4", frames=36, fps=12)

        def underreporting_probe(path: Path, *, require_mp4: bool = False) -> VideoProbe:
            return probe_video(path, require_mp4=require_mp4).model_copy(
                update={"duration_seconds": 0.0}
            )

        monkeypatch.setattr(runner, "probe_video", underreporting_probe)
        with pytest.raises(PipelineRunError) as info:
            _run(clip, tmp_path / "run", FakeDetectorFactory(), max_duration_seconds=1.0)
        assert info.value.code == "video_limit_exceeded"
        assert "Decoded frames run past" in info.value.message
        assert not (tmp_path / "run" / ANNOTATED_VIDEO_FILE).exists()
