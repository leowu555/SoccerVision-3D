from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from soccervision.errors import InvalidVideoError
from soccervision.vision.video import (
    AnnotatedVideoWriter,
    VideoDecoder,
    probe_video,
    select_output_codec,
)
from support import make_test_video


class TestProbe:
    def test_extracts_metadata(self, sample_mp4: Path) -> None:
        probe = probe_video(sample_mp4)
        assert (probe.width, probe.height) == (96, 64)
        assert probe.fps == pytest.approx(12.0)
        assert probe.frame_count == 12
        assert probe.duration_seconds == pytest.approx(1.0, abs=0.1)
        assert "mp4" in probe.container_format.split(",")
        assert probe.video_codec == "mpeg4"
        assert probe.size_bytes == sample_mp4.stat().st_size
        assert probe.has_audio is False

    def test_rejects_missing_and_empty_files(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidVideoError, match="not found"):
            probe_video(tmp_path / "missing.mp4")
        empty = tmp_path / "empty.mp4"
        empty.touch()
        with pytest.raises(InvalidVideoError, match="empty"):
            probe_video(empty)

    def test_rejects_text_renamed_to_mp4(self, tmp_path: Path) -> None:
        fake = tmp_path / "notes.mp4"
        fake.write_text("this is definitely not a video file " * 20)
        with pytest.raises(InvalidVideoError):
            probe_video(fake)

    def test_rejects_non_mp4_container_when_required(self, tmp_path: Path) -> None:
        avi = make_test_video(tmp_path / "clip.avi", container_format="avi")
        renamed = tmp_path / "clip.mp4"
        renamed.write_bytes(avi.read_bytes())
        with pytest.raises(InvalidVideoError, match="MP4"):
            probe_video(renamed)
        assert probe_video(avi, require_mp4=False).frame_count == 12

    def test_rejects_truncated_mp4(self, sample_mp4: Path, tmp_path: Path) -> None:
        truncated = tmp_path / "truncated.mp4"
        truncated.write_bytes(sample_mp4.read_bytes()[:200])
        with pytest.raises(InvalidVideoError):
            probe_video(truncated)


class TestDecodeEncode:
    def test_decoder_yields_rgb_frames_with_pts_timestamps(self, sample_mp4: Path) -> None:
        with VideoDecoder(sample_mp4) as decoder:
            frames = list(decoder.frames())
        assert [f.index for f in frames] == list(range(12))
        assert all(f.image.shape == (64, 96, 3) and f.image.dtype == np.uint8 for f in frames)
        assert all(f.timestamp_source == "pts" for f in frames)
        assert [f.timestamp_seconds for f in frames] == pytest.approx([i / 12 for i in range(12)])

    def test_writer_preserves_timestamps(self, sample_mp4: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.mp4"
        with VideoDecoder(sample_mp4) as decoder:
            source = list(decoder.frames())
            with AnnotatedVideoWriter(
                out, width=96, height=64, frame_rate=decoder.frame_rate, time_base=decoder.time_base
            ) as writer:
                for frame in source:
                    writer.write(frame.image, frame.pts or 0)
        with VideoDecoder(out) as decoder:
            written = list(decoder.frames())
        assert len(written) == len(source)
        assert [f.timestamp_seconds for f in written] == pytest.approx(
            [f.timestamp_seconds for f in source]
        )

    def test_writer_pads_odd_dimensions(self, tmp_path: Path) -> None:
        out = tmp_path / "odd.mp4"
        with AnnotatedVideoWriter(
            out, width=33, height=21, frame_rate=Fraction(10), time_base=Fraction(1, 10)
        ) as writer:
            for i in range(3):
                writer.write(np.zeros((21, 33, 3), dtype=np.uint8), i)
        assert writer.padded
        probe = probe_video(out)
        assert (probe.width, probe.height) == (34, 22)

    def test_codec_selection_prefers_h264_when_available(self) -> None:
        codec = select_output_codec()
        assert codec in {"libx264", "h264_videotoolbox", "mpeg4"}
