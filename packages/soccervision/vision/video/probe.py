"""Media inspection. Validates the actual container and stream, not the file extension."""

from pathlib import Path

import av
import av.error
from av.rational import AVRational

from soccervision.errors import InvalidVideoError
from soccervision.schemas.video import FrameRateMode, VideoProbe

# ISO base media file format: bytes 4-8 hold the first box type.
_ISO_BMFF_LEADING_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"}
_MP4_DEMUXER_ALIASES = {"mp4", "mov"}
_VFR_TOLERANCE = 0.01


def has_mp4_signature(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(12)
    return len(head) >= 8 and head[4:8] in _ISO_BMFF_LEADING_BOXES


def _frame_rate_mode(average: AVRational | None, base: AVRational | None) -> FrameRateMode:
    if not average or not base:
        return "unknown"
    return (
        "variable"
        if abs(float(average) - float(base)) / float(base) > _VFR_TOLERANCE
        else "constant"
    )


def probe_video(path: Path, *, require_mp4: bool = True) -> VideoProbe:
    """Inspect a video file and verify its first frame decodes.

    Raises:
        InvalidVideoError: if the file is missing, not a supported container, has no video
            stream, or cannot be decoded.
    """
    if not path.is_file():
        raise InvalidVideoError(f"Video file not found: {path.name}")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise InvalidVideoError("Video file is empty.")
    if require_mp4 and not has_mp4_signature(path):
        raise InvalidVideoError(
            "File content is not an MP4 container (missing ISO-BMFF signature)."
        )

    try:
        container = av.open(str(path), mode="r")
    except (av.error.FFmpegError, UnicodeDecodeError) as exc:
        raise InvalidVideoError(f"Could not open media: {exc}") from exc

    with container:
        format_name = container.format.name
        if require_mp4 and not _MP4_DEMUXER_ALIASES & set(format_name.split(",")):
            raise InvalidVideoError(f"Unsupported container '{format_name}'; only MP4 is accepted.")
        if not container.streams.video:
            raise InvalidVideoError("File has no video stream.")

        stream = container.streams.video[0]
        codec_context = stream.codec_context
        width, height = codec_context.width, codec_context.height
        if width <= 0 or height <= 0:
            raise InvalidVideoError("Video stream reports invalid dimensions.")

        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if not rate:
            raise InvalidVideoError("Could not determine the video frame rate.")
        time_base = stream.time_base
        if time_base is None:
            raise InvalidVideoError("Video stream has no time base.")

        start_time_seconds = (
            float(stream.start_time * time_base) if stream.start_time is not None else 0.0
        )
        if stream.duration:
            duration_seconds = float(stream.duration * time_base)
        elif container.duration:
            duration_seconds = container.duration / av.time_base
        else:
            duration_seconds = 0.0

        frame_count: int | None = stream.frames or None
        frame_rate_mode = _frame_rate_mode(stream.average_rate, stream.base_rate)
        has_audio = bool(container.streams.audio)
        codec_name = codec_context.name

        try:
            first_frame = next(container.decode(stream), None)
        except av.error.FFmpegError as exc:
            raise InvalidVideoError(
                f"Video stream could not be decoded ({codec_name}): {exc}"
            ) from exc
        if first_frame is None:
            raise InvalidVideoError("Video stream contains no decodable frames.")

    if frame_count is None:
        frame_count = _count_video_packets(path) or None

    return VideoProbe(
        container_format=format_name,
        video_codec=codec_name,
        width=width,
        height=height,
        fps=float(rate),
        duration_seconds=max(duration_seconds, 0.0),
        frame_count=frame_count,
        time_base=f"{time_base.numerator}/{time_base.denominator}",
        start_time_seconds=start_time_seconds,
        frame_rate_mode=frame_rate_mode,
        has_audio=has_audio,
        size_bytes=size_bytes,
    )


def _count_video_packets(path: Path) -> int:
    """Count video packets without decoding; used when the container has no frame index."""
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) if packet.size > 0)
