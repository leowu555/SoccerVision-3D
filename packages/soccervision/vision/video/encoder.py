"""Annotated video encoding.

Prefers H.264 (yuv420p) in MP4 with ``+faststart`` so the result plays in all major browsers.
Falls back to MPEG-4 Part 2 only if no H.264 encoder works, and reports that the output is not
browser-playable.
"""

import contextlib
import functools
import io
import logging
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self

import av
import av.container
import av.error
import av.video
import numpy as np
import numpy.typing as npt

from soccervision.errors import VideoEncodingError

logger = logging.getLogger(__name__)

H264_ENCODERS = ("libx264", "h264_videotoolbox")
FALLBACK_ENCODER = "mpeg4"
PIXEL_FORMAT = "yuv420p"

_ENCODER_OPTIONS: dict[str, dict[str, str]] = {
    "libx264": {"crf": "23", "preset": "veryfast"},
    "h264_videotoolbox": {"realtime": "0"},
    "mpeg4": {"qscale": "4"},
}


def _add_video_stream(
    container: av.container.OutputContainer,
    codec_name: str,
    rate: Fraction | int,
    options: dict[str, str] | None = None,
) -> av.video.VideoStream:
    stream = container.add_stream(codec_name, rate=rate, options=options or {})
    if not isinstance(stream, av.video.VideoStream):
        raise VideoEncodingError(f"Codec {codec_name} is not a video encoder.")
    return stream


@functools.cache
def encoder_works(codec_name: str) -> bool:
    """Check that an encoder exists *and* can encode, by writing two tiny frames in memory."""
    try:
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4") as container:
            stream = _add_video_stream(container, codec_name, 30)
            stream.width, stream.height, stream.pix_fmt = 64, 64, PIXEL_FORMAT
            for i in range(2):
                frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
                frame = frame.reformat(format=PIXEL_FORMAT)
                frame.pts = i
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        return buffer.tell() > 0
    except (av.error.FFmpegError, ValueError, OSError, VideoEncodingError) as exc:
        logger.debug("Encoder %s unavailable: %s", codec_name, exc)
        return False


def select_output_codec(preferred: tuple[str, ...] = H264_ENCODERS) -> str:
    for name in (*preferred, FALLBACK_ENCODER):
        if encoder_works(name):
            return name
    raise VideoEncodingError("No usable video encoder found in the bundled FFmpeg build.")


def is_browser_playable(codec_name: str) -> bool:
    return codec_name in H264_ENCODERS


class AnnotatedVideoWriter:
    """Writes RGB frames with caller-supplied PTS, preserving source timing."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        frame_rate: Fraction,
        time_base: Fraction,
        codec_name: str | None = None,
    ) -> None:
        self.path = path
        self.source_width = width
        self.source_height = height
        # yuv420p requires even dimensions; pad right/bottom so pixel coordinates are unchanged.
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        self.padded = (self.width, self.height) != (width, height)
        self.frame_rate = frame_rate
        self.time_base = time_base
        self.codec_name = codec_name or select_output_codec()
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.VideoStream | None = None
        self._last_pts: int | None = None
        self.frames_written = 0

    @property
    def browser_playable(self) -> bool:
        return is_browser_playable(self.codec_name)

    def __enter__(self) -> Self:
        try:
            self._container = av.open(
                str(self.path), mode="w", format="mp4", options={"movflags": "+faststart"}
            )
            stream = _add_video_stream(
                self._container,
                self.codec_name,
                self.frame_rate,
                _ENCODER_OPTIONS.get(self.codec_name),
            )
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = PIXEL_FORMAT
            stream.time_base = self.time_base
            stream.codec_context.time_base = self.time_base
            self._stream = stream
        except (av.error.FFmpegError, ValueError, VideoEncodingError) as exc:
            self._close_quietly()
            raise VideoEncodingError(
                f"Could not initialise {self.codec_name} encoder: {exc}"
            ) from exc
        return self

    def write(self, image_rgb: npt.NDArray[np.uint8], pts: int) -> None:
        if self._container is None or self._stream is None:
            raise RuntimeError("AnnotatedVideoWriter must be used as a context manager")
        if self.padded:
            image_rgb = np.pad(
                image_rgb,
                (
                    (0, self.height - image_rgb.shape[0]),
                    (0, self.width - image_rgb.shape[1]),
                    (0, 0),
                ),
            )
        # Encoders require strictly increasing PTS; nudge duplicates from malformed sources.
        if self._last_pts is not None and pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image_rgb), format="rgb24")
        frame = frame.reformat(format=PIXEL_FORMAT)
        frame.pts = pts
        frame.time_base = self.time_base
        try:
            self._container.mux(self._stream.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncodingError(f"Encoding frame failed: {exc}") from exc
        self.frames_written += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._container is None or self._stream is None:
            return
        try:
            if exc_type is None:
                self._container.mux(self._stream.encode(None))
            self._container.close()
        except av.error.FFmpegError as err:
            if exc_type is None:
                raise VideoEncodingError(f"Finalising video failed: {err}") from err
        finally:
            self._container = None
            self._stream = None

    def _close_quietly(self) -> None:
        if self._container is not None:
            with contextlib.suppress(av.error.FFmpegError):
                self._container.close()
            self._container = None
