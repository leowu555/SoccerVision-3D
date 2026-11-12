"""Sequential frame decoding with presentation timestamps.

Timestamps are ``(pts - stream.start_time) * time_base``: seconds since the first presentable video
frame. The annotated output video is written with the same relative timestamps, so a detection's
``timestamp_seconds`` equals the annotated video's ``currentTime`` in a browser. Variable frame
rate is preserved because real PTS values are used rather than ``index / fps``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import av
import av.container
import av.error
import av.video
import numpy as np
import numpy.typing as npt

from soccervision.errors import InvalidVideoError
from soccervision.schemas.detection import TimestampSource


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    index: int
    pts: int | None
    """Presentation timestamp in ``time_base`` units, relative to the stream start."""
    timestamp_seconds: float
    timestamp_source: TimestampSource
    image: npt.NDArray[np.uint8]
    """RGB, shape (height, width, 3)."""


class VideoDecoder:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._container: av.container.InputContainer | None = None
        self._stream: av.video.VideoStream | None = None

    def __enter__(self) -> Self:
        try:
            self._container = av.open(str(self._path), mode="r")
        except av.error.FFmpegError as exc:
            raise InvalidVideoError(f"Could not open video for decoding: {exc}") from exc
        if not self._container.streams.video:
            self._container.close()
            raise InvalidVideoError("File has no video stream.")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._stream = None

    @property
    def stream(self) -> av.video.VideoStream:
        if self._stream is None:
            raise RuntimeError("VideoDecoder must be used as a context manager")
        return self._stream

    @property
    def time_base(self) -> Fraction:
        tb = self.stream.time_base
        if tb is None:
            raise InvalidVideoError("Video stream has no time base.")
        return Fraction(tb.numerator, tb.denominator)

    @property
    def frame_rate(self) -> Fraction:
        stream = self.stream
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if not rate:
            raise InvalidVideoError("Could not determine the video frame rate.")
        return Fraction(rate.numerator, rate.denominator)

    def frames(self) -> Iterator[DecodedFrame]:
        if self._container is None:
            raise RuntimeError("VideoDecoder must be used as a context manager")
        stream = self.stream
        time_base = self.time_base
        frame_interval_pts = 1 / (self.frame_rate * time_base)
        start_pts = stream.start_time or 0
        last_pts: int | None = None

        try:
            for index, frame in enumerate(self._container.decode(stream)):
                source: TimestampSource = "pts"
                if frame.pts is None:
                    source = "estimated"
                    rel_pts = 0 if last_pts is None else last_pts + round(frame_interval_pts)
                else:
                    rel_pts = max(frame.pts - start_pts, 0)
                last_pts = rel_pts
                yield DecodedFrame(
                    index=index,
                    pts=rel_pts,
                    timestamp_seconds=float(rel_pts * time_base),
                    timestamp_source=source,
                    image=cast(npt.NDArray[np.uint8], frame.to_ndarray(format="rgb24")),
                )
        except av.error.FFmpegError as exc:
            raise InvalidVideoError(f"Decoding failed: {exc}") from exc
