"""Exception hierarchy shared by the library, CLI, and API.

Each error carries a stable machine-readable ``code`` so API clients can branch on it without
parsing messages.
"""


class SoccerVisionError(Exception):
    """Base class for all expected, user-reportable failures."""

    code = "soccervision_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidVideoError(SoccerVisionError):
    """The input is not a decodable video in a supported container."""

    code = "invalid_video"


class VideoLimitExceededError(SoccerVisionError):
    """The input exceeds a configured size or duration limit."""

    code = "video_limit_exceeded"


class DeviceUnavailableError(SoccerVisionError):
    """An explicitly requested compute device is not available."""

    code = "device_unavailable"


class ModelUnavailableError(SoccerVisionError):
    """The detector cannot be loaded (missing dependency, missing weights, or load failure)."""

    code = "model_unavailable"


class InferenceError(SoccerVisionError):
    """The detector failed while running on a frame."""

    code = "inference_failed"


class VideoEncodingError(SoccerVisionError):
    """The annotated output video could not be written."""

    code = "video_encoding_failed"
