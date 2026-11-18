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


_MAX_REPORTED_ISSUES = 20


class _IssueListError(SoccerVisionError):
    """An error that reports every validation issue found, not just the first."""

    def __init__(self, summary: str, issues: list[str]) -> None:
        shown = "\n".join(f"  - {issue}" for issue in issues[:_MAX_REPORTED_ISSUES])
        hidden = len(issues) - _MAX_REPORTED_ISSUES
        more = f"\n  ... and {hidden} more" if hidden > 0 else ""
        super().__init__(f"{summary} ({len(issues)} issue(s)):\n{shown}{more}")
        self.issues = issues


class InvalidDatasetError(_IssueListError):
    """An evaluation annotation file is malformed or internally inconsistent."""

    code = "invalid_dataset"


class InvalidPredictionsError(_IssueListError):
    """A predictions file does not match its evaluation dataset."""

    code = "invalid_predictions"


class PredictionExportError(_IssueListError):
    """Predictions could not be exported for an evaluation dataset (images, mapping or output)."""

    code = "prediction_export_failed"


class EvaluationUnavailableError(SoccerVisionError):
    """The evaluation dependencies are not installed."""

    code = "evaluation_unavailable"
