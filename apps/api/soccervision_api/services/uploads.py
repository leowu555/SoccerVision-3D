"""Upload intake: bounded copy to private storage, content validation, metadata extraction."""

import logging
import re
import shutil
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from soccervision.errors import InvalidVideoError
from soccervision.schemas.video import VideoMetadata
from soccervision.vision.video import probe_video
from soccervision_api.errors import ApiError
from soccervision_api.persistence import Database
from soccervision_api.settings import Settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4"}
SOURCE_FILENAME = "source.mp4"
_CHUNK_BYTES = 1 << 20
_MAX_FILENAME_LENGTH = 120


def sanitize_filename(raw: str | None) -> str:
    """Display-safe filename: basename only, NFKC-normalized, restricted character set."""
    name = Path((raw or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" .")
    if not name:
        return "upload.mp4"
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, suffix = Path(name).stem, Path(name).suffix
        name = stem[: _MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


def store_upload(
    *, stream: BinaryIO, filename: str | None, settings: Settings, db: Database
) -> VideoMetadata:
    display_name = sanitize_filename(filename)
    if Path(display_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ApiError(415, "unsupported_media_type", "Only .mp4 files are accepted.")

    video_id = uuid.uuid4().hex
    video_dir = settings.upload_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=False)
    partial = video_dir / ".upload.partial"
    try:
        size = _bounded_copy(stream, partial, settings.max_upload_bytes)
        if size == 0:
            raise ApiError(422, "invalid_video", "Uploaded file is empty.")
        try:
            probe = probe_video(partial, require_mp4=True)
        except InvalidVideoError as exc:
            raise ApiError(422, "invalid_video", exc.message) from exc
        if probe.duration_seconds > settings.max_video_duration_seconds:
            raise ApiError(
                422,
                "video_limit_exceeded",
                f"Video is {probe.duration_seconds:.1f}s long; the maximum is "
                f"{settings.max_video_duration_seconds:.0f}s.",
            )
        final = video_dir / SOURCE_FILENAME
        partial.replace(final)
        video = VideoMetadata(
            video_id=video_id,
            filename=display_name,
            duration_seconds=probe.duration_seconds,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            frame_count=probe.frame_count,
            created_at=datetime.now(UTC),
            source_path=f"{video_id}/{SOURCE_FILENAME}",
            container_format=probe.container_format,
            video_codec=probe.video_codec,
            size_bytes=probe.size_bytes,
            frame_rate_mode=probe.frame_rate_mode,
            has_audio=probe.has_audio,
        )
        db.insert_video(video)
    except BaseException:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise
    logger.info(
        "Stored upload %s (%s, %.1fs, %dx%d)",
        video_id,
        display_name,
        video.duration_seconds,
        video.width,
        video.height,
    )
    return video


def _bounded_copy(source: BinaryIO, destination: Path, max_bytes: int) -> int:
    written = 0
    with destination.open("wb") as out:
        while chunk := source.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise ApiError(
                    413,
                    "upload_too_large",
                    f"File exceeds the {max_bytes / (1024 * 1024):.0f} MiB upload limit.",
                )
            out.write(chunk)
    return written


def source_path(settings: Settings, video: VideoMetadata) -> Path:
    return settings.upload_dir / video.source_path


def delete_video(*, video_id: str, settings: Settings, db: Database) -> bool:
    """Delete a video's rows, upload and artifacts. Raises JobConflictError if a job is active."""
    if not db.delete_video(video_id):
        return False
    for directory in (settings.upload_dir / video_id, settings.artifact_dir / video_id):
        shutil.rmtree(directory, ignore_errors=True)
    logger.info("Deleted video %s and its artifacts", video_id)
    return True
