import functools
from pathlib import Path

from soccervision.pipeline import load_detections, load_run_metadata
from soccervision.schemas.artifacts import RunMetadata
from soccervision.schemas.job import PipelineJob
from soccervision_api.errors import ApiError
from soccervision_api.schemas import FrameCount
from soccervision_api.settings import Settings


def resolve_artifact(settings: Settings, job: PipelineJob, name: str) -> Path:
    """Resolve an artifact path recorded on a job, refusing anything outside the artifact root."""
    relative = job.artifact_paths.get(name)
    if relative is None:
        raise ApiError(404, "artifact_not_found", f"Job {job.job_id} has no '{name}' artifact.")
    root = settings.artifact_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ApiError(404, "artifact_not_found", "Artifact path is invalid.")
    if not path.is_file():
        raise ApiError(
            410, "artifact_missing", f"Artifact '{name}' was recorded but no longer exists on disk."
        )
    return path


@functools.lru_cache(maxsize=32)
def _frame_counts_cached(path: str, mtime_ns: int) -> tuple[FrameCount, ...]:
    artifact = load_detections(Path(path))
    return tuple(
        FrameCount(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            detections=len(frame.detections),
        )
        for frame in artifact.frames
        if frame.detections is not None
    )


def frame_counts(path: Path) -> list[FrameCount]:
    return list(_frame_counts_cached(str(path), path.stat().st_mtime_ns))


def run_metadata(path: Path) -> RunMetadata:
    return load_run_metadata(path)
