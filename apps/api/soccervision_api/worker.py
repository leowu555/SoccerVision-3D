"""Worker subprocess entrypoint: ``python -m soccervision_api.worker <job_id>``.

Runs exactly one already-claimed (PROCESSING) job through the shared pipeline and records the
outcome in the database. Exit code 0 means the job reached a terminal state that was recorded.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from soccervision.config import DetectorConfig, PipelineConfig
from soccervision.errors import SoccerVisionError
from soccervision.pipeline import PipelineRunError, ProgressUpdate, run_detection_pipeline
from soccervision.schemas.job import JobStatus
from soccervision.vision.detection import DetectorFactory, ObjectDetector, create_detector
from soccervision_api.persistence import Database
from soccervision_api.services.jobs import run_dir_for
from soccervision_api.services.uploads import source_path
from soccervision_api.settings import Settings

logger = logging.getLogger("soccervision_api.worker")

_PROGRESS_MIN_INTERVAL_SECONDS = 0.5


class JobAbandonedError(SoccerVisionError):
    """The job left PROCESSING while this worker was still running it (e.g. API restart)."""

    code = "job_abandoned"


def default_detector_factory(settings: Settings) -> DetectorFactory:
    def factory(config: DetectorConfig) -> ObjectDetector:
        return create_detector(
            config, weights_dir=settings.weights_dir, allow_download=settings.allow_weight_download
        )

    return factory


def execute_job(
    job_id: str, *, settings: Settings, db: Database, detector_factory: DetectorFactory
) -> JobStatus:
    job = db.get_job(job_id)
    if job is None:
        raise LookupError(f"Job {job_id} does not exist")
    if job.status is not JobStatus.PROCESSING:
        logger.warning("Job %s is %s, not processing; nothing to do", job_id, job.status.value)
        return job.status

    video = db.get_video(job.video_id)
    if video is None:
        db.transition(
            job_id,
            JobStatus.FAILED,
            error_code="video_missing",
            error_message="Video record not found.",
        )
        return JobStatus.FAILED

    run_dir = run_dir_for(settings, job)
    relative_run_dir = run_dir.relative_to(settings.artifact_dir)
    last_report = 0.0

    def report(update: ProgressUpdate) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= _PROGRESS_MIN_INTERVAL_SECONDS:
            last_report = now
            if not db.update_progress(job_id, update.fraction):
                raise JobAbandonedError(f"Job {job_id} is no longer processing; stopping.")

    try:
        config = PipelineConfig.model_validate(job.configuration)
        result = run_detection_pipeline(
            input_path=source_path(settings, video),
            run_dir=run_dir,
            config=config,
            detector_factory=detector_factory,
            video_id=video.video_id,
            run_id=job.job_id,
            progress=report,
            require_mp4=True,
        )
    except PipelineRunError as exc:
        artifacts = (
            {"metadata": str(relative_run_dir / exc.metadata_path.name)}
            if exc.metadata_path
            else {}
        )
        db.transition(
            job_id,
            JobStatus.FAILED,
            error_code=exc.code,
            error_message=exc.message,
            artifact_paths=artifacts,
        )
        return JobStatus.FAILED

    artifacts = {
        "annotated_video": str(relative_run_dir / result.annotated_video_path.name),
        "detections": str(relative_run_dir / result.detections_path.name),
        "metadata": str(relative_run_dir / result.metadata_path.name),
    }
    if not db.transition(job_id, JobStatus.COMPLETED, progress=1.0, artifact_paths=artifacts):
        logger.warning(
            "Job %s finished but was no longer PROCESSING; result not recorded as completed", job_id
        )
        return JobStatus.FAILED
    return JobStatus.COMPLETED


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m soccervision_api.worker <job_id>", file=sys.stderr)
        return 2
    job_id = args[0]
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    db = Database(settings.database_path)
    logger.info("Worker starting job %s (data_dir=%s)", job_id, Path(settings.data_dir).resolve())
    try:
        status = execute_job(
            job_id, settings=settings, db=db, detector_factory=default_detector_factory(settings)
        )
    except KeyboardInterrupt:
        db.transition(
            job_id,
            JobStatus.FAILED,
            error_code="interrupted",
            error_message="Interrupted: the worker was stopped before finishing.",
        )
        return 130
    logger.info("Job %s finished with status %s", job_id, status.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
