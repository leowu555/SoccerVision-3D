"""SQLite persistence for videos and jobs.

Stdlib ``sqlite3`` (WAL mode) is sufficient for a single-host Phase 1 deployment where the API
process and worker subprocesses share one file. Each operation opens a short-lived connection so
the class is safe to use from threads and separate processes. Job status changes are conditional
``UPDATE ... WHERE status IN (...)`` statements, so concurrent writers cannot violate the state
machine. Migrate to SQLAlchemy/Alembic + PostgreSQL when multi-host workers are introduced.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soccervision.schemas.job import JobStatus, PipelineJob, allowed_sources, ensure_transition
from soccervision.schemas.video import VideoMetadata, VideoProcessingStatus

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id          TEXT PRIMARY KEY,
    filename          TEXT NOT NULL,
    source_path       TEXT NOT NULL,
    duration_seconds  REAL NOT NULL,
    width             INTEGER NOT NULL,
    height            INTEGER NOT NULL,
    fps               REAL NOT NULL,
    frame_count       INTEGER,
    container_format  TEXT NOT NULL,
    video_codec       TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    frame_rate_mode   TEXT NOT NULL,
    has_audio         INTEGER NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    video_id          TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    status            TEXT NOT NULL,
    progress          REAL NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    completed_at      TEXT,
    error_code        TEXT,
    error_message     TEXT,
    pipeline_version  TEXT NOT NULL,
    configuration     TEXT NOT NULL,
    artifact_paths    TEXT NOT NULL DEFAULT '{}',
    worker_pid        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_video_created ON jobs(video_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
"""

_ACTIVE = (JobStatus.PENDING.value, JobStatus.PROCESSING.value)
_VIDEO_STATUS_FOR_JOB = {
    JobStatus.PENDING: VideoProcessingStatus.QUEUED,
    JobStatus.PROCESSING: VideoProcessingStatus.PROCESSING,
    JobStatus.COMPLETED: VideoProcessingStatus.COMPLETED,
    JobStatus.FAILED: VideoProcessingStatus.FAILED,
}


class JobConflictError(Exception):
    def __init__(self, active_job_id: str) -> None:
        super().__init__(f"Job {active_job_id} is already queued or running for this video")
        self.active_job_id = active_job_id


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema v{version} is newer than this code (v{SCHEMA_VERSION})."
                )
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------------------------------------------------------------- videos

    def insert_video(self, video: VideoMetadata) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO videos (video_id, filename, source_path, duration_seconds, width, height,
                       fps, frame_count, container_format, video_codec, size_bytes, frame_rate_mode,
                       has_audio, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video.video_id,
                    video.filename,
                    video.source_path,
                    video.duration_seconds,
                    video.width,
                    video.height,
                    video.fps,
                    video.frame_count,
                    video.container_format,
                    video.video_codec,
                    video.size_bytes,
                    video.frame_rate_mode,
                    int(video.has_audio),
                    video.created_at.isoformat(),
                ),
            )

    def get_video(self, video_id: str) -> VideoMetadata | None:
        with self.connect() as conn:
            row = conn.execute(
                f"{self._video_select()} WHERE v.video_id = ?", (video_id,)
            ).fetchone()
        return self._video_from_row(row) if row else None

    def list_videos(self, limit: int = 50) -> list[VideoMetadata]:
        with self.connect() as conn:
            rows = conn.execute(
                f"{self._video_select()} ORDER BY v.created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._video_from_row(row) for row in rows]

    @staticmethod
    def _video_select() -> str:
        return """SELECT v.*, (SELECT j.status FROM jobs j WHERE j.video_id = v.video_id
                              ORDER BY j.created_at DESC, j.rowid DESC LIMIT 1) AS latest_status
                  FROM videos v"""

    @staticmethod
    def _video_from_row(row: sqlite3.Row) -> VideoMetadata:
        latest = row["latest_status"]
        status = (
            _VIDEO_STATUS_FOR_JOB[JobStatus(latest)] if latest else VideoProcessingStatus.UPLOADED
        )
        return VideoMetadata(
            video_id=row["video_id"],
            filename=row["filename"],
            source_path=row["source_path"],
            duration_seconds=row["duration_seconds"],
            width=row["width"],
            height=row["height"],
            fps=row["fps"],
            frame_count=row["frame_count"],
            container_format=row["container_format"],
            video_codec=row["video_codec"],
            size_bytes=row["size_bytes"],
            frame_rate_mode=row["frame_rate_mode"],
            has_audio=bool(row["has_audio"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            processing_status=status,
        )

    # ------------------------------------------------------------------ jobs

    def create_job(self, job: PipelineJob) -> PipelineJob:
        """Insert a PENDING job unless the video already has an active one."""
        if job.status is not JobStatus.PENDING:
            raise ValueError("New jobs must start in PENDING")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                active = conn.execute(
                    "SELECT job_id FROM jobs WHERE video_id = ? AND status IN (?, ?)",
                    (job.video_id, *_ACTIVE),
                ).fetchone()
                if active:
                    raise JobConflictError(active["job_id"])
                conn.execute(
                    """INSERT INTO jobs (job_id, video_id, status, progress, created_at,
                           pipeline_version, configuration, artifact_paths)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job.job_id,
                        job.video_id,
                        job.status.value,
                        job.progress,
                        job.created_at.isoformat(),
                        job.pipeline_version,
                        json.dumps(job.configuration),
                        json.dumps(job.artifact_paths),
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return job

    def get_job(self, job_id: str) -> PipelineJob | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs_for_video(self, video_id: str) -> list[PipelineJob]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE video_id = ? ORDER BY created_at DESC, rowid DESC",
                (video_id,),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def latest_job(self, video_id: str, status: JobStatus | None = None) -> PipelineJob | None:
        query = "SELECT * FROM jobs WHERE video_id = ?"
        params: list[Any] = [video_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return self._job_from_row(row) if row else None

    def next_pending_job(self, exclude: set[str] | None = None) -> PipelineJob | None:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at, rowid",
                (JobStatus.PENDING.value,),
            ).fetchall()
        for row in rows:
            if not exclude or row["job_id"] not in exclude:
                return self._job_from_row(row)
        return None

    def transition(
        self,
        job_id: str,
        target: JobStatus,
        *,
        progress: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        artifact_paths: dict[str, str] | None = None,
        worker_pid: int | None = None,
    ) -> bool:
        """Atomically move a job to ``target`` if its current state allows it.

        Returns False (and changes nothing) when the job is missing or the transition is not
        allowed from its *current* state, e.g. a worker finishing a job that was already failed
        by restart reconciliation.
        """
        sources = allowed_sources(target)
        assignments = ["status = ?"]
        params: list[Any] = [target.value]
        if target is JobStatus.PROCESSING:
            assignments.append("started_at = ?")
            params.append(_now())
        if target in (JobStatus.COMPLETED, JobStatus.FAILED):
            assignments.append("completed_at = ?")
            params.append(_now())
        for column, value in (
            ("progress", progress),
            ("error_code", error_code),
            ("error_message", error_message),
            ("worker_pid", worker_pid),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if artifact_paths is not None:
            assignments.append("artifact_paths = ?")
            params.append(json.dumps(artifact_paths))

        placeholders = ", ".join("?" for _ in sources)
        params.extend([job_id, *(s.value for s in sources)])
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ? AND status IN ({placeholders})",
                params,
            )
            return cursor.rowcount == 1

    def set_worker_pid(self, job_id: str, pid: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET worker_pid = ? WHERE job_id = ?", (pid, job_id))

    def update_progress(self, job_id: str, progress: float) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET progress = ? WHERE job_id = ? AND status = ?",
                (min(max(progress, 0.0), 1.0), job_id, JobStatus.PROCESSING.value),
            )
            return cursor.rowcount == 1

    def fail_interrupted_jobs(self, message: str) -> list[str]:
        """Mark every PROCESSING job FAILED. Call only when no worker from this DB can be alive."""
        ensure_transition(JobStatus.PROCESSING, JobStatus.FAILED)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status = ?", (JobStatus.PROCESSING.value,)
            ).fetchall()
            conn.execute(
                """UPDATE jobs SET status = ?, completed_at = ?, error_code = ?, error_message = ?
                   WHERE status = ?""",
                (
                    JobStatus.FAILED.value,
                    _now(),
                    "interrupted",
                    message,
                    JobStatus.PROCESSING.value,
                ),
            )
            conn.execute("COMMIT")
        return [row["job_id"] for row in rows]

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> PipelineJob:
        return PipelineJob(
            job_id=row["job_id"],
            video_id=row["video_id"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=_dt(row["started_at"]),
            completed_at=_dt(row["completed_at"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            pipeline_version=row["pipeline_version"],
            configuration=json.loads(row["configuration"]),
            artifact_paths=json.loads(row["artifact_paths"]),
        )
