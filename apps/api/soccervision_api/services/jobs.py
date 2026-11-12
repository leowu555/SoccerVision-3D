"""Job submission and the bounded local job runner.

Design: the SQLite ``jobs`` table is the queue. A single dispatcher thread in the API process
claims PENDING jobs (PENDING -> PROCESSING) and launches one worker *subprocess* per job, never
more than ``max_concurrent_jobs`` at a time. Model inference therefore never runs inside an HTTP
request or the API's event loop, and a crashing worker (e.g. out of memory) cannot take the API
down with it. Workers report progress and final state directly to the database.

Restart semantics: on startup, PROCESSING jobs are failed with code ``interrupted`` (their worker
belonged to a previous server process); PENDING jobs are simply picked up again.
"""

import logging
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol

from soccervision.config import PipelineConfig
from soccervision.pipeline import PIPELINE_VERSION
from soccervision.schemas.job import JobStatus, PipelineJob, is_terminal
from soccervision_api.persistence import Database
from soccervision_api.settings import Settings

logger = logging.getLogger(__name__)

WORKER_LOG_FILE = "worker.log"
INTERRUPTED_ON_RESTART = "Interrupted: the server restarted while this job was processing."
INTERRUPTED_ON_SHUTDOWN = "Interrupted: the server shut down while this job was processing."


class WorkerHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


WorkerLauncher = Callable[[PipelineJob, Path], WorkerHandle]


def run_dir_for(settings: Settings, job: PipelineJob) -> Path:
    return settings.artifact_dir / job.video_id / job.job_id


def submit_job(db: Database, video_id: str, config: PipelineConfig) -> PipelineJob:
    """Create a PENDING job. Raises JobConflictError if the video already has an active job."""
    job = PipelineJob(
        job_id=uuid.uuid4().hex,
        video_id=video_id,
        status=JobStatus.PENDING,
        progress=0.0,
        created_at=datetime.now(UTC),
        pipeline_version=PIPELINE_VERSION,
        configuration=config.model_dump(mode="json"),
    )
    return db.create_job(job)


def subprocess_launcher(settings: Settings) -> WorkerLauncher:
    def launch(job: PipelineJob, log_path: Path) -> WorkerHandle:
        env = {**os.environ, **settings.as_env(), "PYTHONUNBUFFERED": "1"}
        log: IO[bytes] = log_path.open("ab")
        try:
            return subprocess.Popen(
                [sys.executable, "-m", "soccervision_api.worker", job.job_id],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        finally:
            log.close()

    return launch


@dataclass
class _Running:
    handle: WorkerHandle
    log_path: Path


class JobRunner:
    def __init__(
        self, db: Database, settings: Settings, launcher: WorkerLauncher | None = None
    ) -> None:
        self._db = db
        self._settings = settings
        self._launcher = launcher or subprocess_launcher(settings)
        self._running: dict[str, _Running] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running_job_ids(self) -> set[str]:
        with self._lock:
            return set(self._running)

    def reconcile(self) -> list[str]:
        failed = self._db.fail_interrupted_jobs(INTERRUPTED_ON_RESTART)
        if failed:
            logger.warning(
                "Marked %d interrupted job(s) as failed: %s", len(failed), ", ".join(failed)
            )
        return failed

    def start(self) -> None:
        self.reconcile()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-dispatcher", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        with self._lock:
            running = dict(self._running)
            self._running.clear()
        for job_id, entry in running.items():
            logger.info("Stopping worker for job %s", job_id)
            entry.handle.terminate()
        for job_id, entry in running.items():
            try:
                entry.handle.wait(timeout)
            except subprocess.TimeoutExpired:
                entry.handle.kill()
            self._db.transition(
                job_id,
                JobStatus.FAILED,
                error_code="interrupted",
                error_message=INTERRUPTED_ON_SHUTDOWN,
            )

    def tick(self) -> None:
        """One dispatcher iteration: reap finished workers, then launch queued jobs."""
        self._reap()
        self._launch_pending()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Job dispatcher iteration failed")
            self._wake.wait(self._settings.dispatcher_poll_seconds)
            self._wake.clear()

    def _reap(self) -> None:
        with self._lock:
            finished = {
                job_id: (entry, code)
                for job_id, entry in self._running.items()
                if (code := entry.handle.poll()) is not None
            }
            for job_id in finished:
                del self._running[job_id]
        for job_id, (entry, code) in finished.items():
            job = self._db.get_job(job_id)
            if job is not None and not is_terminal(job.status):
                message = f"Worker process exited unexpectedly (exit code {code}). See {entry.log_path.name}."
                logger.error("Job %s: %s", job_id, message)
                self._db.transition(
                    job_id, JobStatus.FAILED, error_code="worker_crashed", error_message=message
                )

    def _launch_pending(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if len(self._running) >= self._settings.max_concurrent_jobs:
                    return
                exclude = set(self._running)
            job = self._db.next_pending_job(exclude=exclude)
            if job is None:
                return
            if not self._db.transition(job.job_id, JobStatus.PROCESSING, progress=0.0):
                continue
            run_dir = run_dir_for(self._settings, job)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / WORKER_LOG_FILE
            try:
                handle = self._launcher(job, log_path)
            except Exception as exc:
                logger.exception("Could not launch worker for job %s", job.job_id)
                self._db.transition(
                    job.job_id,
                    JobStatus.FAILED,
                    error_code="worker_launch_failed",
                    error_message=f"Could not start worker process: {exc}",
                )
                continue
            self._db.set_worker_pid(job.job_id, handle.pid)
            with self._lock:
                self._running[job.job_id] = _Running(handle=handle, log_path=log_path)
            logger.info("Launched worker pid=%s for job %s", handle.pid, job.job_id)
