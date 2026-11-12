import sys
import time
from pathlib import Path
from subprocess import Popen

import pytest
from fastapi.testclient import TestClient

from soccervision.config import DetectorConfig
from soccervision.schemas.job import JobStatus, PipelineJob
from soccervision_api.persistence import Database
from soccervision_api.services.jobs import INTERRUPTED_ON_RESTART, JobRunner
from soccervision_api.settings import Settings
from soccervision_api.worker import execute_job
from support import FakeDetector, FakeDetectorFactory


def _db(client: TestClient) -> Database:
    db: Database = client.app.state.db  # type: ignore[attr-defined]
    return db


def _settings(client: TestClient) -> Settings:
    settings: Settings = client.app.state.settings  # type: ignore[attr-defined]
    return settings


def _claim(db: Database, job_id: str) -> None:
    assert db.transition(job_id, JobStatus.PROCESSING)


class TestAnalyzeEndpoint:
    def test_creates_pending_job_with_recorded_configuration(
        self, client: TestClient, uploaded_video: str
    ) -> None:
        response = client.post(
            f"/api/v1/videos/{uploaded_video}/analyze",
            json={"confidence_threshold": 0.35, "inference_stride": 2},
        )
        assert response.status_code == 202
        job = response.json()
        assert job["status"] == "pending"
        assert job["progress"] == 0.0
        assert job["configuration"]["detector"]["confidence_threshold"] == 0.35
        assert job["configuration"]["inference_stride"] == 2
        assert job["configuration"]["max_duration_seconds"] == 5.0
        assert client.get(f"/api/v1/jobs/{job['job_id']}").json()["status"] == "pending"
        assert (
            client.get(f"/api/v1/videos/{uploaded_video}").json()["processing_status"] == "queued"
        )

    def test_rejects_second_active_job(self, client: TestClient, uploaded_video: str) -> None:
        first = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()
        second = client.post(f"/api/v1/videos/{uploaded_video}/analyze")
        assert second.status_code == 409
        assert second.json()["error"]["details"]["job_id"] == first["job_id"]

    def test_rejects_invalid_overrides(self, client: TestClient, uploaded_video: str) -> None:
        for body in (
            {"confidence_threshold": 3},
            {"inference_stride": 0},
            {"detector": "yolo"},
            {"x": 1},
        ):
            response = client.post(f"/api/v1/videos/{uploaded_video}/analyze", json=body)
            assert response.status_code == 422, body

    def test_unknown_video_and_job(self, client: TestClient) -> None:
        assert client.post("/api/v1/videos/nope/analyze").status_code == 404
        assert client.get("/api/v1/jobs/nope").json()["error"]["code"] == "job_not_found"


class TestWorkerExecution:
    def test_successful_job_completes_with_artifacts(
        self, client: TestClient, uploaded_video: str
    ) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        db = _db(client)
        _claim(db, job_id)
        status = execute_job(
            job_id, settings=_settings(client), db=db, detector_factory=FakeDetectorFactory()
        )
        assert status is JobStatus.COMPLETED
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "completed"
        assert job["progress"] == 1.0
        assert job["started_at"] and job["completed_at"]
        assert set(job["artifact_urls"]) == {"annotated_video", "detections", "metadata"}
        assert (
            client.get(f"/api/v1/videos/{uploaded_video}").json()["processing_status"]
            == "completed"
        )

    def test_failing_job_surfaces_error(self, client: TestClient, uploaded_video: str) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        db = _db(client)
        _claim(db, job_id)
        status = execute_job(
            job_id,
            settings=_settings(client),
            db=db,
            detector_factory=FakeDetectorFactory(fail_on_call=2),
        )
        assert status is JobStatus.FAILED
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "failed"
        assert job["error_code"] == "internal_error"
        assert "simulated detector crash" in job["error_message"]
        assert set(job["artifact_urls"]) == {"metadata"}

    def test_worker_ignores_unclaimed_job(self, client: TestClient, uploaded_video: str) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        status = execute_job(
            job_id,
            settings=_settings(client),
            db=_db(client),
            detector_factory=FakeDetectorFactory(),
        )
        assert status is JobStatus.PENDING

    def test_completion_after_reconciliation_is_not_recorded(
        self, client: TestClient, uploaded_video: str
    ) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        db = _db(client)
        _claim(db, job_id)
        db.fail_interrupted_jobs("restarted")
        assert not db.transition(job_id, JobStatus.COMPLETED)
        assert db.get_job(job_id).status is JobStatus.FAILED  # type: ignore[union-attr]

    def test_worker_stops_when_job_is_reconciled_mid_run(
        self, client: TestClient, uploaded_video: str
    ) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        db, settings = _db(client), _settings(client)
        _claim(db, job_id)
        fakes = FakeDetectorFactory()

        def factory(config: DetectorConfig) -> FakeDetector:
            db.fail_interrupted_jobs("restarted")
            return fakes(config)

        status = execute_job(job_id, settings=settings, db=db, detector_factory=factory)
        assert status is JobStatus.FAILED
        job = db.get_job(job_id)
        assert job is not None and job.error_message == "restarted"
        assert fakes.instances[0].calls == 1
        run_dir = settings.artifact_dir / uploaded_video / job_id
        assert [p.name for p in run_dir.iterdir()] == ["metadata.json"]


class TestDatabaseTransitions:
    def test_conditional_transitions(self, client: TestClient, uploaded_video: str) -> None:
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        db = _db(client)
        assert not db.transition(job_id, JobStatus.COMPLETED)  # PENDING -> COMPLETED forbidden
        assert db.transition(job_id, JobStatus.PROCESSING)
        assert not db.transition(job_id, JobStatus.PROCESSING)
        assert db.update_progress(job_id, 0.4)
        assert db.transition(job_id, JobStatus.COMPLETED, progress=1.0)
        assert not db.transition(job_id, JobStatus.FAILED)
        assert not db.update_progress(job_id, 0.1)
        assert db.get_job(job_id).progress == 1.0  # type: ignore[union-attr]


class _FakeHandle:
    def __init__(self, code: int | None) -> None:
        self.pid = 4242
        self.code = code
        self.terminated = False

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = -15

    def kill(self) -> None:
        self.code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.code or 0


class TestJobRunner:
    def _submit(self, client: TestClient, video_id: str) -> str:
        job_id: str = client.post(f"/api/v1/videos/{video_id}/analyze").json()["job_id"]
        return job_id

    def test_restart_fails_processing_jobs_and_keeps_pending(
        self, client: TestClient, uploaded_video: str, sample_mp4: Path
    ) -> None:
        db, settings = _db(client), _settings(client)
        interrupted = self._submit(client, uploaded_video)
        _claim(db, interrupted)
        with sample_mp4.open("rb") as fh:
            other_video = client.post(
                "/api/v1/videos", files={"file": ("b.mp4", fh, "video/mp4")}
            ).json()
        queued = self._submit(client, other_video["video_id"])
        run_dir = settings.artifact_dir / uploaded_video / interrupted
        run_dir.mkdir(parents=True)
        partial = run_dir / ".annotated.mp4.partial"
        partial.write_bytes(b"half-written")
        (run_dir / "worker.log").write_text("log")

        JobRunner(db, settings, launcher=lambda job, log: _FakeHandle(None)).reconcile()

        failed = db.get_job(interrupted)
        assert failed is not None and failed.status is JobStatus.FAILED
        assert failed.error_code == "interrupted" and failed.error_message == INTERRUPTED_ON_RESTART
        assert not partial.exists()
        assert (run_dir / "worker.log").exists()
        assert db.get_job(queued).status is JobStatus.PENDING  # type: ignore[union-attr]

    def test_concurrency_is_bounded(
        self, client: TestClient, uploaded_video: str, sample_mp4: Path
    ) -> None:
        db, settings = _db(client), _settings(client)
        launched: list[PipelineJob] = []

        def launcher(job: PipelineJob, log: Path) -> _FakeHandle:
            launched.append(job)
            return _FakeHandle(None)

        self._submit(client, uploaded_video)
        with sample_mp4.open("rb") as fh:
            second = client.post(
                "/api/v1/videos", files={"file": ("b.mp4", fh, "video/mp4")}
            ).json()
        self._submit(client, second["video_id"])

        runner = JobRunner(
            db, settings.model_copy(update={"max_concurrent_jobs": 1}), launcher=launcher
        )
        runner.tick()
        runner.tick()
        assert len(launched) == 1
        assert db.get_job(launched[0].job_id).status is JobStatus.PROCESSING  # type: ignore[union-attr]

    def test_crashed_worker_marks_job_failed(self, client: TestClient, uploaded_video: str) -> None:
        db, settings = _db(client), _settings(client)
        job_id = self._submit(client, uploaded_video)

        def launcher(job: PipelineJob, log: Path) -> Popen[bytes]:
            return Popen([sys.executable, "-c", "import sys; sys.exit(3)"])

        runner = JobRunner(db, settings, launcher=launcher)
        runner.tick()
        deadline = time.monotonic() + 10
        while runner.running_job_ids and time.monotonic() < deadline:
            time.sleep(0.05)
            runner.tick()
        job = db.get_job(job_id)
        assert job is not None and job.status is JobStatus.FAILED
        assert job.error_code == "worker_crashed"
        assert "exit code 3" in (job.error_message or "")

    def test_launch_failure_marks_job_failed(self, client: TestClient, uploaded_video: str) -> None:
        db, settings = _db(client), _settings(client)
        job_id = self._submit(client, uploaded_video)

        def launcher(job: PipelineJob, log: Path) -> _FakeHandle:
            raise OSError("no python")

        JobRunner(db, settings, launcher=launcher).tick()
        job = db.get_job(job_id)
        assert job is not None and job.status is JobStatus.FAILED
        assert job.error_code == "worker_launch_failed"

    def test_stop_terminates_running_workers(self, client: TestClient, uploaded_video: str) -> None:
        db, settings = _db(client), _settings(client)
        job_id = self._submit(client, uploaded_video)
        handle = _FakeHandle(None)
        runner = JobRunner(db, settings, launcher=lambda job, log: handle)
        runner.tick()
        runner.stop(timeout=1)
        assert handle.terminated
        assert db.get_job(job_id).status is JobStatus.FAILED  # type: ignore[union-attr]


@pytest.mark.parametrize("stride", [1, 4])
def test_end_to_end_with_fake_detector(
    client: TestClient, uploaded_video: str, stride: int
) -> None:
    job_id = client.post(
        f"/api/v1/videos/{uploaded_video}/analyze", json={"inference_stride": stride}
    ).json()["job_id"]
    db = _db(client)
    _claim(db, job_id)
    execute_job(job_id, settings=_settings(client), db=db, detector_factory=FakeDetectorFactory())
    results = client.get(f"/api/v1/videos/{uploaded_video}/results").json()
    assert results["run"]["stats"]["analyzed_frames"] == len(range(0, 12, stride))
