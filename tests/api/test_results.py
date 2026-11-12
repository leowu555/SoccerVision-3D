from fastapi.testclient import TestClient

from soccervision.schemas.job import JobStatus
from soccervision_api.persistence import Database
from soccervision_api.worker import execute_job
from support import FakeDetectorFactory


def _run_job(client: TestClient, video_id: str, factory: FakeDetectorFactory | None = None) -> str:
    db: Database = client.app.state.db  # type: ignore[attr-defined]
    job_id: str = client.post(
        f"/api/v1/videos/{video_id}/analyze", json={"inference_stride": 2}
    ).json()["job_id"]
    assert db.transition(job_id, JobStatus.PROCESSING)
    execute_job(
        job_id,
        settings=client.app.state.settings,  # type: ignore[attr-defined]
        db=db,
        detector_factory=factory or FakeDetectorFactory(),
    )
    return job_id


def test_results_before_any_analysis(client: TestClient, uploaded_video: str) -> None:
    response = client.get(f"/api/v1/videos/{uploaded_video}/results")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "results_not_found"
    assert "No analysis has been requested" in response.json()["error"]["message"]


def test_results_while_pending(client: TestClient, uploaded_video: str) -> None:
    job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
    response = client.get(f"/api/v1/videos/{uploaded_video}/results")
    assert response.status_code == 404
    assert "pending" in response.json()["error"]["message"]
    response = client.get(f"/api/v1/videos/{uploaded_video}/results", params={"job_id": job_id})
    assert response.status_code == 409


def test_results_after_completion(client: TestClient, uploaded_video: str) -> None:
    job_id = _run_job(client, uploaded_video)
    response = client.get(f"/api/v1/videos/{uploaded_video}/results")
    assert response.status_code == 200
    body = response.json()
    assert body["job"]["job_id"] == job_id
    assert body["video"]["processing_status"] == "completed"
    stats = body["run"]["stats"]
    assert (stats["decoded_frames"], stats["analyzed_frames"], stats["skipped_frames"]) == (
        12,
        6,
        6,
    )
    assert stats["total_detections"] == 12
    assert body["run"]["model"]["is_synthetic"] is True
    # Only analyzed frames appear in the per-frame series.
    assert [c["frame_index"] for c in body["frame_counts"]] == [0, 2, 4, 6, 8, 10]
    assert all(c["detections"] == 2 for c in body["frame_counts"])


def test_failed_job_has_no_results_but_keeps_previous(
    client: TestClient, uploaded_video: str
) -> None:
    good = _run_job(client, uploaded_video)
    _run_job(client, uploaded_video, FakeDetectorFactory(fail_on_call=1))
    body = client.get(f"/api/v1/videos/{uploaded_video}/results").json()
    assert body["job"]["job_id"] == good
    assert body["video"]["processing_status"] == "failed"


def test_artifact_download(client: TestClient, uploaded_video: str) -> None:
    job_id = _run_job(client, uploaded_video)
    video = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert video.content[4:8] == b"ftyp"
    partial = client.get(
        f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4", headers={"Range": "bytes=0-9"}
    )
    assert partial.status_code == 206
    detections = client.get(f"/api/v1/jobs/{job_id}/artifacts/detections.json").json()
    assert detections["artifact_type"] == "detections"
    assert len(detections["frames"]) == 12


def test_artifact_name_whitelist(client: TestClient, uploaded_video: str) -> None:
    job_id = _run_job(client, uploaded_video)
    for name in ("worker.log", "..%2F..%2Fsoccervision.db", "source.mp4"):
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts/{name}")
        assert response.status_code == 404, name


def test_missing_artifact_file_reports_gone(client: TestClient, uploaded_video: str) -> None:
    job_id = _run_job(client, uploaded_video)
    settings = client.app.state.settings  # type: ignore[attr-defined]
    (settings.artifact_dir / uploaded_video / job_id / "annotated.mp4").unlink()
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/annotated.mp4")
    assert response.status_code == 410


def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["pipeline_name"] == "person-detection"
    assert body["model"]["weights_present"] is False
    assert body["limits"]["max_upload_bytes"] == 2 * 1024 * 1024
