from pathlib import Path

from fastapi.testclient import TestClient

from soccervision.schemas.job import JobStatus
from soccervision_api.services.uploads import sanitize_filename
from support import make_test_video


def _upload(
    client: TestClient, path: Path, name: str = "clip.mp4", content_type: str = "video/mp4"
):  # type: ignore[no-untyped-def]
    with path.open("rb") as fh:
        return client.post("/api/v1/videos", files={"file": (name, fh, content_type)})


class TestUpload:
    def test_valid_upload_returns_metadata(self, client: TestClient, sample_mp4: Path) -> None:
        response = _upload(client, sample_mp4, name="Derby Highlights.mp4")
        assert response.status_code == 201
        body = response.json()
        assert body["upload_status"] == "stored"
        assert body["original_filename"] == "Derby Highlights.mp4"
        meta = body["metadata"]
        assert meta["video_id"] == body["video_id"]
        assert (meta["width"], meta["height"], meta["frame_count"]) == (96, 64, 12)
        assert meta["fps"] == 12.0
        assert meta["processing_status"] == "uploaded"
        assert meta["source_url"] == f"/api/v1/videos/{body['video_id']}/source"
        assert "source_path" not in meta

    def test_stored_outside_public_paths_with_generated_name(
        self, client: TestClient, sample_mp4: Path
    ) -> None:
        video_id = _upload(client, sample_mp4, name="../../etc/passwd.mp4").json()["video_id"]
        settings = client.app.state.settings  # type: ignore[attr-defined]
        stored = settings.upload_dir / video_id / "source.mp4"
        assert stored.is_file()
        assert stored.read_bytes() == sample_mp4.read_bytes()
        assert client.get(f"/api/v1/videos/{video_id}").json()["filename"] == "passwd.mp4"

    def test_rejects_wrong_extension(self, client: TestClient, sample_mp4: Path) -> None:
        response = _upload(client, sample_mp4, name="clip.mov")
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_media_type"

    def test_rejects_non_video_with_mp4_extension(self, client: TestClient, tmp_path: Path) -> None:
        fake = tmp_path / "fake.mp4"
        fake.write_bytes(b"<html>not a video</html>" * 50)
        response = _upload(client, fake)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_video"
        settings = client.app.state.settings  # type: ignore[attr-defined]
        assert list(settings.upload_dir.iterdir()) == []

    def test_rejects_other_container_renamed_to_mp4(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        avi = make_test_video(tmp_path / "clip.avi", container_format="avi")
        response = _upload(client, avi, name="clip.mp4")
        assert response.status_code == 422

    def test_rejects_too_long_video(self, client: TestClient, tmp_path: Path) -> None:
        long_clip = make_test_video(tmp_path / "long.mp4", frames=72, fps=12)  # 6 s > 5 s limit
        response = _upload(client, long_clip)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "video_limit_exceeded"

    def test_rejects_too_large_upload(self, client: TestClient, tmp_path: Path) -> None:
        big = tmp_path / "big.mp4"
        big.write_bytes(b"\x00" * (4 * 1024 * 1024))
        response = _upload(client, big)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "upload_too_large"

    def test_requires_file_field(self, client: TestClient) -> None:
        response = client.post("/api/v1/videos")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestVideoQueries:
    def test_list_and_get(self, client: TestClient, uploaded_video: str) -> None:
        listing = client.get("/api/v1/videos").json()["videos"]
        assert [v["video_id"] for v in listing] == [uploaded_video]
        assert client.get(f"/api/v1/videos/{uploaded_video}").status_code == 200

    def test_unknown_video(self, client: TestClient) -> None:
        response = client.get("/api/v1/videos/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "video_not_found"

    def test_source_supports_range_requests(self, client: TestClient, uploaded_video: str) -> None:
        response = client.get(
            f"/api/v1/videos/{uploaded_video}/source", headers={"Range": "bytes=0-99"}
        )
        assert response.status_code == 206
        assert len(response.content) == 100
        assert response.headers["content-type"] == "video/mp4"


class TestDeleteVideo:
    def test_removes_rows_upload_and_artifacts(
        self, client: TestClient, uploaded_video: str
    ) -> None:
        settings = client.app.state.settings  # type: ignore[attr-defined]
        db = client.app.state.db  # type: ignore[attr-defined]
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]
        assert db.transition(job_id, JobStatus.PROCESSING)
        assert db.transition(job_id, JobStatus.FAILED, error_code="x", error_message="x")
        artifacts = settings.artifact_dir / uploaded_video / job_id
        artifacts.mkdir(parents=True)
        (artifacts / "metadata.json").write_text("{}")

        response = client.delete(f"/api/v1/videos/{uploaded_video}")

        assert response.status_code == 204
        assert not (settings.upload_dir / uploaded_video).exists()
        assert not (settings.artifact_dir / uploaded_video).exists()
        assert db.get_job(job_id) is None
        assert client.get(f"/api/v1/videos/{uploaded_video}").status_code == 404
        assert client.get("/api/v1/videos").json()["videos"] == []

    def test_refuses_while_job_is_active(self, client: TestClient, uploaded_video: str) -> None:
        settings = client.app.state.settings  # type: ignore[attr-defined]
        job_id = client.post(f"/api/v1/videos/{uploaded_video}/analyze").json()["job_id"]

        response = client.delete(f"/api/v1/videos/{uploaded_video}")

        assert response.status_code == 409
        assert response.json()["error"]["details"]["job_id"] == job_id
        assert (settings.upload_dir / uploaded_video / "source.mp4").is_file()

    def test_unknown_video(self, client: TestClient) -> None:
        response = client.delete("/api/v1/videos/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "video_not_found"


def test_sanitize_filename() -> None:
    assert sanitize_filename("..\\..\\evil.mp4") == "evil.mp4"
    assert sanitize_filename(None) == "upload.mp4"
    assert sanitize_filename("<script>.mp4") == "_script_.mp4"
    assert len(sanitize_filename("a" * 500 + ".mp4")) <= 120
