from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from soccervision_api.app import create_app
from soccervision_api.persistence import Database
from soccervision_api.services.jobs import JobRunner
from soccervision_api.settings import Settings


class _NoLaunchRunner(JobRunner):
    """Runner whose dispatcher never starts; tests drive jobs explicitly."""

    def wake(self) -> None:
        pass


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    settings = api_settings.resolved()
    runner = _NoLaunchRunner(Database(settings.database_path), settings)
    app = create_app(settings, runner=runner, start_runner=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def uploaded_video(client: TestClient, sample_mp4: Path) -> str:
    with sample_mp4.open("rb") as fh:
        response = client.post("/api/v1/videos", files={"file": ("clip.mp4", fh, "video/mp4")})
    assert response.status_code == 201, response.text
    video_id: str = response.json()["video_id"]
    return video_id
