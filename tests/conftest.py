from collections.abc import Iterator
from pathlib import Path

import pytest

from support import make_test_video


@pytest.fixture
def sample_mp4(tmp_path: Path) -> Path:
    return make_test_video(tmp_path / "clip.mp4")


@pytest.fixture
def api_settings(tmp_path: Path) -> Iterator[object]:
    pytest.importorskip("fastapi")
    from soccervision_api.settings import Settings

    yield Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        weights_dir=tmp_path / "models",
        allow_weight_download=False,
        max_upload_bytes=2 * 1024 * 1024,
        max_video_duration_seconds=5.0,
        dispatcher_poll_seconds=0.05,
    )
