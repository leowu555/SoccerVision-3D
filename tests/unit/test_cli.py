import json
from pathlib import Path

import pytest

from soccervision import cli
from soccervision.pipeline import METADATA_FILE, load_run_metadata
from support import FakeDetectorFactory


def test_cli_runs_shared_pipeline(
    sample_mp4: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    factory = FakeDetectorFactory()
    monkeypatch.setattr(cli, "create_detector", lambda config, **_: factory(config))
    out = tmp_path / "out"
    code = cli.main(
        ["analyze", "-i", str(sample_mp4), "-o", str(out), "--stride", "2", "--confidence", "0.6"]
    )
    assert code == cli.EXIT_OK
    summary = json.loads(capsys.readouterr().out)
    assert summary["stats"]["analyzed_frames"] == 6
    meta = load_run_metadata(out / METADATA_FILE)
    assert meta.configuration["detector"]["confidence_threshold"] == 0.6
    assert factory.instances[0].calls == 6


def test_cli_refuses_to_overwrite(
    sample_mp4: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "create_detector", lambda config, **_: FakeDetectorFactory()(config))
    out = tmp_path / "out"
    assert cli.main(["analyze", "-i", str(sample_mp4), "-o", str(out)]) == cli.EXIT_OK
    assert cli.main(["analyze", "-i", str(sample_mp4), "-o", str(out)]) == cli.EXIT_USAGE
    assert (
        cli.main(["analyze", "-i", str(sample_mp4), "-o", str(out), "--overwrite"]) == cli.EXIT_OK
    )


def test_cli_reports_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.mp4"
    code = cli.main(["analyze", "-i", str(missing), "-o", str(tmp_path / "out"), "--no-download"])
    assert code == cli.EXIT_FAILED
    assert "invalid_video" in capsys.readouterr().err


def test_cli_rejects_bad_config(sample_mp4: Path, tmp_path: Path) -> None:
    assert (
        cli.main(["analyze", "-i", str(sample_mp4), "-o", str(tmp_path), "--confidence", "2"])
        == cli.EXIT_USAGE
    )
