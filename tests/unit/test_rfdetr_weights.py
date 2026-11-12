import hashlib
from pathlib import Path

import pytest

from soccervision.config import DetectorConfig
from soccervision.errors import ModelUnavailableError
from soccervision.vision.detection import rfdetr
from soccervision.vision.detection.rfdetr import RFDETR_VARIANTS, RFDETRDetector, verify_weights


def _forbid_rfdetr_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> None:
        raise AssertionError(f"{name} must not be imported when the weights are rejected")

    monkeypatch.setattr(rfdetr.importlib, "import_module", fail)


def _detector(weights_dir: Path) -> RFDETRDetector:
    return RFDETRDetector(
        DetectorConfig(name="rfdetr-nano", device="cpu"),
        device="cpu",
        weights_dir=weights_dir,
        allow_download=False,
    )


def test_missing_weights_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_rfdetr_import(monkeypatch)
    with pytest.raises(ModelUnavailableError, match="not found"):
        _detector(tmp_path)


def test_corrupt_weights_are_rejected_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_rfdetr_import(monkeypatch)
    weights = tmp_path / RFDETR_VARIANTS["rfdetr-nano"].weights_filename
    weights.write_bytes(b"truncated checkpoint")

    with pytest.raises(ModelUnavailableError, match="integrity check"):
        _detector(tmp_path)
    assert weights.read_bytes() == b"truncated checkpoint"


def test_verify_weights_returns_md5_on_match(tmp_path: Path) -> None:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"checkpoint")
    expected = hashlib.md5(b"checkpoint", usedforsecurity=False).hexdigest()
    variant = rfdetr.RFDETRVariant("RFDETRNano", "w.pth", expected)
    assert verify_weights(weights, variant, "rfdetr-nano") == expected


def test_pinned_md5s_match_installed_rfdetr_registry() -> None:
    model_weights = pytest.importorskip("rfdetr.assets.model_weights")
    for variant in RFDETR_VARIANTS.values():
        assert model_weights.ModelWeights.get_md5(variant.weights_filename) == variant.weights_md5
