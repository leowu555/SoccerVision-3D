"""API configuration, loaded from ``SOCCERVISION_*`` environment variables (and ``.env``)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from soccervision.config import DetectorName, DeviceName

ENV_PREFIX = "SOCCERVISION_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, env_file=".env", extra="ignore")

    data_dir: Path = Field(
        default=Path("data"), description="Root for uploads, artifacts, and the job DB."
    )
    weights_dir: Path = Field(
        default=Path("models"), description="Directory holding model checkpoints."
    )
    allow_weight_download: bool = Field(
        default=True,
        description="Allow RF-DETR to download missing pretrained weights on first use.",
    )

    max_upload_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    max_video_duration_seconds: float = Field(default=600.0, gt=0)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    dispatcher_poll_seconds: float = Field(default=1.0, gt=0)

    default_detector: DetectorName = "rfdetr-nano"
    default_device: DeviceName = "auto"
    default_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    default_inference_stride: int = Field(default=1, ge=1)

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed browser origins.",
    )
    log_level: str = "INFO"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "soccervision.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def resolved(self) -> "Settings":
        """Return a copy with absolute paths so worker subprocesses agree on locations."""
        return self.model_copy(
            update={"data_dir": self.data_dir.resolve(), "weights_dir": self.weights_dir.resolve()}
        )

    def as_env(self) -> dict[str, str]:
        """Serialize to environment variables for worker subprocesses."""
        env: dict[str, str] = {}
        for name, value in self.model_dump().items():
            env[f"{ENV_PREFIX}{name.upper()}"] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )
        return env

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
