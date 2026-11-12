import functools
import importlib.util

from fastapi import APIRouter

from soccervision import __version__
from soccervision.devices import detect_available_devices
from soccervision.pipeline import PIPELINE_NAME, PIPELINE_VERSION
from soccervision.vision.detection import weights_path_for
from soccervision_api.api.deps import SettingsDep
from soccervision_api.schemas import (
    AnalysisDefaults,
    DeviceStatus,
    HealthResponse,
    Limits,
    ModelStatus,
)

router = APIRouter(tags=["system"])


@functools.cache
def _devices() -> DeviceStatus:
    availability = detect_available_devices()
    return DeviceStatus(
        torch_installed=availability.torch_installed,
        cuda=availability.cuda,
        mps=availability.mps,
        cpu=availability.cpu,
    )


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        version=__version__,
        pipeline_name=PIPELINE_NAME,
        pipeline_version=PIPELINE_VERSION,
        devices=_devices(),
        model=ModelStatus(
            detector=settings.default_detector,
            ml_dependencies_installed=importlib.util.find_spec("rfdetr") is not None,
            weights_present=weights_path_for(
                settings.default_detector, settings.weights_dir
            ).is_file(),
            weights_download_allowed=settings.allow_weight_download,
        ),
        limits=Limits(
            max_upload_bytes=settings.max_upload_bytes,
            max_video_duration_seconds=settings.max_video_duration_seconds,
            max_concurrent_jobs=settings.max_concurrent_jobs,
        ),
        defaults=AnalysisDefaults(
            detector=settings.default_detector,
            device=settings.default_device,
            confidence_threshold=settings.default_confidence_threshold,
            inference_stride=settings.default_inference_stride,
        ),
    )
