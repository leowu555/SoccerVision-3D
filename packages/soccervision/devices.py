"""Compute device discovery and selection.

Torch is an optional dependency, so this module must import cleanly without it.
"""

import importlib.util
import logging
from dataclasses import dataclass

from soccervision.config import DeviceName
from soccervision.errors import DeviceUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceAvailability:
    torch_installed: bool
    cuda: bool
    mps: bool
    cpu: bool = True


def detect_available_devices() -> DeviceAvailability:
    if importlib.util.find_spec("torch") is None:
        return DeviceAvailability(torch_installed=False, cuda=False, mps=False)

    import torch

    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return DeviceAvailability(torch_installed=True, cuda=cuda, mps=mps)


def resolve_device(requested: DeviceName, availability: DeviceAvailability | None = None) -> str:
    """Map a requested device to a concrete one, failing loudly if an explicit request is impossible."""
    available = availability if availability is not None else detect_available_devices()

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not available.cuda:
            raise DeviceUnavailableError("CUDA was requested but no CUDA device is available.")
        return "cuda"
    if requested == "mps":
        if not available.mps:
            raise DeviceUnavailableError(
                "Apple MPS was requested but is not available "
                "(requires Apple Silicon and an MPS-enabled torch)."
            )
        return "mps"

    if available.cuda:
        device = "cuda"
    elif available.mps:
        device = "mps"
    else:
        device = "cpu"
    logger.info("Automatic device selection chose %s", device)
    return device
