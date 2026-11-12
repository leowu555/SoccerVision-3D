import pytest

from soccervision.devices import DeviceAvailability, resolve_device
from soccervision.errors import DeviceUnavailableError

CPU_ONLY = DeviceAvailability(torch_installed=True, cuda=False, mps=False)
APPLE = DeviceAvailability(torch_installed=True, cuda=False, mps=True)
NVIDIA = DeviceAvailability(torch_installed=True, cuda=True, mps=False)
NO_TORCH = DeviceAvailability(torch_installed=False, cuda=False, mps=False)


@pytest.mark.parametrize(
    ("availability", "expected"),
    [(CPU_ONLY, "cpu"), (APPLE, "mps"), (NVIDIA, "cuda"), (NO_TORCH, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    availability: DeviceAvailability, expected: str
) -> None:
    assert resolve_device("auto", availability) == expected


def test_explicit_cpu_always_works() -> None:
    assert resolve_device("cpu", NVIDIA) == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_fails_loudly(device: str) -> None:
    with pytest.raises(DeviceUnavailableError):
        resolve_device(device, CPU_ONLY)  # type: ignore[arg-type]
