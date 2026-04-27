"""Check whether the local Python environment can import project GPU dependencies."""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    module: str
    label: str
    required: bool = False


PROBES = (
    Probe("numpy", "NumPy", required=True),
    Probe("torch", "PyTorch"),
    Probe("triton", "Triton"),
    Probe("cutlass", "NVIDIA CUTLASS Python package"),
    Probe("cutlass.cute", "NVIDIA CuTe DSL"),
)


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "imported"))


def main() -> int:
    print(f"Python platform: {platform.platform()}")
    failed_required = False

    for probe in PROBES:
        try:
            module = importlib.import_module(probe.module)
        except Exception as exc:
            status = "missing"
            if probe.required:
                failed_required = True
                status = "missing required"
            print(f"{probe.label}: {status} ({exc})")
            continue

        print(f"{probe.label}: {_version(module)}")

        if probe.module == "torch":
            cuda = getattr(module, "cuda", None)
            cuda_available = bool(cuda and cuda.is_available())
            cuda_version = getattr(getattr(module, "version", None), "cuda", None)
            print(f"PyTorch CUDA available: {cuda_available}")
            print(f"PyTorch CUDA version: {cuda_version}")

    return 1 if failed_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
