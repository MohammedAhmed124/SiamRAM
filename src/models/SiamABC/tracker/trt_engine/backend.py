"""TensorRT backend discovery shared by all compiled model components."""

from __future__ import annotations

import importlib.util
import os
import platform
from typing import Literal

import torch

TensorRTBackend = Literal["torch_tensorrt", "raw"]


def module_available(name: str) -> bool:
    """Return whether *name* can be imported without importing it eagerly."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def select_trt_backend() -> TensorRTBackend:
    """Select Torch-TensorRT when usable, otherwise raw TensorRT.

    ``SIAMRAM_TRT_BACKEND`` can force ``torch_tensorrt`` or ``raw`` for testing
    and deployment. Automatic selection deliberately checks only availability;
    native ABI/linker failures during import are caught by ``import_torch_trt``.
    """
    requested = os.environ.get("SIAMRAM_TRT_BACKEND", "auto").strip().lower()
    aliases = {
        "torch": "torch_tensorrt",
        "torch-trt": "torch_tensorrt",
        "torch_tensorrt": "torch_tensorrt",
        "raw": "raw",
        "tensorrt": "raw",
        "onnx": "raw",
        "auto": "auto",
        "": "auto",
    }
    normalized = aliases.get(requested)
    if normalized is None:
        raise ValueError(
            "SIAMRAM_TRT_BACKEND must be auto, torch_tensorrt, or raw; "
            f"got {requested!r}"
        )
    if normalized == "torch_tensorrt":
        import_torch_trt()
        return "torch_tensorrt"
    if normalized == "raw":
        require_raw_trt()
        return "raw"

    if platform.machine().lower() in {"aarch64", "arm64"}:
        require_raw_trt()
        return "raw"
    if module_available("torch_tensorrt"):
        try:
            torch_trt = import_torch_trt()
            if _major_minor(torch.__version__) == _major_minor(
                str(getattr(torch_trt, "__version__", ""))
            ):
                return "torch_tensorrt"
        except (ImportError, OSError, RuntimeError):
            pass
    require_raw_trt()
    return "raw"


def import_torch_trt():
    """Import Torch-TensorRT, surfacing ABI/linker errors to backend selection."""
    import torch_tensorrt

    return torch_tensorrt


def _major_minor(version: str) -> tuple[int, int] | None:
    """Extract a package major/minor pair, ignoring CUDA/local suffixes."""
    parts = version.split("+", 1)[0].split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def require_raw_trt() -> None:
    """Raise a useful error unless raw TensorRT and ONNX export are available."""
    missing = [name for name in ("tensorrt", "onnx") if not module_available(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Raw TensorRT fallback requires JetPack TensorRT plus ONNX export; "
            f"missing: {joined}"
        )


def backend_version(backend: TensorRTBackend) -> str:
    """Return a cache-fingerprintable version for the selected backend."""
    if backend == "torch_tensorrt":
        module = import_torch_trt()
    else:
        import tensorrt as module
    return str(getattr(module, "__version__", "unknown"))
