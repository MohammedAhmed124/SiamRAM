from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from models.SiamABC.tracker.trt_engine import backend


def test_auto_falls_back_when_torch_tensorrt_import_has_linker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIAMRAM_TRT_BACKEND", raising=False)

    def available(name: str) -> bool:
        return name in {"torch_tensorrt", "tensorrt", "onnx"}

    with (
        patch.object(backend, "module_available", side_effect=available),
        patch.object(backend, "import_torch_trt", side_effect=OSError("bad ABI")),
    ):
        assert backend.select_trt_backend() == "raw"


def test_forced_torch_tensorrt_surfaces_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIAMRAM_TRT_BACKEND", "torch_tensorrt")
    with (
        patch.object(backend, "import_torch_trt", side_effect=OSError("bad ABI")),
        pytest.raises(OSError, match="bad ABI"),
    ):
        backend.select_trt_backend()


def test_auto_prefers_raw_on_aarch64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIAMRAM_TRT_BACKEND", raising=False)
    with (
        patch.object(backend.platform, "machine", return_value="aarch64"),
        patch.object(backend, "require_raw_trt"),
        patch.object(backend, "import_torch_trt") as import_torch_trt,
    ):
        assert backend.select_trt_backend() == "raw"
        import_torch_trt.assert_not_called()


def test_auto_rejects_mismatched_torch_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIAMRAM_TRT_BACKEND", raising=False)
    incompatible = type("TorchTRT", (), {"__version__": "2.8.0"})()
    with (
        patch.object(backend.platform, "machine", return_value="x86_64"),
        patch.object(backend, "module_available", return_value=True),
        patch.object(backend, "import_torch_trt", return_value=incompatible),
        patch.object(backend, "require_raw_trt"),
    ):
        assert backend.select_trt_backend() == "raw"


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not backend.module_available("tensorrt")
    or not backend.module_available("onnx"),
    reason="requires CUDA, TensorRT, and ONNX",
)
def test_raw_engine_build_execute_and_cache(tmp_path) -> None:
    from models.SiamABC.tracker.trt_engine.raw_trt import build_raw_trt_module

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    ).to(device).eval()
    sample = torch.randn(1, 3, 16, 16, device=device)
    cache_path = tmp_path / "smoke.engine"

    engine = build_raw_trt_module(
        model,
        (sample,),
        device=device,
        input_names=["input"],
        output_names=["output"],
        cache_path=cache_path,
    )
    with torch.no_grad():
        expected = model(sample)
        actual = engine(sample)
    torch.cuda.synchronize(device)
    assert cache_path.is_file()
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)

    cached = build_raw_trt_module(
        model,
        (sample,),
        device=device,
        input_names=["input"],
        output_names=["output"],
        cache_path=cache_path,
    )
    cached_actual = cached(sample)
    torch.cuda.synchronize(device)
    assert torch.allclose(cached_actual, expected, rtol=1e-3, atol=1e-3)
