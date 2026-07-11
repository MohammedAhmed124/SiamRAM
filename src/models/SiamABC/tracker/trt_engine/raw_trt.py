"""ONNX-to-TensorRT compilation without the Torch-TensorRT C++ extension."""

from __future__ import annotations

import io
import os
import platform
from pathlib import Path
from typing import Sequence

import tensorrt as trt
import torch
import torch.nn as nn

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_MIB = 1 << 20
_JETSON_WORKSPACE_BYTES = 256 * _MIB
_JETSON_TACTIC_DRAM_BYTES = 512 * _MIB
_DESKTOP_WORKSPACE_BYTES = 1 << 30
_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}
if hasattr(trt.DataType, "INT64"):
    _TRT_TO_TORCH[trt.DataType.INT64] = torch.int64


class RawTRTModule:
    """Callable raw TensorRT engine using Torch CUDA tensors as bindings."""

    def __init__(
        self,
        engine: trt.ICudaEngine,
        device: torch.device,
        input_names: Sequence[str],
        output_names: Sequence[str],
    ) -> None:
        self.engine = engine
        self.device = device
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT failed to create an execution context")
        self.stream = torch.cuda.Stream(device=device)
        self.input_names = tuple(input_names)
        self.output_names = tuple(output_names)
        engine_names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        missing = (set(self.input_names) | set(self.output_names)) - engine_names
        if missing:
            raise RuntimeError(f"TensorRT engine is missing bindings: {sorted(missing)}")

    def eval(self) -> RawTRTModule:
        return self

    def to(self, *args, **kwargs) -> RawTRTModule:
        return self

    def half(self) -> RawTRTModule:
        return self

    def float(self) -> RawTRTModule:
        return self

    def __call__(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.input_names):
            raise RuntimeError(
                f"RawTRTModule expected {len(self.input_names)} inputs, got {len(inputs)}"
            )

        held_inputs: list[torch.Tensor] = []
        for name, tensor in zip(self.input_names, inputs):
            if not torch.is_tensor(tensor):
                raise TypeError(f"TensorRT input {name!r} must be a torch.Tensor")
            bound = tensor.to(self.device, dtype=torch.float32).contiguous()
            held_inputs.append(bound)
            bound.record_stream(self.stream)
            if not self.context.set_input_shape(name, tuple(bound.shape)):
                raise RuntimeError(
                    f"TensorRT rejected shape {tuple(bound.shape)} for input {name!r}"
                )
            self.context.set_tensor_address(name, int(bound.data_ptr()))

        outputs: list[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output {name!r} unresolved: {shape}")
            trt_dtype = self.engine.get_tensor_dtype(name)
            try:
                dtype = _TRT_TO_TORCH[trt_dtype]
            except KeyError as exc:
                raise RuntimeError(
                    f"Unsupported TensorRT output dtype {trt_dtype} for {name!r}"
                ) from exc
            output = torch.empty(shape, dtype=dtype, device=self.device)
            outputs.append(output)
            self.context.set_tensor_address(name, int(output.data_ptr()))

        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for output in outputs:
            output.record_stream(self.stream)
        current_stream.wait_stream(self.stream)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _export_onnx(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> bytes:
    buffer = io.BytesIO()
    kwargs = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "opset_version": 17,
        "do_constant_folding": True,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(module, example_inputs, buffer, dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(module, example_inputs, buffer, **kwargs)
    return buffer.getvalue()


def _build_serialized_engine(
    onnx_bytes: bytes,
    *,
    fp16: bool,
    workspace_bytes: int,
    tactic_dram_bytes: int | None,
) -> bytes:
    builder = trt.Builder(_TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, _TRT_LOGGER)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if tactic_dram_bytes is not None and hasattr(trt.MemoryPoolType, "TACTIC_DRAM"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.TACTIC_DRAM, tactic_dram_bytes
        )
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 requested but TensorRT reports no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")
    return bytes(serialized)


def _memory_limit_from_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value_mb = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer number of MiB") from exc
    if value_mb <= 0:
        raise ValueError(f"{name} must be a positive integer number of MiB")
    return value_mb * _MIB


def _resolve_builder_memory_limits(
    workspace_bytes: int | None,
    tactic_dram_bytes: int | None,
) -> tuple[int, int | None]:
    """Return bounded builder pools, with conservative defaults on Jetson."""
    jetson = platform.machine().lower() in {"aarch64", "arm64"}
    if workspace_bytes is None:
        workspace_bytes = _memory_limit_from_env("SIAMRAM_TRT_WORKSPACE_MB")
    if workspace_bytes is None:
        workspace_bytes = (
            _JETSON_WORKSPACE_BYTES if jetson else _DESKTOP_WORKSPACE_BYTES
        )

    if tactic_dram_bytes is None:
        tactic_dram_bytes = _memory_limit_from_env("SIAMRAM_TRT_TACTIC_DRAM_MB")
    if tactic_dram_bytes is None and jetson:
        tactic_dram_bytes = _JETSON_TACTIC_DRAM_BYTES
    return workspace_bytes, tactic_dram_bytes


def _deserialize(serialized: bytes) -> trt.ICudaEngine:
    runtime = trt.Runtime(_TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the engine")
    return engine


def _validate_finite(module: RawTRTModule, inputs: tuple[torch.Tensor, ...]) -> None:
    with torch.no_grad():
        result = module(*inputs)
    outputs = result if isinstance(result, tuple) else (result,)
    for output in outputs:
        if not bool(torch.isfinite(output.float()).all().item()):
            raise RuntimeError("Raw TensorRT engine produced non-finite output")


def build_raw_trt_module(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    input_names: Sequence[str],
    output_names: Sequence[str],
    fp16: bool = False,
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
    workspace_bytes: int | None = None,
    tactic_dram_bytes: int | None = None,
) -> RawTRTModule:
    """Build or load a static-shape raw TensorRT engine."""
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        try:
            cached = RawTRTModule(
                _deserialize(cache_path.read_bytes()), device, input_names, output_names
            )
            _validate_finite(cached, example_inputs)
            return cached
        except (OSError, RuntimeError):
            cache_path.unlink(missing_ok=True)

    workspace_bytes, tactic_dram_bytes = _resolve_builder_memory_limits(
        workspace_bytes, tactic_dram_bytes
    )

    prepared = module.eval().cpu().float()
    export_inputs = tuple(
        item.detach().to(device="cpu", dtype=torch.float32) for item in example_inputs
    )
    onnx_bytes = _export_onnx(prepared, export_inputs, input_names, output_names)
    del export_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    serialized = _build_serialized_engine(
        onnx_bytes,
        fp16=fp16,
        workspace_bytes=workspace_bytes,
        tactic_dram_bytes=tactic_dram_bytes,
    )
    prepared.to(device)
    inputs = tuple(item.to(device, dtype=torch.float32) for item in example_inputs)
    wrapper = RawTRTModule(_deserialize(serialized), device, input_names, output_names)
    _validate_finite(wrapper, inputs)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(serialized)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return wrapper
