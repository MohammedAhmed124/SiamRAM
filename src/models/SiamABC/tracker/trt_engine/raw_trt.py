"""
raw_trt
=======
Raw-TensorRT engine builder used everywhere torch_tensorrt used to be.

torch_tensorrt is a compiled C++ extension that must be ABI-matched to the exact
torch build, which makes it uninstallable on the Jetson. This module reaches the
same place with only ``torch`` + the ``tensorrt`` Python module (both present on
JetPack): export the nn.Module to ONNX, build a TensorRT engine from those bytes,
and run it with torch CUDA tensors as the I/O buffers (their ``data_ptr()`` is
handed straight to TensorRT, so there is no pycuda/cuda-python dependency and no
host<->device copies).

Engines are static-shape: each call site builds one engine per concrete input
shape. I/O is always FP32; ``fp16=True`` only flips the builder's FP16 flag, so
TensorRT runs internal layers in half precision while keeping FP32 in/out.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import tensorrt as trt
import torch
import torch.nn as nn

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}
if hasattr(trt.DataType, "INT64"):
    _TRT_TO_TORCH[trt.DataType.INT64] = torch.int64


class TRTModule:
    """
    Callable wrapper around a deserialized TensorRT engine.

    Behaves like the eager nn.Module it replaced: ``module(*inputs) -> output`` (a
    single tensor, or a tuple when the engine has multiple outputs). Inputs are
    bound by name to torch CUDA tensors; outputs are freshly allocated torch
    tensors. Execution runs on the current torch CUDA stream and synchronises
    before returning, so results are immediately usable by eager code.
    """

    def __init__(
        self,
        engine: "trt.ICudaEngine",
        device: torch.device,
        input_names: Sequence[str],
        output_names: Sequence[str],
    ) -> None:
        self.engine = engine
        self.device = device
        self.context = engine.create_execution_context()
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    # No-ops so call sites that treat engines like nn.Modules keep working.
    def eval(self) -> "TRTModule":
        return self

    def to(self, *args, **kwargs) -> "TRTModule":
        return self

    def half(self) -> "TRTModule":
        return self

    def float(self) -> "TRTModule":
        return self

    def __call__(self, *inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(inputs) != len(self.input_names):
            raise RuntimeError(
                f"TRTModule expected {len(self.input_names)} inputs "
                f"({self.input_names}), got {len(inputs)}"
            )

        held: List[torch.Tensor] = []  # keep input tensors alive through execute
        for name, tensor in zip(self.input_names, inputs):
            tensor = tensor.to(device=self.device, dtype=torch.float32).contiguous()
            held.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        outputs: List[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            out = torch.empty(shape, dtype=dtype, device=self.device)
            outputs.append(out)
            self.context.set_tensor_address(name, int(out.data_ptr()))

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _export_onnx(
    module: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> bytes:
    buffer = io.BytesIO()
    with torch.no_grad():
        try:
            torch.onnx.export(
                module,
                tuple(example_inputs),
                buffer,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            # Older torch has no `dynamo` kwarg; its default exporter is the legacy one.
            torch.onnx.export(
                module,
                tuple(example_inputs),
                buffer,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=17,
                do_constant_folding=True,
            )
    return buffer.getvalue()


def _build_serialized_engine(
    onnx_bytes: bytes,
    fp16: bool,
    workspace_bytes: int,
) -> "trt.IHostMemory":
    builder = trt.Builder(_TRT_LOGGER)

    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, _TRT_LOGGER)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 engine requested but the platform lacks fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    return serialized


def _smoke(module: TRTModule, example_inputs: Tuple[torch.Tensor, ...]) -> None:
    with torch.no_grad():
        output = module(*example_inputs)
    tensors = output if isinstance(output, tuple) else (output,)
    for tensor in tensors:
        if not bool(torch.isfinite(tensor.float()).all().item()):
            raise RuntimeError("TensorRT engine produced non-finite output")


def build_trt_module(
    module: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    input_names: Sequence[str],
    output_names: Sequence[str],
    fp16: bool = False,
    cache_path: Optional[Path] = None,
    rebuild_cache: bool = False,
    workspace_bytes: int = 1 << 30,
) -> TRTModule:
    """
    Build (or load) a static-shape TensorRT engine for *module* and return a
    callable :class:`TRTModule`.

    *example_inputs* fix the engine's input shapes and seed the post-build smoke
    test. When *cache_path* is given, the serialized engine bytes are cached there
    and reused on later runs (unless *rebuild_cache*).
    """
    runtime = trt.Runtime(_TRT_LOGGER)

    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        engine = runtime.deserialize_cuda_engine(cache_path.read_bytes())
        if engine is not None:
            wrapper = TRTModule(engine, device, input_names, output_names)
            _smoke(wrapper, example_inputs)
            return wrapper

    onnx_bytes = _export_onnx(module, example_inputs, input_names, output_names)
    serialized = _build_serialized_engine(onnx_bytes, fp16, workspace_bytes)

    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization returned None")
    wrapper = TRTModule(engine, device, input_names, output_names)
    _smoke(wrapper, example_inputs)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(bytes(serialized))
        os.replace(temporary, cache_path)

    return wrapper
