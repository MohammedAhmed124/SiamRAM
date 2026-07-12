"""
build_osnet_trt
===============
TensorRT compilation for the OSNet appearance-descriptor backbone.

Mirrors the SiamABC backbone engine (see ``siamabc.py``):

  torch.jit.trace (TorchScript backend — purely trace-based, no symbolic shape
  guards, so no ``ConstraintViolationError``) → ``torch_tensorrt.compile`` with
  a single **static** input shape ``(1, 3, 256, 128)``.

Why a static batch of 1 (and not a dynamic batch profile)?
  OSNet's classification head does ``v = v.view(v.size(0), -1)`` after the
  global-average-pool. With a *dynamic* batch dimension, Torch-TensorRT's
  TorchScript converter has to materialize shape ops for the dynamic
  ``size(0)`` and mishandles that reshape ("Index to unsqueeze is out of
  bounds"). A static batch of 1 folds ``size(0)`` to a constant, so the engine
  builds cleanly — exactly the fixed-batch approach the SiamABC backbone uses.
  The caller (``_OSNetDescriptorExtractor.extract_batch``) loops per crop, so
  multi-candidate calls just issue several sub-millisecond engine runs.

Logging uses the shared SiamRAM console helpers so the output matches the rest
of the TRT compilation while third-party TensorRT chatter stays hidden.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

import torch
import torch.nn as nn

from utils.console import (compile_progress, quiet_external_logs, siamram_log,
                           silence_noisy_libraries)

from .backend import TensorRTBackend, import_torch_trt, select_trt_backend
from .raw_trt import build_raw_trt_module

log = logging.getLogger(__name__)
silence_noisy_libraries()


def build_osnet_trt(
    model: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    input_h: int,
    input_w: int,
    cache_dir: str = "",
    cache_prefix: str = "osnet",
    rebuild_cache: bool = False,
    backend: TensorRTBackend | None = None,
) -> nn.Module:
    """
    Compile a torchreid OSNet model into a static batch-1 TensorRT engine.

    Parameters
    ----------
    model      : torchreid OSNet, already ``.eval()`` (and ``.half()`` if fp16).
    dtype      : torch.float16 or torch.float32 — drives both the engine
                 precision and the dummy dtype.
    device     : CUDA device the engine is built and run on.
    input_h/w  : fixed OSNet crop resolution (256×128).

    Returns the compiled module; call it on a ``(1, 3, H, W)`` tensor to get a
    single ``(1, D)`` feature row. The caller loops per crop.
    """
    backend = backend or select_trt_backend()

    mod = model.eval()
    mod = mod.float() if backend == "raw" else (
        mod.half() if dtype == torch.float16 else mod.float()
    )

    enabled_precisions: Set[torch.dtype] = (
        {torch.float16} if dtype == torch.float16 else {torch.float32}
    )

    input_dtype = torch.float32 if backend == "raw" else dtype
    dummy = torch.randn(1, 3, input_h, input_w, device=device, dtype=input_dtype)
    cache_path = (
        Path(cache_dir) / f"{cache_prefix}.{'engine' if backend == 'raw' else 'ts'}"
        if cache_dir
        else None
    )

    compile_progress().stage("compiling osnet descriptor engine")
    try:
        if (
            backend == "torch_tensorrt"
            and cache_path is not None
            and cache_path.exists()
            and not rebuild_cache
        ):
            try:
                engine = torch.jit.load(str(cache_path), map_location=device).eval()
                with torch.no_grad():
                    engine(dummy)
                return engine
            except Exception as exc:
                siamram_log(
                    f"cached OSNet engine load failed ({cache_path.name}): {exc}; rebuilding",
                    phase="DESC",
                    status="warn",
                    indent=1,
                )

        if backend == "raw":
            engine = build_raw_trt_module(
                mod,
                (dummy,),
                device=device,
                input_names=["crop"],
                output_names=["descriptor"],
                fp16=(dtype == torch.float16),
                cache_path=cache_path,
                rebuild_cache=rebuild_cache,
            )
        else:
            with quiet_external_logs():
                torch_tensorrt = import_torch_trt()
                with torch.no_grad():
                    scripted = torch.jit.trace(mod, dummy)
                engine = torch_tensorrt.compile(
                    scripted,
                    inputs=[
                        torch_tensorrt.Input(
                            shape=(1, 3, input_h, input_w), dtype=dtype
                        )
                    ],
                    enabled_precisions=enabled_precisions,
                    truncate_long_and_double=True,
                )

        with torch.no_grad():
            got = engine(dummy).float()
            want = mod(dummy).float()
        tolerance = 2e-2 if dtype == torch.float16 else 1e-3
        if got.shape != want.shape:
            raise RuntimeError(
                f"OSNet TensorRT shape {tuple(got.shape)} != {tuple(want.shape)}"
            )
        if not torch.allclose(got, want, rtol=tolerance, atol=tolerance):
            max_diff = float((got - want).abs().max())
            raise RuntimeError(f"OSNet TensorRT validation failed (max diff {max_diff:.4g})")

        if backend == "torch_tensorrt" and cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.jit.save(engine, str(cache_path))
            except Exception as exc:
                siamram_log(
                    f"cached OSNet engine save failed ({cache_path.name}): {exc}",
                    phase="DESC",
                    status="warn",
                    indent=1,
                )

        return engine
    finally:
        compile_progress().complete()
