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

from utils.console import (
    compile_progress,
    quiet_external_logs,
    siamram_log,
    silence_noisy_libraries,
)

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
    with quiet_external_logs():
        import torch_tensorrt

    mod = model.eval()
    mod = mod.half() if dtype == torch.float16 else mod.float()

    enabled_precisions: Set[torch.dtype] = (
        {torch.float16} if dtype == torch.float16 else {torch.float32}
    )

    dummy = torch.randn(1, 3, input_h, input_w, device=device, dtype=dtype)
    cache_path = (
        Path(cache_dir) / f"{cache_prefix}.ts"
        if cache_dir
        else None
    )

    # Advance the shared one-time compile progress bar (the same bar SiamABC
    # uses). stage()/complete() auto-open a lone session if SiamABC's has already
    # closed, so OSNet still renders a clean 0-100% bar. Cache load/save are
    # silent so they don't break the single line; failures still warn.
    compile_progress().stage("compiling osnet descriptor engine")
    try:
        if cache_path is not None and cache_path.exists() and not rebuild_cache:
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

        with quiet_external_logs():
            with torch.no_grad():
                scripted = torch.jit.trace(mod, dummy)

            engine = torch_tensorrt.compile(
                scripted,
                inputs=[
                    torch_tensorrt.Input(
                        shape=(1, 3, input_h, input_w),
                        dtype=dtype,
                    )
                ],
                enabled_precisions=enabled_precisions,
                truncate_long_and_double=True,
            )

            # Warm the engine so the first real frame doesn't pay the lazy-init cost.
            with torch.no_grad():
                engine(dummy)

        if cache_path is not None:
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
