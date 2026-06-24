"""
build_osnet_trt
===============
Raw-TensorRT engine (ONNX -> TensorRT, see raw_trt.py) for the OSNet
appearance-descriptor backbone, at a single static input shape ``(1, 3, 256,
128)``. The caller (``_OSNetDescriptorExtractor.extract_batch``) loops per crop,
so multi-candidate calls just issue several sub-millisecond engine runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from utils.console import compile_progress, silence_noisy_libraries

from .raw_trt import build_trt_module

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
    model      : torchreid OSNet, already ``.eval()``.
    dtype      : torch.float16 or torch.float32 — only selects the FP16 builder
                 flag; engine I/O is always FP32.
    device     : CUDA device the engine is built and run on.
    input_h/w  : fixed OSNet crop resolution (256×128).

    Returns the compiled module; call it on a ``(1, 3, H, W)`` tensor to get a
    single ``(1, D)`` feature row. The caller loops per crop.
    """
    dummy = torch.randn(1, 3, input_h, input_w, device=device, dtype=torch.float32)
    cache_path = Path(cache_dir) / f"{cache_prefix}.engine" if cache_dir else None

    compile_progress().stage("compiling osnet descriptor engine")
    try:
        return build_trt_module(
            model.eval().float().to(device),
            (dummy,),
            device=device,
            input_names=["crop"],
            output_names=["descriptor"],
            fp16=(dtype == torch.float16),
            cache_path=cache_path,
            rebuild_cache=rebuild_cache,
        )
    finally:
        compile_progress().complete()
