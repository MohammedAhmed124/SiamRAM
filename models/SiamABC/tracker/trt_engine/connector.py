"""
TRT connect_model patch for TRTSiamABCNet
=========================================

Drop-in replacement for the connect_model compilation section.

Two root problems solved
------------------------
1. BoxTower.forward uses torch.cuda.Stream context managers.
   TRT's dynamo tracer cannot lower stream synchronisation into a TensorRT
   graph.  Solution: reproduce the serial (non-stream) forward logic in the
   wrapper so the exported graph contains only pure tensor ops.

2. AdaptiveBatchNorm receives `lam` as a live tensor.
   If AdaptiveBatchNorm contains `if lam == 0:` (or any Python-level branch
   on lam's *value*), torch.export raises a guard error.  Even without that,
   lam as a dynamic scalar prevents TRT from constant-folding the BN blend.
   Solution: store lam as a register_buffer (a constant in the exported
   graph).  torch.export sees it as a compile-time constant and folds the
   entire lam-weighted blend into fixed BN parameters.

Why two static engines instead of one dynamic engine
-----------------------------------------------------
lam only ever takes two values at runtime:
  • 0.0          — TTA off (most frames)
  • norm_lambda  — TTA on
Building one engine per value lets TRT constant-fold the BN blend for each
case, which is both faster and numerically exact.  Dispatch is O(1).

Usage (inside TRTSiamABCNet.__init__)
--------------------------------------
    self._connect_engines = _build_connect_engines(
        model,
        s_feat_shape=s_feat_shape,
        t_feat_shape=t_feat_shape,
        norm_lambda=norm_lambda,
        device=self._device,
    )

Usage (inside TRTSiamABCNet.track)
------------------------------------
    bbox_pred, cls_pred = _dispatch_connect(
        self._connect_engines, lam_val, self._norm_lambda,
        search_org=sf.float().contiguous(),
        search=s_mixed.float(),
        kernel=t_mixed.float(),
    )
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from utils.console import quiet_external_logs, silence_noisy_libraries

silence_noisy_libraries()
with quiet_external_logs():
    import torch_tensorrt
    from torch_tensorrt.dynamo import compile as trt_dynamo_compile

log = logging.getLogger(__name__)


class _ConnectModelForExport(nn.Module):
    """
    Wraps BoxTower so it is fully TRT-traceable:

    (a) CUDA streams removed — TRT owns internal parallelism; torch.cuda.stream
        context managers are not lowerable to TRT graph nodes.

    (b) lam stored as register_buffer — torch.export treats buffers as
        compile-time constants, so any Python-level branch inside
        AdaptiveBatchNorm on lam's *value* is resolved at export time and
        constant-folded by TRT.  Two engines are built (lam=0, lam=norm_lambda)
        so each engine sees exactly one concrete lam value.

    forward signature has only tensor inputs — no Python scalars — which is
    required for torch.export and TRT dynamo compilation.
    """

    def __init__(
        self,
        net: nn.Module,
        lam_val: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        bt = copy.deepcopy(net.connect_model)

        for attr in ("_cls_stream", "_reg_stream"):
            if hasattr(bt, attr):
                delattr(bt, attr)

        self.cls_encode = bt.cls_encode
        self.reg_encode = bt.reg_encode
        self.cls_dw = bt.cls_dw
        self.reg_dw = bt.reg_dw
        self.bbox_tower = bt.bbox_tower
        self.cls_tower = bt.cls_tower
        self.bbox_pred = bt.bbox_pred
        self.cls_pred = bt.cls_pred
        self.adjust = bt.adjust
        self.bias = bt.bias

        self.register_buffer(
            "lam",
            torch.tensor(lam_val, dtype=torch.float32, device=device),
        )

    def forward(
        self,
        search_org: torch.Tensor,
        search: torch.Tensor,
        kernel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = kernel.reshape(kernel.size(0), kernel.size(1), -1)

        cls_x = self.cls_encode(search)
        cls_dw = self.cls_dw(z, cls_x, search_org, self.lam)
        c = self.cls_tower(cls_dw, self.lam)
        cls = 0.1 * self.cls_pred(c)

        reg_x = self.reg_encode(search)
        reg_dw = self.reg_dw(z, reg_x, search_org, self.lam)
        x_reg = self.bbox_tower(reg_dw, self.lam)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)

        return x, cls


def _build_connect_engines(
    model: nn.Module,
    s_feat_shape: Tuple[int, ...],
    t_feat_shape: Tuple[int, ...],
    norm_lambda: float,
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    _, C_s, h_s, w_s = s_feat_shape
    _, _, h_t, w_t = t_feat_shape

    dummy_search_org = torch.randn(1, C_s, h_s, w_s, device=device, dtype=torch.float32)
    dummy_search = torch.randn(1, C_s, h_s, w_s, device=device, dtype=torch.float32)
    dummy_kernel = torch.randn(1, C_s, h_t, w_t, device=device, dtype=torch.float32)

    trt_input_set = [
        torch_tensorrt.Input(shape=(1, C_s, h_s, w_s), dtype=torch.float32),
        torch_tensorrt.Input(shape=(1, C_s, h_s, w_s), dtype=torch.float32),
        torch_tensorrt.Input(shape=(1, C_s, h_t, w_t), dtype=torch.float32),
    ]

    engines: Dict[str, torch.nn.Module] = {}

    for tag, lam_val in (("zero", 0.0), ("lambda", norm_lambda)):
        log.info(
            "TRTSiamABCNet: building connect_model TRT engine [lam=%.4f] …", lam_val
        )

        module = (
            _ConnectModelForExport(model, lam_val=lam_val, device=device)
            .eval()
            .float()
            .to(device)
        )

        exported = torch.export.export(
            module,
            args=(dummy_search_org, dummy_search, dummy_kernel),
        )

        engine = trt_dynamo_compile(
            exported,
            inputs=trt_input_set,
            enabled_precisions={torch.float32},
            optimization_level=3,
            use_fast_partitioner=True,
        )

        with torch.no_grad():
            engine(dummy_search_org, dummy_search, dummy_kernel)

        engines[tag] = engine
        del module, exported         
        torch.cuda.empty_cache()     

        log.info("TRTSiamABCNet: connect_model engine [lam=%.4f] ready.", lam_val)

    return engines


def _dispatch_connect(
    engines: Dict[str, torch.nn.Module],
    lam_val: float,
    norm_lambda: float,
    search_org: torch.Tensor,
    search: torch.Tensor,
    kernel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tag = "zero" if abs(lam_val) < 1e-6 else "lambda"
    return engines[tag](search_org, search, kernel)
