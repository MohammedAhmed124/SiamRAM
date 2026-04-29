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
  • 0.0   — TTA off (most frames)
  • norm_lambda — TTA on
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
    lam_val = float(lam.item())
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
from torch_tensorrt.dynamo import compile as trt_dynamo_compile
import torch_tensorrt

from ...model.blocks import AdaptiveBatchNorm, AdaptiveSequential  # adjust import path

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Wrapper: serial BoxTower forward + lam as a buffer constant                 #
# --------------------------------------------------------------------------- #

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
        net:     nn.Module,          # SiamABCNet (or any object with .connect_model)
        lam_val: float,
        device:  torch.device,
    ) -> None:
        super().__init__()

        bt = copy.deepcopy(net.connect_model)

        # Remove cached CUDA streams left over from previous warm-up calls.
        # Their presence doesn't affect correctness here, but deleting them
        # keeps the module state clean for export.
        for attr in ("_cls_stream", "_reg_stream"):
            if hasattr(bt, attr):
                delattr(bt, attr)

        self.cls_encode  = bt.cls_encode
        self.reg_encode  = bt.reg_encode
        self.cls_dw      = bt.cls_dw
        self.reg_dw      = bt.reg_dw
        self.bbox_tower  = bt.bbox_tower   # AdaptiveSequential
        self.cls_tower   = bt.cls_tower    # AdaptiveSequential
        self.bbox_pred   = bt.bbox_pred
        self.cls_pred    = bt.cls_pred
        self.adjust      = bt.adjust
        self.bias        = bt.bias

        # lam is a compile-time constant in the exported graph.
        self.register_buffer(
            "lam",
            torch.tensor(lam_val, dtype=torch.float32, device=device),
        )

    def forward(
        self,
        search_org: torch.Tensor,   # raw search features  (B, C, h_s, w_s)
        search:     torch.Tensor,   # s_mixed              (B, C, h_s, w_s)
        kernel:     torch.Tensor,   # t_mixed              (B, C, h_t, w_t)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Serial reproduction of BoxTower.forward — no CUDA streams.
        Both cls and reg branches are computed sequentially; TRT schedules
        its own kernel-level parallelism internally.
        """
        z = kernel.reshape(kernel.size(0), kernel.size(1), -1)

        # ---- classification branch ---- #
        cls_x  = self.cls_encode(search)
        cls_dw = self.cls_dw(z, cls_x, search_org, self.lam)
        c      = self.cls_tower(cls_dw, self.lam)
        cls    = 0.1 * self.cls_pred(c)

        # ---- regression branch ---- #
        reg_x  = self.reg_encode(search)
        reg_dw = self.reg_dw(z, reg_x, search_org, self.lam)
        x_reg  = self.bbox_tower(reg_dw, self.lam)
        x      = self.adjust * self.bbox_pred(x_reg) + self.bias
        x      = torch.exp(x)

        return x, cls  # (bbox_pred, cls_pred)


# --------------------------------------------------------------------------- #
# Engine builder                                                               #
# --------------------------------------------------------------------------- #

def _build_connect_engines(
    model:        nn.Module,
    s_feat_shape: Tuple[int, ...],   # (B, C_s, h_s, w_s)  from shape probe
    t_feat_shape: Tuple[int, ...],   # (B, C_s, h_t, w_t)  from shape probe
    norm_lambda:  float,
    device:       torch.device,
) -> Dict[str, torch.nn.Module]:
    """
    Build two static-shape TRT FP32 engines for connect_model:
      engines["zero"]   — lam = 0.0
      engines["lambda"] — lam = norm_lambda

    Returns a dict keyed by a string tag; use _dispatch_connect() to call
    the right engine at runtime.

    Why static shapes:
      Both spatial sizes (h_s×w_s for search, h_t×w_t for kernel) are fixed
      throughout a tracking session.  Static-shape engines let TRT select the
      most aggressive kernel implementations and perform constant-folding on
      the lam-weighted BN blend.

    Why FP32:
      cls_pred feeds hard thresholds (0.55, 0.70, 0.80) in SiamRAMTracker.
      FP16 precision loss near sigmoid(0) causes borderline-good scores
      (~0.56) to round below threshold, triggering spurious occlusion entry.
    """
    _, C_s, h_s, w_s = s_feat_shape
    _, _,   h_t, w_t = t_feat_shape

    # Concrete dummy inputs sized exactly as at runtime.
    dummy_search_org = torch.randn(1, C_s, h_s, w_s, device=device, dtype=torch.float32)
    dummy_search     = torch.randn(1, C_s, h_s, w_s, device=device, dtype=torch.float32)
    dummy_kernel     = torch.randn(1, C_s, h_t, w_t, device=device, dtype=torch.float32)

    # Static-shape TRT Input descriptors (one per tensor input to forward).
    trt_inputs = [
        torch_tensorrt.Input(shape=(1, C_s, h_s, w_s), dtype=torch.float32),  # search_org
        torch_tensorrt.Input(shape=(1, C_s, h_s, w_s), dtype=torch.float32),  # search
        torch_tensorrt.Input(shape=(1, C_s, h_t, w_t), dtype=torch.float32),  # kernel
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

        # torch.export with concrete dummy inputs.
        # lam is a buffer → seen as a constant; any Python-level if/else
        # inside AdaptiveBatchNorm is resolved for this concrete lam value
        # and eliminated from the exported graph.
        exported = torch.export.export(
            module,
            args=(dummy_search_org, dummy_search, dummy_kernel),
        )

        engine = trt_dynamo_compile(
            exported,
            inputs=trt_inputs,
            enabled_precisions={torch.float32},
            optimization_level=3,
            use_fast_partitioner=True,
        )

        # Warm up: run one forward pass so TRT JIT-compiles any lazy kernels.
        with torch.no_grad():
            engine(dummy_search_org, dummy_search, dummy_kernel)

        engines[tag] = engine
        log.info(
            "TRTSiamABCNet: connect_model engine [lam=%.4f] ready.", lam_val
        )

    return engines


# --------------------------------------------------------------------------- #
# Runtime dispatcher                                                           #
# --------------------------------------------------------------------------- #

def _dispatch_connect(
    engines:      Dict[str, torch.nn.Module],
    lam_val:      float,
    norm_lambda:  float,
    search_org:   torch.Tensor,
    search:       torch.Tensor,
    kernel:       torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call the correct static-lam engine based on the runtime lam value.

    lam == 0  → engines["zero"]    (TTA off, most frames)
    otherwise → engines["lambda"]  (TTA on)

    A small epsilon (1e-6) guards against floating-point representation
    artefacts when lam_val comes from float(tensor.item()).
    """
    tag = "zero" if abs(lam_val) < 1e-6 else "lambda"
    return engines[tag](search_org, search, kernel)


# --------------------------------------------------------------------------- #
# Integration guide (annotated diff for TRTSiamABCNet)                       #
# --------------------------------------------------------------------------- #
"""
Replace the entire "Step 3" block in TRTSiamABCNet.__init__ with:

    # ------------------------------------------------------------------ #
    # Step 3 — TRT: connect_model — two static-lam FP32 engines           #
    # ------------------------------------------------------------------ #
    log.info("TRTSiamABCNet: building connect_model TRT engines…")
    self._connect_engines = _build_connect_engines(
        model,
        s_feat_shape=s_feat_shape,
        t_feat_shape=t_feat_shape,
        norm_lambda=norm_lambda,
        device=self._device,
    )
    log.info("TRTSiamABCNet: connect_model engines ready.")


Replace the connect_model call in track() with:

    bbox_pred, cls_pred = _dispatch_connect(
        self._connect_engines,
        lam_val=float(lam.item()),
        norm_lambda=self._norm_lambda,
        search_org=sf.float().contiguous(),
        search=s_mixed.float(),
        kernel=t_mixed.float(),
    )


Remove:
    • _warm_connect_model (warm-up is now inside _build_connect_engines)
    • The old _ConnectModel class (no longer used)
    • The torch.compile connect_mod block
"""