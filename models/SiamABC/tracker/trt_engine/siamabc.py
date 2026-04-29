"""
TRTSiamABCNet
=============
Drop-in TensorRT replacement for SiamABCNet for use inside SiamABCTracker.

Architecture
------------
get_features      → fully TRT-compiled (encoder + neck)

track             → SPLIT into two stages:
  Stage 1 (TRT)  : polarized_self_attention + attention_neck
                   Single dynamic-shape engine covering both spatial resolutions.
                   One engine = one kernel family = numerically consistent
                   t_mixed / s_mixed for BoxTower cross-correlation.
  Stage 2 (torch.compile, FP32) : connect_model  (BoxTower + AdaptiveBatchNorm)
                   AdaptiveBatchNorm has Python-level control flow that TRT's
                   dynamo tracer cannot lower; torch.compile handles it fine.
                   IMPORTANT: connect_model is kept in FP32 to preserve the
                   classification score precision that all SiamRAMTracker
                   thresholds (conf_threshold, reacq_threshold, etc.) depend on.

Why not cast connect_model to FP16?
  The cls_pred sigmoid output feeds hard thresholds (0.55, 0.70, 0.80) in
  SiamRAMTracker. FP16 reduces pre-sigmoid logit precision near 0, causing
  borderline-good scores (~0.56) to round below threshold, triggering spurious
  occlusion entry every frame and halving tracking performance. The encoder/neck
  (the bulk of compute) still runs in FP16 for speed.

Why a single dynamic-shape attention engine (not two separate engines)?
  Two separate TRT engines for 8x8 and 16x16 pick different CUDA kernel
  implementations (different tile sizes, Softmax reduction strategies). This
  puts t_mixed and s_mixed in slightly different numerical spaces, which corrupts
  the matmul in CorrelationConcatAtt. A single engine with a dynamic shape
  profile uses one consistent kernel family for both spatial sizes.

Bug fixes applied vs previous version
--------------------------------------
  1. Engine built AFTER shape probing — template_hw/search_hw now come from the
     actual probed shapes (h_t, h_s), not hardcoded 8/16.
  2. adjust_channels default corrected from 512 → 256 (SiamABCNet default).
     channels passed to engine = 256*2 = 512, not 1024.
  3. Dead two-engine code (_trt_attn_template / _trt_attn_search) removed.
  4. _AttentionNeck now deep-copies weights (was holding live references).
  5. _warm_connect_model dummy_kernel uses C_s (neck output channels = 256),
     not C_t (raw feature channels from encoder).

Delegated / no-op:
    modules()                   → original model (AdaptiveBatchNorm discovery)
    invalidate_template_cache() → no-op (TRT engines are stateless)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch_tensorrt

from .trt_utils import _AttentionNeck, _FeatureExtractorModule, _cast_module, _trt_input
from ...model import constants

log = logging.getLogger(__name__)
from .connector import _build_connect_engines, _dispatch_connect

logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
logging.getLogger("torch_tensorrt.dynamo.conversion").setLevel(logging.ERROR)


class TRTSiamABCNet:
    """
    TensorRT + torch.compile wrapper for SiamABCNet.

    Parameters
    ----------
    model          : trained SiamABCNet — already .cuda().eval(), kept in FP32
    template_size  : spatial H/W of template crop (e.g. 64)
    instance_size  : spatial H/W of search  crop  (e.g. 128)
    norm_lambda    : TTA blending weight
    cuda_id        : GPU index
    fp16           : use FP16 for TRT encoder/neck engines ONLY —
                     attention engine is always FP32 (numerical consistency),
                     connect_model is always FP32 (score threshold precision)
    adjust_channels: neck output channels — must match SiamABCNet config
                     (FIX 2: default corrected to 256, was wrongly 512)
    min/opt/max_batch: TRT dynamic-shape batch-size range (1/1/1 for tracking)
    """

    def __init__(
        self,
        model: nn.Module,
        template_size: int,
        instance_size: int,
        norm_lambda: float = 0.1,
        cuda_id: int = 0,
        fp16: bool = True,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 1,
        adjust_channels: int = 256,  # FIX 2: was 512 — SiamABCNet default is 256
    ) -> None:

        self._model = model.eval()  # original FP32 — never mutated
        self._norm_lambda = norm_lambda
        self._device = torch.device(f"cuda:{cuda_id}")
        self._fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self._template_size = template_size
        self._instance_size = instance_size

        enabled_precisions: Set[torch.dtype] = (
            {torch.float16} if fp16 else {torch.float32}
        )
        batch = (min_batch, opt_batch, max_batch)

        with torch.no_grad():
            _t = torch.randn(opt_batch, 3, template_size, template_size,
                             device=self._device, dtype=torch.float32)
            _s = torch.randn(opt_batch, 3, instance_size, instance_size,
                             device=self._device, dtype=torch.float32)
            t_feat_shape: Tuple[int, ...] = tuple(model.get_features(_t).shape)
            s_feat_shape: Tuple[int, ...] = tuple(model.get_features(_s).shape)

        _, C, h_t, w_t = t_feat_shape
        _, _, h_s, w_s = s_feat_shape

        self._trt_feat_template = self._compile_feat(
            model, template_size, *batch, enabled_precisions,
        )
        self._trt_feat_search = self._compile_feat(
            model, instance_size, *batch, enabled_precisions,
        )

        _attn_mod = _AttentionNeck(model).eval().to(self._device).float()
        self._trt_attn = torch.compile(_attn_mod, dynamic=True, fullgraph=True)

        # Warm both shapes so Inductor compiles both specializations now, not lazily
        with torch.no_grad():
            for hw in (h_t, h_s):
                _dummy = torch.randn(1, adjust_channels * 2, hw, hw,
                                     device=self._device, dtype=torch.float32)
                self._trt_attn(_dummy)

        self._connect_engines = _build_connect_engines(
            model,
            s_feat_shape=s_feat_shape,
            t_feat_shape=t_feat_shape,
            norm_lambda=norm_lambda,
            device=self._device,
        )

    # ---------------------------------------------------------------------- #
    # Compilation helpers                                                     #
    # ---------------------------------------------------------------------- #

    def _compile_feat(
        self,
        model: nn.Module,
        crop_size: int,
        min_batch: int,
        opt_batch: int,
        max_batch: int,
        enabled_precisions: Set[torch.dtype],
    ) -> torch.nn.Module:
        mod = _FeatureExtractorModule(model).eval().to(self._device)
        _cast_module(mod, self._dtype)

        return torch_tensorrt.compile(
            mod,
            inputs=[
                _trt_input(
                    (1, 3, crop_size, crop_size),
                    min_batch, opt_batch, max_batch,
                    self._dtype,
                )
            ],
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
        )

    # ---------------------------------------------------------------------- #
    # Interface expected by SiamABCTracker                                   #
    # ---------------------------------------------------------------------- #

    def get_features(self, crop: torch.Tensor) -> torch.Tensor:
        """
        Mirrors SiamABCNet.get_features(crop).
        Dispatches to the correct TRT engine by spatial size.
        Always returns float32 so downstream code works correctly.
        """
        _b, _c, h, _w = crop.shape
        crop = crop.to(dtype=self._dtype, device=self._device)

        if h == self._template_size:
            out = self._trt_feat_template(crop)
        elif h == self._instance_size:
            out = self._trt_feat_search(crop)
        else:
            raise ValueError(
                f"get_features: unexpected crop height {h}. "
                f"Expected {self._template_size} (template) or "
                f"{self._instance_size} (search)."
            )
        return out.float().contiguous()

    def track(
        self,
        search_features: torch.Tensor,
        dynamic_search_features: torch.Tensor,
        template_features: torch.Tensor,
        dynamic_template_features: torch.Tensor,
        lam: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:

        def _cast(t: torch.Tensor) -> torch.Tensor:
            return t.to(dtype=self._dtype, device=self._device)

        sf = _cast(search_features)
        dsf = _cast(dynamic_search_features)
        tf = _cast(template_features)
        dtf = _cast(dynamic_template_features)

        # Concatenate pairs and pass through single attention engine.
        # .float() ensures FP32 input to the FP32 attention engine.
        # .contiguous() forces NCHW — TRT outputs can be channel-last which
        # silently corrupts conv-based modules downstream.
        t_combined = torch.cat([tf.float(), dtf.float()], dim=1).contiguous()
        s_combined = torch.cat([dsf.float(), sf.float()], dim=1).contiguous()

        t_mixed = self._trt_attn(t_combined).contiguous()  # (B, C, h_t, w_t)
        s_mixed = self._trt_attn(s_combined).contiguous()  # (B, C, h_s, w_s)

        bbox_pred, cls_pred = _dispatch_connect(
            self._connect_engines,
            lam_val=float(lam.item()),
            norm_lambda=self._norm_lambda,
            search_org=sf.float().contiguous(),
            search=s_mixed.float(),
            kernel=t_mixed.float(),
        )

        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred.float(),
            constants.TARGET_CLASSIFICATION_KEY: cls_pred.float(),
            constants.TRACKER_TARGET_SEARCH_SIM_SCORE: None,
            constants.TRACKER_ATTENTION_MAP: s_mixed.float(),
        }

    def modules(self):
        """Delegated to the original FP32 model for AdaptiveBatchNorm discovery."""
        return self._model.modules()

    def invalidate_template_cache(self) -> None:
        """No-op — TRT engines are stateless."""


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def get_trt_tracker(
    config,
    weights_path: str,
    lambda_tta: float = 0.1,
    fp16: bool = True,
    cuda_id: int = 0,
):
    """
    Drop-in replacement for get_tracker() that returns a TRT-compiled tracker.

    Example
    -------
        tracker = get_trt_tracker(config, "weights/model.pth",
                                  lambda_tta=0.1, fp16=True)
    """
    from hydra.utils import instantiate
    from pytorch_toolbelt.utils import transfer_weights

    from ..SiamABC_Tracker import SiamABCTracker  # noqa: F401

    model: nn.Module = instantiate(
        config["model"], inference_mode=True, norm_lambda=lambda_tta
    )
    checkpoint = torch.load(weights_path, map_location=f"cuda:{cuda_id}")
    state_dict = {
        k.lstrip("module").lstrip("."): v
        for k, v in checkpoint.items()
        if k.startswith("module.")
    }
    transfer_weights(model, state_dict)
    model = model.to(f"cuda:{cuda_id}").eval()

    tracking_cfg = config["tracker"]
    template_size = int(tracking_cfg["template_size"])
    instance_size = int(tracking_cfg["instance_size"])

    trt_model = TRTSiamABCNet(
        model=model,
        template_size=template_size,
        instance_size=instance_size,
        norm_lambda=lambda_tta,
        cuda_id=cuda_id,
        fp16=fp16,
        # adjust_channels defaults to 256 — override here if your config differs
    )

    tracker: SiamABCTracker = instantiate(config["tracker"], model=trt_model)
    return tracker
