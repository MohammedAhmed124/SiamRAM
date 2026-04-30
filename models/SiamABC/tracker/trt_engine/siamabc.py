"""
TRTSiamABCNet
=============
Drop-in TensorRT replacement for SiamABCNet for use inside SiamABCTracker.

Architecture
------------
get_features      → ONE dynamic-shape TRT engine (encoder + neck).
                    Built via torch.jit.trace + torch_tensorrt.compile
                    (TorchScript backend).  Handles both template (e.g. 128)
                    and search (e.g. 256) crop sizes in a single engine.

                    Why TorchScript and not torch.export / dynamo?
                      Every torch.export path — strict=True or strict=False —
                      runs produce_guards after tracing and raises
                      ConstraintViolationError when the backbone's stride-2
                      divisibility guard Ne(Mod(((size-1)//2), 2), 0) is not
                      satisfied for every integer in [min_hw, max_hw].
                      The TorchScript backend is purely trace-based: it records
                      ops on a dummy input and never inspects symbolic shape
                      guards.  A single dynamic-shape TRT profile then covers
                      both discrete crop sizes without any guard validation.

track             → SPLIT into two stages:
  Stage 1 (torch.compile, FP32) : polarized_self_attention + attention_neck
                   Single dynamic-shape compiled module — one kernel family =
                   numerically consistent t_mixed / s_mixed for cross-corr.
  Stage 2 (FP32) : connect_model (BoxTower + AdaptiveBatchNorm)
                   AdaptiveBatchNorm has Python-level control flow TRT cannot
                   lower.  Kept FP32 for classification score precision.

Why not FP16 for connect_model?
  cls_pred sigmoid feeds hard thresholds (0.55, 0.70, 0.80).  FP16 logit
  precision near 0 causes borderline scores (~0.56) to fall below threshold
  → spurious occlusion every frame.  Encoder/neck still run FP16 for speed.

Bug fixes
---------
  1. Engine built after shape probing (shapes from actual forward pass).
  2. adjust_channels default corrected 512 → 256.
  3. Dead two-engine attention code removed.
  4. _AttentionNeck deep-copies weights.
  5. _warm_connect_model dummy uses C_s not C_t.
  6. Single dynamic backbone TRT engine via TorchScript backend — no
     torch.export, no symbolic guard validation, no ConstraintViolationError.

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

from ...model import constants
from ...model.adaptive_batch_norm import AdaptiveBatchNorm
from .connector import _build_connect_engines, _dispatch_connect
from .trt_utils import _AttentionNeck, _FeatureExtractorModule, _cast_module

log = logging.getLogger(__name__)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
logging.getLogger("torch_tensorrt.dynamo.conversion").setLevel(logging.ERROR)


class TRTSiamABCNet:
    """
    TensorRT + torch.compile wrapper for SiamABCNet.

    Parameters
    ----------
    model           : trained SiamABCNet — .cuda().eval(), kept FP32
    template_size   : spatial H/W of template crop (e.g. 128)
    instance_size   : spatial H/W of search  crop  (e.g. 256)
    norm_lambda     : TTA blending weight
    cuda_id         : GPU index
    fp16            : FP16 for TRT backbone engine only;
                      attention + connect_model are always FP32
    adjust_channels : neck output channels — must match SiamABCNet config
                      (default corrected to 256, was wrongly 512)
    min/opt/max_batch: TRT batch-size range (1/1/1 for single-object tracking)
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
        adjust_channels: int = 256,
    ) -> None:

        # self._model = model.eval()
        self._norm_lambda = norm_lambda
        self._device = torch.device(f"cuda:{cuda_id}")
        self._fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self._template_size = template_size
        self._instance_size = instance_size

        enabled_precisions: Set[torch.dtype] = (
            {torch.float16} if fp16 else {torch.float32}
        )

        # ------------------------------------------------------------------ #
        # Probe output shapes (needed for connect engine + attention warmup)  #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            _t = torch.randn(opt_batch, 3, template_size, template_size,
                             device=self._device, dtype=torch.float32)
            _s = torch.randn(opt_batch, 3, instance_size, instance_size,
                             device=self._device, dtype=torch.float32)
            t_feat_shape: Tuple[int, ...] = tuple(model.get_features(_t).shape)
            s_feat_shape: Tuple[int, ...] = tuple(model.get_features(_s).shape)

        _, C, h_t, w_t = t_feat_shape
        _, _, h_s, w_s = s_feat_shape

        # ------------------------------------------------------------------ #
        # Single dynamic-shape TRT backbone engine (TorchScript backend)     #
        # ------------------------------------------------------------------ #
        self._trt_feat = self._compile_feat(
            model,
            min_hw=template_size,
            opt_hw=template_size,
            max_hw=instance_size,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            enabled_precisions=enabled_precisions,
        )

        # ------------------------------------------------------------------ #
        # Attention + neck — torch.compile, FP32, dynamic shape              #
        # ------------------------------------------------------------------ #
        _attn_mod = _AttentionNeck(model).eval().to(self._device).float()
        self._trt_attn = torch.compile(_attn_mod, dynamic=True, fullgraph=True)

        # Warm both sizes eagerly so Inductor compiles both specializations now
        with torch.no_grad():
            for hw in (h_t, h_s):
                _dummy = torch.randn(1, adjust_channels * 2, hw, hw,
                                     device=self._device, dtype=torch.float32)
                self._trt_attn(_dummy)

        # ------------------------------------------------------------------ #
        # BoxTower connect engines (FP32)                                     #
        # ------------------------------------------------------------------ #
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
        min_hw: int,
        opt_hw: int,
        max_hw: int,
        min_batch: int,
        opt_batch: int,
        max_batch: int,
        enabled_precisions: Set[torch.dtype],
    ) -> torch.nn.Module:
        """
        Trace the backbone+neck with TorchScript, then compile a single
        dynamic-shape TRT engine covering [min_hw, max_hw].

        torch.jit.trace records concrete ops on the dummy tensor without
        building a symbolic shape graph — no produce_guards, no
        ConstraintViolationError.  The resulting ScriptModule is handed to
        torch_tensorrt.compile which builds one TRT engine with a dynamic
        spatial profile.
        """
        mod = _FeatureExtractorModule(model).eval().to(self._device)
        _cast_module(mod, self._dtype)

        dummy = torch.randn(opt_batch, 3, opt_hw, opt_hw,
                            device=self._device, dtype=self._dtype)

        with torch.no_grad():
            scripted = torch.jit.trace(mod, dummy)

        return torch_tensorrt.compile(
            scripted,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(min_batch, 3, min_hw, min_hw),
                    opt_shape=(opt_batch, 3, opt_hw, opt_hw),
                    max_shape=(max_batch, 3, max_hw, max_hw),
                    dtype=self._dtype,
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
        Both template and search crops routed through the single TRT engine.
        Always returns float32 so downstream code works correctly.
        """
        _b, _c, h, _w = crop.shape
        if h not in (self._template_size, self._instance_size):
            raise ValueError(
                f"get_features: unexpected crop height {h}. "
                f"Expected {self._template_size} (template) or "
                f"{self._instance_size} (search)."
            )
        crop = crop.to(dtype=self._dtype, device=self._device)
        return self._trt_feat(crop).float().contiguous()

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

        sf  = _cast(search_features)
        dsf = _cast(dynamic_search_features)
        tf  = _cast(template_features)
        dtf = _cast(dynamic_template_features)

        # .float()      → FP32 for attention engine
        # .contiguous() → enforce NCHW; TRT/Inductor can emit channel-last
        #                  tensors which silently corrupt downstream conv ops
        t_combined = torch.cat([tf.float(),  dtf.float()], dim=1).contiguous()
        s_combined = torch.cat([dsf.float(), sf.float()],  dim=1).contiguous()

        t_mixed = self._trt_attn(t_combined).contiguous()
        s_mixed = self._trt_attn(s_combined).contiguous()

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
            constants.TARGET_CLASSIFICATION_KEY:   cls_pred.float(),
            constants.TRACKER_TARGET_SEARCH_SIM_SCORE: None,
            constants.TRACKER_ATTENTION_MAP:        s_mixed.float(),
        }

    # def modules(self):
    #     """Delegated to original FP32 model for AdaptiveBatchNorm discovery."""
    #     return self._model.modules()



    def modules(self):
        """
        Minimal shim so SiamABCTracker.__init__ can discover _norm_lambda via
            next((m._norm_lambda for m in self.net.modules()
                if isinstance(m, AdaptiveBatchNorm)), 0.1)
        No FP32 model is retained — only a bare AdaptiveBatchNorm shell whose
        sole attribute is _norm_lambda.
        """
        proxy = AdaptiveBatchNorm.__new__(AdaptiveBatchNorm)
        proxy._norm_lambda = self._norm_lambda
        yield proxy

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
    )

    tracker: SiamABCTracker = instantiate(config["tracker"], model=trt_model)
    return tracker