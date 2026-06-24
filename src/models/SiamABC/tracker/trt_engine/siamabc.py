"""
TRTSiamABCNet
=============
Drop-in TensorRT replacement for SiamABCNet for use inside SiamABCTracker.

Every stage is a raw-TensorRT engine (ONNX -> TensorRT, see raw_trt.py) — no
torch_tensorrt, so this runs wherever only the `tensorrt` Python module is
available (e.g. Jetson). Each stage uses static-shape engines, one per concrete
input shape:

  get_features  → encoder+neck engine per crop size (template, search).
  track stage 1 → polarized_self_attention + attention_neck engine per branch
                  size, dispatched by spatial size (_NeckEngineByHW).
  track stage 2 → connect_model (BoxTower + AdaptiveBatchNorm), two engines
                  (lam=0, lam=norm_lambda); see connector.py.

Precision: engine I/O is always FP32. fp16 only flips the builder's FP16 flag
for the backbone (internal half precision, FP32 in/out). Neck and connect stay
FP32 — cls_pred feeds hard score thresholds where FP16 logit noise causes
spurious occlusion.

Delegated / no-op:
    modules()                   → AdaptiveBatchNorm proxy for lambda discovery
    invalidate_template_cache() → no-op (TRT engines are stateless)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.console import (
    compile_progress,
    quiet_external_logs,
    siamram_log,
    silence_noisy_libraries,
)

from ...model import constants
from ...model.adaptive_batch_norm import AdaptiveBatchNorm
from ..tracker_setup import load_model
from .cache_utils import siamabc_cache_prefix
from .connector import _build_connect_engines, _dispatch_connect
from .raw_trt import TRTModule, build_trt_module
from .trt_utils import _AttentionNeck, _FeatureExtractorModule

log = logging.getLogger(__name__)
silence_noisy_libraries()

with quiet_external_logs():
    import tensorrt


class _NeckEngineByHW:
    """
    Dispatches the attention neck to its per-spatial-size TensorRT engine.

    The TRT path builds one static engine per branch size; this routes each call to
    the right one by the input's trailing dim. The neck is only ever called with the
    template (h_t) and search (h_s) sizes (see TRTSiamABCNet.track), and the feature
    maps are square, so the trailing dim alone selects the engine.
    """

    def __init__(self, engines: Dict[int, torch.nn.Module]) -> None:
        self._engines = engines

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hw = int(x.shape[-1])
        engine = self._engines.get(hw)
        if engine is None:
            raise RuntimeError(
                f"attention-neck engine missing for spatial size {hw} "
                f"(have {sorted(self._engines)})"
            )
        return engine(x)


BACKBONE_MODES = {
    "dynamic_fp16",
    "static_fp16",
    "static_fp16_fp32_neck",
    "dynamic_fp32",
}


def _resolve_backbone_mode(backbone_mode: str, fp16: bool) -> str:
    mode = str(backbone_mode or "").strip().lower()
    if not mode:
        mode = "dynamic_fp16" if fp16 else "dynamic_fp32"
    if mode not in BACKBONE_MODES:
        choices = ", ".join(sorted(BACKBONE_MODES))
        raise ValueError(f"Unsupported SiamABC backbone mode {mode!r}. Expected: {choices}")
    return mode


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
    fp16            : Legacy precision switch used when backbone_mode is empty.
    backbone_mode   : Explicit backbone implementation. Attention and
                      connect_model remain FP32 in every mode.
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
        backbone_mode: str = "",
        disable_tf32: bool = False,
        engine_cache_dir: str = "",
        cache_prefix: str = "siamabc",
        connector_cache_prefix: str = "",
        rebuild_cache: bool = False,
    ) -> None:

        self._norm_lambda = norm_lambda
        self._device = torch.device(f"cuda:{cuda_id}")
        self._backbone_mode = _resolve_backbone_mode(backbone_mode, fp16)
        self._fp16 = self._backbone_mode != "dynamic_fp32"
        self._dtype = torch.float16 if self._fp16 else torch.float32
        self._disable_tf32 = bool(disable_tf32)
        self._template_size = template_size
        self._instance_size = instance_size
        self._min_batch = min_batch
        self._opt_batch = opt_batch
        self._max_batch = max_batch
        self._engine_cache_dir = Path(engine_cache_dir) if engine_cache_dir else None
        self._cache_prefix = cache_prefix
        self._connector_cache_prefix = connector_cache_prefix or cache_prefix
        self._rebuild_cache = bool(rebuild_cache)

        with torch.no_grad():
            _t = torch.randn(
                opt_batch, 3, template_size, template_size,
                device=self._device, dtype=torch.float32,
            )
            _s = torch.randn(
                opt_batch, 3, instance_size, instance_size,
                device=self._device, dtype=torch.float32,
            )
            t_feat_shape: Tuple[int, ...] = tuple(model.get_features(_t).shape)
            s_feat_shape: Tuple[int, ...] = tuple(model.get_features(_s).shape)

        _, C, h_t, w_t = t_feat_shape
        _, _, h_s, w_s = s_feat_shape

        # One shared compile session for the whole TRT phase. begin() no-ops if a
        # caller (run_inference) already opened a larger session that also covers
        # the OSNet descriptor, so SiamABC + OSNet render as a single bar.
        compile_progress().begin(3)
        compile_progress().stage("compiling backbone engine")
        self._trt_feat_by_hw: Dict[int, TRTModule] = {}
        for tag, hw in (("template", template_size), ("search", instance_size)):
            self._trt_feat_by_hw[hw] = build_trt_module(
                _FeatureExtractorModule(model).eval().to(self._device).float(),
                (torch.randn(1, 3, hw, hw, device=self._device),),
                device=self._device,
                input_names=["crop"],
                output_names=["features"],
                fp16=self._fp16,
                cache_path=self._cache_path(f"features_{tag}", "engine"),
                rebuild_cache=self._rebuild_cache,
            )
        compile_progress().complete()
        compile_progress().stage("compiling attention neck")

        _attn_mod = _AttentionNeck(model).eval().to(self._device).float()
        with quiet_external_logs():
            self._trt_attn = self._build_attention_neck(_attn_mod, C, h_t, h_s)
            with torch.no_grad():
                for hw in (h_t, h_s):
                    _dummy = torch.randn(
                        1,
                        C * 2,
                        hw, hw,
                        device=self._device,
                        dtype=torch.float32,
                    )
                    self._trt_attn(_dummy)   # trigger compilation / warm the engine
        compile_progress().complete()
        compile_progress().stage("compiling connect engines")

        with quiet_external_logs():
            self._connect_engines = _build_connect_engines(
                model,
                s_feat_shape=s_feat_shape,
                t_feat_shape=t_feat_shape,
                norm_lambda=norm_lambda,
                device=self._device,
                cache_dir=str(self._engine_cache_dir) if self._engine_cache_dir else "",
                cache_prefix=self._connector_cache_prefix,
                rebuild_cache=self._rebuild_cache,
            )
        compile_progress().complete()
        with quiet_external_logs():
            self.get_features(torch.randn(1, 3, template_size, template_size, device=self._device))

    def _cache_path(
        self,
        name: str,
        suffix: str,
    ) -> Optional[Path]:
        if self._engine_cache_dir is None:
            return None
        self._engine_cache_dir.mkdir(parents=True, exist_ok=True)
        return self._engine_cache_dir / f"{self._cache_prefix}_{name}.{suffix}"

    def _build_attention_neck(
        self,
        module: nn.Module,
        C: int,
        h_t: int,
        h_s: int,
    ) -> _NeckEngineByHW:
        """
        Build one static TensorRT engine per attention-neck branch size and
        dispatch by spatial size (_NeckEngineByHW). The neck is only ever called at
        the template (h_t) and search (h_s) sizes.
        """
        engines: Dict[int, TRTModule] = {}
        for hw in sorted({int(h_t), int(h_s)}):
            engines[hw] = self._compile_attention_neck_engine(module, C, hw)
        return _NeckEngineByHW(engines)

    def _compile_attention_neck_engine(
        self,
        module: nn.Module,
        C: int,
        hw: int,
    ) -> TRTModule:
        """Compile the FP32 attention neck to one static TensorRT engine at *hw*."""
        hw = int(hw)
        return build_trt_module(
            module.eval().to(self._device).float(),
            (torch.randn(1, C * 2, hw, hw, device=self._device),),
            device=self._device,
            input_names=["pair"],
            output_names=["mixed"],
            fp16=False,
            cache_path=self._cache_path(f"neck_{hw}", "engine"),
            rebuild_cache=self._rebuild_cache,
        )

    def get_features(
        self,
        crop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mirrors SiamABCNet.get_features(crop).
        Always returns float32 so downstream code works correctly.
        """
        _b, _c, h, _w = crop.shape
        if h not in (self._template_size, self._instance_size):
            raise ValueError(
                f"get_features: unexpected crop height {h}. "
                f"Expected {self._template_size} (template) or "
                f"{self._instance_size} (search)."
            )
        crop = crop.to(dtype=torch.float32, device=self._device).contiguous()
        features = self._trt_feat_by_hw[h](crop)
        return features.float().contiguous()

    def track(
        self,
        search_features: torch.Tensor,
        dynamic_search_features: torch.Tensor,
        template_features: torch.Tensor,
        dynamic_template_features: torch.Tensor,
        lam: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:

        def _to_fp32(
            t: torch.Tensor,
        ) -> torch.Tensor:
            return t.to(dtype=torch.float32, device=self._device)

        sf = _to_fp32(search_features)
        dsf = _to_fp32(dynamic_search_features)
        tf = _to_fp32(template_features)
        dtf = _to_fp32(dynamic_template_features)

        t_combined = torch.cat([tf, dtf], dim=1).contiguous()
        s_combined = torch.cat([dsf, sf], dim=1).contiguous()

        t_mixed = self._trt_attn(t_combined).contiguous()
        s_mixed = self._trt_attn(s_combined).contiguous()

        bbox_pred, cls_pred = _dispatch_connect(
            self._connect_engines,
            lam_val=float(lam.item()),
            norm_lambda=self._norm_lambda,
            search_org=sf.contiguous(),
            search=s_mixed,
            kernel=t_mixed,
        )

        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred.float(),
            constants.TARGET_CLASSIFICATION_KEY: cls_pred.float(),
            constants.TRACKER_TARGET_SEARCH_SIM_SCORE: None,
            constants.TRACKER_ATTENTION_MAP: s_mixed.float(),
        }

    def modules(
        self,
    ):
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

    def invalidate_template_cache(
        self,
    ) -> None:
        """No-op — TRT engines are stateless."""


def get_trt_tracker(
    config,
    weights_path: str,
    lambda_tta: float = 0.1,
    fp16: bool = True,
    cuda_id: int = 0,
    backbone_mode: str = "",
    disable_tf32: bool = False,
    engine_cache_dir: str = "",
    rebuild_cache: bool = False,
):
    from hydra.utils import instantiate

    from ..SiamABC_Tracker import SiamABCTracker
    from ..tracker_setup import normalize_tracker_config_aliases

    normalize_tracker_config_aliases(config)
    siamram_log("Loading model weights", phase="TRT", status="load")
    # Building the torchvision backbone with the legacy `pretrained=` argument
    # emits harmless UserWarnings from native/3rd-party code; keep the load step
    # quiet so it doesn't interrupt our clean status lines.
    with quiet_external_logs():
        model: nn.Module = instantiate(
            config["model"], inference_mode=True, norm_lambda=lambda_tta
        )
        model = load_model(
            model,
            weights_path,
            map_location=f"cuda:{cuda_id}",
            strict=False,
            require_inference_weights=True,
        )
        model = model.to(f"cuda:{cuda_id}").eval()

    tracking_cfg = config["tracker"]
    template_size = int(tracking_cfg["template_size"])
    instance_size = int(tracking_cfg["instance_size"])
    resolved_backbone_mode = _resolve_backbone_mode(backbone_mode, fp16)
    cache_prefix = _siamabc_cache_prefix(
        config=config,
        weights_path=weights_path,
        template_size=template_size,
        instance_size=instance_size,
        lambda_tta=lambda_tta,
        fp16=fp16,
        cuda_id=cuda_id,
        backbone_mode=resolved_backbone_mode,
        disable_tf32=disable_tf32,
    )
    connector_cache_prefix = _siamabc_cache_prefix(
        config=config,
        weights_path=weights_path,
        template_size=template_size,
        instance_size=instance_size,
        lambda_tta=lambda_tta,
        fp16=False,
        cuda_id=cuda_id,
        cache_scope="connector",
    )

    siamram_log(
        "Starting TensorRT compilation (first run can take 1-3 min)",
        phase="TRT",
        status="info",
    )
    trt_model = TRTSiamABCNet(
        model=model,
        template_size=template_size,
        instance_size=instance_size,
        norm_lambda=lambda_tta,
        cuda_id=cuda_id,
        fp16=fp16,
        backbone_mode=resolved_backbone_mode,
        disable_tf32=disable_tf32,
        engine_cache_dir=engine_cache_dir,
        cache_prefix=cache_prefix,
        connector_cache_prefix=connector_cache_prefix,
        rebuild_cache=rebuild_cache,
    )

    tracker: SiamABCTracker = instantiate(
        config["tracker"],
        model=trt_model,
        cuda_id=cuda_id,
    )
    return tracker


def _siamabc_cache_prefix(
    config,
    weights_path: str,
    template_size: int,
    instance_size: int,
    lambda_tta: float,
    fp16: bool,
    cuda_id: int,
    backbone_mode: str = "",
    disable_tf32: bool = False,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    cache_scope: str = "backbone",
) -> str:
    if cache_scope not in {"backbone", "connector"}:
        raise ValueError(f"Unsupported SiamABC cache scope: {cache_scope!r}")
    resolved_mode = (
        _resolve_backbone_mode(backbone_mode, fp16)
        if cache_scope == "backbone"
        else "connector_fp32"
    )
    properties = torch.cuda.get_device_properties(cuda_id)
    gpu_identity = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory": int(properties.total_memory),
    }
    return siamabc_cache_prefix(
        model_config=config["model"],
        weights_path=weights_path,
        template_size=template_size,
        instance_size=instance_size,
        lambda_tta=lambda_tta,
        backbone_mode=resolved_mode,
        cuda_id=cuda_id,
        disable_tf32=disable_tf32 if cache_scope == "backbone" else False,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        software_versions={
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": str(getattr(torch.version, "cuda", "unknown")),
            "tensorrt": getattr(tensorrt, "__version__", "unknown"),
        },
        gpu_identity=gpu_identity,
    )
