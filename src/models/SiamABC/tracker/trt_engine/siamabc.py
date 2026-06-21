"""
TRTSiamABCNet
=============
Drop-in TensorRT replacement for SiamABCNet for use inside SiamABCTracker.

Architecture
------------
get_features      → Selectable TensorRT backbone:
                    dynamic_fp16/dynamic_fp32 use one dynamic encoder+neck
                    engine; static_fp16 uses exact-shape encoder+neck engines;
                    static_fp16_fp32_neck uses exact-shape FP16 encoder engines
                    followed by the original eager FP32 neck.

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
  2. Attention warm-up channels are probed from the real model output.
  3. Dead two-engine attention code removed.
  4. _AttentionNeck deep-copies weights.
  5. _warm_connect_model dummy uses C_s not C_t.
  6. Backbone TRT engines use the TorchScript backend — no torch.export,
     symbolic guard validation, or ConstraintViolationError.

Delegated / no-op:
    modules()                   → AdaptiveBatchNorm proxy for lambda discovery
    invalidate_template_cache() → no-op (TRT engines are stateless)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

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
from .trt_utils import (
    _AttentionNeck,
    _EncoderModule,
    _FeatureExtractorModule,
    _NeckModule,
    _cast_module,
)

log = logging.getLogger(__name__)
silence_noisy_libraries()

with quiet_external_logs():
    import tensorrt
    import torch_tensorrt


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

        enabled_precisions: Set[torch.dtype] = (
            {torch.float16} if self._fp16 else {torch.float32}
        )

        with torch.no_grad():
            _t = torch.randn(
                opt_batch,
                3,
                template_size,
                template_size,
                device=self._device,
                dtype=torch.float32,
            )
            _s = torch.randn(
                opt_batch,
                3,
                instance_size,
                instance_size,
                device=self._device,
                dtype=torch.float32,
            )
            t_feat_shape: Tuple[int, ...] = tuple(model.get_features(_t).shape)
            s_feat_shape: Tuple[int, ...] = tuple(model.get_features(_s).shape)
            if self._backbone_mode == "static_fp16_fp32_neck":
                t_encoder_shape: Tuple[int, ...] = tuple(model.encoder(_t).shape)
                s_encoder_shape: Tuple[int, ...] = tuple(model.encoder(_s).shape)

        _, C, h_t, w_t = t_feat_shape
        _, _, h_s, w_s = s_feat_shape

        # One shared compile session for the whole TRT phase. begin() no-ops if a
        # caller (run_inference) already opened a larger session that also covers
        # the OSNet descriptor, so SiamABC + OSNet render as a single bar.
        compile_progress().begin(3)
        compile_progress().stage("compiling backbone engine")
        self._trt_feat: Optional[torch.nn.Module] = None
        self._trt_feat_by_hw: Dict[int, torch.nn.Module] = {}
        self._fp32_neck: Optional[torch.nn.Module] = None
        if self._backbone_mode in {"dynamic_fp16", "dynamic_fp32"}:
            self._trt_feat = self._load_or_compile_backbone_engine(
                module=_FeatureExtractorModule(model),
                cache_name="features_dynamic",
                min_hw=template_size,
                opt_hw=template_size,
                max_hw=instance_size,
                enabled_precisions=enabled_precisions,
                validation_specs=(
                    (template_size, t_feat_shape),
                    (instance_size, s_feat_shape),
                ),
            )
        elif self._backbone_mode == "static_fp16":
            for tag, hw, shape in (
                ("template", template_size, t_feat_shape),
                ("search", instance_size, s_feat_shape),
            ):
                self._trt_feat_by_hw[hw] = self._load_or_compile_backbone_engine(
                    module=_FeatureExtractorModule(model),
                    cache_name=f"features_{tag}",
                    min_hw=hw,
                    opt_hw=hw,
                    max_hw=hw,
                    enabled_precisions=enabled_precisions,
                    validation_specs=((hw, shape),),
                )
        else:
            self._fp32_neck = _NeckModule(model).eval().to(self._device).float()
            for tag, hw, shape in (
                ("template", template_size, t_encoder_shape),
                ("search", instance_size, s_encoder_shape),
            ):
                self._trt_feat_by_hw[hw] = self._load_or_compile_backbone_engine(
                    module=_EncoderModule(model),
                    cache_name=f"encoder_{tag}",
                    min_hw=hw,
                    opt_hw=hw,
                    max_hw=hw,
                    enabled_precisions=enabled_precisions,
                    validation_specs=((hw, shape),),
                )
        compile_progress().complete()
        compile_progress().stage("compiling attention neck")

        _attn_mod = _AttentionNeck(model).eval().to(self._device).float()
        with quiet_external_logs():
            self._trt_attn = torch.compile(_attn_mod, dynamic=True, fullgraph=True)
            with torch.no_grad():
                for hw in (h_t, h_s):
                    _dummy = torch.randn(
                        1,
                        C * 2,
                        hw, hw,
                        device=self._device,
                        dtype=torch.float32,
                    )
                    self._trt_attn(_dummy)   # trigger actual compilation
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

    def _load_torchscript_engine(
        self,
        path: Optional[Path],
        validation_specs: Sequence[Tuple[int, Tuple[int, ...]]],
    ) -> Optional[torch.nn.Module]:
        if path is None or self._rebuild_cache or not path.exists():
            return None
        try:
            # Cache load/save are silent so they don't interrupt the single-line
            # compile progress bar; failures below still surface as warn lines.
            engine = torch.jit.load(str(path), map_location=self._device).eval()
            self._validate_torchscript_engine(engine, validation_specs)
            return engine
        except Exception as exc:
            siamram_log(
                f"cached engine load failed ({path.name}): {exc}; rebuilding",
                phase="TRT",
                status="warn",
                indent=1,
            )
            return None

    def _save_torchscript_engine(
        self,
        engine: torch.nn.Module,
        path: Optional[Path],
    ) -> None:
        if path is None:
            return
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.jit.save(engine, str(temporary))
            os.replace(temporary, path)
        except Exception as exc:
            siamram_log(
                f"cached engine save failed ({path.name}): {exc}",
                phase="TRT",
                status="warn",
                indent=1,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_torchscript_engine(
        self,
        engine: torch.nn.Module,
        validation_specs: Sequence[Tuple[int, Tuple[int, ...]]],
    ) -> None:
        with torch.no_grad():
            for hw, expected_shape in validation_specs:
                batch = int(expected_shape[0])
                dummy = torch.randn(
                    batch,
                    3,
                    hw,
                    hw,
                    device=self._device,
                    dtype=self._dtype,
                )
                output = engine(dummy)
                if not torch.is_tensor(output):
                    raise RuntimeError(
                        f"backbone cache returned {type(output).__name__}, expected Tensor"
                    )
                if tuple(output.shape) != tuple(expected_shape):
                    raise RuntimeError(
                        f"backbone cache shape {tuple(output.shape)} != {expected_shape}"
                    )
                # TRT FP16 engines legitimately return FP32 outputs, and
                # get_features casts to float regardless, so only require a finite
                # floating-point tensor rather than an exact dtype match.
                if not output.is_floating_point():
                    raise RuntimeError(
                        f"backbone cache returned non-float dtype {output.dtype}"
                    )
                if not bool(torch.isfinite(output.float()).all().item()):
                    raise RuntimeError("backbone cache returned non-finite values")

    def _load_or_compile_backbone_engine(
        self,
        module: nn.Module,
        cache_name: str,
        min_hw: int,
        opt_hw: int,
        max_hw: int,
        enabled_precisions: Set[torch.dtype],
        validation_specs: Sequence[Tuple[int, Tuple[int, ...]]],
    ) -> torch.nn.Module:
        cache_path = self._cache_path(cache_name, "ts")
        engine = self._load_torchscript_engine(cache_path, validation_specs)
        if engine is not None:
            return engine

        with quiet_external_logs():
            engine = self._compile_backbone_engine(
                module=module,
                min_hw=min_hw,
                opt_hw=opt_hw,
                max_hw=max_hw,
                enabled_precisions=enabled_precisions,
            )
        self._validate_torchscript_engine(engine, validation_specs)
        self._save_torchscript_engine(engine, cache_path)
        return engine

    def _compile_backbone_engine(
        self,
        module: nn.Module,
        min_hw: int,
        opt_hw: int,
        max_hw: int,
        enabled_precisions: Set[torch.dtype],
    ) -> torch.nn.Module:
        """
        Trace a backbone module with TorchScript, then compile it for the
        requested static or dynamic spatial profile.

        torch.jit.trace records concrete ops on the dummy tensor without
        building a symbolic shape graph — no produce_guards, no
        ConstraintViolationError.  The resulting ScriptModule is handed to
        torch_tensorrt.compile which builds one TRT engine with a dynamic
        spatial profile.
        """
        module = module.eval().to(self._device)
        _cast_module(module, self._dtype)

        dummy = torch.randn(
            self._opt_batch,
            3,
            opt_hw,
            opt_hw,
            device=self._device,
            dtype=self._dtype,
        )

        with torch.no_grad():
            scripted = torch.jit.trace(module, dummy)

        return torch_tensorrt.compile(
            scripted,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(self._min_batch, 3, min_hw, min_hw),
                    opt_shape=(self._opt_batch, 3, opt_hw, opt_hw),
                    max_shape=(self._max_batch, 3, max_hw, max_hw),
                    dtype=self._dtype,
                )
            ],
            enabled_precisions=enabled_precisions,
            disable_tf32=self._disable_tf32,
            truncate_long_and_double=True,
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
        crop = crop.to(dtype=self._dtype, device=self._device)
        if self._trt_feat is not None:
            features = self._trt_feat(crop)
        else:
            features = self._trt_feat_by_hw[h](crop)
            if self._fp32_neck is not None:
                features = self._fp32_neck(features.float().contiguous())
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
            "torch_tensorrt": getattr(torch_tensorrt, "__version__", "unknown"),
            "tensorrt": getattr(tensorrt, "__version__", "unknown"),
        },
        gpu_identity=gpu_identity,
    )
