"""
General utility functions for SiamRAM.

This module is the shared toolkit that every other part of SiamRAM relies on.
It sits at the very bottom of the dependency stack — nothing here imports from
the rest of the project, but almost everything in the project imports from here.

Concretely, it provides six categories of helpers:

1. APPEARANCE DESCRIPTORS
   _extract_descriptor() — builds appearance embeddings ϕ(It, b) with the active
   descriptor backend for one bbox or a batch of bboxes. Used by the RAM/DRM
   memory buffers and by Phase-2 reacquisition scoring.

2. GEOMETRIC CHECKS
   _iou() — lightweight single-pair IoU for the RAM admission gate (Section 3.2).
   calc_iou() — batched tensor IoU for the training loss.

3. SIMILARITY METRIC
   _cos_sim() — cosine similarity between two descriptor vectors, used for the
   DRM promotion consensus (Eq. 8) and Phase-2 candidate scoring (Eq. 10).

4. BOUNDING BOX FORMAT CONVERTERS
   xyxy_to_xywh(), convert_xywh_to_xyxy() — convert between the two box formats
   used in different parts of the pipeline.

5. IMAGE CROPPING & PADDING
   extend_bbox(), ensure_bbox_boundaries(), clamp_bbox(), get_extended_crop(),
   handle_empty_bbox() — together these produce the correctly-sized, correctly-padded
   search-region and template crops that SiamABC ingests every frame (Section 2.1).

6. NETWORK SUPPORT UTILITIES
   make_grid() — pre-computes the pixel-coordinate lookup table used by BoxCoder.
   get_regression_weight_label() — builds training weight maps for the regression head.
   squared_size(), limit(), unravel_index(), to_device() — small but frequently
   needed arithmetic and device-management helpers.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torch import Tensor
from torch.nn import Module
from utils.console import quiet_external_logs, siamram_log

BBoxLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
DescriptorOutput = Union[Optional[np.ndarray], list[Optional[np.ndarray]]]


def _preprocess_patches_to_tensor(
    rgb_patches: list[np.ndarray],
    out_hw: Tuple[int, int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Batched GPU-side patch preprocessing shared by both descriptor backends.

    Recipe is identical to SiamABC's own ``_preprocess_image``
    (RGB→CHW → /255 → ImageNet ``(x-mean)/std``), so every appearance embedding
    in the system is normalized the same way. Doing it in one batched pass —
    single ``np.stack``, one uint8 host→device transfer, then float-cast and
    normalize on the GPU — avoids the per-image PIL/torchvision loop that the
    torchreid ``FeatureExtractor`` would otherwise run on the CPU.

    Only the *recipe* is shared, never the crop: each backend passes its own
    ``out_hw`` (OSNet 256×128, siamese 128×128) and the patches are different
    pixels, so sharing the resized tensor across backends would corrupt one of
    them. ``mean``/``std`` are float32 ``(1,3,1,1)`` tensors on ``device``; the
    arithmetic is done in float32 and the result cast to ``dtype`` last for
    precision.
    """
    out_h, out_w = out_hw
    resized = [
        cv2.resize(p, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for p in rgb_patches
    ]
    arr = np.stack(resized, axis=0)  # (N, H, W, 3) uint8
    t = torch.from_numpy(arr).to(device)
    t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    t = t.sub_(mean).div_(std)
    return t.to(dtype)


class _OSNetDescriptorExtractor:
    """
    Lazy OSNet descriptor extractor.

    Preprocessing mirrors the SiamABC pipeline: batched ``cv2`` resize → one
    uint8 host→device transfer → on-GPU ``/255`` + ImageNet normalize (no
    per-image PIL/torchvision loop). The OSNet backbone can optionally run in
    FP16 and/or as a TensorRT engine compiled the same way as the SiamABC
    backbone (TorchScript trace + ``torch_tensorrt.compile``).

    The torchreid ``FeatureExtractor`` is used only to build/load the weighted
    model; its ``__call__`` preprocessing is bypassed entirely — we drive
    ``extractor.model`` directly with our own preprocessed tensor.
    """

    _IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # torchreid/OSNet convention is (H, W) = (256, 128). Kept fixed so the
    # ImageNet-pretrained weights see their training resolution.
    _INPUT_H: int = 256
    _INPUT_W: int = 128

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        fp16: bool = False,
        trt_compile: bool = False,
        max_batch: int = 8,
    ) -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            raise RuntimeError(
                "OSNet descriptor backend requires torchreid. "
                "Install it in the active environment first."
            ) from exc

        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        with quiet_external_logs():
            extractor = FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                image_size=(self._INPUT_H, self._INPUT_W),
                device=str(self._device),
                verbose=False,
            )
        model = extractor.model.eval()

        self._fp16 = bool(fp16) and self._device.type == "cuda"
        self._dtype = torch.float16 if self._fp16 else torch.float32
        if self._fp16:
            model = model.half()
        self._model = model

        self._mean = torch.as_tensor(
            self._IMAGENET_MEAN, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._std = torch.as_tensor(
            self._IMAGENET_STD, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._max_batch = max(1, int(max_batch))

        self._engine = None
        if trt_compile:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "trt_compile_osnet requires a CUDA device; got "
                    f"'{self._device}'."
                )
            # Lazy import: keeps utils free of any torch_tensorrt / models.*
            # dependency unless TRT is actually requested.
            from models.SiamABC.tracker.trt_engine.osnet import build_osnet_trt

            self._engine = build_osnet_trt(
                model=self._model,
                dtype=self._dtype,
                device=self._device,
                input_h=self._INPUT_H,
                input_w=self._INPUT_W,
            )

        self._forward = self._engine if self._engine is not None else self._model
        # The TRT engine is a static batch-1 graph (see build_osnet_trt), so feed
        # it one crop at a time. The eager backbone handles any batch, but we
        # still bound it by the candidate cap to keep memory predictable.
        self._chunk = 1 if self._engine is not None else self._max_batch
        # Dedicated non-default CUDA stream for the async (Tier-3) overlap path.
        # OSNet(frame N) runs here while SiamABC(frame N+1) runs on its own
        # non-default stream — both must be non-default or the legacy default
        # stream would serialize them.
        self._stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

    def _run_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the backbone over a preprocessed tensor, chunked to the engine's
        batch limit. Returns the raw (un-normalized) feature batch on-device."""
        feats: list[torch.Tensor] = []
        for i in range(0, tensor.shape[0], self._chunk):
            out = self._forward(tensor[i : i + self._chunk])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out)
            feats.append(out)
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def extract_batch(self, rgb_patches: list[np.ndarray]) -> np.ndarray:
        if not rgb_patches:
            return np.zeros((0, 0), dtype=np.float32)

        tensor = _preprocess_patches_to_tensor(
            rgb_patches,
            (self._INPUT_H, self._INPUT_W),
            self._mean,
            self._std,
            self._device,
            self._dtype,
        )
        features_t = self._run_forward(tensor).float()
        features_t = torch.nn.functional.normalize(features_t, p=2, dim=1)
        return features_t.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch_async(self, rgb_patches: list[np.ndarray]) -> dict:
        """
        Queue the OSNet forward on the side stream and return a pending handle
        WITHOUT synchronizing. The raw feature batch stays on the GPU until
        `resolve_batch` is called. Falls back to a synchronous result on CPU.
        """
        if not rgb_patches:
            return {"empty": True}
        if self._stream is None:
            return {"result": self.extract_batch(rgb_patches)}

        with torch.cuda.stream(self._stream):
            tensor = _preprocess_patches_to_tensor(
                rgb_patches,
                (self._INPUT_H, self._INPUT_W),
                self._mean,
                self._std,
                self._device,
                self._dtype,
            )
            features_t = self._run_forward(tensor)
        return {"gpu": features_t}

    @torch.no_grad()
    def resolve_batch(self, pending: dict) -> np.ndarray:
        """Block on the side stream and return the L2-normalized numpy batch."""
        if pending.get("empty"):
            return np.zeros((0, 0), dtype=np.float32)
        if "result" in pending:
            return pending["result"]
        # Wait for the queued OSNet forward to finish before reading it back.
        self._stream.synchronize()
        features_t = torch.nn.functional.normalize(
            pending["gpu"].float(), p=2, dim=1
        )
        return features_t.detach().cpu().numpy().astype(np.float32)


class _SiameseDescriptorExtractor:
    """
    Descriptor backend that reuses the tracker's already-loaded SiamABC encoder
    to produce appearance embeddings.

    Compared to OSNet this:
    - has no extra weights to download or extra GPU memory to allocate (the
      encoder is already resident for the visual tracker);
    - is "Siamese" in the same sense FEAR XS is: a small backbone trained to
      produce a discriminative template embedding;
    - emits whatever channel dim the underlying SiamABC neck/encoder projects
      to. Cosine similarity works on any dim, so consumers don't need to know.

    feature_source:
        "neck"    — use model.get_features(crop). 256-d (with default config).
                    Lower-dim, more semantic compression. Recommended default.
        "encoder" — use model.feature_extractor(crop). 1024-d (with default
                    config). More raw info but noisier and 4x heavier.

    comparison_mode:
        "xcorr"  — emit the full spatial (C, H, W) feature map, per-cell
                   L2-normalized. Comparisons use 2D cross-correlation
                   (`_cos_sim` dispatches automatically on shape). This is
                   how SiamABC was trained — the right Siamese contract.
        "pooled" — global average pool to a (C,) vector then L2-normalize.
                   Comparisons fall back to vector cosine. Cheaper but
                   discards the spatial info SiamABC actually uses.
        "similarity" — flatten the whole (C, H, W) map to one vector and
                   L2-normalize, then compare by plain cosine. Keeps spatial
                   detail (unlike "pooled") but, unlike "xcorr", has no
                   translation tolerance — every cell must line up.

    Both feature_source options were trained for Siamese cross-correlation,
    not pooled-vector ReID, so the "pooled" mode handicaps them. Default is
    "xcorr".
    """

    _IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    _IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    _CROP_SIZE: int = 128
    _ALLOWED_SOURCES: Tuple[str, ...] = ("neck", "encoder")
    _ALLOWED_COMPARISONS: Tuple[str, ...] = ("xcorr", "pooled", "similarity")

    def __init__(
        self,
        siam_tracker: Any,
        feature_source: str = "neck",
        comparison_mode: str = "xcorr",
    ) -> None:
        if siam_tracker is None:
            raise RuntimeError(
                "Siamese descriptor backend requires a siam_tracker reference. "
                "Ensure SiamRAMExperimentTracker is passing siam_tracker= to "
                "configure_descriptor_backend()."
            )
        model = getattr(siam_tracker, "net", None)
        if model is None or not hasattr(model, "feature_extractor"):
            raise RuntimeError(
                "Siamese descriptor backend: siam_tracker.net is missing or "
                "does not expose a feature_extractor(crop) method."
            )
        source = feature_source.strip().lower() if feature_source else "neck"
        if source not in self._ALLOWED_SOURCES:
            raise ValueError(
                f"Unsupported siamese_feature_source='{feature_source}'. "
                f"Expected one of: {self._ALLOWED_SOURCES}."
            )
        if source == "neck" and not hasattr(model, "get_features"):
            raise RuntimeError(
                "siamese_feature_source='neck' requires siam_tracker.net to "
                "expose get_features(crop); falling back to 'encoder' is "
                "available via config."
            )
        mode = comparison_mode.strip().lower() if comparison_mode else "xcorr"
        if mode not in self._ALLOWED_COMPARISONS:
            raise ValueError(
                f"Unsupported siamese_comparison_mode='{comparison_mode}'. "
                f"Expected one of: {self._ALLOWED_COMPARISONS}."
            )

        self._model = model
        self._feature_source = source
        self._comparison_mode = mode
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")
        self._mean = torch.as_tensor(
            self._IMAGENET_MEAN, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._std = torch.as_tensor(
            self._IMAGENET_STD, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_batch(self, rgb_patches: list[np.ndarray]) -> np.ndarray:
        if not rgb_patches:
            return np.zeros((0, 0), dtype=np.float32)

        tensor = _preprocess_patches_to_tensor(
            rgb_patches,
            (self._CROP_SIZE, self._CROP_SIZE),
            self._mean,
            self._std,
            self._device,
            torch.float32,
        )

        was_training = self._model.training
        self._model.eval()
        try:
            if self._feature_source == "neck":
                features = self._model.get_features(tensor)
            else:
                features = self._model.feature_extractor(tensor)
        finally:
            if was_training:
                self._model.train()

        if self._comparison_mode == "xcorr":
            # Per-spatial-cell L2 normalization. Each (C,) channel vector at
            # every spatial location ends up unit-length, so downstream
            # cross-correlation produces a response map whose values are
            # bounded in [-1, 1] per cell. The caller / _cos_sim dispatches
            # on the 3D shape to do the conv2d-based comparison.
            normalized = torch.nn.functional.normalize(features, p=2, dim=1)
            return normalized.detach().cpu().numpy().astype(np.float32)

        if self._comparison_mode == "similarity":
            # Whole-map cosine: flatten the full (C, H, W) feature map into one
            # long vector and L2-normalize it, so the standard 1D cosine path
            # compares the entire spatial layout at once. Keeps spatial detail
            # (unlike "pooled", which averages it away) but has NO translation
            # tolerance (unlike "xcorr", which slides and takes the peak).
            flat = features.flatten(start_dim=1)
            normalized = torch.nn.functional.normalize(flat, p=2, dim=1)
            return normalized.detach().cpu().numpy().astype(np.float32)

        # comparison_mode == "pooled": global avg pool then L2-normalize the
        # resulting vector so the standard cosine path applies.
        pooled = features.mean(dim=(2, 3))
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().numpy().astype(np.float32)


_OSNET_EXTRACTOR: Optional[_OSNetDescriptorExtractor] = None
_SIAMESE_EXTRACTOR: Optional[_SiameseDescriptorExtractor] = None
_SIAM_TRACKER_REF: Any = None
_SIAMESE_FEATURE_SOURCE: str = "neck"
_SIAMESE_COMPARISON_MODE: str = "xcorr"
_DESCRIPTOR_BACKEND: str = "osnet"
_OSNET_MODEL_NAME: str = "osnet_x1_0"
_OSNET_MODEL_PATH: str = ""
_OSNET_PRETRAINED_CHECKPOINT: str = "imagenet"
_OSNET_DEVICE: str = "auto"
_OSNET_FP16: bool = False
_TRT_COMPILE_OSNET: bool = False
_OSNET_MAX_CANDIDATE_BATCH: int = 0
_SUPPORTED_DESCRIPTOR_BACKENDS: tuple[str, ...] = (
    "osnet",
    "siamese",
    "pixel_descriptors",
)
_DESCRIPTOR_BACKEND_LABELS: dict[str, str] = {
    "osnet": "osnet",
    "siamese": "siamese",
    "pixel_descriptors": "pixel descriptors",
}
_OSNET_PRETRAINED_DEFAULTS: set[str] = {"", "default", "imagenet", "torchreid_imagenet"}
_OSNET_REID_PRESET_TO_DRIVE: dict[str, dict[str, str]] = {
    "reid_market1501": {
        "model_name": "osnet_x1_0",
        "filename": "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        "file_id": "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    },
    "reid_dukemtmcreid": {
        "model_name": "osnet_x1_0",
        "filename": "osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        "file_id": "1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    },
    "reid_msmt17": {
        "model_name": "osnet_x1_0",
        "filename": "osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        "file_id": "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    },
    "reid_msmt17_combineall": {
        "model_name": "osnet_x1_0",
        "filename": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "file_id": "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
    },
}
_OSNET_REID_MIN_BYTES = 1_000_000


def _canonical_descriptor_backend(name: str) -> str:
    """Normalize user-facing descriptor backend names to internal keys."""
    raw = str(name or "osnet").strip().lower()
    key = "_".join(raw.replace("-", " ").split())
    aliases = {
        "osnet": "osnet",
        "siamese": "siamese",
        "pixel": "pixel_descriptors",
        "pixels": "pixel_descriptors",
        "pixel_descriptor": "pixel_descriptors",
        "pixel_descriptors": "pixel_descriptors",
    }
    if key in aliases:
        return aliases[key]

    options = ", ".join(_DESCRIPTOR_BACKEND_LABELS[k] for k in _SUPPORTED_DESCRIPTOR_BACKENDS)
    raise ValueError(
        f"Unsupported descriptor_backend='{name}'. Supported values: {options}."
    )


def _ensure_osnet_reid_checkpoint(preset_key: str, preset_cfg: dict[str, str]) -> str:
    root_dir = Path(__file__).resolve().parents[1]
    reid_dir = root_dir / "checkpoints" / "reid"
    reid_dir.mkdir(parents=True, exist_ok=True)

    out_file = reid_dir / preset_cfg["filename"]
    if out_file.exists() and out_file.stat().st_size >= _OSNET_REID_MIN_BYTES:
        return str(out_file)

    try:
        import gdown
    except Exception as exc:
        raise RuntimeError(
            "gdown is required to fetch OSNet ReID presets. "
            "Install dependencies or set osnet_model_path manually."
        ) from exc

    if out_file.exists():
        out_file.unlink()

    url = f"https://drive.google.com/uc?id={preset_cfg['file_id']}"
    siamram_log(
        f"downloading preset '{preset_key}' → {out_file}",
        phase="DESC",
        status="build",
    )
    output_path = gdown.download(url=url, output=str(out_file), quiet=False)
    if not output_path or not out_file.exists() or out_file.stat().st_size < _OSNET_REID_MIN_BYTES:
        raise RuntimeError(
            f"Failed to download OSNet ReID preset '{preset_key}' from {url}. "
            "Try again later or set osnet_model_path to a local checkpoint."
        )

    return str(out_file)


def _resolve_osnet_model_path(
    osnet_model_name: str,
    osnet_model_path: str,
    osnet_pretrained_checkpoint: str,
) -> str:
    explicit_path = osnet_model_path.strip()
    if explicit_path:
        explicit_file = Path(explicit_path).expanduser()
        if not explicit_file.is_file():
            raise FileNotFoundError(
                f"osnet_model_path='{osnet_model_path}' does not exist or is not a file."
            )
        return str(explicit_file.resolve())

    preset_key = osnet_pretrained_checkpoint.strip().lower()
    if preset_key in _OSNET_PRETRAINED_DEFAULTS:
        return ""

    if preset_key in {"custom", "local"}:
        raise ValueError(
            "osnet_pretrained_checkpoint is set to 'custom' but osnet_model_path is empty. "
            "Set osnet_model_path to a local ReID checkpoint file."
        )

    if preset_key not in _OSNET_REID_PRESET_TO_DRIVE:
        options = sorted(_OSNET_REID_PRESET_TO_DRIVE.keys())
        options.extend(["custom", "imagenet"])
        raise ValueError(
            f"Unsupported osnet_pretrained_checkpoint='{osnet_pretrained_checkpoint}'. "
            f"Supported values: {', '.join(options)}."
        )

    preset_cfg = _OSNET_REID_PRESET_TO_DRIVE[preset_key]
    expected_name = preset_cfg["model_name"].lower()
    current_name = osnet_model_name.strip().lower()
    if current_name != expected_name:
        raise ValueError(
            f"osnet_pretrained_checkpoint='{preset_key}' requires "
            f"osnet_model_name='{preset_cfg['model_name']}', but got '{osnet_model_name}'."
        )

    return _ensure_osnet_reid_checkpoint(preset_key, preset_cfg)


def configure_descriptor_backend(
    descriptor_backend: str = "osnet",
    osnet_model_name: str = "osnet_x1_0",
    osnet_model_path: str = "",
    osnet_pretrained_checkpoint: str = "imagenet",
    osnet_device: str = "auto",
    siam_tracker: Any = None,
    siamese_feature_source: str = "neck",
    siamese_comparison_mode: str = "xcorr",
    osnet_fp16: bool = False,
    trt_compile_osnet: bool = False,
    osnet_max_candidate_batch: int = 0,
) -> None:
    """
    Configure global descriptor extraction backend options.

    Resets the lazy extractor instances so the next descriptor request uses the
    newly configured settings. The `siam_tracker` and `siamese_feature_source`
    arguments are only consulted by the `siamese` backend; they let the
    descriptor reuse the visual tracker's encoder without re-loading weights.

    Supported `descriptor_backend` values:
        "osnet"   — torchreid OSNet (separate ReID network, OSNet-x1_0 = 512-d).
        "siamese" — reuse siam_tracker.net's encoder + global avg pool. No
                    extra weights. Dim depends on `siamese_feature_source`.
        "pixel descriptors" — legacy 384-d grayscale + HSV histogram descriptor
                    from commit 3f01ab9. No neural net, CPU-only, very fast.

    Supported `siamese_feature_source` values (ignored when backend != siamese):
        "neck"    — model.get_features(crop), neck-projected (~256-d). Default.
        "encoder" — model.feature_extractor(crop), raw encoder (~1024-d).

    Supported `siamese_comparison_mode` values (ignored when backend != siamese):
        "xcorr"  — keep the (C, H, W) feature map and compare via 2D
                   cross-correlation peak. The Siamese-native comparison.
                   Default.
        "pooled" — global-avg-pool to (C,) and compare via cosine. Cheaper but
                   throws away the spatial structure SiamABC was trained on.
        "similarity" — flatten the whole (C, H, W) map to one vector and compare
                   via cosine. Keeps spatial detail (unlike "pooled") but has no
                   translation tolerance (unlike "xcorr").
    """
    global _OSNET_EXTRACTOR, _SIAMESE_EXTRACTOR, _SIAM_TRACKER_REF
    global _SIAMESE_FEATURE_SOURCE, _SIAMESE_COMPARISON_MODE
    global _DESCRIPTOR_BACKEND
    global _OSNET_MODEL_NAME, _OSNET_MODEL_PATH, _OSNET_PRETRAINED_CHECKPOINT
    global _OSNET_DEVICE, _OSNET_FP16, _TRT_COMPILE_OSNET, _OSNET_MAX_CANDIDATE_BATCH

    _DESCRIPTOR_BACKEND = _canonical_descriptor_backend(descriptor_backend)
    _OSNET_MODEL_NAME = osnet_model_name.strip()
    _OSNET_PRETRAINED_CHECKPOINT = osnet_pretrained_checkpoint.strip().lower()
    _OSNET_MODEL_PATH = (
        _resolve_osnet_model_path(
            osnet_model_name=_OSNET_MODEL_NAME,
            osnet_model_path=osnet_model_path,
            osnet_pretrained_checkpoint=_OSNET_PRETRAINED_CHECKPOINT,
        )
        if _DESCRIPTOR_BACKEND == "osnet"
        else osnet_model_path.strip()
    )
    _OSNET_DEVICE = osnet_device.strip().lower()
    _OSNET_FP16 = bool(osnet_fp16)
    _TRT_COMPILE_OSNET = bool(trt_compile_osnet)
    _OSNET_MAX_CANDIDATE_BATCH = max(0, int(osnet_max_candidate_batch))
    _OSNET_EXTRACTOR = None
    _SIAMESE_EXTRACTOR = None
    _SIAM_TRACKER_REF = siam_tracker
    _SIAMESE_FEATURE_SOURCE = (
        siamese_feature_source.strip().lower() if siamese_feature_source else "neck"
    )
    _SIAMESE_COMPARISON_MODE = (
        siamese_comparison_mode.strip().lower() if siamese_comparison_mode else "xcorr"
    )


def _get_osnet_extractor() -> _OSNetDescriptorExtractor:
    global _OSNET_EXTRACTOR
    if _OSNET_EXTRACTOR is None:
        if _OSNET_DEVICE in ("", "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = _OSNET_DEVICE
        # TRT engine batch profile upper bound. Tie it to the candidate cap when
        # set; otherwise fall back to a generous default and let extract_batch
        # chunk anything larger.
        max_batch = _OSNET_MAX_CANDIDATE_BATCH if _OSNET_MAX_CANDIDATE_BATCH > 0 else 16
        _OSNET_EXTRACTOR = _OSNetDescriptorExtractor(
            model_name=_OSNET_MODEL_NAME,
            model_path=_OSNET_MODEL_PATH,
            device=device,
            fp16=_OSNET_FP16,
            trt_compile=_TRT_COMPILE_OSNET,
            max_batch=max_batch,
        )
    return _OSNET_EXTRACTOR


def _get_siamese_extractor() -> _SiameseDescriptorExtractor:
    global _SIAMESE_EXTRACTOR
    if _SIAMESE_EXTRACTOR is None:
        _SIAMESE_EXTRACTOR = _SiameseDescriptorExtractor(
            _SIAM_TRACKER_REF,
            feature_source=_SIAMESE_FEATURE_SOURCE,
            comparison_mode=_SIAMESE_COMPARISON_MODE,
        )
    return _SIAMESE_EXTRACTOR


def _as_bbox_batch(bbox: BBoxLike) -> tuple[np.ndarray, bool]:
    arr = np.asarray(bbox, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError("Single bbox input must have shape (4,).")
        return arr.reshape(1, 4), True
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr, False
    raise ValueError("Bbox input must have shape (4,) or (N, 4).")


def _extract_descriptor(
    frame: np.ndarray,
    bbox: BBoxLike,
    size: int = 16,
    w_gray: float = 0.4,
    w_color: float = 0.6,
    _PROC_SIZE: int = 64,
) -> DescriptorOutput:
    """
    Extract appearance descriptors for one box or a batch of boxes.

    The legacy arguments (`size`, `w_gray`, `w_color`, `_PROC_SIZE`) are used by
    the "pixel descriptors" backend and ignored by neural backends.
    """
    rgb_patches, valid_indices, n_boxes, single_input = _build_patches(frame, bbox)

    desc_batch = _compute_descriptor_batch(
        rgb_patches,
        pixel_size=size,
        pixel_w_gray=w_gray,
        pixel_w_color=w_color,
        pixel_proc_size=_PROC_SIZE,
    )
    return _scatter_descriptor_batch(
        desc_batch, valid_indices, n_boxes, single_input
    )


def _build_patches(
    frame: np.ndarray,
    bbox: BBoxLike,
) -> tuple[list[np.ndarray], list[int], int, bool]:
    """
    Crop + BGR→RGB every requested box. Shared by the synchronous and async
    descriptor entry points so cropping logic stays in one place.

    Returns (rgb_patches, valid_indices, n_boxes, single_input) where
    valid_indices maps each produced patch back to its slot in the n_boxes-long
    output, and single_input mirrors whether the caller passed one bbox.
    """
    bbox_batch, single_input = _as_bbox_batch(bbox)
    h_fr, w_fr = frame.shape[:2]

    valid_indices: list[int] = []
    rgb_patches: list[np.ndarray] = []

    for idx, bb in enumerate(bbox_batch):
        x, y, w, h = map(int, bb)
        w = max(1, w)
        h = max(1, h)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_fr, x + w)
        y2 = min(h_fr, y + h)

        if x1 >= x2 or y1 >= y2:
            continue

        patch_bgr = frame[y1:y2, x1:x2]
        if patch_bgr.size == 0:
            continue

        rgb_patches.append(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
        valid_indices.append(idx)

    return rgb_patches, valid_indices, len(bbox_batch), single_input


def _extract_pixel_descriptor_from_rgb_patch(
    patch_rgb: np.ndarray,
    *,
    size: int = 16,
    w_gray: float = 0.4,
    w_color: float = 0.6,
    proc_size: int = 64,
) -> np.ndarray:
    """
    Legacy 384-d pixel descriptor from commit 3f01ab9.

    It concatenates a normalized 16x16 grayscale texture patch with an 8x4x4 HSV
    color histogram, then L2-normalizes the whole vector.
    """
    small = cv2.resize(
        patch_rgb,
        (int(proc_size), int(proc_size)),
        interpolation=cv2.INTER_LINEAR,
    )

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    p = cv2.resize(gray, (int(size), int(size))).flatten().astype(np.float32)
    p /= np.linalg.norm(p) + 1e-8

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h_hist = (
        cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        .flatten()
        .astype(np.float32)
    )
    h_hist /= np.linalg.norm(h_hist) + 1e-8

    desc = np.concatenate([float(w_gray) * p, float(w_color) * h_hist])
    desc /= np.linalg.norm(desc) + 1e-8
    return desc.astype(np.float32, copy=False)


def _extract_pixel_descriptor_batch(
    rgb_patches: list[np.ndarray],
    *,
    size: int = 16,
    w_gray: float = 0.4,
    w_color: float = 0.6,
    proc_size: int = 64,
) -> np.ndarray:
    if not rgb_patches:
        return np.zeros((0, int(size) * int(size) + 128), dtype=np.float32)
    descs = [
        _extract_pixel_descriptor_from_rgb_patch(
            patch,
            size=size,
            w_gray=w_gray,
            w_color=w_color,
            proc_size=proc_size,
        )
        for patch in rgb_patches
    ]
    return np.stack(descs, axis=0).astype(np.float32, copy=False)


def _compute_descriptor_batch(
    rgb_patches: list[np.ndarray],
    *,
    pixel_size: int = 16,
    pixel_w_gray: float = 0.4,
    pixel_w_color: float = 0.6,
    pixel_proc_size: int = 64,
) -> np.ndarray:
    """Synchronously run the active backend over the cropped patches."""
    if not rgb_patches:
        return np.zeros((0, 0), dtype=np.float32)
    if _DESCRIPTOR_BACKEND == "pixel_descriptors":
        return _extract_pixel_descriptor_batch(
            rgb_patches,
            size=pixel_size,
            w_gray=pixel_w_gray,
            w_color=pixel_w_color,
            proc_size=pixel_proc_size,
        )
    if _DESCRIPTOR_BACKEND == "siamese":
        return _get_siamese_extractor().extract_batch(rgb_patches)
    if _DESCRIPTOR_BACKEND != "osnet":
        options = ", ".join(_DESCRIPTOR_BACKEND_LABELS[k] for k in _SUPPORTED_DESCRIPTOR_BACKENDS)
        raise RuntimeError(
            f"Unsupported descriptor_backend='{_DESCRIPTOR_BACKEND}'. "
            f"Supported values: {options}."
        )
    return _get_osnet_extractor().extract_batch(rgb_patches)


def _scatter_descriptor_batch(
    desc_batch,
    valid_indices: list[int],
    n_boxes: int,
    single_input: bool,
) -> DescriptorOutput:
    """Place each computed descriptor back into its original box slot."""
    output: list[Optional[np.ndarray]] = [None] * n_boxes
    if len(desc_batch):
        for local_i, global_i in enumerate(valid_indices):
            output[global_i] = desc_batch[local_i]
    return output[0] if single_input else output


def descriptor_async_supported() -> bool:
    """
    True when the active backend can run OSNet on a side CUDA stream, overlapping
    its forward with the next SiamABC forward. Only the OSNet backend on CUDA
    qualifies — siamese reuses the SiamABC engine and pixel descriptors are CPU
    feature extraction.
    """
    if _DESCRIPTOR_BACKEND != "osnet":
        return False
    if not torch.cuda.is_available():
        return False
    return _OSNET_DEVICE in ("", "auto") or _OSNET_DEVICE.startswith("cuda")


def _extract_descriptor_async(frame: np.ndarray, bbox: BBoxLike) -> dict:
    """
    Begin descriptor extraction without blocking. For the OSNet/CUDA backend the
    forward is queued on a side stream and a pending handle is returned; resolve
    it later with `_resolve_descriptor`. For any other backend it falls back to
    a synchronous compute stored in the handle (so callers stay uniform).
    """
    rgb_patches, valid_indices, n_boxes, single_input = _build_patches(frame, bbox)
    handle = {
        "valid_indices": valid_indices,
        "n_boxes": n_boxes,
        "single": single_input,
    }
    if _DESCRIPTOR_BACKEND == "osnet" and descriptor_async_supported():
        handle["pending"] = _get_osnet_extractor().extract_batch_async(rgb_patches)
        handle["async"] = True
    else:
        handle["batch"] = _compute_descriptor_batch(rgb_patches)
        handle["async"] = False
    return handle


def _resolve_descriptor(handle: dict) -> DescriptorOutput:
    """Finish a `_extract_descriptor_async` handle into the usual output format."""
    if handle.get("async"):
        desc_batch = _get_osnet_extractor().resolve_batch(handle["pending"])
    else:
        desc_batch = handle["batch"]
    return _scatter_descriptor_batch(
        desc_batch, handle["valid_indices"], handle["n_boxes"], handle["single"]
    )


def _iou(
    a,
    b,
) -> float:
    """
    Compute the Intersection-over-Union (IoU) overlap ratio between two bounding boxes.

    What is this and why does it exist?
    ------------------------------------
    IoU is the standard geometric measure of how much two boxes overlap. In SiamRAM it
    serves as a "sanity check" for temporal coherence: if the tracker's new box barely
    overlaps the previous one, something has likely gone wrong (a distractor swap, a
    missed frame, or the target leaving the scene). Specifically:

      - RAM admission gate (Section 3.2, Eq. 7): a box is only stored in the RAM buffer
        if IoU(b, b̂_{t-1}) ≥ τ_ram = 0.40. This prevents distractor-driven jumps from
        corrupting the memory.
      - Occlusion detection pipeline: IoU checks between consecutive boxes help confirm
        that the tracker has remained locked on the same object across frames.

    This function is intentionally kept lightweight — it operates on plain Python lists
    or NumPy arrays rather than tensors, so it can be called on the CPU in the main
    tracking loop without any GPU round-trips. For batched training use, see calc_iou().

    Formula:
        IoU = area(A ∩ B) / area(A ∪ B)
            = intersection / (area_A + area_B - intersection)

    Args:
        a: First bounding box in [x, y, w, h] format (top-left corner + size).
           Can be a list, tuple, or numpy array.
        b: Second bounding box in [x, y, w, h] format.
           Must be in the same coordinate space as `a`.

    Returns:
        float:
            Overlap ratio in [0.0, 1.0]. A value of 0.0 means the boxes do not overlap
            at all; 1.0 means they are identical. The +1e-8 denominator guard prevents
            division by zero for degenerate zero-area boxes.
    """
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / (union + 1e-8)


_XCORR_MAX_SHIFT: int = 2


def _xcorr_sim_2d(
    a: np.ndarray,
    b: np.ndarray,
    max_shift: int = _XCORR_MAX_SHIFT,
) -> float:
    """
    Spatial cross-correlation similarity between two (C, H, W) feature maps.

    Both inputs are expected to be per-spatial-cell L2-normalized (each (C,)
    channel vector at each (h, w) position is unit length). We treat one map
    as a sliding kernel against a padded copy of the other and take the peak
    response of conv2d. Dividing by H*W puts the result on roughly the same
    scale as cosine similarity (typically in [-1, 1]).

    This is the comparison the SiamABC encoder was actually trained for —
    template-search cross-correlation — so it preserves the spatial structure
    that global average pooling discards.
    """
    a_t = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
    b_t = torch.as_tensor(b, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
    _, _, h, w = a_t.shape
    pad = max(0, int(max_shift))
    search = (
        torch.nn.functional.pad(b_t, [pad, pad, pad, pad]) if pad > 0 else b_t
    )
    response = torch.nn.functional.conv2d(search, a_t)
    denom = float(h * w) + 1e-8
    return float(response.max()) / denom


def _xcorr_sim_2d_many_to_one(
    arr: np.ndarray,
    ref: np.ndarray,
    max_shift: int = _XCORR_MAX_SHIFT,
) -> np.ndarray:
    """
    Batched cross-correlation: ref shape (C, H, W), arr shape (N, C, H, W).
    Returns (N,) array of per-entry peak responses normalized by H*W.

    Implementation reuses a single conv2d call with the kernel = ref and the
    input = padded arr, so it scales linearly with N at one CUDA/CPU launch.
    """
    a_t = torch.as_tensor(arr, dtype=torch.float32)  # (N, C, H, W)
    b_t = torch.as_tensor(ref, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
    _, _, h, w = a_t.shape
    pad = max(0, int(max_shift))
    inputs = (
        torch.nn.functional.pad(a_t, [pad, pad, pad, pad]) if pad > 0 else a_t
    )
    response = torch.nn.functional.conv2d(inputs, b_t)  # (N, 1, 2p+1, 2p+1)
    peaks = response.amax(dim=(1, 2, 3))  # (N,)
    denom = float(h * w) + 1e-8
    return (peaks / denom).cpu().numpy().astype(np.float64)


def _cos_sim(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute the cosine similarity between two appearance descriptor vectors.

    What is this and why does it exist?
    ------------------------------------
    Once we have two appearance descriptors (from _extract_descriptor), we need a
    single number that answers "how similar are these two patches?" Cosine similarity is
    the right tool here because the descriptors are L2-normalised — cosine similarity on
    unit vectors is equivalent to their dot product, and it is invariant to overall
    brightness changes, making it robust to illumination shifts across frames.

    This function is the core comparison primitive used in two key places:

      1. DRM promotion consensus (Section 3.2, Eq. 8): we count how many of the last
         W=5 RAM descriptors have cosine similarity ≥ τ_sim = 0.60 with the current
         frame's descriptor. If at least m_min = 3 do, the entry is promoted to the
         stable DRM bank. This filters out frames captured during brief blur or partial
         occlusion.

      2. Phase-2 DAM appearance scoring (Section 4.3, Eq. 10): each YOLO candidate ci
         is compared against every DRM anchor ψk, and S*(ci) = max_k cos(ψk, ϕ(It, ci))
         gives the candidate's best appearance match score. Combined with the velocity
         consistency term V(ci), this score ranks candidates for reacquisition.

    Args:
        a (np.ndarray):
            First descriptor vector. Expected to be L2-normalised output from
            _extract_descriptor, but un-normalised vectors are handled safely.
        b (np.ndarray):
            Second descriptor vector. Same requirements as `a`.

    Returns:
        float:
            Cosine similarity in [-1.0, 1.0]. In practice, since all descriptor
            components are non-negative (grayscale intensities, histogram counts),
            values will typically lie in [0.0, 1.0]. A value near 1.0 means the two
            patches look nearly identical; near 0.0 means no visual similarity.

    Shape-dispatch:
        - 1D × 1D vectors  → standard cosine = dot product of L2-normalized vectors.
                              Used by OSNet, pixel descriptors, and siamese vector modes.
        - 3D × 3D maps     → 2D cross-correlation peak (`_xcorr_sim_2d`). Used by
                              the siamese "xcorr" mode where descriptors are spatial
                              (C, H, W) feature maps with per-cell L2 normalization.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return float(
            np.dot(a_arr, b_arr)
            / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8)
        )
    if a_arr.ndim == 3 and b_arr.ndim == 3:
        return _xcorr_sim_2d(a_arr, b_arr)
    raise ValueError(
        "Unsupported descriptor shapes for _cos_sim: "
        f"a.shape={a_arr.shape}, b.shape={b_arr.shape}. "
        "Expected either two 1D vectors or two 3D (C, H, W) feature maps."
    )


def xyxy_to_xywh(
    boxes,
):
    """
    Convert a bounding box from corner format [x1, y1, x2, y2] to
    top-left-plus-size format [x, y, w, h].

    What is this and why does it exist?
    ------------------------------------
    Different parts of the SiamRAM pipeline use different box conventions:

      - YOLO (Phase 1 reacquisition, Section 4.3) outputs boxes as [x1, y1, x2, y2]
        — the absolute pixel coordinates of the top-left and bottom-right corners.
      - SiamABC, the RAM buffer, and all tracker-internal logic use [x, y, w, h]
        — the top-left corner plus the width and height.

    This function is the bridge between these two worlds. It is called whenever a YOLO
    detection needs to be handed off to the Siamese tracker or stored in the DAM buffer.

    Args:
        boxes:
            A sequence of four values [x1, y1, x2, y2] representing the top-left corner
            (x1, y1) and bottom-right corner (x2, y2) of the box in pixel coordinates.
            Can be a list, tuple, or numpy array.

    Returns:
        List[float]:
            The same box expressed as [x, y, w, h], where x=x1, y=y1, w=x2-x1, h=y2-y1.
            Note that if x2 < x1 or y2 < y1 (malformed input), width or height will be
            negative — callers should validate inputs if this is a concern.
    """
    x1, y1, x2, y2 = boxes
    return [x1, y1, x2 - x1, y2 - y1]


def convert_xywh_to_xyxy(
    bbox: NDArray,
) -> NDArray:
    """
    Convert a bounding box from top-left-plus-size format [x, y, w, h] to
    corner format [x1, y1, x2, y2].

    What is this and why does it exist?
    ------------------------------------
    This is the inverse of xyxy_to_xywh(). It is needed whenever tracker-internal
    [x, y, w, h] boxes must be passed to routines that expect explicit corners — for
    example, when computing the Gaussian label function in box_coder.py (which takes
    [x1, y1, x2, y2]), or when drawing boxes with certain OpenCV/visualization APIs.

    Args:
        bbox (NDArray):
            A numpy array of four values [x, y, w, h], where (x, y) is the top-left
            corner and (w, h) are the width and height in pixels.

    Returns:
        NDArray:
            A numpy array [x1, y1, x2, y2] = [x, y, x+w, y+h].
            The dtype matches the input array.
    """
    return np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])


def to_device(
    x: Union[torch.Tensor, torch.nn.Module],
    cuda_id: int = 0,
) -> Tensor | Module:
    """
    Move a PyTorch tensor or module to a GPU if one is available, otherwise keep on CPU.

    What is this and why does it exist?
    ------------------------------------
    SiamRAM runs in real-time, and the heavy components — the ResNet backbone, the
    attention modules, and the BoxTower head (Section 2.1) — must live on GPU to meet
    latency requirements. At the same time, the code should degrade gracefully on
    CPU-only machines (e.g. during debugging or on edge devices without a GPU).

    This helper centralises the device check so that model initialisation code doesn't
    need to repeat `if torch.cuda.is_available()` everywhere. It is typically called
    once per component at startup, after the model is loaded.

    Args:
        x (Union[torch.Tensor, torch.nn.Module]):
            The tensor or module to move. Works for any PyTorch object that supports
            the .cuda() method.
        cuda_id (int):
            The index of the GPU to use when multiple GPUs are present. Default 0
            (the first GPU). Ignored if no CUDA device is available.

    Returns:
        Tensor | Module:
            The same object, now residing on the requested GPU, or unchanged on CPU
            if CUDA is not available.
    """
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def extend_bbox(
    bbox: NDArray,
    image_width: int,
    image_height: int,
    offset: float = 1.1,
) -> NDArray:
    """
    Expand a bounding box outward to include surrounding context.

    What is this and why does it exist?
    ------------------------------------
    SiamABC (Section 2.1) does not process the raw object crop alone — it needs context
    around the object so the network can distinguish the target from its immediate
    background. The template crops (static template z and dynamic template z̃) and the
    search region crop x are all generated from an extended bounding box, not from a
    tight crop.

    The expansion is multiplicative: if offset=1.1, the box grows 10% of its own size
    outward on each side. The `offset` can be a tuple for non-uniform expansion:
      - A single float: equal expansion on all four sides.
      - A 2-tuple (w_offset, h_offset): independent horizontal and vertical expansion.
      - A 4-tuple (left, right, top, bottom): full per-side control.

    Note that the returned coordinates may be negative or exceed the image boundaries.
    This is intentional — callers should pass the result through ensure_bbox_boundaries()
    or get_extended_crop() (which handles out-of-bounds regions with padding).

    Args:
        bbox (NDArray):
            The original target box in [x, y, w, h] format (pixel coordinates).
        image_width (int):
            Width of the source image in pixels. Passed for reference but not used to
            clamp in this function — clamping is handled downstream.
        image_height (int):
            Height of the source image in pixels. Same note as image_width.
        offset (float or tuple):
            Expansion factor. Default 1.1 expands each side by 10% of the box's
            corresponding dimension. See above for tuple formats.

    Returns:
        NDArray (int32):
            The expanded box [x_new, y_new, w_new, h_new]. May contain negative values
            or values exceeding the image size if the original box is near the border.
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array(
        [x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]
    ).astype("int32")


def ensure_bbox_boundaries(
    bbox: NDArray,
    img_shape: Tuple[int, int],
) -> NDArray:
    """
    Clamp a bounding box so it lies entirely within the image boundaries.

    What is this and why does it exist?
    ------------------------------------
    After extending a box with extend_bbox() or receiving a prediction from the network,
    the coordinates may fall outside the [0, W] × [0, H] image domain. Passing such
    coordinates to cv2.resize(), array slicing, or _extract_descriptor() would cause
    crashes or silently corrupt data. This function is the safety net that prevents that.

    The clamping logic preserves the bottom-right corner independently: it clips both
    the top-left and the bottom-right corners separately before recomputing width and
    height, which means the box can shrink but never expand beyond the image edge.

    This function is called internally by get_extended_crop() and clamp_bbox(), and
    should be called any time a box is computed from network outputs or arithmetic
    before being used for image access.

    Args:
        bbox (NDArray):
            The box to clamp, in [x, y, w, h] format (pixel coordinates, may be
            out-of-bounds).
        img_shape (Tuple[int, int]):
            The (Height, Width) of the image, as returned by image.shape[:2].

    Returns:
        NDArray (int32):
            The clamped box [x, y, w, h], guaranteed to satisfy:
                0 ≤ x ≤ W, 0 ≤ y ≤ H, x+w ≤ W, y+h ≤ H, w ≥ 0, h ≥ 0.
            Note that width or height may become 0 if the entire box was outside the
            image — use clamp_bbox() if a minimum area guarantee is also needed.
    """
    x1, y1, w, h = bbox
    x2_raw = x1 + w
    y2_raw = y1 + h
    x1 = min(max(0, x1), img_shape[1])
    y1 = min(max(0, y1), img_shape[0])
    x2 = min(max(0, x2_raw), img_shape[1])
    y2 = min(max(0, y2_raw), img_shape[0])
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def clamp_bbox(
    bbox: NDArray,
    shape: Tuple[int, int],
    min_side: int = 3,
) -> NDArray:
    """
    Clamp a bounding box to image boundaries AND enforce a minimum side length.

    What is this and why does it exist?
    ------------------------------------
    This is the stricter sibling of ensure_bbox_boundaries(). While that function
    prevents out-of-bounds access, it can still return a zero-area box if the target
    has drifted entirely off-screen. A zero-area box would cause division-by-zero in
    squared_size(), a crash in cv2.resize() inside get_extended_crop(), or a silent
    NaN in _extract_descriptor().

    By guaranteeing at least min_side pixels in each dimension, clamp_bbox() ensures
    that every downstream operation receives a valid, non-degenerate input — even in
    edge cases like the target temporarily exiting the frame (which can occur during
    occlusion before the Ego-Motion Block (Section 4.2) compensates).

    Args:
        bbox (NDArray):
            The box to clamp, in [x, y, w, h] format. May be out-of-bounds or have
            small dimensions.
        shape (Tuple[int, int]):
            Image dimensions as (Height, Width).
        min_side (int):
            The minimum allowed value for both width and height. Default 3 pixels.
            If the clamped width or height is below this, it is set to min_side and
            the origin is shifted inward to keep the box within the image.

    Returns:
        NDArray:
            A box [x, y, w, h] satisfying: w ≥ min_side, h ≥ min_side, and all
            coordinates within image bounds.
    """
    bbox = ensure_bbox_boundaries(bbox, img_shape=shape)
    x, y, w, h = bbox
    img_h, img_w = shape[0], shape[1]
    if w < min_side:
        w = min_side
        x -= max(0, x + w - img_w)
    if h < min_side:
        h = min_side
        y -= max(0, y + h - img_h)
    return np.array([x, y, w, h])


def get_extended_crop(
    image,
    bbox,
    crop_size,
    context,
    padding_value=None,
):
    """
    Extract a fixed-size square crop of a target region from a video frame, with
    intelligent padding for regions that extend beyond the image boundary.

    What is this and why does it exist?
    ------------------------------------
    This is the core image extraction routine for SiamABC's four input crops
    (Section 2.1, Figure 2):
      - Static template z (first-frame crop of the target).
      - Dynamic template z̃ (updated from the RAM buffer, Section 2.2).
      - Search region x (centred on the current position estimate).
      - Dynamic search x̃ (from the most recent high-confidence memory frame).

    All four crops must be exactly crop_size × crop_size pixels before being passed
    to the ResNet backbone. When the context window extends beyond the frame edge
    (common for targets near the image border, or when the EKF predicts a position
    near the edge during reacquisition), we cannot simply crop — we need to pad the
    missing area. This function does exactly that: it crops as much as the frame
    allows, then fills the remainder with `padding_value` (typically the per-channel
    mean of the image, so the padding is "invisible" to batch normalisation).

    It also returns the target's location expressed relative to the new crop — this
    "bbox_in_crop" is what gets passed to the box coder's encode() function during
    training, since the network sees the crop, not the full frame.

    Step-by-step:
    -------------
    1. Compute how much of the requested context window falls outside each edge.
    2. Slice the valid portion from the image.
    3. If any padding is needed, apply cv2.copyMakeBorder with a constant fill.
    4. Resize the padded crop to exactly crop_size × crop_size.
    5. Scale the target bbox coordinates from the original image space into the
       resized crop space.
    6. Clamp the in-crop bbox to the crop boundaries.

    Args:
        image (np.ndarray):
            The full video frame in BGR format, shape (H, W, 3).
        bbox:
            The target's current bounding box in [x, y, w, h] pixel coordinates,
            in the full-frame coordinate system.
        crop_size (int):
            The side length of the output square crop in pixels (e.g. 255 or 256).
            The output is always exactly crop_size × crop_size.
        context:
            The extended context box in [x, y, w, h] format — the region of the
            image we want to capture before resizing. Typically produced by
            extend_bbox(). May have negative coordinates or exceed image dimensions.
        padding_value:
            The BGR pixel value used to fill areas that fall outside the image.
            If None, cv2.copyMakeBorder defaults to black (0, 0, 0). Typically set
            to the image's per-channel mean for neutral padding.

    Returns:
        Tuple of three values:
          - resized_crop (np.ndarray): The crop_size × crop_size BGR image patch,
            ready to be passed to the ResNet backbone.
          - bbox_in_crop (NDArray): The target box [x, y, w, h] expressed in
            the coordinate system of the resized crop (i.e. values in [0, crop_size]).
          - context_rect: The original context box that was used (same as `context`),
            returned for the caller's reference.
    """

    pad_left = max(-context[0], 0)
    pad_top = max(-context[1], 0)
    pad_right = max(context[0] + context[2] - image.shape[1], 0)
    pad_bottom = max(context[1] + context[3] - image.shape[0], 0)

    crop = image[
        context[1] + pad_top: context[1] + context[3] - pad_bottom,
        context[0] + pad_left: context[0] + context[2] - pad_right,
    ]

    if pad_top or pad_bottom or pad_left or pad_right:
        if not crop.flags["C_CONTIGUOUS"]:
            crop = np.ascontiguousarray(crop)
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=padding_value,
        )
    elif not crop.flags["C_CONTIGUOUS"]:
        crop = np.ascontiguousarray(crop)

    resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    sx = crop_size / crop.shape[1]
    sy = crop_size / crop.shape[0]
    padded_bbox = np.array(
        [
            (bbox[0] - context[0]) * sx,
            (bbox[1] - context[1]) * sy,
            bbox[2] * sx,
            bbox[3] * sy,
        ]
    )

    padded_bbox = ensure_bbox_boundaries(
        padded_bbox.astype(np.int32), resized.shape[:2]
    )
    return resized, padded_bbox, context



def handle_empty_bbox(
    bbox: NDArray,
    min_bbox: int = 3,
) -> NDArray:
    """
    Prevents a bounding box from mathematically collapsing into a zero-area point or line.

    Why we need this exactly:
    When tracking objects over time, especially during rapid camera movements (handled by 
    SiamRAM's Ego-Motion block) or when the target moves far away, the network's predicted 
    width or height might round down to zero[cite: 1]. If a box has 0 area, downstream operations 
    like Intersection-over-Union (IoU) calculations or feature cropping will divide by zero 
    and crash the entire pipeline. This acts as a hard safety net to ensure the box always 
    maintains a minimal pixel footprint.

    Inputs:
        bbox (NDArray): The predicted bounding box in [x, y, width, height] format.
        min_bbox (int): The absolute minimum allowable size (in pixels) for width and height. Default is 3.

    Outputs:
        NDArray: The safely adjusted bounding box where width and height are guaranteed 
        to be at least `min_bbox`.
    """
    bbox[2] = max(bbox[2], min_bbox)
    bbox[3] = max(bbox[3], min_bbox)
    return bbox


def get_regression_weight_label(
    bbox,
    image_size: int = 255,
    map_size: int = 25,
    r_pos: int = 2,
    r_neg: int = 0,
) -> torch.Tensor:
    """
    Creates a spatial "importance map" to tell the SiamABC neural network where to focus 
    its learning during training.

    Why we need this exactly:
    The SiamABC Box Tower predicts regression offsets for every single cell on a grid[cite: 1]. 
    However, we don't care if the network is bad at predicting the box from a pixel in the 
    far corner of the background. We only want to train it to make highly accurate box 
    predictions from pixels that are actually *inside* or very close to the target object. 
    This function generates a weight map that assigns a value of 1.0 to the dead-center 
    of the object, 0.5 to the immediate edges, and 0 to the irrelevant background.

    Inputs:
        bbox: The ground-truth bounding box.
        image_size (int): The dimension of the raw input search crop (e.g., 255x255).
        map_size (int): The dimension of the network's output feature map (e.g., 25x25).
        r_pos (int): The radius around the center considered "highly important" (weight 1.0).
        r_neg (int): The radius marking the boundary of the "neutral zone" (weight 0.5).

    Outputs:
        torch.Tensor: A 2D grid of shape (map_size, map_size) containing the loss weights 
        for the regression head.
    """
    bbox_c_x, bbox_c_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
    sz_x, sz_y = np.floor(float(bbox_c_x / image_size * map_size)), np.floor(
        float(bbox_c_y / image_size * map_size)
    )
    x, y = np.meshgrid(np.arange(0, map_size) - sz_x, np.arange(0, map_size) - sz_y)

    dist_to_center = np.abs(x) + np.abs(y)
    label = np.where(
        dist_to_center <= r_pos,
        np.ones_like(y),
        np.where(dist_to_center < r_neg, 0.5 * np.ones_like(y), np.zeros_like(y)),
    )
    return torch.from_numpy(label)


@torch.no_grad()
def make_grid(
    score_size: int,
    total_stride: int,
    instance_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-computes a static translation map between the network's internal matrix and 
    the actual video frame pixels.

    Why we need this exactly:
    SiamABC doesn't directly spit out absolute pixel coordinates (like "x=500, y=300"). 
    Instead, its Box Tower head looks at a compressed feature map (e.g., 16x16) and 
    predicts *distances* to the box edges from each of those grid cells[cite: 1]. To turn those 
    predicted distances back into a real bounding box we can draw on the screen, we 
    first need to know exactly which real-world image pixel corresponds to each of 
    those 16x16 grid cells. Because this mapping never changes, we calculate it 
    once here to save precious CPU/GPU cycles during real-time tracking.

    Inputs:
        score_size (int): The width/height of the network's output feature map (e.g., 16).
        total_stride (int): The downsampling factor of the ResNet backbone. E.g., a stride 
                            of 16 means 1 feature map cell equals 16 image pixels.
        instance_size (int): The total resolution of the search image crop (e.g., 255).

    Outputs:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors (grid_x, grid_y) representing the 
        absolute X and Y pixel coordinates for every cell in the feature map.
    """
    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )

    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    grid_x = torch.from_numpy(grid_x[np.newaxis, :, :])
    grid_y = torch.from_numpy(grid_y[np.newaxis, :, :])
    return grid_x, grid_y


def limit(
    radius: Union[torch.Tensor, float],
) -> Union[torch.Tensor, float]:
    """
    Mathematically restrains object scale changes so the tracker doesn't predict 
    impossible physical deformations.

    Why we need this exactly:
    During standard tracking, we apply a "penalty" to predictions that suggest the target 
    suddenly grew or shrank by a massive, unnatural amount in just 1/30th of a second. 
    This function processes the ratio of the new box size to the old box size. By forcing 
    the result to always be >= 1.0 (e.g., if the ratio is 0.5, it flips to 2.0), it 
    creates a symmetric penalty curve. This prevents visual distractors from tricking the 
    tracker into collapsing the box down to a single pixel or blowing it up to cover 
    the whole screen.

    Inputs:
        radius (Union[torch.Tensor, float]): The calculated ratio between the proposed 
                                             bounding box scale and the previous frame's scale.

    Outputs:
        Union[torch.Tensor, float]: The constrained ratio, mathematically guaranteed to 
        be >= 1.0.
    """
    if isinstance(radius, torch.Tensor):
        return torch.maximum(radius, 1.0 / radius)
    return np.maximum(radius, 1.0 / radius)


def squared_size(
    w: int,
    h: int,
) -> Union[torch.Tensor, float]:
    """
    Calculates the exact dimension of the square image crop we need to feed into the 
    SiamABC network.

    Why we need this exactly:
    Real-world objects are usually rectangles (like cars or walking people), but neural 
    networks like SiamABC's ResNet backbone are heavily optimized to process perfect 
    squares. We cannot just stretch the image to make it square, or it distorts the 
    target's appearance. Instead, we use this formula to calculate a square that perfectly 
    contains the rectangular object, plus a specific, mathematically consistent amount of 
    background "context" padding. This context is critical for the network to understand 
    the object's edges.

    Inputs:
        w (int): The current estimated width of the target object.
        h (int): The current estimated height of the target object.

    Outputs:
        Union[torch.Tensor, float]: The length of one side of the required square crop.
    """
    pad = (w + h) * 0.5
    size = (w + pad) * (h + pad)
    if isinstance(size, torch.Tensor):
        return torch.sqrt(size)
    return np.sqrt(size)


def unravel_index(
    index: Any,
    shape: Tuple[int, int],
) -> Tuple[int, ...]:
    """
    Translates a 1D flat computer index back into human-readable 2D map coordinates (row, col).

    Why we need this exactly:
    When the SiamABC network outputs its classification map (the confidence scores), we use 
    PyTorch's argmax function to find the absolute highest score. However, PyTorch flattens 
    the 2D map into a 1D list and returns a single number (e.g., "The highest score is at 
    index 145"). To actually locate the target and pull its bounding box coordinates, we 
    must translate "145" back into its 2D grid position (e.g., "Row 9, Column 1").

    Inputs:
        index (Any): The flat 1D integer index returned by argmax.
        shape (Tuple[int, int]): The original dimensions (Height, Width) of the feature map.

    Outputs:
        Tuple[int, ...]: A tuple representing the exact multidimensional coordinates 
        (e.g., (row, col)).
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def calc_iou(
    reg_target: torch.Tensor,
    pred: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Calculates how well our predicted bounding boxes overlap with the true bounding boxes 
    (Intersection-over-Union) at high speed across massive batches of data.

    Why we need this exactly:
    IoU is the gold-standard metric for accuracy in object tracking. In SiamRAM, we need 
    to calculate IoU for two major reasons: 
    1. During training, to calculate the loss and teach the network to draw tighter boxes.
    2. During live tracking, specifically in the Distractor-Aware Memory (DAM) module[cite: 1]. 
       The RAM buffer has a strict admission gate: it will only store a new appearance memory 
       if the IoU between the current frame and the previous frame is >= 0.40 (Eq 7)[cite: 1]. 
       This function is highly optimized using vector math to compute these overlaps instantly.

    Inputs:
        reg_target (torch.Tensor): A batch of ground-truth bounding box coordinates.
        pred (torch.Tensor): A batch of predicted bounding box coordinates.
        smooth (float): A tiny mathematical buffer added to the calculation. It prevents 
                        divide-by-zero crashes if the network accidentally predicts an 
                        impossible box with 0 area. Default is 1.0.

    Outputs:
        torch.Tensor: A tensor containing the final IoU overlap scores (ranging from 0.0 to 1.0) 
        for every box in the batch.
    """
    target_area = (reg_target[..., 0] + reg_target[..., 2]) * (
        reg_target[..., 1] + reg_target[..., 3]
    )
    pred_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])

    w_intersect = torch.min(pred[..., 0], reg_target[..., 0]) + torch.min(
        pred[..., 2], reg_target[..., 2]
    )
    h_intersect = torch.min(pred[..., 3], reg_target[..., 3]) + torch.min(
        pred[..., 1], reg_target[..., 1]
    )

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    return (area_intersect + smooth) / (area_union + smooth)
