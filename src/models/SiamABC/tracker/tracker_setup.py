"""
Utility functions for setting up and initialising SiamABC trackers.

This module provides helper functions to load model checkpoints and
instantiate tracker objects with appropriate configurations.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate

from utils.console import siamram_log

from .SiamABC_Tracker import SiamABCTracker

INFERENCE_WEIGHT_PREFIXES = (
    "encoder.",
    "neck.",
    "polarized_self_attention.",
    "attention_neck.",
    "connect_model.",
)

TRACKER_CONFIG_ALIASES = {
    "score_window_type": "windowing",
    "score_grid_stride": "total_stride",
    "score_grid_size": "score_size",
    "template_update_enabled": "dynamic_update",
    "bbox_smoothing_enabled": "smooth",
    "template_context_padding": "template_bbox_offset",
    "search_context_scale": "search_context",
    "search_crop_size": "instance_size",
    "template_crop_size": "template_size",
    "template_update_interval": "N",
    "warmup_template_update_interval": "warm_up_n",
    "template_memory_window": "memory_window_size",
    "template_admit_conf_threshold": "dynamic_update_threshold",
    "running_confidence_floor": "running_confidence_floor_value",
    "template_admit_iou_threshold": "iou_threshold",
    "template_warmup_frames": "warmup_frames",
    "warmup_template_memory_window": "warmup_window_size",
}

HANNING_WINDOW_PENALTY_ALIASES = {
    "enabled": "hanning_window_enabled",
    "window_type": "windowing",
    "influence": "window_influence",
    "size_penalty_enabled": "size_penalty_enabled",
    "size_penalty_k": "penalty_k",
    "bbox_size_smoothing_enabled": "smooth",
    "bbox_size_smoothing_lr": "lr",
}

ARCHITECTURAL_CONFIG_ALIASES = {
    "score_grid_stride": "total_stride",
    "score_grid_size": "score_size",
    "template_context_padding": "template_bbox_offset",
    "search_context_scale": "search_context",
    "search_crop_size": "instance_size",
    "template_crop_size": "template_size",
}

DYNAMIC_TEMPLATE_ALIASES = {
    "enabled": "dynamic_update",
    "update_interval": "N",
    "warmup_update_interval": "warm_up_n",
    "memory_window": "memory_window_size",
    "admit_conf_threshold": "dynamic_update_threshold",
    "running_confidence_floor": "running_confidence_floor_value",
    "admit_iou_threshold": "iou_threshold",
    "warmup_frames": "warmup_frames",
    "warmup_memory_window": "warmup_window_size",
}


def _apply_tracker_aliases(tracker_cfg, source_cfg, aliases) -> None:
    for friendly_key, legacy_key in aliases.items():
        if friendly_key in source_cfg:
            tracker_cfg[legacy_key] = source_cfg[friendly_key]


def normalize_tracker_config_aliases(config) -> None:
    """Populate legacy SiamABC tracker keys from readable grouped config."""
    if "tracker" not in config:
        return
    tracker_cfg = config["tracker"] if isinstance(config, dict) else config.tracker
    _apply_tracker_aliases(tracker_cfg, tracker_cfg, TRACKER_CONFIG_ALIASES)

    penalty_cfg = tracker_cfg.get("hanning_window_penalty", None)
    if penalty_cfg is not None:
        _apply_tracker_aliases(
            tracker_cfg, penalty_cfg, HANNING_WINDOW_PENALTY_ALIASES
        )

    architectural_cfg = tracker_cfg.get("architectural_config", None)
    if architectural_cfg is not None:
        _apply_tracker_aliases(
            tracker_cfg, architectural_cfg, ARCHITECTURAL_CONFIG_ALIASES
        )

    dynamic_cfg = tracker_cfg.get("dynamic_template", None)
    if dynamic_cfg is not None:
        _apply_tracker_aliases(tracker_cfg, dynamic_cfg, DYNAMIC_TEMPLATE_ALIASES)


def _required_inference_prefixes(model: nn.Module) -> tuple[str, ...]:
    prefixes = list(INFERENCE_WEIGHT_PREFIXES)
    if getattr(model, "iou_head", None) is not None:
        prefixes.append("iou_head.")
    return tuple(prefixes)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    map_location: Optional[Union[int, str]] = None,
    strict: bool = True,
    require_inference_weights: bool = False,
) -> nn.Module:
    """
    Load weights from a checkpoint into a model.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the .pth checkpoint file.
        map_location (Optional[Union[int, str]]): Device to map weights to.
        strict (bool): Whether to enforce strict matching of state_dict keys.
        require_inference_weights (bool): Require every tensor used by the
            inference path, while still allowing optional training/quality heads
            to be absent.

    Returns:
        nn.Module: The model with loaded weights.
    """
    map_location = f"cuda:{map_location}" if type(map_location) is int else map_location
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {checkpoint_path!r}")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {checkpoint_path!r}")

    state_dict = {}
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        compatible_state = {}
        shape_mismatch = []
        unexpected = []

        for key, value in state_dict.items():
            if key not in model_state:
                unexpected.append(key)
                continue
            if model_state[key].shape != value.shape:
                shape_mismatch.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
                continue
            compatible_state[key] = value

        missing = [key for key in model_state.keys() if key not in compatible_state]
        required_prefixes = _required_inference_prefixes(model)
        required_missing = [
            key
            for key in missing
            if key.startswith(required_prefixes)
        ]
        if require_inference_weights and required_missing:
            raise RuntimeError(
                "Checkpoint is missing required SiamABC inference weights "
                f"({len(required_missing)} tensors; examples: {required_missing[:5]}). "
                f"Refusing to run with a partially initialized model: {checkpoint_path}"
            )

        model.load_state_dict(compatible_state, strict=False)

        total = max(1, len(model_state))
        loaded = len(compatible_state)
        pct = 100.0 * loaded / total
        siamram_log(
            f"Loaded {loaded}/{total} tensors ({pct:.1f}%) from {checkpoint_path}",
            phase="CKPT",
            status="ready",
        )
        if shape_mismatch:
            siamram_log(
                f"shape mismatches: {len(shape_mismatch)} "
                f"(examples: {shape_mismatch[:3]})",
                phase="CKPT",
                status="warn",
            )
        if unexpected:
            siamram_log(
                f"unexpected keys in checkpoint: {len(unexpected)} "
                f"(examples: {unexpected[:3]})",
                phase="CKPT",
                status="warn",
            )
        if missing:
            siamram_log(
                f"missing keys in checkpoint: {len(missing)} "
                f"(examples: {missing[:3]})",
                phase="CKPT",
                status="warn",
            )
    return model


def get_tracker(
    config,
    weights_path: str,
    lambda_tta: float = 0.1,
    continuous=False,
) -> SiamABCTracker:
    """
    Initialise a SiamABCTracker instance from a configuration.

    Args:
        config (Dict): Hydra-style configuration dictionary.
        weights_path (str): Path to the model weights.
        lambda_tta (float): Blending factor for Adaptive BatchNorm.
        continuous (bool): Whether to use continuous TTA updates.

    Returns:
        SiamABCTracker: An initialised tracker instance.
    """

    normalize_tracker_config_aliases(config)
    model = instantiate(config["model"], inference_mode=True, norm_lambda=lambda_tta)

    model = load_model(
        model,
        weights_path,
        strict=False,
        require_inference_weights=True,
    ).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker
