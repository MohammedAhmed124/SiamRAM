"""
Utility functions for setting up and initialising SiamABC trackers.

This module provides helper functions to load model checkpoints and
instantiate tracker objects with appropriate configurations.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate

from .SiamABC_Tracker import SiamABCTracker


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    map_location: Optional[Union[int, str]] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load weights from a checkpoint into a model.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the .pth checkpoint file.
        map_location (Optional[Union[int, str]]): Device to map weights to.
        strict (bool): Whether to enforce strict matching of state_dict keys.

    Returns:
        nn.Module: The model with loaded weights.
    """
    map_location = f"cuda:{map_location}" if type(map_location) is int else map_location
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
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
        model.load_state_dict(compatible_state, strict=False)

        total = max(1, len(model_state))
        loaded = len(compatible_state)
        pct = 100.0 * loaded / total
        print(
            f"[load_model] Loaded {loaded}/{total} tensors ({pct:.1f}%) from {checkpoint_path}"
        )
        if shape_mismatch:
            print(
                f"[load_model][warn] shape mismatches: {len(shape_mismatch)} "
                f"(examples: {shape_mismatch[:3]})"
            )
        if unexpected:
            print(
                f"[load_model][warn] unexpected keys in checkpoint: {len(unexpected)} "
                f"(examples: {unexpected[:3]})"
            )
        if missing:
            print(
                f"[load_model][warn] missing keys in checkpoint: {len(missing)} "
                f"(examples: {missing[:3]})"
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

    model = instantiate(config["model"], inference_mode=True, norm_lambda=lambda_tta)

    model = load_model(model, weights_path, strict=False).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker
