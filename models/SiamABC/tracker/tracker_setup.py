"""
Utility functions for setting up and initialising SiamABC trackers.

This module provides helper functions to load model checkpoints and
instantiate tracker objects with appropriate configurations.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_toolbelt.utils import transfer_weights

from .SiamABC_Tracker import SiamABCTracker


def load_model(
    model: nn.Module, checkpoint_path: str, map_location: Optional[Union[int, str]] = None, strict: bool = True
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
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = {
        k.lstrip("module").lstrip("."): v for k, v in checkpoint.items() if k.startswith("module.")
    }

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        transfer_weights(model, state_dict)
    return model


def get_tracker(config, weights_path: str, lambda_tta: float = 0.1, continuous=False) -> SiamABCTracker:
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
