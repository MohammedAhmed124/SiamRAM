

from typing import Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_toolbelt.utils import transfer_weights

from .SiamABC_Tracker import SiamABCTracker


def load_model(
    model: nn.Module, checkpoint_path: str, map_location: Optional[Union[int, str]] = None, strict: bool = True
) -> nn.Module:
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

    # inference_mode=True causes SiamABCNet to instantiate the BatchNorm layers inside
    # connect_model (cls_dw, reg_dw, bbox_tower, cls_tower) directly as AdaptiveBatchNorm,
    # so no post-hoc layer replacement is needed.
    model = instantiate(config["model"], inference_mode=True, norm_lambda=lambda_tta)

    model = load_model(model, weights_path, strict=False).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker
