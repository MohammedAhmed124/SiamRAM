

import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from typing import Optional, Union
from pytorch_toolbelt.utils import transfer_weights
from ..model.custom_bn import replace_layers_adaptive_bn
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







def get_tracker(config, weights_path: str, lambda_tta: int = 0.1 , continuous = False) -> SiamABCTracker:
    
    model = instantiate(config["model"])
        
    replace_layers_adaptive_bn(model.connect_model.cls_dw, lambda_tta, continuous)
    replace_layers_adaptive_bn(model.connect_model.reg_dw,  lambda_tta, continuous)
    replace_layers_adaptive_bn(model.connect_model.bbox_tower,  lambda_tta, continuous)
    replace_layers_adaptive_bn(model.connect_model.cls_tower,  lambda_tta, continuous)
    
    model = load_model(model, weights_path, strict=False).cuda().eval()
    tracker: SiamABCTracker = instantiate(config["tracker"], model=model)
    return tracker
