import math
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .utils import make_grid

TrackerEncodeResult = namedtuple("TrackerEncodeResult", ["regression_map", "classification_label"])
TrackerDecodeResult = namedtuple("TrackerDecodeResult", ["bbox", "pred_coords"])


class BoxCoder(ABC):
    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        super().__init__()
        self.tracker_config = tracker_config
        self.grid_x, self.grid_y = make_grid(
            tracker_config["score_size"], tracker_config["total_stride"], tracker_config["instance_size"]
        )

    def to_device(self, device: Union[str, int]) -> "BoxCoder":
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        return self

    @abstractmethod
    def encode(self, bboxes: np.array) -> TrackerEncodeResult:
        """

        :param bboxes: np.array - [x, y, w, h]
        :return: encoded_info: TrackerEncodeResult - regression_map: np.array(batch, 4, 16, 16),
                                                     classification_label: np.array(batch, 1, 16, 16),
        """
        pass

    @abstractmethod
    def decode(
        self,
        regression_map: torch.Tensor,
        classification_map: torch.Tensor,
        use_sigmoid: bool = True,
    ) -> TrackerDecodeResult:
        """
        :param regression_map: torch.Tensor(batch, 4, 16, 16) - Regression output from a tracking net
        :param classification_map: torch.Tensor(batch, 1, 16, 16) - Classification output from a tracking net
        :param use_sigmoid: torch.Tensor - Use sigmoid or not (for classification_labels we don`t need it)
        :return: decoded_info: TrackerDecodeResult - bbox, pred_coords
        """
        pass


class SiamABCBoxCoder(BoxCoder):
    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        super().__init__(tracker_config=tracker_config)

    @torch.no_grad()
    def encode(self, bboxes: torch.Tensor) -> TrackerEncodeResult:
        """
        :param bboxes: torch.Tensor(batch, 4) - Boxes in xywh format
        :return: encoded_info: TrackerEncodeResult - regression_map: torch.Tensor(batch, 4, 16, 16),
                                                     classification_label: torch.Tensor(batch, 1, 16, 16
                                                     )
        """
        bboxes = bboxes.unsqueeze(-1).unsqueeze(-1)
        left = self.grid_x - bboxes[:, 0]
        top = self.grid_y - bboxes[:, 1]
        right = bboxes[:, 0] + bboxes[:, 2] - self.grid_x
        bottom = bboxes[:, 1] + bboxes[:, 3] - self.grid_y
        regression_map = torch.stack((left, top, right, bottom), dim=1).float()
        regression_map_min, _ = torch.min(regression_map, dim=1)
        classification_label = (regression_map_min.unsqueeze(1) > 0).float()
        return TrackerEncodeResult(regression_map=regression_map, classification_label=classification_label)


    @torch.no_grad()
    def decode(self, regression_map, classification_map, use_sigmoid=True, pred_location=None):
        if use_sigmoid:
            classification_map = classification_map.float().sigmoid()
        cls_map = classification_map[0, 0]  # [H, W] — skip batch dim, no loop

        if pred_location is None:
            pred_location = torch.stack([
                self.grid_x - regression_map[:, 0],
                self.grid_y - regression_map[:, 1],
                self.grid_x + regression_map[:, 2],
                self.grid_y + regression_map[:, 3],
            ], dim=1)
        loc = pred_location[0] 

        flat_idx = torch.argmax(cls_map)
        H, W = cls_map.shape
        r_max = (flat_idx // W).item()
        c_max = (flat_idx  % W).item()

        x1, y1, x2, y2 = loc[:, r_max, c_max]  # shape [4] — one index op
        bbox = torch.stack([x1, y1, x2 - x1, y2 - y1])

        return TrackerDecodeResult(bbox=bbox.unsqueeze(0), pred_coords=[(r_max, c_max)])

def get_box_coder(tracker_config: Dict[str, Any], tracker_name: str = "SiamABC") -> Optional[BoxCoder]:
    """

    :param tracker_config: Dict[str, Any]
    :param tracker_name: str - name of the tracker
    :return: box_coder: BoxCoder - box coder instance
    """
    if tracker_name == "SiamABC":
        return SiamABCBoxCoder(tracker_config=tracker_config)
    return None




def gauss_1d(sz, sigma, center, end_pad=0, density=False) -> torch.Tensor:
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss



def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False) -> torch.Tensor:
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)




def gaussian_label_function(target_bb, sigma_factor=0.1, kernel_sz=1, feat_sz=16, image_sz=256, end_pad_if_even=True, density=False, uni_bias=0) -> torch.Tensor:
    """Construct Gaussian label function.
    target_bb: [b x [x1,y1,x2,y2]]
    
    """

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = (target_bb[:, 0:2] +target_bb[:, 2:4]) * 0.5 
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label

if __name__ == '__main__':
    tracker_config = {
        "score_size": 16,
        "total_stride": 16,
        "instance_size": 256
    }
    
    box_coder = SiamABCBoxCoder(tracker_config=tracker_config)
    bboxes = torch.tensor([[54, 60, 84, 90]])
    encoded  = box_coder.encode(bboxes)
    print()