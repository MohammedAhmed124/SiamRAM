"""
Bounding box encoding and decoding for Siamese trackers.

This module provides classes to convert between ground-truth bounding boxes
and model output maps (regression and classification), as well as utilities
for generating Gaussian labels.
"""
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
    """
    Abstract base class for bounding box encoding and decoding.

    Handles the conversion of bounding boxes to spatial maps for training
    and the decoding of network outputs back into image-coordinate boxes.
    """

    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        """
        Initialise the BoxCoder with a tracker configuration.

        Args:
            tracker_config (Dict[str, Any]): Configuration containing
                'score_size', 'total_stride', and 'instance_size'.
        """
        super().__init__()
        self.tracker_config = tracker_config
        self.grid_x, self.grid_y = make_grid(
            tracker_config["score_size"], tracker_config["total_stride"], tracker_config["instance_size"]
        )

    def to_device(self, device: Union[str, int]) -> "BoxCoder":
        """
        Move the internal coordinate grids to the specified device.

        Args:
            device (Union[str, int]): Target device (e.g., 'cuda:0').

        Returns:
            BoxCoder: The instance itself.
        """
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
    """
    Box coder implementation for the SiamABC model.

    Implements the encoding of boxes into 4-channel regression maps (l, t, r, b)
    and single-channel classification labels.
    """

    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        """
        Initialise the SiamABCBoxCoder.

        Args:
            tracker_config (Dict[str, Any]): Configuration for the coder.
        """
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
        """
        Decode model outputs into a bounding box in image coordinates.

        Args:
            regression_map (torch.Tensor): Predicted regression offsets.
            classification_map (torch.Tensor): Predicted classification scores.
            use_sigmoid (bool): Whether to apply sigmoid to classification_map.
            pred_location (Optional[torch.Tensor]): Pre-computed absolute locations.

        Returns:
            TrackerDecodeResult: Decoded box [x, y, w, h] and max-score coordinates.
        """
        if use_sigmoid:
            classification_map = classification_map.float().sigmoid()
        cls_map = classification_map[0, 0]

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
        c_max = (flat_idx % W).item()

        x1, y1, x2, y2 = loc[:, r_max, c_max]
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
    """
    Generate a 1D Gaussian distribution.

    Args:
        sz (int): Size of the output tensor.
        sigma (float): Standard deviation of the Gaussian.
        center (torch.Tensor): Center position(s).
        end_pad (int): Optional padding.
        density (bool): Whether to normalise to unit area.

    Returns:
        torch.Tensor: 1D Gaussian values.
    """
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False) -> torch.Tensor:
    """
    Generate a 2D Gaussian distribution.

    Args:
        sz (Tuple[int, int]): Size of the output map.
        sigma (Union[float, Tuple[float, float]]): Standard deviation.
        center (torch.Tensor): Center coordinates.
        end_pad (Tuple[int, int]): Optional padding.
        density (bool): Whether to normalise to unit area.

    Returns:
        torch.Tensor: 2D Gaussian values.
    """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
        gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor=0.1, kernel_sz=1, feat_sz=16, image_sz=256, end_pad_if_even=True,
                            density=False, uni_bias=0) -> torch.Tensor:
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

    target_center = (target_bb[:, 0:2] + target_bb[:, 2:4]) * 0.5
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
    encoded = box_coder.encode(bboxes)
    print()
