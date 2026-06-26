"""
Base tracker classes and state management.

This module defines the abstract base class for all trackers in the
SiamABC package, along with a TrackingState container for keeping
track of historical frames and bounding boxes.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy._typing import NDArray
from torch import Tensor

from utils.box_coder import TrackerDecodeResult
from utils.utils import limit, squared_size, to_device


class TrackingState:
    """
    Container for maintaining the internal state of a tracking session.

    Attributes:
        frame_h (int): Height of the current video frame.
        frame_w (int): Width of the current video frame.
        bbox (NDArray): Most recent predicted bounding box.
        pred_score (float): Confidence score of the last prediction.
        paths (Deque): Historical sequence of predicted bounding boxes.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialise an empty TrackingState.
        """
        super().__init__()
        self.frame_h = 0
        self.frame_w = 0
        self.bbox: Optional[NDArray] = None
        self.pred_score = None
        self.mapping: Optional[NDArray] = None
        self.prev_size: Optional[NDArray] = None
        self.mean_color = None
        self.paths: Deque[NDArray] = deque(maxlen=70)

    def save_frame_shape(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Store the dimensions of the input frame.

        Args:
            frame (np.ndarray): Input image array.
        """
        self.frame_h = frame.shape[0]
        self.frame_w = frame.shape[1]


class Tracker(ABC):
    """
    Abstract base class for Siamese trackers.

    Provides common utilities for image preprocessing, device management,
    and state handling that are shared across different tracker variants.
    """

    def __init__(
        self,
        model: nn.Module,
        cuda_id: Union[int, str] = 0,
        **tracking_config: Any,
    ) -> None:
        """
        Initialise the base tracker.

        Args:
            model (nn.Module): The neural network model used for tracking.
            cuda_id (Union[int, str]): GPU device identifier.
            **tracking_config: Hyperparameters and flags for the tracker.
        """
        super().__init__()

        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._norm_mean = _mean.cuda(cuda_id)
        self._norm_std = _std.cuda(cuda_id)

        self.cuda_id = cuda_id
        tracking_config = (
            tracking_config
            if "tracking_config" not in tracking_config.keys()
            else tracking_config["tracking_config"]
        )
        self.tracking_config = tracking_config
        self.tracking_state = TrackingState()
        self.net = model
        self.box_coder = self.get_box_coder(tracking_config, cuda_id)
        self._template_features = None
        self.window = self._get_tracking_window(
            tracking_config["windowing"], tracking_config["score_size"]
        )
        self.memory_window_size = (
            tracking_config["memory_window_size"]
            if tracking_config["memory_window_size"]
            else 50
        )
        self.running_confidence_floor_value = tracking_config[
            "running_confidence_floor_value"
        ]
        self.warmup_frames = tracking_config["warmup_frames"]
        self.warmup_window_size = tracking_config["warmup_window_size"]
        self.to_device(cuda_id)

    @staticmethod
    def _array_to_batch(
        x: np.ndarray,
    ) -> torch.Tensor:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    @abstractmethod
    def get_box_coder(
        self,
        tracking_config,
        cuda_id: str | int = 0,
    ):
        """
        Abstract method to retrieve the appropriate box coder.
        """
        pass

    def to_device(
        self,
        cuda_id,
    ):
        """
        Move the tracker and its components to the specified device.

        Args:
            cuda_id (Union[int, str]): Destination device identifier.
        """
        self.cuda_id = cuda_id
        self.window = to_device(self.window, cuda_id)
        self.box_coder = self.box_coder.to_device(self.cuda_id)

    @staticmethod
    def _get_tracking_window(
        windowing: str,
        score_size: int,
    ) -> torch.Tensor:
        if windowing == "cosine":
            return torch.from_numpy(
                np.outer(np.hanning(score_size), np.hanning(score_size))
            )
        return torch.ones(int(score_size), int(score_size))

    def _rescale_bbox(
        self,
        bbox: np.array,
        padded_box,
    ) -> np.array:

        instance_size = self.tracking_config["instance_size"]
        w_scale = padded_box[2] / instance_size
        h_scale = padded_box[3] / instance_size

        if not all(math.isfinite(x) for x in [w_scale, h_scale] + list(bbox)):
            return [0, 0, 3, 3]

        new_w = max(3, round(bbox[2] * w_scale))
        new_h = max(3, round(bbox[3] * h_scale))

        new_x = round(bbox[0] * w_scale + padded_box[0])
        new_y = round(bbox[1] * h_scale + padded_box[1])

        return [int(new_x), int(new_y), int(new_w), int(new_h)]

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        x = torch.from_numpy(image[:, :, :3]).permute(2, 0, 1).unsqueeze(0).float()
        x = x.to(self.cuda_id, non_blocking=True).div_(255.0)
        return x.sub_(self._norm_mean).div_(self._norm_std)

    def reset(
        self,
    ) -> None:
        """
        Clear cached features and reset the tracker to its initial state.
        """
        self._template_features = None

    def initialize(
        self,
        image: NDArray,
        rect: NDArray,
        **kwargs,
    ) -> None:
        pass

    def update(
        self,
        search: NDArray,
        *kw,
    ):
        return {"bbox": self.tracking_state.bbox}

    def _smooth_size(
        self,
        size: NDArray,
        prev_size: NDArray,
        lr: float,
    ) -> Tuple[float, float]:
        size = size * lr
        prev_size = prev_size * (1 - lr)
        w = prev_size[0] + lr * (size[0] + prev_size[0])
        h = prev_size[1] + lr * (size[1] + prev_size[1])
        return w, h

    def _get_point_offset(
        self,
        pred_bbox: NDArray,
    ) -> Tuple[float, float]:
        pred_xs = pred_bbox[0] + (pred_bbox[2] / 2)
        pred_ys = pred_bbox[1] + (pred_bbox[3] / 2)

        diff_xs = pred_xs - self.tracking_config["instance_size"] // 2
        diff_ys = pred_ys - self.tracking_config["instance_size"] // 2
        return diff_xs, diff_ys

    def _postprocess_bbox(
        self,
        decoded_info: TrackerDecodeResult,
        cls_score: NDArray,
        penalty: Any = None,
    ) -> NDArray:
        pred_bbox = np.squeeze(decoded_info.bbox.cpu().numpy())
        if not self.tracking_config.get("smooth", False):
            return pred_bbox

        prev_size = self.tracking_state.prev_size
        assert prev_size is not None
        r_max, c_max = decoded_info.pred_coords[0]
        lr = (
            penalty[r_max, c_max] * cls_score[r_max, c_max] * self.tracking_config["lr"]
        ).item()

        pred_size = np.array(pred_bbox[2:])
        pred_w, pred_h = self._smooth_size(pred_size, prev_size=prev_size, lr=lr)
        predicted_bbox = np.array([pred_bbox[0], pred_bbox[1], pred_w, pred_h])
        return predicted_bbox

    def _confidence_postprocess(
        self,
        cls_score: NDArray,
        regression_map: torch.Tensor,
    ) -> tuple[NDArray, None, None] | tuple[Any, NDArray, Tensor]:
        if not self.tracking_config.get("smooth", False):
            return cls_score, None, None
        prev_size = self.tracking_state.prev_size
        assert prev_size is not None

        pred_location_ = torch.stack(
            [
                self.box_coder.grid_x - regression_map[:, 0],
                self.box_coder.grid_y - regression_map[:, 1],
                self.box_coder.grid_x + regression_map[:, 2],
                self.box_coder.grid_y + regression_map[:, 3],
            ],
            dim=1,
        )

        pred_location = pred_location_[0]

        s_c = limit(
            squared_size(
                pred_location[2] - pred_location[0], pred_location[3] - pred_location[1]
            )
            / (squared_size(prev_size[0], prev_size[1]))
        )

        r_c = limit(
            (prev_size[0] / prev_size[1])
            / (
                (pred_location[2] - pred_location[0])
                / (pred_location[3] - pred_location[1])
            )
        )

        penalty = torch.exp(-(r_c * s_c - 1) * self.tracking_config["penalty_k"])
        pscore = penalty * cls_score

        pscore = (
            pscore * (1 - self.tracking_config["window_influence"])
            + self.window * self.tracking_config["window_influence"]
        )
        return pscore, penalty.cpu().numpy(), pred_location_
