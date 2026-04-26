from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from numpy._typing import NDArray
from torch import Tensor

from utils.box_coder import TrackerDecodeResult
from utils.utils import limit, squared_size, to_device


class TrackingState:
    def __init__(self) -> None:
        super().__init__()
        self.frame_h = 0
        self.frame_w = 0
        self.bbox: Optional[NDArray] = None
        self.pred_score = None
        self.mapping: Optional[NDArray] = None
        self.prev_size = None
        self.mean_color = None

    def save_frame_shape(self, frame: np.ndarray) -> None:
        self.frame_h = frame.shape[0]
        self.frame_w = frame.shape[1]


class Tracker(ABC):
    def __init__(self, model: nn.Module, cuda_id: Union[int, str] = 0, **tracking_config: Any) -> None:
        super().__init__()

        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._norm_mean = _mean.cuda(cuda_id)
        self._norm_std = _std.cuda(cuda_id)

        self.cuda_id = cuda_id
        tracking_config = tracking_config if 'tracking_config' not in tracking_config.keys() else tracking_config[
            'tracking_config']
        print(tracking_config)
        self.tracking_config = tracking_config
        self.tracking_state = TrackingState()
        self.net = model
        self.box_coder = self.get_box_coder(tracking_config, cuda_id)
        self._template_features = None
        self._template_transform = self._get_default_transform(img_size=tracking_config["template_size"])
        self._search_transform = self._get_default_transform(img_size=tracking_config["instance_size"])
        self._dynamic_search_transform = self._get_default_transform(img_size=tracking_config["instance_size"])
        self.window = self._get_tracking_window(tracking_config["windowing"], tracking_config["score_size"])
        self.memory_window_size = tracking_config["memory_window_size"] if tracking_config["memory_window_size"] else 50
        self.running_confidence_floor_value = tracking_config["running_confidence_floor_value"]
        self.to_device(cuda_id)

    @staticmethod
    def _array_to_batch(x: np.ndarray) -> torch.Tensor:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    @abstractmethod
    def get_box_coder(self, tracking_config, cuda_id: int = 0):
        pass

    def to_device(self, cuda_id):
        self.cuda_id = cuda_id
        self.window = to_device(self.window, cuda_id)
        self.box_coder = self.box_coder.to_device(self.cuda_id)

    @staticmethod
    def _get_tracking_window(windowing: str, score_size: int) -> torch.Tensor:
        """

        :param windowing: str - window creation type
        :param score_size: int - size of classification map
        :return: window: np.array
        """
        if windowing == "cosine":
            return torch.from_numpy(np.outer(np.hanning(score_size), np.hanning(score_size)))
        return torch.ones(int(score_size), int(score_size))

    @staticmethod
    def _get_default_transform(img_size):
        pipeline = albu.Compose(
            [
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def process(a):
            r = pipeline(image=a)
            return r["image"]

        return process

    def _rescale_bbox(self, bbox: np.array, padded_box) -> np.array:
        w_scale = padded_box[2] / self.tracking_config["instance_size"]
        h_scale = padded_box[3] / self.tracking_config["instance_size"]
        bbox[0] = round(bbox[0] * w_scale + padded_box[0])
        bbox[1] = round(bbox[1] * h_scale + padded_box[1])
        bbox[2] = max(3, round(bbox[2] * w_scale))
        bbox[3] = max(3, round(bbox[3] * h_scale))
        return list(map(int, bbox))

    def _preprocess_image(self, image: np.ndarray, transform: Callable = None) -> torch.Tensor:
        # Non-blocking CPU→GPU transfer
        x = torch.from_numpy(
            np.ascontiguousarray(image[:, :, :3].transpose(2, 0, 1))
        ).unsqueeze(0).float()
        x = x.to(self.cuda_id, non_blocking=True) / 255.0
        # return (x - self._norm_mean) / self._norm_std
        return x.sub_(self._norm_mean).div_(self._norm_std)

    def reset(self) -> None:
        self._template_features = None

    def initialize(self, image: np.ndarray, rect: np.array, **kwargs) -> None:
        """
        args:
            img(np.ndarray): RGB image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        pass

    def update(self, image: np.ndarray, *kw) -> Dict[str, Any]:
        """
        args:
            img(np.ndarray): RGB image
        return:
            bbox(np.array):[x, y, width, height]
        """
        return {"bbox": self.tracking_state.bbox}

    def _smooth_size(self, size: np.array, prev_size: np.array, lr: float) -> Tuple[float, float]:
        """
        Tracking smoothing logic matches the code of Siamese Tracking
        https://www.robots.ox.ac.uk/~luca/siamese-fc.html
        :param size: np.array([w, h]) - predicted bbox size
        :param prev_size: np.array([w, h]) - bbox size on previous frame
        :param lr: float - smoothing learning rate
        :return: Tuple[float, float] - smoothed size
        """
        size = size * lr
        prev_size = prev_size * (1 - lr)
        w = prev_size[0] + lr * (size[0] + prev_size[0])
        h = prev_size[1] + lr * (size[1] + prev_size[1])
        return w, h

    def _get_point_offset(self, pred_bbox: np.array) -> Tuple[float, float]:
        pred_xs = pred_bbox[0] + (pred_bbox[2] / 2)
        pred_ys = pred_bbox[1] + (pred_bbox[3] / 2)

        diff_xs = pred_xs - self.tracking_config["instance_size"] // 2
        diff_ys = pred_ys - self.tracking_config["instance_size"] // 2
        return diff_xs, diff_ys

    def _postprocess_bbox(
        self, decoded_info: TrackerDecodeResult, cls_score: np.array, penalty: Any = None
    ) -> np.array:
        pred_bbox = np.squeeze(decoded_info.bbox.cpu().numpy())
        if not self.tracking_config.get("smooth", False):
            return pred_bbox

        prev_size = self.tracking_state.prev_size
        r_max, c_max = decoded_info.pred_coords[0]
        # size learning rate
        lr = (penalty[r_max, c_max] * cls_score[r_max, c_max] * self.tracking_config["lr"]).item()

        pred_size = np.array(pred_bbox[2:])
        pred_w, pred_h = self._smooth_size(pred_size, prev_size=prev_size, lr=lr)
        predicted_bbox = np.array([pred_bbox[0], pred_bbox[1], pred_w, pred_h])
        return predicted_bbox

    def _confidence_postprocess(
        self, cls_score: np.ndarray, regression_map: torch.Tensor
    ) -> tuple[ndarray, None, None] | tuple[Any, ndarray, Tensor]:
        """

        :param cls_score: torch.Tensor - classification score
        :param pred_location: torch.Tensor - predicted locations
        :param prev_size: np.array - size from previous frame
        :return: penalty_score: np.ndarray - updated cls_score
        """
        if not self.tracking_config.get("smooth", False):
            return cls_score, None, None
        prev_size = self.tracking_state.prev_size

        pred_location_ = torch.stack([
            self.box_coder.grid_x - regression_map[:, 0],
            self.box_coder.grid_y - regression_map[:, 1],
            self.box_coder.grid_x + regression_map[:, 2],
            self.box_coder.grid_y + regression_map[:, 3],
        ], dim=1)  # shape [1,4,H,W]

        pred_location = pred_location_[0]  # [4,H,W]

        s_c = limit(
            squared_size(pred_location[2] - pred_location[0], pred_location[3] - pred_location[1])
            / (squared_size(prev_size[0], prev_size[1]))
        )

        r_c = limit(
            (prev_size[0] / prev_size[1])
            / ((pred_location[2] - pred_location[0]) / (pred_location[3] - pred_location[1]))
        )

        penalty = torch.exp(-(r_c * s_c - 1) * self.tracking_config["penalty_k"])
        pscore = penalty * cls_score

        pscore = (
            pscore * (1 - self.tracking_config["window_influence"])
            + self.window * self.tracking_config["window_influence"]
        )
        return pscore, penalty.cpu().numpy(), pred_location_
