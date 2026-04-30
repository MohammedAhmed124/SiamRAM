"""
General utility functions for SiamRAM.

This module contains bounding box conversion helpers, coordinate grid
generation, image cropping utilities, and IoU calculation functions.
"""

from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torch import Tensor
from torch.nn import Module


def _extract_descriptor(
    frame: np.ndarray,
    bbox,
    size: int = 16,
    w_gray: float = 0.4,
    w_color: float = 0.6,
    _PROC_SIZE: int = 64,
) -> Optional[np.ndarray]:
    x, y, w, h = map(int, bbox)
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)
    patch = frame[y: y + h, x: x + w]
    if patch.size == 0:
        return None

    small = cv2.resize(patch, (_PROC_SIZE, _PROC_SIZE), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    p = cv2.resize(gray, (size, size)).flatten().astype(np.float32)
    p /= np.linalg.norm(p) + 1e-8

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_hist = (
        cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        .flatten()
        .astype(np.float32)
    )
    h_hist /= np.linalg.norm(h_hist) + 1e-8

    desc = np.concatenate([w_gray * p, w_color * h_hist])
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


def _iou(
    a,
    b,
) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / (union + 1e-8)


def _cos_sim(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def xyxy_to_xywh(
    boxes,
):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format.

    Args:
        boxes (List[float]): Bounding box coordinates.

    Returns:
        List[float]: Converted bounding box.
    """
    x1, y1, x2, y2 = boxes
    return [x1, y1, x2 - x1, y2 - y1]


def convert_xywh_to_xyxy(
    bbox: NDArray,
) -> NDArray:
    """
    Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (NDArray): Bounding box in xywh format.

    Returns:
        NDArray: Bounding box in xyxy format.
    """
    return np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])


def to_device(
    x: Union[torch.Tensor, torch.nn.Module],
    cuda_id: int = 0,
) -> Tensor | Module:
    """
    Move a tensor or module to a CUDA device if available.

    Args:
        x (Union[Tensor, Module]): Object to move.
        cuda_id (int): CUDA device index.

    Returns:
        Union[Tensor, Module]: Object on the target device.
    """
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def extend_bbox(
    bbox: NDArray,
    image_width: int,
    image_height: int,
    offset: float = 1.1,
) -> NDArray:
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
    Extract a cropped and resized patch from an image with padding.

    Args:
        image (np.ndarray): Source image.
        bbox (np.ndarray): Target bounding box.
        crop_size (int): Size of the output square crop.
        context (np.ndarray): Context window to crop.
        padding_value (Optional[np.ndarray]): Value for border padding.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (resized_crop, bbox_in_crop, context_rect).
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
    Ensure a bounding box has a minimum width and height.

    Args:
        bbox (NDArray): Bounding box [x, y, w, h].
        min_bbox (int): Minimum side length.

    Returns:
        NDArray: Adjusted bounding box.
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
    Generate a Gaussian regression weight label map.

    Args:
        bbox (NDArray): Target bounding box.
        image_size (int): Size of the input image.
        map_size (int): Size of the output score map.
        r_pos, r_neg (int): Radius for positive and negative locations.

    Returns:
        torch.Tensor: Weight map.
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
    Limit the scale ratio to avoid extreme values.

    Args:
        radius (Union[Tensor, float]): Scale ratio.

    Returns:
        Union[Tensor, float]: Limited scale ratio.
    """
    if isinstance(radius, torch.Tensor):
        return torch.maximum(radius, 1.0 / radius)
    return np.maximum(radius, 1.0 / radius)


def squared_size(
    w: int,
    h: int,
) -> Union[torch.Tensor, float]:
    """
    Compute the side length of a square with equivalent area after padding.

    Args:
        w, h (int): Original width and height.

    Returns:
        Union[Tensor, float]: Equivalent square side length.
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
    Convert a flat index to a multi-dimensional index.

    Args:
        index (Any): Flat index.
        shape (Tuple[int, int]): Dimensions of the target tensor.

    Returns:
        Tuple[int, ...]: Multi-dimensional index.
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
    Compute the Intersection-over-Union (IoU) between two boxes.

    Args:
        reg_target (torch.Tensor): Ground-truth boxes.
        pred (torch.Tensor): Predicted boxes.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: IoU values.
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
