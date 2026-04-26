from typing import Any, Optional, Tuple, Union, cast

import cv2
import numpy as np
import torch
from numpy._typing import NDArray
from torch import Tensor
from torch.nn import Module


def _extract_descriptor(frame: np.ndarray, bbox, size=16) -> Optional[np.ndarray]:
    """phi(I_t, b) = normalised concat of grayscale patch + HSV histogram."""
    x, y, w, h = map(int, bbox)
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)
    patch = frame[y:y + h, x:x + w]
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    p = cv2.resize(gray, (size, size)).flatten().astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                          [0, 180, 0, 256, 0, 256]).flatten().astype(np.float32)
    desc = np.concatenate([p, h_hist])
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


def _iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / (union + 1e-8)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes
    return [x1, y1, x2 - x1, y2 - y1]


def convert_xywh_to_xyxy(bbox: NDArray) -> NDArray:
    return np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])


def to_device(x: Union[torch.Tensor, torch.nn.Module], cuda_id: int = 0) -> Tensor | Module:
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def extend_bbox(
    bbox: NDArray,
    image_width: int,
    image_height: int,
    offset: float = 1.1
) -> NDArray:
    """
    Expands a bounding box outward by a given percentage to include surrounding context.

    Why we need this:
    In object tracking, we rarely just want the tight crop of the object. We need to see the
    area around the object (the context) to help predict where it might move in the next frame
    or to give the neural network background information to distinguish the object from its surroundings.

    How it works:
    The offset acts as a multiplier. An offset of 0.1 means "add 10% of the box's width/height
    to the edges". You can pass a single float (applies to all sides), a tuple of 2 (width_offset,
    height_offset), or a tuple of 4 (left, right, top, bottom).

    Note: The `image_width` and `image_height` parameters are not actually used in the function body,
    but are kept in the signature. To safely bound the box to the image size, use `ensure_bbox_boundaries` afterward.

    Args:
        bbox (np.array): The original bounding box in [x, y, width, height] format.
        image_width (int): Width of the source image.
        image_height (int): Height of the source image.
        offset (float | tuple): The percentage to expand by. Default 1.1 (110% expansion).

    Returns:
        np.array: The newly expanded bounding box in [x, y, w, h] format, cast to integers.

    Example:
        >>> bbox = np.array([100, 100, 50, 50]) # Box at x=100, y=100, w=50, h=50
        >>> # Extend by 0.1 (10% of 50 = 5 pixels)
        >>> extend_bbox(bbox, 1920, 1080, offset=0.1)
        array([95, 95, 60, 60]) # x and y moved left/up by 5, w and h grew by 10 (5 on each side)
    """

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

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def ensure_bbox_boundaries(
    bbox: NDArray,
    img_shape: Tuple[int, int]
) -> NDArray:
    """
    Trims a bounding box so it does not spill over the edges of the image.

    Why we need this:
    When you expand a bounding box (like we do in `extend_bbox`) near the edge of an image,
    the coordinates might become negative, or the width/height might push the box outside the
    image boundaries. If you try to slice an image array with these out-of-bounds coordinates,
    your code will crash or return an empty array.

    How it works:
    It checks the top-left corner (x1, y1) and the bottom-right corner (x2, y2). It forces
    them to be at least 0, and at most the width or height of the image. It then recalculates
    the width and height based on these safe coordinates.

    Args:
        bbox (np.array): The bounding box in [x, y, width, height] format.
        img_shape (Tuple[int, int]): The dimensions of the image in (height, width) format.

    Returns:
        np.array: A safe bounding box strictly contained within the image, in [x, y, w, h] format.

    Example:
        >>> img_shape = (1080, 1920) # (height, width)
        >>> bbox = np.array([-10, 500, 100, 100]) # Starts 10 pixels off the left edge
        >>> ensure_bbox_boundaries(bbox, img_shape)
        array([0, 500, 90, 100]) # Trimmed at x=0, making the width 90 instead of 100
    """
    x1, y1, w, h = bbox
    x2_raw = x1 + w  # ← compute BEFORE any clamping
    y2_raw = y1 + h
    x1 = min(max(0, x1), img_shape[1])
    y1 = min(max(0, y1), img_shape[0])
    x2 = min(max(0, x2_raw), img_shape[1])  # ← uses raw, not clamped x1
    y2 = min(max(0, y2_raw), img_shape[0])
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def clamp_bbox(
    bbox: NDArray,
    shape: Tuple[int, int],
    min_side: int = 3
) -> NDArray:
    """
    Ensures a bounding box is within image bounds AND forces it to have a minimum width and height.

    Why we need this:
    If an object is just barely peeking onto the screen (e.g., 1 pixel wide), trimming it with
    `ensure_bbox_boundaries` might result in a 0-width or 1-width box. Deep learning trackers
    often break or divide by zero when dealing with boxes that are effectively flat lines. This
    guarantees the box is mathematically usable.

    How it works:
    First, it safely trims the box to the image. Then, if the width or height is less than
    `min_side`, it forces it to be `min_side`. If forcing it to be bigger pushes it off the
    right/bottom edge of the image, it shifts the entire box slightly to the left/up so it fits.

    Args:
        bbox (np.array): The bounding box in [x, y, width, height] format.
        shape (Tuple[int, int]): The image dimensions in (height, width) format.
        min_side (int): The absolute minimum pixel size allowed for width or height. Default is 3.

    Returns:
        np.array: The clamped bounding box in [x, y, w, h] format.

    Example:
        >>> shape = (100, 100)
        >>> bbox = np.array([99, 50, 1, 10]) # A 1-pixel wide box at the very right edge
        >>> clamp_bbox(bbox, shape, min_side=3)
        array([97, 50, 3, 10]) # Forced to be 3 pixels wide, and shifted left to x=97 so it fits.
    """
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


def get_extended_crop(image, bbox, crop_size, context, padding_value=None):
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))

    pad_left = max(-context[0], 0)
    pad_top = max(-context[1], 0)
    pad_right = max(context[0] + context[2] - image.shape[1], 0)
    pad_bottom = max(context[1] + context[3] - image.shape[0], 0)

    crop = image[
        context[1] + pad_top: context[1] + context[3] - pad_bottom,
        context[0] + pad_left: context[0] + context[2] - pad_right,
    ]

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=padding_value)

    # Scale bbox manually — 4 multiplications vs full albumentations pipeline
    sx = crop_size / crop.shape[1]
    sy = crop_size / crop.shape[0]
    padded_bbox = np.array([
        (bbox[0] - context[0]) * sx,
        (bbox[1] - context[1]) * sy,
        bbox[2] * sx,
        bbox[3] * sy,
    ])

    resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    padded_bbox = ensure_bbox_boundaries(padded_bbox.astype(np.int32), resized.shape[:2])
    return resized, padded_bbox, context


def handle_empty_bbox(bbox: NDArray, min_bbox: int = 3) -> NDArray:
    bbox[2] = max(bbox[2], min_bbox)
    bbox[3] = max(bbox[3], min_bbox)
    return bbox


def get_regression_weight_label(
    bbox, image_size: int = 255, map_size: int = 25, r_pos: int = 2, r_neg: int = 0
) -> torch.Tensor:
    bbox_c_x, bbox_c_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
    sz_x, sz_y = np.floor(float(bbox_c_x / image_size * map_size)), np.floor(float(bbox_c_y / image_size * map_size))
    x, y = np.meshgrid(np.arange(0, map_size) - sz_x, np.arange(0, map_size) - sz_y)

    dist_to_center = np.abs(x) + np.abs(y)
    label = np.where(
        dist_to_center <= r_pos,
        np.ones_like(y),
        np.where(dist_to_center < r_neg, 0.5 * np.ones_like(y), np.zeros_like(y)),
    )
    return torch.from_numpy(label)


@torch.no_grad()
def make_grid(score_size: int, total_stride: int, instance_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each element of feature map on input search image
    :return: H*W*2 (position for each element)
    """

    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )

    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    grid_x = torch.from_numpy(grid_x[np.newaxis, :, :])
    grid_y = torch.from_numpy(grid_y[np.newaxis, :, :])
    return grid_x, grid_y


def limit(radius: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    if isinstance(radius, torch.Tensor):
        return torch.maximum(radius, 1.0 / radius)
    return np.maximum(radius, 1.0 / radius)


def squared_size(w: int, h: int) -> Union[torch.Tensor, float]:
    pad = (w + h) * 0.5
    size = (w + pad) * (h + pad)
    if isinstance(size, torch.Tensor):
        return torch.sqrt(size)
    return np.sqrt(size)


def unravel_index(index: Any, shape: Tuple[int, int]) -> Tuple[int, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
