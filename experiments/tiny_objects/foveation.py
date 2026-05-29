"""Hyperbolic foveation utilities for tiny-object tracking experiments.

The implementation mirrors the Telescope transform described in
``arxiv.org_html_2604.06332v1.md`` while keeping a NumPy/OpenCV fallback path
so it can wrap the existing SiamRAM/SiamABC trackers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FoveationConfig:
    alpha: float = 3.5
    blend_exponent: float = 2.0
    radius_beta: float = 5.0
    min_radius_px: float = 16.0
    max_radius_norm: float = 1.25
    inverse_iters: int = 8
    inverse_eta: float = 1
    use_torch_inverse: bool = True
    border_mode: int = cv2.BORDER_REPLICATE
    border_value: tuple[int, int, int] = (128, 128, 128)
    interpolation: int = cv2.INTER_LINEAR


@dataclass
class FoveationState:
    center_xy: tuple[float, float]
    center_norm: tuple[float, float]
    radius_norm: float
    radius_px: float
    image_size: tuple[int, int]
    alpha: float
    blend_exponent: float
    radius_beta: float
    inverse_iters: int

    def to_row(self, prefix: str = "tiny") -> dict[str, Any]:
        row = asdict(self)
        return {f"{prefix}_{key}": value for key, value in row.items()}


def bbox_center_xy(bbox: Any) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with four values, got {bbox!r}")
    return np.array([arr[0] + arr[2] / 2.0, arr[1] + arr[3] / 2.0], dtype=np.float64)


def _as_bbox(bbox: Any) -> np.ndarray:
    arr = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with four values, got {bbox!r}")
    return arr[:4].astype(np.float64)


def _xy_to_norm(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    denom = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float64)
    return 2.0 * points / denom - 1.0


def _norm_to_xy(points_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points_norm, dtype=np.float64)
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float64)
    return (points + 1.0) * 0.5 * scale


def phi_norm(
    points_norm: np.ndarray,
    center_norm: np.ndarray,
    radius_norm: float,
    *,
    alpha: float,
    blend_exponent: float,
) -> np.ndarray:
    """Map normalized Euclidean coordinates into foveated coordinates."""

    x = np.asarray(points_norm, dtype=np.float64)
    center = np.asarray(center_norm, dtype=np.float64).reshape(1, 2)
    delta = x - center
    r = np.linalg.norm(delta, axis=-1, keepdims=True)
    safe_r = np.maximum(r, 1e-8)
    h_x = center + (np.tanh(float(alpha) * safe_r) / safe_r) * delta
    radius = max(float(radius_norm), 1e-8)
    weight = np.power(1.0 - np.minimum(r / radius, 1.0), float(blend_exponent))
    return (1.0 - weight) * x + weight * h_x


def inverse_phi_norm(
    points_foveated_norm: np.ndarray,
    center_norm: np.ndarray,
    radius_norm: float,
    *,
    alpha: float,
    blend_exponent: float,
    inverse_iters: int,
    eta: float,
) -> np.ndarray:
    """Fixed-point Newton-style inverse used by Telescope for grid sampling."""

    y = np.asarray(points_foveated_norm, dtype=np.float64)
    x = y.copy()
    for _ in range(int(inverse_iters)):
        phi_x = phi_norm(x, center_norm, radius_norm, alpha=alpha, blend_exponent=blend_exponent)
        x = x + float(eta) * (y - phi_x)
    return np.clip(x, -1.0, 1.0)


def inverse_phi_norm_fast(
    points_foveated_norm: np.ndarray,
    center_norm: np.ndarray,
    radius_norm: float,
    *,
    alpha: float,
    blend_exponent: float,
    inverse_iters: int,
    eta: float,
) -> np.ndarray:
    """Torch-accelerated fixed-point inverse for dense remap grids.

    CUDA is used when available. If PyTorch is unavailable for a lightweight
    geometry-only import, this falls back to the NumPy implementation.
    """

    try:
        import torch
    except ImportError:
        return inverse_phi_norm(
            points_foveated_norm,
            center_norm,
            radius_norm,
            alpha=alpha,
            blend_exponent=blend_exponent,
            inverse_iters=inverse_iters,
            eta=eta,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y = torch.as_tensor(points_foveated_norm, dtype=torch.float32, device=device)
        center = torch.as_tensor(center_norm, dtype=torch.float32, device=device).reshape(1, 2)
        x = y.clone()
        radius = max(float(radius_norm), 1e-8)
        for _ in range(int(inverse_iters)):
            delta = x - center
            r = torch.norm(delta, dim=-1, keepdim=True)
            safe_r = torch.clamp(r, min=1e-8)
            h_x = center + (torch.tanh(float(alpha) * safe_r) / safe_r) * delta
            weight = torch.pow(1.0 - torch.clamp(r / radius, max=1.0), float(blend_exponent))
            phi_x = (1.0 - weight) * x + weight * h_x
            x = x + float(eta) * (y - phi_x)
        return torch.clamp(x, -1.0, 1.0).cpu().numpy()


class HyperbolicFoveator:
    """Create foveated frames and map boxes between original and warped space."""

    def __init__(self, config: FoveationConfig | None = None) -> None:
        self.config = config or FoveationConfig()
        self._grid_cache: dict[tuple[int, int], np.ndarray] = {}
        self._torch_grid_cache: dict[tuple[int, int, str], Any] = {}

    def state_from_bbox(self, bbox: Any, frame_shape: tuple[int, int]) -> FoveationState:
        height, width = frame_shape[:2]
        box = _as_bbox(bbox)
        center_xy = bbox_center_xy(box)
        center_xy[0] = np.clip(center_xy[0], 0.0, max(width - 1, 1))
        center_xy[1] = np.clip(center_xy[1], 0.0, max(height - 1, 1))
        center_norm = _xy_to_norm(center_xy.reshape(1, 2), width, height).reshape(2)
        box_norm_size = np.array(
            [
                2.0 * max(box[2], 1.0) / max(width - 1, 1),
                2.0 * max(box[3], 1.0) / max(height - 1, 1),
            ],
            dtype=np.float64,
        )
        min_radius_norm = 2.0 * float(self.config.min_radius_px) / max(max(width - 1, 1), max(height - 1, 1))
        radius_norm = float(self.config.radius_beta) * float(np.max(box_norm_size))
        radius_norm = float(np.clip(radius_norm, min_radius_norm, float(self.config.max_radius_norm)))
        radius_px = 0.5 * radius_norm * max(width - 1, height - 1)
        return FoveationState(
            center_xy=(float(center_xy[0]), float(center_xy[1])),
            center_norm=(float(center_norm[0]), float(center_norm[1])),
            radius_norm=radius_norm,
            radius_px=float(radius_px),
            image_size=(int(width), int(height)),
            alpha=float(self.config.alpha),
            blend_exponent=float(self.config.blend_exponent),
            radius_beta=float(self.config.radius_beta),
            inverse_iters=int(self.config.inverse_iters),
        )

    def _dest_grid(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float64)
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid = np.stack([grid_x, grid_y], axis=-1)
        self._grid_cache[key] = grid
        return grid

    def _inverse_grid_torch_tensor(self, state: FoveationState, width: int, height: int):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (width, height, str(device))
        y = self._torch_grid_cache.get(cache_key)
        if y is None:
            xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
            ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            y = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
            self._torch_grid_cache[cache_key] = y
        center = torch.as_tensor(state.center_norm, dtype=torch.float32, device=device).reshape(1, 2)
        radius = max(float(state.radius_norm), 1e-8)
        with torch.no_grad():
            x = y.clone()
            for _ in range(int(state.inverse_iters)):
                delta = x - center
                r = torch.norm(delta, dim=-1, keepdim=True)
                safe_r = torch.clamp(r, min=1e-8)
                h_x = center + (torch.tanh(float(state.alpha) * safe_r) / safe_r) * delta
                weight = torch.pow(1.0 - torch.clamp(r / radius, max=1.0), float(state.blend_exponent))
                phi_x = (1.0 - weight) * x + weight * h_x
                x = x + float(self.config.inverse_eta) * (y - phi_x)
            return torch.clamp(x, -1.0, 1.0).reshape(height, width, 2)

    def _inverse_grid_torch(self, dest: np.ndarray, state: FoveationState, width: int, height: int) -> np.ndarray:
        try:
            return self._inverse_grid_torch_tensor(state, width, height).cpu().numpy()
        except ImportError:
            return inverse_phi_norm(
                dest.reshape(-1, 2),
                np.asarray(state.center_norm, dtype=np.float64),
                state.radius_norm,
                alpha=state.alpha,
                blend_exponent=state.blend_exponent,
                inverse_iters=state.inverse_iters,
                eta=float(self.config.inverse_eta),
            )

    def warp_image_torch(self, frame: np.ndarray, state: FoveationState) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        height, width = frame.shape[:2]
        if (width, height) != state.image_size:
            raise ValueError(f"State image size {state.image_size} does not match frame {(width, height)}")
        if not torch.cuda.is_available():
            return self.warp_image_cv2(frame, state)

        source_norm = self._inverse_grid_torch_tensor(state, width, height).unsqueeze(0)
        frame_t = torch.as_tensor(np.ascontiguousarray(frame), dtype=torch.float32, device=source_norm.device)
        frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)
        mode = "nearest" if int(self.config.interpolation) == cv2.INTER_NEAREST else "bilinear"
        warped_t = F.grid_sample(
            frame_t,
            source_norm,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
        return (
            warped_t.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def warp_image_cv2(self, frame: np.ndarray, state: FoveationState) -> np.ndarray:
        height, width = frame.shape[:2]
        if (width, height) != state.image_size:
            raise ValueError(f"State image size {state.image_size} does not match frame {(width, height)}")
        dest = self._dest_grid(width, height)
        if self.config.use_torch_inverse:
            source_norm = self._inverse_grid_torch(dest, state, width, height).reshape(height, width, 2)
        else:
            source_norm = inverse_phi_norm(
                dest.reshape(-1, 2),
                np.asarray(state.center_norm, dtype=np.float64),
                state.radius_norm,
                alpha=state.alpha,
                blend_exponent=state.blend_exponent,
                inverse_iters=state.inverse_iters,
                eta=float(self.config.inverse_eta),
            ).reshape(height, width, 2)
        source_xy = _norm_to_xy(source_norm, width, height)
        return cv2.remap(
            frame,
            source_xy[..., 0].astype(np.float32),
            source_xy[..., 1].astype(np.float32),
            interpolation=int(self.config.interpolation),
            borderMode=int(self.config.border_mode),
            borderValue=tuple(int(v) for v in self.config.border_value),
        )

    def warp_image(self, frame: np.ndarray, state: FoveationState) -> np.ndarray:
        if self.config.use_torch_inverse:
            try:
                import torch
            except ImportError:
                return self.warp_image_cv2(frame, state)
            if torch.cuda.is_available() and frame.ndim == 3:
                return self.warp_image_torch(frame, state)
        return self.warp_image_cv2(frame, state)

    def transform_points(self, points_xy: Any, state: FoveationState) -> np.ndarray:
        width, height = state.image_size
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        norm = _xy_to_norm(points, width, height)
        warped_norm = phi_norm(
            norm,
            np.asarray(state.center_norm, dtype=np.float64),
            state.radius_norm,
            alpha=state.alpha,
            blend_exponent=state.blend_exponent,
        )
        return _norm_to_xy(warped_norm, width, height)

    def inverse_points(self, points_foveated_xy: Any, state: FoveationState) -> np.ndarray:
        width, height = state.image_size
        points = np.asarray(points_foveated_xy, dtype=np.float64).reshape(-1, 2)
        norm = _xy_to_norm(points, width, height)
        source_norm = inverse_phi_norm(
            norm,
            np.asarray(state.center_norm, dtype=np.float64),
            state.radius_norm,
            alpha=state.alpha,
            blend_exponent=state.blend_exponent,
            inverse_iters=state.inverse_iters,
            eta=float(self.config.inverse_eta),
        )
        return _norm_to_xy(source_norm, width, height)

    def warp_bbox(self, bbox: Any, state: FoveationState) -> np.ndarray:
        x, y, w, h = _as_bbox(bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        center = np.array([[cx, cy]], dtype=np.float64)
        warped_center = self.transform_points(center, state)[0]

        eps_w = max(w * 0.1, 1.0)
        eps_h = max(h * 0.1, 1.0)

        pt_w = np.array([[cx + eps_w, cy]], dtype=np.float64)
        pt_h = np.array([[cx, cy + eps_h]], dtype=np.float64)

        warped_pt_w = self.transform_points(pt_w, state)[0]
        warped_pt_h = self.transform_points(pt_h, state)[0]

        mag_x = np.linalg.norm(warped_pt_w - warped_center) / eps_w
        mag_y = np.linalg.norm(warped_pt_h - warped_center) / eps_h

        new_w = max(1.0, w * mag_x)
        new_h = max(1.0, h * mag_y)

        return np.array(
            [warped_center[0] - new_w / 2.0, warped_center[1] - new_h / 2.0, new_w, new_h],
            dtype=np.float64,
        )

    def unwarp_bbox(self, bbox_foveated: Any, state: FoveationState) -> np.ndarray:
        x, y, w, h = _as_bbox(bbox_foveated)
        cx, cy = x + w / 2.0, y + h / 2.0

        center_foveated = np.array([[cx, cy]], dtype=np.float64)
        source_center = self.inverse_points(center_foveated, state)[0]

        eps_w = max(w * 0.1, 1.0)
        eps_h = max(h * 0.1, 1.0)

        pt_w_fov = np.array([[cx + eps_w, cy]], dtype=np.float64)
        pt_h_fov = np.array([[cx, cy + eps_h]], dtype=np.float64)

        source_pt_w = self.inverse_points(pt_w_fov, state)[0]
        source_pt_h = self.inverse_points(pt_h_fov, state)[0]

        inv_mag_x = np.linalg.norm(source_pt_w - source_center) / eps_w
        inv_mag_y = np.linalg.norm(source_pt_h - source_center) / eps_h

        new_w = max(1.0, w * inv_mag_x)
        new_h = max(1.0, h * inv_mag_y)

        return np.array(
            [source_center[0] - new_w / 2.0, source_center[1] - new_h / 2.0, new_w, new_h],
            dtype=np.float64,
        )

    def roundtrip_error_px(self, points_xy: Any, state: FoveationState) -> float:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        recovered = self.inverse_points(self.transform_points(points, state), state)
        return float(np.linalg.norm(recovered - points, axis=1).max()) if len(points) else 0.0
