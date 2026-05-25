"""
BoT-SORT-inspired camera-motion estimator used as an optional homography engine.

This is a focused extraction of the branch experiment implementation:
- target/dynamic-region masking
- grid-distributed GFTT points
- forward/backward LK filtering
- RANSAC transform estimation
- confidence gating + fallback to last valid transform
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


BBox = tuple[float, float, float, float]


@dataclass
class MotionConfig:
    model: str = "partial_affine"
    work_scale: float = 0.75
    bbox_pad: float = 0.50
    grid_rows: int = 6
    grid_cols: int = 8
    max_corners_per_cell: int = 24
    quality_level: float = 0.01
    min_distance: int = 7
    block_size: int = 7
    lk_win_size: int = 21
    lk_max_level: int = 3
    fb_max_error: float = 1.5
    ransac_reproj_threshold: float = 3.0
    ransac_confidence: float = 0.99
    ransac_max_iters: int = 3000
    min_inliers: int = 30
    min_inlier_ratio: float = 0.35
    max_median_residual: float = 4.0
    min_coverage_cells: int = 6
    max_translation_frac: float = 0.25
    min_scale: float = 0.70
    max_scale: float = 1.40
    max_rotation_deg: float = 25.0
    residual_mad_multiplier: float = 4.0
    residual_min_threshold: int = 18
    residual_max_fraction: float = 0.22
    residual_min_area: int = 12
    dynamic_dilate: int = 9
    feature_detect_interval: int = 2
    min_points_to_redetect: int = 120
    max_track_points: int = 320
    enable_dynamic_mask: bool = True
    residual_update_interval: int = 2


@dataclass
class Confidence:
    valid: bool
    reason: str
    num_detected: int = 0
    num_tracked: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    median_residual: float = float("inf")
    coverage_cells: int = 0
    coverage_ratio: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    max_corner_displacement: float = 0.0


@dataclass
class MotionStats:
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    max_corner_displacement: float = 0.0


@dataclass
class StepResult:
    transform_original: np.ndarray
    confidence: Confidence
    used_fallback: bool
    runtime_ms: float


def to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = matrix.astype(np.float64)
    return h


def from_homogeneous(h: np.ndarray, model: str) -> np.ndarray:
    if model == "homography":
        return h.astype(np.float64)
    return h[:2, :].astype(np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 2)
    h = to_homogeneous(transform)
    pts = points.reshape(-1, 2).astype(np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = pts_h @ h.T
    warped[:, 0] /= np.maximum(warped[:, 2], 1e-9)
    warped[:, 1] /= np.maximum(warped[:, 2], 1e-9)
    return warped[:, :2].astype(np.float32)


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    x, y, w, h = bbox
    return x * scale, y * scale, w * scale, h * scale


def pad_bbox(
    bbox: BBox,
    pad: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    px = w * pad
    py = h * pad
    x1 = max(0, int(round(x - px)))
    y1 = max(0, int(round(y - py)))
    x2 = min(width, int(round(x + w + px)))
    y2 = min(height, int(round(y + h + py)))
    return x1, y1, x2, y2


def warp_gray(
    previous: np.ndarray,
    transform: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    if transform.shape == (3, 3):
        return cv2.warpPerspective(previous, transform, (width, height), flags=cv2.INTER_LINEAR)
    return cv2.warpAffine(previous, transform, (width, height), flags=cv2.INTER_LINEAR)


class CameraMotionEstimator:
    def __init__(self, first_frame: np.ndarray, config: MotionConfig):
        self.cfg = config
        self.frame_shape_original = first_frame.shape[:2]
        self.prev_gray = self._preprocess(first_frame)
        self.height, self.width = self.prev_gray.shape[:2]
        self.last_transform = self._identity_transform()
        self.dynamic_mask = np.zeros_like(self.prev_gray, dtype=np.uint8)
        self._frame_idx: int = 0
        self._cached_prev_points: np.ndarray = np.empty((0, 2), dtype=np.float32)

    def _identity_transform(self) -> np.ndarray:
        if self.cfg.model == "homography":
            return np.eye(3, dtype=np.float64)
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if abs(self.cfg.work_scale - 1.0) > 1e-6:
            gray = cv2.resize(
                gray,
                None,
                fx=self.cfg.work_scale,
                fy=self.cfg.work_scale,
                interpolation=cv2.INTER_AREA,
            )
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _bbox_ignore_mask(self, bbox: Optional[BBox]) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if bbox is None:
            return mask
        x1, y1, x2, y2 = pad_bbox(
            scale_bbox(bbox, self.cfg.work_scale),
            self.cfg.bbox_pad,
            self.width,
            self.height,
        )
        mask[y1:y2, x1:x2] = 255
        return mask

    def _detect_grid_corners(self, valid_mask: np.ndarray) -> np.ndarray:
        points: list[np.ndarray] = []
        cell_h = int(math.ceil(self.height / self.cfg.grid_rows))
        cell_w = int(math.ceil(self.width / self.cfg.grid_cols))
        for gy in range(self.cfg.grid_rows):
            for gx in range(self.cfg.grid_cols):
                x1 = gx * cell_w
                y1 = gy * cell_h
                x2 = min(self.width, x1 + cell_w)
                y2 = min(self.height, y1 + cell_h)
                roi_mask = valid_mask[y1:y2, x1:x2]
                if roi_mask.size == 0 or cv2.countNonZero(roi_mask) < 20:
                    continue
                roi_prev = self.prev_gray[y1:y2, x1:x2]
                cell_points = cv2.goodFeaturesToTrack(
                    roi_prev,
                    maxCorners=self.cfg.max_corners_per_cell,
                    qualityLevel=self.cfg.quality_level,
                    minDistance=self.cfg.min_distance,
                    mask=roi_mask,
                    blockSize=self.cfg.block_size,
                    useHarrisDetector=False,
                )
                if cell_points is not None:
                    cell_points[:, 0, 0] += float(x1)
                    cell_points[:, 0, 1] += float(y1)
                    points.append(cell_points)
        if not points:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.vstack(points).astype(np.float32)

    def _cap_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) > 0:
            pts = pts[np.isfinite(pts).all(axis=1)]
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        max_pts = max(4, int(self.cfg.max_track_points))
        n = len(pts)
        if n <= max_pts:
            return pts
        step = max(1, int(math.ceil(n / max_pts)))
        return np.ascontiguousarray(pts[::step][:max_pts], dtype=np.float32)

    def _should_redetect(self) -> bool:
        if len(self._cached_prev_points) < max(4, int(self.cfg.min_points_to_redetect)):
            return True
        interval = max(1, int(self.cfg.feature_detect_interval))
        if interval <= 1:
            return True
        return (self._frame_idx % interval) == 0

    def _track_points(
        self,
        curr_gray: np.ndarray,
        points_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts_prev = np.asarray(points_prev, dtype=np.float32).reshape(-1, 2)
        if len(pts_prev) > 0:
            pts_prev = pts_prev[np.isfinite(pts_prev).all(axis=1)]
        if len(pts_prev) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        pts_prev = np.ascontiguousarray(pts_prev, dtype=np.float32).reshape(-1, 1, 2)

        lk_params = dict(
            winSize=(self.cfg.lk_win_size, self.cfg.lk_win_size),
            maxLevel=self.cfg.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        try:
            points_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, pts_prev, None, **lk_params
            )
        except cv2.error:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        if points_curr is None or status_fwd is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        points_curr = np.ascontiguousarray(points_curr, dtype=np.float32)
        try:
            points_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, self.prev_gray, points_curr, None, **lk_params
            )
        except cv2.error:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        if points_back is None or status_back is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        prev_flat = pts_prev.reshape(-1, 2)
        curr_flat = points_curr.reshape(-1, 2)
        back_flat = points_back.reshape(-1, 2)
        fb_error = np.linalg.norm(prev_flat - back_flat, axis=1)
        valid = (
            (status_fwd.reshape(-1) == 1)
            & (status_back.reshape(-1) == 1)
            & np.isfinite(curr_flat).all(axis=1)
            & (fb_error <= self.cfg.fb_max_error)
        )
        return prev_flat[valid], curr_flat[valid]

    def _estimate_transform(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(prev_pts) < 4:
            return self._identity_transform(), np.zeros((len(prev_pts),), dtype=bool)

        if self.cfg.model == "homography":
            matrix, inlier_mask = cv2.findHomography(
                prev_pts,
                curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
                maxIters=self.cfg.ransac_max_iters,
                confidence=self.cfg.ransac_confidence,
            )
            if matrix is None:
                return self._identity_transform(), np.zeros((len(prev_pts),), dtype=bool)
            return matrix.astype(np.float64), inlier_mask.reshape(-1).astype(bool)

        if self.cfg.model == "full_affine":
            matrix, inlier_mask = cv2.estimateAffine2D(
                prev_pts,
                curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
                maxIters=self.cfg.ransac_max_iters,
                confidence=self.cfg.ransac_confidence,
                refineIters=10,
            )
        else:
            matrix, inlier_mask = cv2.estimateAffinePartial2D(
                prev_pts,
                curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac_reproj_threshold,
                maxIters=self.cfg.ransac_max_iters,
                confidence=self.cfg.ransac_confidence,
                refineIters=10,
            )
        if matrix is None or inlier_mask is None:
            return self._identity_transform(), np.zeros((len(prev_pts),), dtype=bool)
        return matrix.astype(np.float64), inlier_mask.reshape(-1).astype(bool)

    def _coverage_cells(self, points: np.ndarray) -> tuple[int, float]:
        if len(points) == 0:
            return 0, 0.0
        cell_h = max(1, int(math.ceil(self.height / self.cfg.grid_rows)))
        cell_w = max(1, int(math.ceil(self.width / self.cfg.grid_cols)))
        cells = set()
        for x, y in points:
            gx = min(self.cfg.grid_cols - 1, max(0, int(x // cell_w)))
            gy = min(self.cfg.grid_rows - 1, max(0, int(y // cell_h)))
            cells.add((gx, gy))
        count = len(cells)
        return count, count / float(self.cfg.grid_rows * self.cfg.grid_cols)

    def _transform_to_original(self, transform: np.ndarray) -> np.ndarray:
        if abs(self.cfg.work_scale - 1.0) <= 1e-6:
            return transform.copy()
        s = np.array(
            [
                [self.cfg.work_scale, 0.0, 0.0],
                [0.0, self.cfg.work_scale, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        h_original = np.linalg.inv(s) @ to_homogeneous(transform) @ s
        return from_homogeneous(h_original, self.cfg.model)

    def _motion_stats(self, transform: np.ndarray) -> MotionStats:
        h = to_homogeneous(transform)
        dx = float(h[0, 2])
        dy = float(h[1, 2])
        a = float(h[0, 0])
        b = float(h[1, 0])
        scale = math.sqrt(max(a * a + b * b, 1e-9))
        rotation = math.degrees(math.atan2(b, a))
        corners = np.array(
            [
                [0.0, 0.0],
                [self.width - 1.0, 0.0],
                [self.width - 1.0, self.height - 1.0],
                [0.0, self.height - 1.0],
            ],
            dtype=np.float32,
        )
        warped = transform_points(corners, transform)
        max_corner_disp = float(np.linalg.norm(warped - corners, axis=1).max())
        return MotionStats(
            dx=dx,
            dy=dy,
            scale=scale,
            rotation_deg=rotation,
            max_corner_displacement=max_corner_disp,
        )

    def _evaluate(
        self,
        transform: np.ndarray,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        inliers: np.ndarray,
        num_detected: int,
    ) -> Confidence:
        num_tracked = int(len(prev_pts))
        num_inliers = int(inliers.sum())
        inlier_ratio = num_inliers / max(1, num_tracked)
        predicted = transform_points(prev_pts, transform)
        residuals = (
            np.linalg.norm(predicted - curr_pts, axis=1)
            if num_tracked
            else np.array([], dtype=np.float32)
        )
        median_residual = float(np.median(residuals[inliers])) if num_inliers else float("inf")
        coverage_cells, coverage_ratio = self._coverage_cells(
            prev_pts[inliers] if num_inliers else np.empty((0, 2))
        )
        stats = self._motion_stats(transform)

        checks = [
            (num_tracked >= self.cfg.min_inliers, f"too_few_tracks:{num_tracked}"),
            (num_inliers >= self.cfg.min_inliers, f"too_few_inliers:{num_inliers}"),
            (inlier_ratio >= self.cfg.min_inlier_ratio, f"low_inlier_ratio:{inlier_ratio:.3f}"),
            (median_residual <= self.cfg.max_median_residual, f"high_residual:{median_residual:.3f}"),
            (coverage_cells >= self.cfg.min_coverage_cells, f"low_coverage:{coverage_cells}"),
        ]
        if self.cfg.model == "homography":
            max_allowed = self.cfg.max_translation_frac * math.hypot(self.width, self.height)
            checks.append(
                (
                    stats.max_corner_displacement <= max_allowed,
                    f"large_corner_warp:{stats.max_corner_displacement:.2f}",
                )
            )
        else:
            max_translation = self.cfg.max_translation_frac * max(self.width, self.height)
            checks.extend(
                [
                    (
                        abs(stats.dx) <= max_translation and abs(stats.dy) <= max_translation,
                        f"large_translation:{stats.dx:.1f},{stats.dy:.1f}",
                    ),
                    (self.cfg.min_scale <= stats.scale <= self.cfg.max_scale, f"bad_scale:{stats.scale:.3f}"),
                    (abs(stats.rotation_deg) <= self.cfg.max_rotation_deg, f"large_rotation:{stats.rotation_deg:.2f}"),
                ]
            )

        valid = all(ok for ok, _ in checks)
        reason = "ok" if valid else ";".join(reason for ok, reason in checks if not ok)
        return Confidence(
            valid=valid,
            reason=reason,
            num_detected=int(num_detected),
            num_tracked=num_tracked,
            num_inliers=num_inliers,
            inlier_ratio=float(inlier_ratio),
            median_residual=median_residual,
            coverage_cells=coverage_cells,
            coverage_ratio=float(coverage_ratio),
            dx=float(stats.dx),
            dy=float(stats.dy),
            scale=float(stats.scale),
            rotation_deg=float(stats.rotation_deg),
            max_corner_displacement=float(stats.max_corner_displacement),
        )

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    def _dynamic_residual_mask(
        self,
        curr_gray: np.ndarray,
        transform: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> np.ndarray:
        aligned_prev = warp_gray(self.prev_gray, transform, (self.width, self.height))
        residual = cv2.absdiff(curr_gray, aligned_prev)
        residual = cv2.GaussianBlur(residual, (5, 5), 0)
        valid_pixels = residual[ignore_mask == 0]
        if valid_pixels.size == 0:
            return np.zeros_like(residual)

        median = float(np.median(valid_pixels))
        mad = float(np.median(np.abs(valid_pixels.astype(np.float32) - median)))
        robust_sigma = mad / 0.6745 if mad > 1e-6 else float(np.std(valid_pixels))
        threshold = max(
            float(self.cfg.residual_min_threshold),
            median + self.cfg.residual_mad_multiplier * robust_sigma,
        )
        mask = np.zeros_like(residual, dtype=np.uint8)
        mask[(residual >= threshold) & (ignore_mask == 0)] = 255
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=1)
        mask = self._remove_small_components(mask, self.cfg.residual_min_area)

        if self.cfg.dynamic_dilate > 1:
            k = self.cfg.dynamic_dilate
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        fraction = cv2.countNonZero(mask) / float(mask.size)
        if fraction > self.cfg.residual_max_fraction:
            return np.zeros_like(mask)
        return mask

    def step(
        self,
        current_frame: np.ndarray,
        prev_target_bbox: Optional[BBox] = None,
    ) -> StepResult:
        start = time.perf_counter()
        self._frame_idx += 1
        curr_gray = self._preprocess(current_frame)
        prev_bbox_mask = self._bbox_ignore_mask(prev_target_bbox)
        ignore_mask = cv2.bitwise_or(prev_bbox_mask, self.dynamic_mask)
        valid_mask = cv2.bitwise_not(ignore_mask)

        used_redetect = self._should_redetect()
        if used_redetect:
            detected = self._detect_grid_corners(valid_mask).reshape(-1, 2)
        else:
            detected = self._cached_prev_points.copy()

        detected = self._cap_points(detected)
        prev_pts, curr_pts = self._track_points(curr_gray, detected)
        if (not used_redetect) and len(prev_pts) < max(4, int(self.cfg.min_points_to_redetect)):
            detected = self._cap_points(self._detect_grid_corners(valid_mask).reshape(-1, 2))
            prev_pts, curr_pts = self._track_points(curr_gray, detected)

        candidate_transform, inliers = self._estimate_transform(prev_pts, curr_pts)
        confidence = self._evaluate(
            candidate_transform, prev_pts, curr_pts, inliers, int(len(detected))
        )

        used_fallback = False
        used_transform = candidate_transform
        if not confidence.valid:
            used_fallback = True
            used_transform = self.last_transform.copy()
        else:
            self.last_transform = candidate_transform.copy()

        used_transform_original = self._transform_to_original(used_transform)
        if self.cfg.enable_dynamic_mask:
            residual_interval = max(1, int(self.cfg.residual_update_interval))
            if (self._frame_idx % residual_interval) == 0:
                self.dynamic_mask = self._dynamic_residual_mask(
                    curr_gray, used_transform, prev_bbox_mask
                )
        else:
            self.dynamic_mask = np.zeros_like(curr_gray, dtype=np.uint8)

        if len(curr_pts) > 0:
            carry_pts = curr_pts
            if len(inliers) == len(curr_pts):
                inlier_pts = curr_pts[inliers.astype(bool)]
                if len(inlier_pts) >= 4:
                    carry_pts = inlier_pts
            self._cached_prev_points = self._cap_points(carry_pts)
        else:
            self._cached_prev_points = np.empty((0, 2), dtype=np.float32)
        self.prev_gray = curr_gray

        runtime_ms = (time.perf_counter() - start) * 1000.0
        return StepResult(
            transform_original=used_transform_original,
            confidence=confidence,
            used_fallback=used_fallback,
            runtime_ms=float(runtime_ms),
        )
