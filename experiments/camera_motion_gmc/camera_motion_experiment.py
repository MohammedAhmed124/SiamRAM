"""
Lightweight camera-motion estimation experiment for AIC-4 aerial videos.

Pipeline:
  expanded bbox exclusion mask
  -> grid-distributed Shi-Tomasi background corners
  -> pyramidal LK tracking with forward/backward filtering
  -> RANSAC motion estimation
  -> confidence gating
  -> optional residual dynamic mask

The runner writes complete per-frame diagnostics both as CSV/JSONL and as
overlays on the annotated inference video frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

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
    candidate_transform: np.ndarray
    transform: np.ndarray
    candidate_transform_original: np.ndarray
    transform_original: np.ndarray
    confidence: Confidence
    candidate_stats_original: MotionStats
    applied_stats_work: MotionStats
    applied_stats_original: MotionStats
    dynamic_mask: np.ndarray
    residual_mask: np.ndarray
    detected_points: np.ndarray
    prev_points: np.ndarray
    curr_points: np.ndarray
    inliers: np.ndarray
    used_fallback: bool
    residual_threshold: float
    dynamic_fraction: float
    runtime_ms: float
    protected_curr_bbox: Optional[BBox] = None


def parse_bbox_line(line: str) -> BBox:
    parts = [float(part.strip()) for part in line.replace("\t", ",").split(",") if part.strip()]
    if len(parts) < 4:
        raise ValueError(f"Expected at least 4 bbox values, got: {line!r}")
    return parts[0], parts[1], parts[2], parts[3]


def load_annotations(path: Path) -> list[BBox]:
    boxes: list[BBox] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                boxes.append(parse_bbox_line(line))
    if not boxes:
        raise ValueError(f"No bboxes found in {path}")
    return boxes


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    x, y, w, h = bbox
    return x * scale, y * scale, w * scale, h * scale


def pad_bbox(bbox: BBox, pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    px = w * pad
    py = h * pad
    x1 = max(0, int(round(x - px)))
    y1 = max(0, int(round(y - py)))
    x2 = min(width, int(round(x + w + px)))
    y2 = min(height, int(round(y + h + py)))
    return x1, y1, x2, y2


def matrix_to_row(matrix: np.ndarray) -> dict[str, float]:
    h = np.eye(3, dtype=np.float64)
    if matrix.shape == (2, 3):
        h[:2, :] = matrix
    else:
        h = matrix.astype(np.float64)
    return {
        "m00": float(h[0, 0]),
        "m01": float(h[0, 1]),
        "m02": float(h[0, 2]),
        "m10": float(h[1, 0]),
        "m11": float(h[1, 1]),
        "m12": float(h[1, 2]),
        "m20": float(h[2, 0]),
        "m21": float(h[2, 1]),
        "m22": float(h[2, 2]),
    }


def prefixed_matrix_to_row(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in matrix_to_row(matrix).items()}


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


def warp_bbox(bbox: BBox, transform: np.ndarray) -> BBox:
    x, y, w, h = bbox
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )
    warped = transform_points(corners, transform)
    x1 = float(np.min(warped[:, 0]))
    y1 = float(np.min(warped[:, 1]))
    x2 = float(np.max(warped[:, 0]))
    y2 = float(np.max(warped[:, 1]))
    return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)


def warp_gray(previous: np.ndarray, transform: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if transform.shape == (3, 3):
        return cv2.warpPerspective(previous, transform, (width, height), flags=cv2.INTER_LINEAR)
    return cv2.warpAffine(previous, transform, (width, height), flags=cv2.INTER_LINEAR)


def draw_text_panel(frame: np.ndarray, lines: Iterable[str]) -> None:
    lines = list(lines)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    pad = 10
    line_h = 18
    width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        width = max(width, tw)
    panel_w = min(frame.shape[1] - 2 * pad, width + 2 * pad)
    panel_h = min(frame.shape[0] - 2 * pad, len(lines) * line_h + 2 * pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 20, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = 8 + pad + 4
    for line in lines:
        cv2.putText(frame, line, (8 + pad, y), font, scale, (240, 250, 245), thickness, cv2.LINE_AA)
        y += line_h


class CameraMotionEstimator:
    def __init__(self, first_frame: np.ndarray, initial_bbox: BBox, config: MotionConfig):
        self.cfg = config
        self.frame_shape_original = first_frame.shape[:2]
        self.prev_gray = self._preprocess(first_frame)
        self.height, self.width = self.prev_gray.shape[:2]
        self.last_transform = self._identity_transform()
        self.dynamic_mask = np.zeros_like(self.prev_gray, dtype=np.uint8)

    def _identity_transform(self) -> np.ndarray:
        if self.cfg.model == "homography":
            return np.eye(3, dtype=np.float64)
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if abs(self.cfg.work_scale - 1.0) > 1e-6:
            gray = cv2.resize(gray, None, fx=self.cfg.work_scale, fy=self.cfg.work_scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _bbox_ignore_mask(self, bbox: Optional[BBox]) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if bbox is None:
            return mask
        x1, y1, x2, y2 = pad_bbox(scale_bbox(bbox, self.cfg.work_scale), self.cfg.bbox_pad, self.width, self.height)
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
                cell_mask = np.zeros_like(valid_mask)
                cell_mask[y1:y2, x1:x2] = valid_mask[y1:y2, x1:x2]
                if cv2.countNonZero(cell_mask) < 20:
                    continue
                cell_points = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=self.cfg.max_corners_per_cell,
                    qualityLevel=self.cfg.quality_level,
                    minDistance=self.cfg.min_distance,
                    mask=cell_mask,
                    blockSize=self.cfg.block_size,
                    useHarrisDetector=False,
                )
                if cell_points is not None:
                    points.append(cell_points)
        if not points:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.vstack(points).astype(np.float32)

    def _track_points(self, curr_gray: np.ndarray, points_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points_prev) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=bool)

        lk_params = dict(
            winSize=(self.cfg.lk_win_size, self.cfg.lk_win_size),
            maxLevel=self.cfg.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        points_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, points_prev, None, **lk_params)
        if points_curr is None or status_fwd is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=bool)
        points_back, status_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, points_curr, None, **lk_params)
        if points_back is None or status_back is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.zeros((0,), dtype=bool)

        prev_flat = points_prev.reshape(-1, 2)
        curr_flat = points_curr.reshape(-1, 2)
        back_flat = points_back.reshape(-1, 2)
        fb_error = np.linalg.norm(prev_flat - back_flat, axis=1)
        valid = (
            (status_fwd.reshape(-1) == 1)
            & (status_back.reshape(-1) == 1)
            & np.isfinite(curr_flat).all(axis=1)
            & (fb_error <= self.cfg.fb_max_error)
        )
        return prev_flat[valid], curr_flat[valid], valid

    def _estimate_transform(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            [[self.cfg.work_scale, 0.0, 0.0], [0.0, self.cfg.work_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        h_original = np.linalg.inv(s) @ to_homogeneous(transform) @ s
        return from_homogeneous(h_original, self.cfg.model)

    def _motion_stats(self, transform: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> MotionStats:
        width = self.width if width is None else width
        height = self.height if height is None else height
        h = to_homogeneous(transform)
        dx = float(h[0, 2])
        dy = float(h[1, 2])
        a = float(h[0, 0])
        b = float(h[1, 0])
        scale = math.sqrt(max(a * a + b * b, 1e-9))
        rotation = math.degrees(math.atan2(b, a))
        corners = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        warped = transform_points(corners, transform)
        max_corner_disp = float(np.linalg.norm(warped - corners, axis=1).max())
        return MotionStats(dx=dx, dy=dy, scale=scale, rotation_deg=rotation, max_corner_displacement=max_corner_disp)

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
        residuals = np.linalg.norm(predicted - curr_pts, axis=1) if num_tracked else np.array([], dtype=np.float32)
        median_residual = float(np.median(residuals[inliers])) if num_inliers else float("inf")
        coverage_cells, coverage_ratio = self._coverage_cells(prev_pts[inliers] if num_inliers else np.empty((0, 2)))
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
            checks.append((stats.max_corner_displacement <= max_allowed, f"large_corner_warp:{stats.max_corner_displacement:.2f}"))
        else:
            max_translation = self.cfg.max_translation_frac * max(self.width, self.height)
            checks.extend(
                [
                    (abs(stats.dx) <= max_translation and abs(stats.dy) <= max_translation, f"large_translation:{stats.dx:.1f},{stats.dy:.1f}"),
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

    def _dynamic_residual_mask(
        self,
        curr_gray: np.ndarray,
        transform: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        aligned_prev = warp_gray(self.prev_gray, transform, (self.width, self.height))
        residual = cv2.absdiff(curr_gray, aligned_prev)
        residual = cv2.GaussianBlur(residual, (5, 5), 0)
        valid_pixels = residual[ignore_mask == 0]
        if valid_pixels.size == 0:
            return np.zeros_like(residual), float(self.cfg.residual_min_threshold), 0.0
        median = float(np.median(valid_pixels))
        mad = float(np.median(np.abs(valid_pixels.astype(np.float32) - median)))
        robust_sigma = mad / 0.6745 if mad > 1e-6 else float(np.std(valid_pixels))
        threshold = max(float(self.cfg.residual_min_threshold), median + self.cfg.residual_mad_multiplier * robust_sigma)
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
            return np.zeros_like(mask), threshold, fraction
        return mask, threshold, fraction

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    def step(
        self,
        current_frame: np.ndarray,
        prev_target_bbox: Optional[BBox] = None,
        curr_target_bbox: Optional[BBox] = None,
    ) -> StepResult:
        start = time.perf_counter()
        curr_gray = self._preprocess(current_frame)
        prev_bbox_mask = self._bbox_ignore_mask(prev_target_bbox)
        ignore_mask = cv2.bitwise_or(prev_bbox_mask, self.dynamic_mask)
        valid_mask = cv2.bitwise_not(ignore_mask)

        detected = self._detect_grid_corners(valid_mask)
        prev_pts, curr_pts, _ = self._track_points(curr_gray, detected)
        candidate_transform, inliers = self._estimate_transform(prev_pts, curr_pts)
        confidence = self._evaluate(candidate_transform, prev_pts, curr_pts, inliers, len(detected))

        used_fallback = False
        used_transform = candidate_transform
        if not confidence.valid:
            used_fallback = True
            used_transform = self.last_transform.copy()
        else:
            self.last_transform = candidate_transform.copy()

        used_transform_original = self._transform_to_original(used_transform)
        candidate_transform_original = self._transform_to_original(candidate_transform)
        if curr_target_bbox is None and prev_target_bbox is not None:
            curr_target_bbox = warp_bbox(prev_target_bbox, used_transform_original)
        curr_bbox_mask = self._bbox_ignore_mask(curr_target_bbox)
        residual_mask, residual_threshold, dynamic_fraction = self._dynamic_residual_mask(curr_gray, used_transform, curr_bbox_mask)
        self.dynamic_mask = residual_mask
        self.prev_gray = curr_gray

        runtime_ms = (time.perf_counter() - start) * 1000.0
        return StepResult(
            candidate_transform=candidate_transform,
            transform=used_transform,
            candidate_transform_original=candidate_transform_original,
            transform_original=used_transform_original,
            confidence=confidence,
            candidate_stats_original=self._motion_stats(
                candidate_transform_original,
                width=self.frame_shape_original[1],
                height=self.frame_shape_original[0],
            ),
            applied_stats_work=self._motion_stats(used_transform),
            applied_stats_original=self._motion_stats(
                used_transform_original,
                width=self.frame_shape_original[1],
                height=self.frame_shape_original[0],
            ),
            dynamic_mask=self.dynamic_mask.copy(),
            residual_mask=residual_mask,
            detected_points=detected.reshape(-1, 2).copy(),
            prev_points=prev_pts.copy(),
            curr_points=curr_pts.copy(),
            inliers=inliers.copy(),
            used_fallback=used_fallback,
            residual_threshold=float(residual_threshold),
            dynamic_fraction=float(dynamic_fraction),
            runtime_ms=float(runtime_ms),
            protected_curr_bbox=curr_target_bbox,
        )


def annotate_frame(
    frame: np.ndarray,
    result: StepResult,
    frame_index: int,
    total_frames: int,
    prev_bbox: Optional[BBox],
    curr_bbox: Optional[BBox],
    output_scale: float,
    model: str,
    ignore_mode: str,
) -> np.ndarray:
    annotated = frame.copy()
    if prev_bbox is not None:
        x, y, w, h = prev_bbox
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (0, 220, 255), 2)
    if curr_bbox is not None:
        x, y, w, h = curr_bbox
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), (255, 220, 0), 2)

    if result.dynamic_mask.size:
        dyn = cv2.resize(result.dynamic_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(annotated)
        color[:, :, 2] = dyn
        annotated = cv2.addWeighted(annotated, 1.0, color, 0.22, 0)

    for i, (p0, p1) in enumerate(zip(result.prev_points, result.curr_points)):
        p0s = tuple(np.round(p0 / output_scale).astype(int))
        p1s = tuple(np.round(p1 / output_scale).astype(int))
        is_inlier = i < len(result.inliers) and bool(result.inliers[i])
        color = (80, 230, 80) if is_inlier else (55, 80, 255)
        cv2.circle(annotated, p1s, 2, color, -1, cv2.LINE_AA)
        if is_inlier:
            cv2.line(annotated, p0s, p1s, color, 1, cv2.LINE_AA)

    c = result.confidence
    status = "VALID" if c.valid and not result.used_fallback else "FALLBACK"
    applied = result.applied_stats_original
    candidate = result.candidate_stats_original
    lines = [
        f"frame {frame_index:04d}/{max(total_frames - 1, 0):04d} | model={model} | ignore={ignore_mode}",
        f"status={status} | reason={c.reason}",
        f"features detected={c.num_detected} tracked={c.num_tracked} inliers={c.num_inliers} ratio={c.inlier_ratio:.2f}",
        f"residual median={c.median_residual:.2f}px threshold={result.residual_threshold:.1f} dyn={100.0 * result.dynamic_fraction:.2f}%",
        f"coverage cells={c.coverage_cells} ({100.0 * c.coverage_ratio:.1f}%) runtime={result.runtime_ms:.1f}ms",
        f"applied/orig dx={applied.dx:.2f} dy={applied.dy:.2f} scale={applied.scale:.4f} rot={applied.rotation_deg:.2f}deg",
        f"candidate/orig dx={candidate.dx:.2f} dy={candidate.dy:.2f} scale={candidate.scale:.4f} rot={candidate.rotation_deg:.2f}deg",
        "green=inlier KLT tracks | red=outliers | yellow=prev bbox | cyan=current bbox | red tint=residual mask",
    ]
    draw_text_panel(annotated, lines)
    return annotated


def stats_to_row(prefix: str, stats: MotionStats) -> dict[str, float]:
    return {
        f"{prefix}_dx": float(stats.dx),
        f"{prefix}_dy": float(stats.dy),
        f"{prefix}_scale": float(stats.scale),
        f"{prefix}_rotation_deg": float(stats.rotation_deg),
        f"{prefix}_max_corner_displacement": float(stats.max_corner_displacement),
    }


def bbox_to_row(prefix: str, bbox: Optional[BBox]) -> dict[str, object]:
    return {
        f"{prefix}_x": None if bbox is None else bbox[0],
        f"{prefix}_y": None if bbox is None else bbox[1],
        f"{prefix}_w": None if bbox is None else bbox[2],
        f"{prefix}_h": None if bbox is None else bbox[3],
    }


def result_to_log_row(frame_index: int, result: StepResult, prev_bbox: Optional[BBox], curr_bbox: Optional[BBox]) -> dict[str, object]:
    c = result.confidence
    row: dict[str, object] = {
        "frame": frame_index,
        "valid": c.valid,
        "used_fallback": result.used_fallback,
        "reason": c.reason,
        "num_detected": c.num_detected,
        "num_tracked": c.num_tracked,
        "num_inliers": c.num_inliers,
        "inlier_ratio": c.inlier_ratio,
        "median_residual": c.median_residual,
        "coverage_cells": c.coverage_cells,
        "coverage_ratio": c.coverage_ratio,
        "candidate_work_dx": c.dx,
        "candidate_work_dy": c.dy,
        "candidate_work_scale": c.scale,
        "candidate_work_rotation_deg": c.rotation_deg,
        "candidate_work_max_corner_displacement": c.max_corner_displacement,
        "residual_threshold": result.residual_threshold,
        "dynamic_fraction": result.dynamic_fraction,
        "runtime_ms": result.runtime_ms,
        "matrix_coordinate_system": "original",
    }
    row.update(bbox_to_row("prev_bbox", prev_bbox))
    row.update(bbox_to_row("curr_bbox", curr_bbox))
    row.update(stats_to_row("candidate_original", result.candidate_stats_original))
    row.update(stats_to_row("applied_work", result.applied_stats_work))
    row.update(stats_to_row("applied_original", result.applied_stats_original))
    row.update(matrix_to_row(result.transform_original))
    row.update(prefixed_matrix_to_row("applied_work", result.transform))
    row.update(prefixed_matrix_to_row("applied_original", result.transform_original))
    row.update(prefixed_matrix_to_row("candidate_work", result.candidate_transform))
    row.update(prefixed_matrix_to_row("candidate_original", result.candidate_transform_original))
    for key, value in list(row.items()):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            row[key] = None
    return row


def find_video(sequence_dir: Path) -> Path:
    videos = sorted([p for p in sequence_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    if not videos:
        raise FileNotFoundError(f"No video file found in {sequence_dir}")
    return videos[0]


def make_output_dir(base: Path, sequence_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{sequence_name}_camera_motion_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def select_target_bboxes(
    mode: str,
    frame_index: int,
    annotations: list[BBox],
    initial_bbox: BBox,
) -> tuple[Optional[BBox], Optional[BBox]]:
    if mode == "none":
        return None, None
    if mode == "annotation":
        prev_bbox = annotations[frame_index - 1] if frame_index - 1 < len(annotations) else initial_bbox
        curr_bbox = annotations[frame_index] if frame_index < len(annotations) else None
        return prev_bbox, curr_bbox
    if mode == "annotation_prev":
        prev_bbox = annotations[frame_index - 1] if frame_index - 1 < len(annotations) else initial_bbox
        return prev_bbox, None
    if mode == "first_bbox":
        return initial_bbox, None
    raise ValueError(f"Unknown ignore mode: {mode}")


def run_sequence(args: argparse.Namespace) -> Path:
    sequence_dir = Path(args.sequence_dir)
    if not sequence_dir.exists():
        raise FileNotFoundError(sequence_dir)
    video_path = Path(args.video) if args.video else find_video(sequence_dir)
    annotation_path = Path(args.annotation) if args.annotation else sequence_dir / "annotation.txt"
    annotations = load_annotations(annotation_path) if annotation_path.exists() else []
    if not annotations and args.initial_bbox is None:
        raise ValueError("No annotations found. Pass --initial-bbox x,y,w,h.")
    initial_bbox = parse_bbox_line(args.initial_bbox) if args.initial_bbox else annotations[0]

    cfg = MotionConfig(
        model=args.model,
        work_scale=args.work_scale,
        bbox_pad=args.bbox_pad,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_median_residual=args.max_median_residual,
    )

    output_base = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "camera_motion_outputs"
    out_dir = make_output_dir(output_base, sequence_dir.name)
    frames_dir = out_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    annotated_video_path = out_dir / f"{sequence_dir.name}_camera_motion_annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {annotated_video_path}")

    estimator = CameraMotionEstimator(first_frame, initial_bbox, cfg)
    csv_path = out_dir / "frame_motion_log.csv"
    jsonl_path = out_dir / "frame_motion_log.jsonl"
    summary_path = out_dir / "summary.json"

    fieldnames: Optional[list[str]] = None
    valid_count = 0
    fallback_count = 0
    runtime_values: list[float] = []
    inlier_values: list[int] = []
    residual_values: list[float] = []
    processed_count = 1

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        csv_writer: Optional[csv.DictWriter] = None
        identity_work = estimator._identity_transform()
        identity_original = estimator._transform_to_original(identity_work)
        identity_stats_work = estimator._motion_stats(identity_work)
        identity_stats_original = estimator._motion_stats(identity_original, width=width, height=height)
        first_result = StepResult(
            candidate_transform=identity_work,
            transform=identity_work,
            candidate_transform_original=identity_original,
            transform_original=identity_original,
            confidence=Confidence(valid=True, reason="initial_frame"),
            candidate_stats_original=identity_stats_original,
            applied_stats_work=identity_stats_work,
            applied_stats_original=identity_stats_original,
            dynamic_mask=np.zeros(estimator.prev_gray.shape, dtype=np.uint8),
            residual_mask=np.zeros(estimator.prev_gray.shape, dtype=np.uint8),
            detected_points=np.empty((0, 2), dtype=np.float32),
            prev_points=np.empty((0, 2), dtype=np.float32),
            curr_points=np.empty((0, 2), dtype=np.float32),
            inliers=np.empty((0,), dtype=bool),
            used_fallback=False,
            residual_threshold=0.0,
            dynamic_fraction=0.0,
            runtime_ms=0.0,
            protected_curr_bbox=annotations[0] if annotations else initial_bbox,
        )
        first_bbox = annotations[0] if annotations else initial_bbox
        row = result_to_log_row(0, first_result, first_bbox, first_bbox)
        fieldnames = list(row.keys())
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerow(row)
        jsonl_file.write(json.dumps(row, allow_nan=False) + "\n")
        first_annotated = annotate_frame(first_frame, first_result, 0, total_frames, first_bbox, first_bbox, cfg.work_scale, cfg.model, args.ignore_mode)
        writer.write(first_annotated)
        if args.save_frames:
            cv2.imwrite(str(frames_dir / "frame_0000.jpg"), first_annotated)

        frame_index = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            prev_bbox, curr_bbox = select_target_bboxes(args.ignore_mode, frame_index, annotations, initial_bbox)

            result = estimator.step(frame, prev_target_bbox=prev_bbox, curr_target_bbox=curr_bbox)
            protected_curr_bbox = result.protected_curr_bbox
            row = result_to_log_row(frame_index, result, prev_bbox, protected_curr_bbox)
            csv_writer.writerow(row)
            jsonl_file.write(json.dumps(row, allow_nan=False) + "\n")

            annotated = annotate_frame(frame, result, frame_index, total_frames, prev_bbox, protected_curr_bbox, cfg.work_scale, cfg.model, args.ignore_mode)
            writer.write(annotated)
            if args.save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{frame_index:04d}.jpg"), annotated)

            valid_count += int(result.confidence.valid)
            fallback_count += int(result.used_fallback)
            runtime_values.append(result.runtime_ms)
            inlier_values.append(result.confidence.num_inliers)
            if math.isfinite(result.confidence.median_residual):
                residual_values.append(result.confidence.median_residual)

            processed_count += 1
            if args.max_frames and processed_count >= args.max_frames:
                break
            frame_index += 1

    cap.release()
    writer.release()

    summary = {
        "sequence_dir": str(sequence_dir),
        "video_path": str(video_path),
        "annotation_path": str(annotation_path) if annotation_path.exists() else None,
        "output_dir": str(out_dir),
        "annotated_video": str(annotated_video_path),
        "csv_log": str(csv_path),
        "jsonl_log": str(jsonl_path),
        "frames_dir": str(frames_dir) if args.save_frames else None,
        "frames_processed": processed_count,
        "video_total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "config": asdict(cfg),
        "ignore_mode": args.ignore_mode,
        "ignore_mode_kind": {
            "annotation": "oracle_previous_and_current_ground_truth_boxes",
            "annotation_prev": "tracker_like_previous_ground_truth_box_current_box_warped_from_camera_motion",
            "first_bbox": "initial_box_only_current_box_warped_from_camera_motion",
            "none": "no_target_masking",
        }[args.ignore_mode],
        "matrix_coordinate_system": "original_resolution_for_unprefixed_m00_to_m22",
        "valid_estimates": valid_count,
        "fallback_estimates": fallback_count,
        "valid_ratio_excluding_initial": valid_count / max(1, processed_count - 1),
        "fallback_ratio_excluding_initial": fallback_count / max(1, processed_count - 1),
        "mean_runtime_ms": float(np.mean(runtime_values)) if runtime_values else 0.0,
        "median_runtime_ms": float(np.median(runtime_values)) if runtime_values else 0.0,
        "median_inliers": float(np.median(inlier_values)) if inlier_values else 0.0,
        "median_residual": float(np.median(residual_values)) if residual_values else None,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run masked KLT + RANSAC camera-motion inference on an AIC video sequence.")
    parser.add_argument(
        "--sequence-dir",
        default=r"E:\Coding projects\AIC-4-Hackathon\contest_release\dataset2\Animal4",
        help="Directory containing video and annotation.txt.",
    )
    parser.add_argument("--video", default=None, help="Optional explicit video path.")
    parser.add_argument("--annotation", default=None, help="Optional explicit annotation path.")
    parser.add_argument("--initial-bbox", default=None, help="Initial bbox as x,y,w,h if no annotation file is available.")
    parser.add_argument("--output-dir", default=None, help="Base output directory. Defaults under this experiment folder.")
    parser.add_argument("--model", choices=["partial_affine", "full_affine", "homography"], default="partial_affine")
    parser.add_argument(
        "--ignore-mode",
        choices=["annotation", "annotation_prev", "first_bbox", "none"],
        default="annotation_prev",
        help=(
            "annotation=oracle prev/current GT boxes; annotation_prev=previous GT box only with current box predicted by camera motion; "
            "first_bbox=initial box only; none=no target masking."
        ),
    )
    parser.add_argument("--work-scale", type=float, default=0.75)
    parser.add_argument("--bbox-pad", type=float, default=0.50)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--max-median-residual", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional processing cap for quick tests. 0 means full video.")
    parser.add_argument("--save-frames", action="store_true", help="Also save every annotated inference frame as JPG.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_sequence(args)


if __name__ == "__main__":
    main()
