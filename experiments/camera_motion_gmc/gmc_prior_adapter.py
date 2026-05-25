"""Valid-only camera-motion search-prior adapter for tracker A/B tests.

This adapter intentionally does not use the residual-motion mask inside the
tracker. It only estimates the background transform from frame t-1 to frame t
and, when valid, shifts the wrapped tracker's search prior to the warped
previous prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from .camera_motion_experiment import CameraMotionEstimator, MotionConfig, warp_bbox


@dataclass
class GMCPriorRecord:
    frame: int
    used_camera_prior: bool
    reason: str
    prev_bbox: list[float]
    search_prior_bbox: list[float]
    pred_bbox: list[float]
    score: float
    motion_valid: bool
    motion_used_fallback: bool
    inlier_ratio: float
    median_residual: Optional[float]
    coverage_cells: int
    applied_dx: float
    applied_dy: float
    applied_scale: float
    applied_rotation_deg: float
    applied_max_corner_displacement: float
    motion_runtime_ms: float


def _as_bbox_list(bbox: Any) -> list[float]:
    arr = np.asarray(bbox, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def _clip_bbox_to_frame(bbox: Any, frame: np.ndarray) -> np.ndarray:
    x, y, w, h = _as_bbox_list(bbox)
    height, width = frame.shape[:2]
    w = max(1.0, min(w, float(width)))
    h = max(1.0, min(h, float(height)))
    x = min(max(0.0, x), max(0.0, float(width) - w))
    y = min(max(0.0, y), max(0.0, float(height) - h))
    return np.array([round(x), round(y), round(w), round(h)], dtype=int)


def _innermost_visual_tracker(tracker):
    """Return the ORTrackWrapper-like object whose search centre can be set."""
    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "tracking_state"):
        return tracker.tracker
    if hasattr(tracker, "base_tracker"):
        base = tracker.base_tracker
        if hasattr(base, "tracker") and hasattr(base.tracker, "tracking_state"):
            return base.tracker
        if hasattr(base, "tracking_state"):
            return base
    if hasattr(tracker, "tracking_state"):
        return tracker
    return None


def _set_search_prior(tracker, prior_bbox: np.ndarray) -> bool:
    visual_tracker = _innermost_visual_tracker(tracker)
    if visual_tracker is None:
        return False
    prior_for_tracker = np.asarray(prior_bbox, dtype=float)
    frame_scale = getattr(tracker, "_frame_scale", 1.0)
    if frame_scale != 1.0 and getattr(tracker, "tracker", None) is visual_tracker:
        prior_for_tracker = prior_for_tracker * float(frame_scale)
    prior_for_tracker = np.round(prior_for_tracker).astype(int)
    visual_tracker.tracking_state.bbox = prior_for_tracker.copy()
    if hasattr(visual_tracker, "_ot"):
        visual_tracker._ot.state = list(prior_for_tracker.astype(float))
    return True


def _state_bbox_to_frame_bbox(tracker, bbox: Any) -> Any:
    if bbox is None:
        return None
    frame_scale = getattr(tracker, "_frame_scale", 1.0)
    if frame_scale != 1.0:
        return np.asarray(bbox, dtype=float) / float(frame_scale)
    return bbox


class GMCPriorTracker:
    """Wrap an existing tracker and shift only its search prior when GMC is valid."""

    def __init__(
        self,
        tracker,
        motion_config: Optional[MotionConfig] = None,
        *,
        min_inlier_ratio: float = 0.35,
        min_coverage_cells: int = 6,
        use_fallback_prior: bool = False,
    ) -> None:
        self.tracker = tracker
        self.motion_config = motion_config or MotionConfig(model="partial_affine")
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_coverage_cells = int(min_coverage_cells)
        self.use_fallback_prior = bool(use_fallback_prior)
        self.motion_estimator: Optional[CameraMotionEstimator] = None
        self.prev_pred_box: Optional[np.ndarray] = None
        self.frame_idx = 0
        self.records: list[GMCPriorRecord] = []
        self.last_motion = None
        self.last_prev_bbox: Optional[np.ndarray] = None
        self.last_search_prior: Optional[np.ndarray] = None
        self.last_used_camera_prior = False

    def __getattr__(self, name: str):
        return getattr(self.tracker, name)

    def initialize(self, frame: np.ndarray, bbox) -> None:
        init_bbox = _clip_bbox_to_frame(bbox, frame)
        self.tracker.initialize(frame, init_bbox)
        self.motion_estimator = CameraMotionEstimator(
            first_frame=frame,
            initial_bbox=tuple(float(v) for v in init_bbox),
            config=self.motion_config,
        )
        self.prev_pred_box = init_bbox.copy()
        self.frame_idx = 0
        self.records.clear()
        self.last_motion = None
        self.last_prev_bbox = init_bbox.copy()
        self.last_search_prior = init_bbox.copy()
        self.last_used_camera_prior = False

    def _should_use_motion(self, motion) -> bool:
        if motion is None:
            return False
        if not motion.confidence.valid:
            return False
        if motion.used_fallback and not self.use_fallback_prior:
            return False
        if motion.confidence.inlier_ratio < self.min_inlier_ratio:
            return False
        if motion.confidence.coverage_cells < self.min_coverage_cells:
            return False
        return True

    def update(self, frame: np.ndarray):
        if self.motion_estimator is None or self.prev_pred_box is None:
            raise RuntimeError("GMCPriorTracker is not initialized. Call initialize() first.")

        self.frame_idx += 1
        prev_box = self.prev_pred_box.copy()
        motion = self.motion_estimator.step(
            current_frame=frame,
            prev_target_bbox=tuple(float(v) for v in prev_box),
            curr_target_bbox=None,
        )
        self.last_motion = motion
        self.last_prev_bbox = prev_box.copy()

        use_camera_prior = self._should_use_motion(motion)
        if use_camera_prior:
            prior_box = _clip_bbox_to_frame(warp_bbox(tuple(float(v) for v in prev_box), motion.transform_original), frame)
            prior_was_set = _set_search_prior(self.tracker, prior_box)
            if not prior_was_set:
                use_camera_prior = False
                prior_box = prev_box.copy()
        else:
            prior_box = prev_box.copy()

        self.last_search_prior = prior_box.copy()
        self.last_used_camera_prior = bool(use_camera_prior)

        result = self.tracker.update(frame)
        raw_pred = np.asarray(result[0], dtype=float).reshape(-1)
        pred_bbox = _clip_bbox_to_frame(raw_pred, frame)
        if raw_pred.size >= 4 and (raw_pred[2] <= 0 or raw_pred[3] <= 0):
            state_box = getattr(self.tracker, "current_bbox", None)
            if state_box is None:
                state_box = getattr(self.tracker, "held_box", None)
            state_box = _state_bbox_to_frame_bbox(self.tracker, state_box)
            self.prev_pred_box = _clip_bbox_to_frame(state_box, frame) if state_box is not None else prior_box.copy()
        else:
            self.prev_pred_box = pred_bbox.copy()

        stats = motion.applied_stats_original
        median_residual = motion.confidence.median_residual
        if not np.isfinite(median_residual):
            median_residual = None

        self.records.append(
            GMCPriorRecord(
                frame=self.frame_idx,
                used_camera_prior=bool(use_camera_prior),
                reason=motion.confidence.reason,
                prev_bbox=_as_bbox_list(prev_box),
                search_prior_bbox=_as_bbox_list(prior_box),
                pred_bbox=_as_bbox_list(pred_bbox),
                score=float(result[1]),
                motion_valid=bool(motion.confidence.valid),
                motion_used_fallback=bool(motion.used_fallback),
                inlier_ratio=float(motion.confidence.inlier_ratio),
                median_residual=None if median_residual is None else float(median_residual),
                coverage_cells=int(motion.confidence.coverage_cells),
                applied_dx=float(stats.dx),
                applied_dy=float(stats.dy),
                applied_scale=float(stats.scale),
                applied_rotation_deg=float(stats.rotation_deg),
                applied_max_corner_displacement=float(stats.max_corner_displacement),
                motion_runtime_ms=float(motion.runtime_ms),
            )
        )
        return result

    def records_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]
