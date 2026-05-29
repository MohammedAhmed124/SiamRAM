"""Telescope-style tiny-object tracker wrapper using foveated I/O only."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from experiments.tiny_objects.foveation import FoveationConfig, HyperbolicFoveator, bbox_center_xy


@dataclass
class TinyObjectRecord:
    frame: int
    pred_bbox: list[float]
    score: float
    runtime_ms: float
    tiny_focus_bbox: list[float]
    tiny_center_xy: list[float]
    tiny_raw_radius_norm: float
    tiny_radius_px: float
    tiny_radius_norm: float
    tiny_radius_ema_alpha: float
    tiny_alpha: float
    tiny_blend_exponent: float
    tiny_foveated_prior_bbox: list[float]
    tiny_foveated_pred_bbox: list[float]
    tiny_predicted_gain: float
    tiny_roundtrip_error_px: float
    tiny_prior_set: bool
    tiny_update_accepted: bool
    tiny_raw_pred_invalid: bool
    tiny_tracker_in_occlusion: bool
    dynamic_template_enabled: bool
    dynamic_template_refreshed: bool
    dynamic_template_best_score: Optional[float]
    dynamic_template_threshold: Optional[float]
    dynamic_template_memory_size: int
    dynamic_template_update_period: Optional[int]
    dynamic_template_idx: Optional[int]
    dynamic_template_bbox: Optional[list[float]]


def _as_bbox_list(bbox: Any) -> list[float]:
    arr = np.asarray(bbox, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def clip_bbox_to_frame(bbox: Any, frame_shape: tuple[int, int], as_int: bool = False) -> np.ndarray:
    x, y, w, h = _as_bbox_list(bbox)
    height, width = frame_shape[:2]
    w = max(1.0, min(w, float(width)))
    h = max(1.0, min(h, float(height)))
    x = min(max(0.0, x), max(0.0, float(width) - w))
    y = min(max(0.0, y), max(0.0, float(height) - h))
    arr = np.array([x, y, w, h], dtype=float)
    return np.round(arr).astype(int) if as_int else arr


def _innermost_visual_tracker(tracker: Any) -> Any:
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


def _set_search_prior(tracker: Any, prior_bbox: np.ndarray) -> bool:
    visual_tracker = _innermost_visual_tracker(tracker)
    if visual_tracker is None:
        return False
    prior_for_tracker = np.asarray(prior_bbox, dtype=float)
    frame_scale = getattr(tracker, "_frame_scale", 1.0)
    if frame_scale != 1.0 and getattr(tracker, "tracker", None) is visual_tracker:
        prior_for_tracker = prior_for_tracker * float(frame_scale)
    visual_tracker.tracking_state.bbox = prior_for_tracker.copy()
    if hasattr(visual_tracker, "_ot"):
        visual_tracker._ot.state = list(prior_for_tracker.astype(float))
    return True


def _state_bbox_to_frame_bbox(tracker: Any, bbox: Any) -> Any:
    if bbox is None:
        return None
    frame_scale = getattr(tracker, "_frame_scale", 1.0)
    if frame_scale != 1.0:
        return np.asarray(bbox, dtype=float) / float(frame_scale)
    return bbox


def _dynamic_template_snapshot(tracker: Any) -> dict[str, Any]:
    visual_tracker = _innermost_visual_tracker(tracker)
    if visual_tracker is None:
        return {
            "enabled": False,
            "best_score": None,
            "threshold": None,
            "memory_size": 0,
            "period": None,
            "idx": None,
            "bbox": None,
            "image": None,
        }
    bbox = _state_bbox_to_frame_bbox(tracker, getattr(visual_tracker, "running_dynamic_bbox", None))
    image = getattr(visual_tracker, "running_dynamic_image", None)
    scores = getattr(visual_tracker, "classification_scores", None)
    return {
        "enabled": bool(getattr(visual_tracker, "dynamic_update", False)),
        "best_score": getattr(visual_tracker, "_best_score", None),
        "threshold": getattr(visual_tracker, "dynamic_update_threshold", None),
        "memory_size": len(scores) if scores is not None else 0,
        "period": getattr(visual_tracker, "N", None),
        "idx": getattr(visual_tracker, "idx", None),
        "bbox": None if bbox is None else np.asarray(bbox, dtype=float).reshape(-1)[:4],
        "image": image,
    }


def _dynamic_template_refreshed(snapshot: dict[str, Any]) -> bool:
    if not snapshot["enabled"]:
        return False
    period = snapshot["period"]
    idx = snapshot["idx"]
    best_score = snapshot["best_score"]
    threshold = snapshot["threshold"]
    if period is None or idx is None or int(period) <= 0:
        return False
    if snapshot["memory_size"] <= 0 or best_score is None or threshold is None:
        return False
    return int(idx) > 0 and int(idx) % int(period) == 0 and float(best_score) >= float(threshold)


class TinyObjectTracker:
    """Wrap a black-box SOT tracker with hyperbolic foveated frames and boxes.
    Includes Dynamic Foveal Widening for occlusion/distractor recovery.
    """

    def __init__(
        self,
        tracker: Any,
        *,
        foveation_config: Optional[FoveationConfig] = None,
        radius_ema_alpha: float = 0.2,
        min_update_score: float = 0.55,
    ) -> None:
        self.tracker = tracker
        self.foveator = HyperbolicFoveator(foveation_config)
        self.radius_ema_alpha = float(np.clip(radius_ema_alpha, 0.0, 1.0))
        self.min_update_score = float(min_update_score)
        self.ema_radius_norm: Optional[float] = None

        self.prev_euclidean_box: Optional[np.ndarray] = None
        self.frame_idx = 0

        # Occlusion Recovery states
        self.consecutive_failures = 0
        self.base_radius_beta = self.foveator.config.radius_beta
        self.current_radius_beta = self.base_radius_beta

        self.records: list[TinyObjectRecord] = []
        self.last_state = None
        self.last_foveated_frame: Optional[np.ndarray] = None
        self.last_focus_bbox: Optional[np.ndarray] = None
        self.last_foveated_prior: Optional[np.ndarray] = None
        self.last_foveated_pred: Optional[np.ndarray] = None
        self.last_dynamic_template: dict[str, Any] = _dynamic_template_snapshot(self.tracker)

    def __getattr__(self, name: str):
        return getattr(self.tracker, name)

    def _apply_radius_ema(self, state: Any, frame_shape: tuple[int, int]) -> tuple[Any, float]:
        raw_radius_norm = float(state.radius_norm)
        if self.ema_radius_norm is None:
            self.ema_radius_norm = raw_radius_norm
        else:
            a = self.radius_ema_alpha
            self.ema_radius_norm = a * raw_radius_norm + (1.0 - a) * float(self.ema_radius_norm)
        state.radius_norm = float(self.ema_radius_norm)
        height, width = frame_shape[:2]
        state.radius_px = 0.5 * state.radius_norm * max(width - 1, height - 1)
        return state, raw_radius_norm

    def initialize(self, frame: np.ndarray, bbox: Any) -> None:
        init_bbox = clip_bbox_to_frame(bbox, frame.shape[:2], as_int=False)
        self.ema_radius_norm = None
        self.consecutive_failures = 0
        self.current_radius_beta = self.base_radius_beta
        self.foveator.config.radius_beta = self.base_radius_beta

        state = self.foveator.state_from_bbox(init_bbox, frame.shape[:2])
        state, _ = self._apply_radius_ema(state, frame.shape[:2])
        foveated_frame = self.foveator.warp_image(frame, state)

        # SOT initialization requires integers
        foveated_bbox_int = clip_bbox_to_frame(self.foveator.warp_bbox(init_bbox, state), frame.shape[:2], as_int=True)
        self.tracker.initialize(foveated_frame, foveated_bbox_int)

        self.prev_euclidean_box = init_bbox.copy()
        self.frame_idx = 0
        self.records.clear()
        self.last_state = state
        self.last_foveated_frame = foveated_frame
        self.last_focus_bbox = init_bbox.copy()

        foveated_bbox_float = clip_bbox_to_frame(self.foveator.warp_bbox(init_bbox, state), frame.shape[:2], as_int=False)
        self.last_foveated_prior = foveated_bbox_float.copy()
        self.last_foveated_pred = foveated_bbox_float.copy()
        self.last_dynamic_template = _dynamic_template_snapshot(self.tracker)

    def update(self, frame: np.ndarray):
        if self.prev_euclidean_box is None:
            raise RuntimeError("TinyObjectTracker is not initialized. Call initialize() first.")

        start = time.perf_counter()
        self.frame_idx += 1

        # 1. DYNAMIC FOVEAL WIDENING ON UNCERTAINTY
        # If the tracker lost confidence last frame, smoothly expand the foveal sweet-spot
        # so the SOT can see a broader, uncompressed context to re-acquire the target.
        target_radius_beta = self.base_radius_beta * (1.0 + 0.35 * min(self.consecutive_failures, 10))
        self.current_radius_beta = 0.8 * self.current_radius_beta + 0.2 * target_radius_beta
        self.foveator.config.radius_beta = self.current_radius_beta

        # Focus is ALWAYS exactly where the tracker last predicted. We never freeze it.
        focus_bbox = self.prev_euclidean_box.copy()
        self.last_focus_bbox = focus_bbox.copy()
        current_center = bbox_center_xy(focus_bbox)

        # 2. FOVEATE
        state = self.foveator.state_from_bbox(focus_bbox, frame.shape[:2])
        state, raw_radius_norm = self._apply_radius_ema(state, frame.shape[:2])
        foveated_frame = self.foveator.warp_image(frame, state)

        foveated_prior = clip_bbox_to_frame(self.foveator.warp_bbox(focus_bbox, state), frame.shape[:2], as_int=False)
        prior_set = _set_search_prior(self.tracker, foveated_prior)

        # 3. TRACKER EXECUTION
        result = self.tracker.update(foveated_frame)
        dynamic_template = _dynamic_template_snapshot(self.tracker)
        dynamic_refreshed = _dynamic_template_refreshed(dynamic_template)
        score = float(result[1]) if len(result) > 1 else 0.0
        tracker_in_occlusion = bool(result[2]) if len(result) > 2 else False

        raw_pred = np.asarray(result[0], dtype=float).reshape(-1)
        raw_pred_invalid = (
            raw_pred.size < 4
            or not np.all(np.isfinite(raw_pred[:4]))
            or raw_pred[2] <= 0.0
            or raw_pred[3] <= 0.0
        )

        # 4. RESOLVE PREDICTION
        if raw_pred_invalid:
            # If SOT outputs garbage, trust its internal memory/coasting state
            state_box = getattr(self.tracker, "held_box", None)
            if state_box is None:
                state_box = getattr(self.tracker, "current_bbox", None)
            state_box = _state_bbox_to_frame_bbox(self.tracker, state_box)
            foveated_pred = (
                clip_bbox_to_frame(state_box, frame.shape[:2], as_int=False)
                if state_box is not None
                else foveated_prior.copy()
            )
        else:
            foveated_pred = clip_bbox_to_frame(raw_pred, frame.shape[:2], as_int=False)

        # 5. UNWARP TO EUCLIDEAN
        euclidean_pred = clip_bbox_to_frame(self.foveator.unwarp_bbox(foveated_pred, state), frame.shape[:2], as_int=False)

        # 6. UPDATE STATES
        # WE ALWAYS FOLLOW THE TRACKER. If it coasts internally, we coast. 
        # If it tracks, we track. This preserves geometric synchronization.
        self.prev_euclidean_box = euclidean_pred.copy()
        
        # If confidence is low, trigger Foveal Widening for the NEXT frame.
        update_accepted = (not raw_pred_invalid) and (not tracker_in_occlusion) and score >= self.min_update_score
        
        if update_accepted:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        self.last_state = state
        self.last_foveated_frame = foveated_frame
        self.last_foveated_prior = foveated_prior.copy()
        self.last_foveated_pred = foveated_pred.copy()
        self.last_dynamic_template = dynamic_template

        # Log construction
        original_diag = max(float(np.hypot(focus_bbox[2], focus_bbox[3])), 1.0)
        foveated_diag = float(np.hypot(foveated_prior[2], foveated_prior[3]))
        roundtrip_error = self.foveator.roundtrip_error_px(current_center.reshape(1, 2), state)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        self.records.append(
            TinyObjectRecord(
                frame=self.frame_idx,
                pred_bbox=_as_bbox_list(euclidean_pred),
                score=score,
                runtime_ms=float(runtime_ms),
                tiny_focus_bbox=_as_bbox_list(focus_bbox),
                tiny_center_xy=[float(v) for v in state.center_xy],
                tiny_raw_radius_norm=float(raw_radius_norm),
                tiny_radius_px=float(state.radius_px),
                tiny_radius_norm=float(state.radius_norm),
                tiny_radius_ema_alpha=float(self.radius_ema_alpha),
                tiny_alpha=float(state.alpha),
                tiny_blend_exponent=float(state.blend_exponent),
                tiny_foveated_prior_bbox=_as_bbox_list(foveated_prior),
                tiny_foveated_pred_bbox=_as_bbox_list(foveated_pred),
                tiny_predicted_gain=float(foveated_diag / original_diag),
                tiny_roundtrip_error_px=float(roundtrip_error),
                tiny_prior_set=bool(prior_set),
                tiny_update_accepted=bool(update_accepted),
                tiny_raw_pred_invalid=bool(raw_pred_invalid),
                tiny_tracker_in_occlusion=bool(tracker_in_occlusion),
                dynamic_template_enabled=bool(dynamic_template["enabled"]),
                dynamic_template_refreshed=bool(dynamic_refreshed),
                dynamic_template_best_score=(
                    None if dynamic_template["best_score"] is None else float(dynamic_template["best_score"])
                ),
                dynamic_template_threshold=(
                    None if dynamic_template["threshold"] is None else float(dynamic_template["threshold"])
                ),
                dynamic_template_memory_size=int(dynamic_template["memory_size"]),
                dynamic_template_update_period=(
                    None if dynamic_template["period"] is None else int(dynamic_template["period"])
                ),
                dynamic_template_idx=None if dynamic_template["idx"] is None else int(dynamic_template["idx"]),
                dynamic_template_bbox=(
                    None if dynamic_template["bbox"] is None else _as_bbox_list(dynamic_template["bbox"])
                ),
            )
        )
        return euclidean_pred, score

    def records_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]