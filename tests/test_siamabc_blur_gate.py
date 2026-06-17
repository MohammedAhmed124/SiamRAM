from __future__ import annotations

from collections import deque

import numpy as np

from models.SiamABC.tracker.SiamABC_Tracker import SiamABCTracker


def _tracker_with_blur_gate(
    *,
    ref: float = 100.0,
    ratio: float = 1.25,
    alpha: float = 0.0,
) -> SiamABCTracker:
    tracker = SiamABCTracker.__new__(SiamABCTracker)
    tracker._template_sharpness_ema = ref
    tracker._blur_gate_ratio = ratio
    tracker._blur_gate_ema_alpha = alpha
    tracker._blur_gate_update_ema_on_selected_only = False
    return tracker


def test_blur_gate_ratio_sets_minimum_sharpness_only() -> None:
    tracker = _tracker_with_blur_gate()

    assert tracker._blur_admit_ok(79.9) is False
    assert tracker._blur_admit_ok(80.0) is True
    assert tracker._blur_admit_ok(250.0) is True


def test_blur_gate_preserves_fractional_ratio_floor() -> None:
    tracker = _tracker_with_blur_gate(ratio=0.6)

    assert tracker._blur_admit_ok(59.9) is False
    assert tracker._blur_admit_ok(60.0) is True
    assert tracker._blur_admit_ok(200.0) is True


def test_blur_gate_can_defer_ema_update_to_selected_templates() -> None:
    tracker = _tracker_with_blur_gate(alpha=1.0)
    tracker._blur_gate_update_ema_on_selected_only = True

    assert tracker._blur_admit_ok(120.0) is True
    assert tracker._template_sharpness_ema == 100.0


def test_selected_template_updates_blur_gate_ema_when_configured() -> None:
    tracker = _tracker_with_blur_gate(alpha=1.0)
    tracker._blur_gate_enabled = True
    tracker._blur_gate_update_ema_on_selected_only = True
    tracker.dynamic_update_threshold = 0.83
    tracker.classification_scores = deque([0.9])
    tracker.template_drm_scores = deque([float("nan")])
    tracker.template_sharpness_scores = deque([140.0])
    tracker.all_memory_imgs = deque(
        [[np.zeros((8, 8, 3), dtype=np.uint8), np.array([1, 1, 4, 4])]]
    )
    tracker.get_template_features = lambda image, bbox: "template"
    tracker.get_search_features = lambda image, bbox: ("search", None, None)
    tracker.net = object()

    tracker.select_representatives()

    assert tracker._template_sharpness_ema == 140.0
