"""
SAMURAI-style constant-velocity Kalman motion prior (arXiv:2411.11922, Sec. 4.1).

A small, dependency-free Kalman filter over the state
``[cx, cy, w, h, vx, vy, vw, vh]`` used to score response-map candidates by
motion consistency (KF-IoU) before peak selection. Noise scales follow the
standard MOT convention (ByteTrack/BoT-SORT): position/velocity standard
deviations proportional to the target height, so the filter adapts to target
scale without per-sequence tuning.

The filter is intentionally minimal: dense 8x8 numpy linear algebra once per
frame is microseconds of work, which keeps it viable on edge devices.
"""

from __future__ import annotations

import numpy as np


class KalmanMotionPrior:
    """Constant-velocity Kalman filter over an axis-aligned box."""

    _STD_POS = 1.0 / 20.0
    _STD_VEL = 1.0 / 160.0

    def __init__(self) -> None:
        ndim = 4
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = 1.0
        self._H = np.eye(ndim, 2 * ndim)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    @staticmethod
    def _to_cxcywh(bbox_xywh) -> np.ndarray:
        x, y, w, h = [float(v) for v in bbox_xywh[:4]]
        return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=float)

    def seed(self, bbox_xywh) -> None:
        """(Re)initialise the state at a box with zero velocity."""
        cx, cy, w, h = self._to_cxcywh(bbox_xywh)
        h_ref = max(h, 1.0)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        std = np.array(
            [2 * self._STD_POS * h_ref] * 4 + [10 * self._STD_VEL * h_ref] * 4
        )
        self.P = np.diag(np.square(std))

    def predict(self) -> None:
        """Advance the state one frame (constant-velocity transition)."""
        if self.x is None:
            return
        h_ref = max(float(self.x[3]), 1.0)
        q_std = np.array(
            [self._STD_POS * h_ref] * 4 + [self._STD_VEL * h_ref] * 4
        )
        Q = np.diag(np.square(q_std))
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + Q

    def update(self, bbox_xywh) -> None:
        """Measurement update with the selected box."""
        if self.x is None:
            self.seed(bbox_xywh)
            return
        z = self._to_cxcywh(bbox_xywh)
        h_ref = max(float(self.x[3]), 1.0)
        R = np.diag(np.square(np.array([self._STD_POS * h_ref] * 4)))
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P

    def predicted_bbox(self) -> np.ndarray | None:
        """Current state as [x, y, w, h] (call after predict())."""
        if self.x is None:
            return None
        cx, cy, w, h = [float(v) for v in self.x[:4]]
        if not all(np.isfinite([cx, cy, w, h])) or w <= 1.0 or h <= 1.0:
            return None
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=float)

    def center_distance(self, bbox_xywh) -> float:
        """Distance between the state center and a box center, in pixels."""
        if self.x is None:
            return float("inf")
        z = self._to_cxcywh(bbox_xywh)
        return float(np.hypot(z[0] - self.x[0], z[1] - self.x[1]))
