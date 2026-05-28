"""
Shared dataclasses for SiamRAM tracker state.

These exist to replace ad-hoc tuple unpacking in per-frame history buffers.
Tuples grow silently as features are added (extra fields tacked onto the
end), which leads to defensive `entry[2] if len(entry) > 2 else None`
patterns scattered through the code. Named fields fix that.
"""

from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class FrameRecord:
    """
    One entry of the per-frame tracking history committed at the end of
    a successful normal update or a recovery commit.

    Fields:
        bbox: predicted bbox in [x, y, w, h] integer pixel form.
        velocity: 2-element vector (px/frame) of the smoothed target velocity.
        H: 3x3 homography estimated this frame, or None if unavailable.
        H_reliable: True when H passed the inlier-ratio gate.
    """

    bbox: np.ndarray
    velocity: np.ndarray
    H: Optional[np.ndarray]
    H_reliable: bool


@dataclass
class VisualState:
    """
    Overlay/debug state produced each frame.

    Fields:
        mode: high-level mode name shown in UI ("tracking", "occluded", ...).
        reason: short explanation for the active mode.
        details: optional numeric/detail string.
    """

    mode: str
    reason: str
    details: str


@dataclass
class CandidateRecord:
    """
    Typed candidate tuple used in occlusion collection windows.

    Fields:
        bbox: candidate bounding box in [x, y, w, h] format.
        descriptor: appearance descriptor for `bbox`.
    """

    bbox: np.ndarray
    descriptor: np.ndarray


class FrameHistory:
    """
    Atomic writer/clearer for synchronized per-frame tracker history deques.

    Several downstream modules assume these histories stay in lockstep. This
    class centralizes mutations so new history signals don't require edits in
    many call sites.
    """

    def __init__(
        self,
        *,
        conf_history: Deque[FrameRecord],
        size_history: Deque[tuple[int, int]],
        center_history: Deque[np.ndarray],
        vel_history: Deque[np.ndarray],
        cam_vel_history: Deque[np.ndarray],
        cam_disp_history: Deque[float],
        spike_step_norms: Deque[float],
    ) -> None:
        self._conf_history = conf_history
        self._size_history = size_history
        self._center_history = center_history
        self._vel_history = vel_history
        self._cam_vel_history = cam_vel_history
        self._cam_disp_history = cam_disp_history
        self._spike_step_norms = spike_step_norms

    def clear(self) -> None:
        """Clear all synchronized history deques together."""
        self._conf_history.clear()
        self._size_history.clear()
        self._center_history.clear()
        self._vel_history.clear()
        self._cam_vel_history.clear()
        self._cam_disp_history.clear()
        self._spike_step_norms.clear()

    def commit_normal(
        self,
        *,
        record: FrameRecord,
        center: np.ndarray,
        velocity: np.ndarray,
        size_wh: tuple[int, int],
        cam_velocity: np.ndarray,
        cam_displacement: float,
        spike_step_norm: Optional[float],
    ) -> None:
        """
        Commit one normal-tracking frame to all lockstep histories.
        """
        self._center_history.append(center.copy())
        self._vel_history.append(velocity.copy())
        self._size_history.append((int(size_wh[0]), int(size_wh[1])))
        self._cam_vel_history.append(cam_velocity.copy())
        self._cam_disp_history.append(float(cam_displacement))
        if spike_step_norm is not None:
            self._spike_step_norms.append(float(spike_step_norm))
        self._conf_history.append(record)

    def commit_recovery(
        self,
        *,
        record: FrameRecord,
        center: np.ndarray,
        cam_velocity: np.ndarray,
        spike_step_norm: Optional[float],
    ) -> None:
        """
        Commit one recovery-confirmed frame.

        Matches legacy behavior: no size/vel/cam-disp append at this step.
        """
        self._center_history.append(center.copy())
        self._cam_vel_history.append(cam_velocity.copy())
        if spike_step_norm is not None:
            self._spike_step_norms.append(float(spike_step_norm))
        self._conf_history.append(record)
