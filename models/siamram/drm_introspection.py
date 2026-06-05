"""
Introspection-based DRM (distractor-resolving memory) update for SiamRAM
(arXiv:2411.17576, "A Distractor-Aware Memory for Visual Object Tracking with
SAM2" -- the paper that originated the RAM/DRM split SiamRAM already uses).

The paper triggers a DRM *anchor* update when the tracker's own multi-hypothesis
output reveals a distractor and tracking is currently reliable. Concretely
(Sec. 3.2.2), DRM is updated only when ALL of the following hold:

  1. The target is present (predicted mask not empty).
  2. At least ``delta = 5`` frames have passed since the last DRM update.
  3. A distractor is detected via *hypothesis divergence*: a box is fitted to the
     output mask and to the union of the output mask with the largest connected
     component of the alternative mask; if the ratio of the two box areas drops
     below ``theta_anc = 0.7`` the alternative hypothesis spans a clearly
     separate region (a competing distractor).
  4. Tracking is reliable: the predicted IoU/score exceeds ``theta_iou = 0.8``
     AND the current target area is within ``theta_area = 20%`` of the median
     area over the last ``theta_M = 10`` frames (size stability).

This module adapts the mask-based mechanism to SiamABC's response map. The
analogue of the "alternative mask" is the SECOND-strongest local peak in the
classification/centerness response map. The hypothesis-divergence signal is the
bbox-area ratio between a box fitted to the primary-peak region and a box fitted
to the union of the primary- and secondary-peak regions: when a strong secondary
peak sits at a spatially separate location, the union box is much larger than the
primary box and the ratio drops below ``theta_anc`` -- exactly the paper's
``Theta_anc`` trigger. A secondary peak must also clear a minimum relative
strength (``secondary_min_ratio``) to count as a genuine competing hypothesis
rather than response-map noise.

Scale-aware divergence (de-degenerating ``theta_anc``)
------------------------------------------------------
The paper fits its divergence boxes to real segmentation masks, which carry the
target's actual spatial extent, so the primary/union area ratio is meaningful. On
SiamABC's coarse 16x16 response map a FIXED 3x3 box (``peak_half_size=1``) plus a
fixed ``suppress_radius=2`` is DEGENERATE: any secondary peak >= 3 cells from the
primary yields a ratio < 0.7, so ``theta_anc=0.7`` fires unconditionally and
``theta_anc`` becomes a dead knob (telemetry confirmed the divergence gate never
blocked). To restore the paper's mask-fitted semantics, the divergence box AND the
non-max suppression can be scaled to the TARGET's grid-cell extent
(``target_half_cells``): the secondary peak must then lie BEYOND the target's own
footprint to count as a distractor, and the ratio reflects the competing peak's
separation RELATIVE to target size. This turns ``theta_anc`` back into a real
selectivity control. With ``target_half_cells=None`` the function reproduces the
exact legacy fixed-box behaviour for A/B comparison.

The class is intentionally tracker-agnostic in the decoupled style of
``auto_template_rate.py`` / ``auto_margin.py``: it consumes scalars/arrays
(peak scores, the divergence ratio, predicted score, current area, frame index)
and emits a single boolean decision. The tracker owns the actual DRM write; this
object only owns the *gating* state (last-update frame index and the size-history
deque). When the feature is disabled the tracker never constructs this object and
the legacy DRM write path in ``AppearanceMemory`` is completely untouched.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


def secondary_peak_divergence(
    cls_score: np.ndarray,
    primary_rc: Tuple[int, int],
    *,
    suppress_radius: int = 2,
    peak_half_size: int = 1,
    target_half_cells: Optional[Tuple[float, float]] = None,
    min_half: int = 1,
) -> Tuple[float, float, float, Optional[Tuple[int, int]]]:
    """Compute the hypothesis-divergence signal from a 2D response map.

    The primary peak is taken at ``primary_rc`` (the tracker's chosen location).
    The neighbourhood of radius ``suppress_radius`` around it is suppressed and
    the strongest remaining cell is taken as the secondary peak (the analogue of
    SAM2's alternative mask). A small ``peak_half_size`` box is fitted to each
    peak; the divergence ratio is

        ratio = area(primary_box) / area(union(primary_box, secondary_box)),

    mirroring the paper's bbox-area ratio between the output-mask box and the
    output-with-alternative union box. A spatially separated, strong secondary
    peak yields a large union box and thus a small ratio.

    Args:
        cls_score: 2D response map (H, W), already in [0, 1].
        primary_rc: (row, col) of the chosen primary peak.
        suppress_radius: Half-width of the window zeroed around the primary peak
            before searching for the secondary peak (non-max suppression). When
            ``target_half_cells`` is given, the effective per-axis suppression is
            grown to at least the target's grid-cell half-extent so the secondary
            peak must lie beyond the target's own footprint.
        peak_half_size: Half-width of the box fitted to each peak location when no
            target-relative scaling is requested.
        target_half_cells: Optional ``(half_rows, half_cols)`` giving the target's
            half-extent in response-map grid cells. When provided, the divergence
            boxes and the non-max suppression are scaled to this extent (the paper's
            mask-fitted-box semantics), making the ratio reflect the secondary
            peak's separation relative to target size and turning ``theta_anc`` into
            a real selectivity knob. When ``None``, the legacy fixed-box behaviour
            is reproduced exactly.
        min_half: Minimum per-axis box/suppression half-extent in cells (floor),
            so a tiny target still fits a non-degenerate box.

    Returns:
        (ratio, primary_score, secondary_score, secondary_rc). ``ratio`` is 1.0
        when no valid secondary peak exists (degenerate map), so it never triggers
        an update. ``secondary_rc`` is the ``(int(row), int(col))`` grid location of
        the detected secondary peak on the main path, or ``None`` for every
        degenerate early-return (no competing peak was located).

    Backward compatibility: with ``target_half_cells=None``, ``peak_half_size=1``
    and ``suppress_radius=2`` the result is byte-for-byte identical to the legacy
    fixed 3x3-box / radius-2 implementation.
    """
    arr = np.asarray(cls_score, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return 1.0, 0.0, 0.0, None

    h, w = arr.shape
    r1, c1 = int(primary_rc[0]), int(primary_rc[1])
    r1 = int(np.clip(r1, 0, h - 1))
    c1 = int(np.clip(c1, 0, w - 1))
    primary_score = float(arr[r1, c1])

    # Per-axis box half-sizes. When the target's grid-cell extent is supplied,
    # scale the divergence boxes to it (paper's mask-fitted-box semantics);
    # otherwise fall back to the fixed ``peak_half_size`` legacy box.
    if target_half_cells is not None:
        hy = max(int(min_half), int(round(target_half_cells[0])))
        hx = max(int(min_half), int(round(target_half_cells[1])))
    else:
        hy = hx = int(max(int(min_half), peak_half_size))

    # Scale the non-max suppression so the secondary peak must lie BEYOND the
    # target's own footprint (otherwise the "secondary peak" is just the target's
    # response shoulder, not a distractor). Split the single radius per-axis.
    sup_r = max(int(suppress_radius), hy)
    sup_c = max(int(suppress_radius), hx)

    suppressed = arr.copy()
    r_lo = max(0, r1 - sup_r)
    r_hi = min(h, r1 + sup_r + 1)
    c_lo = max(0, c1 - sup_c)
    c_hi = min(w, c1 + sup_c + 1)
    suppressed[r_lo:r_hi, c_lo:c_hi] = -np.inf

    if not np.isfinite(suppressed).any():
        # The whole map fell inside the suppression window -> no secondary peak.
        return 1.0, primary_score, 0.0, None

    flat_idx = int(np.argmax(suppressed))
    r2, c2 = np.unravel_index(flat_idx, arr.shape)
    secondary_score = float(arr[r2, c2])
    if not np.isfinite(secondary_score) or secondary_score <= 0.0:
        return 1.0, primary_score, max(0.0, secondary_score), None

    # Box fitted to the primary peak region (half-extent (hy, hx)).
    p_x0 = c1 - hx
    p_y0 = r1 - hy
    p_x1 = c1 + hx
    p_y1 = r1 + hy
    primary_area = float((2 * hx + 1) * (2 * hy + 1))

    # Union box spanning both peak regions (same half-extent around the secondary).
    s_x0 = c2 - hx
    s_y0 = r2 - hy
    s_x1 = c2 + hx
    s_y1 = r2 + hy
    u_x0 = min(p_x0, s_x0)
    u_y0 = min(p_y0, s_y0)
    u_x1 = max(p_x1, s_x1)
    u_y1 = max(p_y1, s_y1)
    union_area = float((u_x1 - u_x0 + 1) * (u_y1 - u_y0 + 1))

    if union_area <= 0.0:
        return 1.0, primary_score, secondary_score, None
    ratio = primary_area / union_area
    return (
        float(np.clip(ratio, 0.0, 1.0)),
        primary_score,
        secondary_score,
        (int(r2), int(c2)),
    )


class DrmIntrospectionUpdater:
    """Introspection gate deciding when to write a DRM distractor-context anchor.

    Faithful to the DRM update protocol of arXiv:2411.17576 (Sec. 3.2.2), adapted
    from masks to SiamABC's response map. ``should_update`` returns ``True`` only
    when the target is present, the minimum frame spacing ``delta`` has elapsed,
    a competing secondary peak is detected (divergence ratio < ``theta_anc`` and
    secondary/primary strength >= ``secondary_min_ratio``), and tracking is
    reliable (score > ``theta_iou`` and the current area is within ``theta_area``
    of the median area over the last ``theta_M`` frames).

    The object holds only gating state (last-update frame index and the rolling
    size history); the tracker performs the actual DRM write. It is constructed
    by the tracker exclusively when the feature is enabled, so when disabled the
    legacy DRM write path is untouched.
    """

    def __init__(
        self,
        *,
        theta_anc: float = 0.7,
        theta_iou: float = 0.8,
        theta_area: float = 0.2,
        theta_M: int = 10,
        delta: int = 5,
        secondary_min_ratio: float = 0.5,
    ) -> None:
        self.theta_anc = float(theta_anc)
        self.theta_iou = float(theta_iou)
        self.theta_area = max(0.0, float(theta_area))
        self.theta_M = max(1, int(theta_M))
        self.delta = max(1, int(delta))
        self.secondary_min_ratio = max(0.0, float(secondary_min_ratio))

        self._areas: deque = deque(maxlen=self.theta_M)
        self._last_update_frame: Optional[int] = None

    def reset(self) -> None:
        """Forget gating state (e.g. on tracker re-init per video)."""
        self._areas.clear()
        self._last_update_frame = None

    def _due(self, frame_idx: int) -> bool:
        """Whether at least ``delta`` frames have passed since the last update."""
        return (
            self._last_update_frame is None
            or (int(frame_idx) - self._last_update_frame) >= self.delta
        )

    def _size_stable(self, area: float) -> bool:
        """Current area within ``theta_area`` of the median of the last theta_M."""
        if not self._areas:
            # No history yet: treat as stable so the very first eligible anchor
            # is not blocked purely by a cold size history.
            return True
        med = float(np.median(np.fromiter(self._areas, dtype=np.float64)))
        if med <= 0.0:
            return True
        return abs(float(area) - med) / med <= self.theta_area

    def observe_size(self, area: float) -> None:
        """Fold the current target area into the rolling size history.

        Called every confident frame so the median-area stability test reflects
        recent target scale, independent of whether an update fires.
        """
        a = float(area)
        if a > 0.0 and np.isfinite(a):
            self._areas.append(a)

    def evaluate(
        self,
        *,
        frame_idx: int,
        target_present: bool,
        divergence_ratio: float,
        primary_score: float,
        secondary_score: float,
        reliability_score: float,
        area: float,
    ) -> str:
        """Run the DRM-update gates in order and name the FIRST one that fails.

        Side-effect free: it does not call ``observe_size`` or ``mark_updated``
        and does not mutate ``self._areas`` (``_size_stable`` only reads it). Runs
        the exact same checks in the exact same order as the legacy
        ``should_update`` so the boolean ``evaluate(...) == "pass"`` is identical
        to the old result; it merely also exposes the blocking reason.

        Returns one of: ``"not_present"``, ``"not_due"``, ``"reliability"``,
        ``"size"``, ``"divergence"``, ``"secondary"`` (the first gate that fails)
        or ``"pass"`` (all gates satisfied).
        """
        if not target_present:
            return "not_present"
        if not self._due(frame_idx):
            return "not_due"

        # Reliability gates first -- updating during unreliable tracking risks
        # corrupting DRM (paper: trigger only during sufficiently stable periods).
        if float(reliability_score) <= self.theta_iou:
            return "reliability"
        if not self._size_stable(area):
            return "size"

        # Hypothesis-divergence / distractor-detection gate.
        distractor_detected = float(divergence_ratio) < self.theta_anc
        if not distractor_detected:
            return "divergence"
        # Require the secondary peak to be a genuine competing hypothesis, not
        # response-map noise: its strength must be a meaningful fraction of the
        # primary peak.
        prim = max(1e-8, float(primary_score))
        if (float(secondary_score) / prim) < self.secondary_min_ratio:
            return "secondary"

        return "pass"

    def should_update(
        self,
        *,
        frame_idx: int,
        target_present: bool,
        divergence_ratio: float,
        primary_score: float,
        secondary_score: float,
        reliability_score: float,
        area: float,
    ) -> bool:
        """Decide whether to write a DRM distractor-context anchor this frame.

        Args:
            frame_idx: Current frame index (for the ``delta`` spacing gate).
            target_present: Whether the target is present this frame.
            divergence_ratio: Primary/union bbox-area ratio from
                ``secondary_peak_divergence``; < ``theta_anc`` flags a distractor.
            primary_score: Strength of the chosen (primary) peak.
            secondary_score: Strength of the competing (secondary) peak.
            reliability_score: Predicted IoU / confidence (paper's ``theta_iou``
                gate); SiamABC's per-frame score serves as the analogue.
            area: Current target box area (w * h), for the size-stability gate.

        Returns:
            True iff all of the paper's DRM-update conditions are satisfied.
        """
        return (
            self.evaluate(
                frame_idx=frame_idx,
                target_present=target_present,
                divergence_ratio=divergence_ratio,
                primary_score=primary_score,
                secondary_score=secondary_score,
                reliability_score=reliability_score,
                area=area,
            )
            == "pass"
        )

    def mark_updated(self, frame_idx: int) -> None:
        """Record that a DRM anchor was written at ``frame_idx``."""
        self._last_update_frame = int(frame_idx)
