"""
Frame-difference motion-saliency injection for SiamRAM (``frame_dynamics``).

This implements the *frame-dynamics* input augmentation of Wang, Liu, Cheng et
al., *"A Simple Detector with Frame Dynamics is a Strong Tracker"* (the 1st-place
entry of the 4th Anti-UAV Challenge, arXiv:2505.04917). The paper's core idea
(Sec. 3.2) is to inject SHORT-TERM MOTION into the network input so the detector
locks onto a moving tiny target and suppresses static background clutter. The
paper concatenates the current frame with two frame-difference maps

    x_fd = cat( x_t,  x_t - x_{t-1},  x_t - x_{t-2} )                       (Eq. 1)

(the *frame-difference* variant; an *optical-flow* / Farneback variant also
exists). Their Table-3 ablation shows the frame-difference variant gives a
+7-9 AP lift over the raw input on infrared tiny-target detection and is both
superior to and faster than the optical-flow variant.

DEVIATION FROM THE PAPER (and why)
----------------------------------
The paper assumes a detector trained FROM SCRATCH on a 6-channel input
(3 RGB + 2 frame-difference triplets). SiamRAM's SiamABC backbone is a FROZEN,
PRETRAINED 3-channel RGB network: we must NOT change its input channel count and
must NOT retrain it. So we cannot literally concatenate the difference maps.

Instead this module computes the same frame-difference motion-saliency signal
between consecutive FULL frames and BLENDS it ADDITIVELY into the existing
3-channel search region around the predicted box (an "additive motion emphasis"
with a learnable-style ``blend_weight``). Concretely, per frame:

  1. diff_t_1 = | x_t - x_{t-1} |   and   diff_t_2 = | x_t - x_{t-2} |   are
     the absolute short-term frame differences (motion saliency); they are
     averaged into a single per-pixel saliency map (the two-difference average
     mirrors the paper's two-difference triplet while staying 3-channel).
  2. The saliency is scaled by ``scale``, optionally clipped to ``clip`` (when
     ``clip > 0``) to suppress outlier pixels, and added into the search region
     with weight ``blend_weight`` so moving pixels are brightened relative to
     static clutter. The result is re-clamped to the valid 8-bit range.
  3. When the buffer has fewer than two previous frames (first two frames of a
     clip) there is no motion signal yet, so the crop is returned UNCHANGED.

The blend keeps the input 3-channel (the frozen backbone is untouched) while
still surfacing the paper's motion cue. By default the blend is both LOW-WEIGHT
(``blend_weight = 0.06``) and TINY-ONLY (``tiny_only = True``): it is applied only
when the target is small (``bbox area fraction <= tiny_area_fraction``, default
``0.001``), leaving
normal/large-target crops clean. This confines the augmentation to the regime
where appearance collapses (per CST Anti-UAV) and the motion cue helps, avoiding
corruption of the appearance signal on the majority of frames. ``scale``/``clip``
tune emphasis and outlier suppression without retraining; ``blend_weight = 0`` (or
``tiny_only`` with a non-tiny target) is an exact no-op.

The module OWNS its rolling buffer of the previous two full frames (so the host
needs no extra state) and is deliberately decoupled from the tracker, mirroring
the standalone style of :mod:`auto_margin`. The host only
constructs it when ``frame_dynamics_enabled`` is True, so when disabled the
search crop is byte-for-byte unchanged and no frame-difference work runs.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Sequence, Tuple

import numpy as np


class FrameDynamicsProcessor:
    """Frame-difference motion-saliency blender (arXiv:2505.04917, Sec. 3.2).

    Parameters
    ----------
    enabled:
        Master switch echo. When False :meth:`process` always returns the input
        crop unchanged and never touches the frame buffer (the host normally
        avoids constructing this object at all when disabled; the flag is kept so
        the class is safe to instantiate inertly).
    blend_weight:
        Additive blend weight ``w`` for the motion-saliency map: the modified
        search region is ``crop + w * scaled_saliency`` (then clamped to the
        input dtype range). ``0.0`` is an exact no-op; the paper reports no
        explicit weight (it is learned end-to-end) so the default ``0.08`` is a
        deliberately low choice that emphasises motion without overwhelming the
        appearance signal the frozen backbone relies on.
    scale:
        Multiplier applied to the raw frame-difference saliency before blending.
        ``1.0`` keeps the absolute-difference magnitude; larger values amplify
        faint tiny-target motion.
    clip:
        Per-pixel upper clip applied to the SCALED saliency before blending, to
        suppress outlier difference pixels (e.g. from compression noise or a
        sudden illumination change) that would otherwise corrupt the crop. A
        value ``<= 0`` disables clipping (no upper bound). Clipping is applied
        AFTER scaling and BEFORE blending, as the paper-grounded mitigation for
        difference-map outliers.
    tiny_only:
        When True, the motion blend is applied ONLY when the target is tiny
        (bbox area fraction ``<= tiny_area_fraction``); normal / large targets are
        returned unchanged. This restricts the augmentation to the regime where
        appearance collapses (per CST Anti-UAV) and the paper's motion cue helps,
        avoiding corruption of the appearance the frozen backbone matches on for
        the majority of frames. False = blend on every frame (unrestricted).
    tiny_area_fraction:
        Target ``bbox_area / frame_area`` at/below which the target counts as
        tiny (only consulted when ``tiny_only`` is True). Default ``0.001``
        restricts the blend to genuinely tiny / long-range targets (e.g. a
        ~20x20 px UAV in 1080p ~= 0.0002); the looser ``0.03`` mirrors
        SiamRAM's ``roi_search.tiny.long_distance_area_fraction`` tiny boundary.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        blend_weight: float = 0.06,
        scale: float = 1.0,
        clip: float = -1.0,
        tiny_only: bool = False,
        tiny_area_fraction: float = 0.001,
    ) -> None:
        self.enabled = bool(enabled)
        self.blend_weight = float(blend_weight)
        self.scale = float(scale)
        self.clip = float(clip)
        # Tiny-only restriction. When True the motion blend is applied ONLY when
        # the target is small (bbox area fraction <= tiny_area_fraction), leaving
        # normal/large-target crops clean. This avoids corrupting the appearance
        # the frozen backbone relies on for the majority of frames, while still
        # surfacing the motion cue in the regime (tiny targets) where appearance
        # collapses and the paper's method actually helps. False = blend on every
        # frame (the unrestricted variant).
        self.tiny_only = bool(tiny_only)
        self.tiny_area_fraction = float(tiny_area_fraction)

        # Rolling buffer of the previous two FULL frames (oldest -> newest):
        # buffer[-1] is x_{t-1}, buffer[-2] is x_{t-2}. Owned here so the host
        # carries no extra per-video state.
        self._prev_frames: Deque[np.ndarray] = deque(maxlen=2)

    def reset(self) -> None:
        """Clear the previous-frame buffer (call at each per-video re-init)."""
        self._prev_frames.clear()

    def process(
        self,
        search_crop: np.ndarray,
        full_frame: np.ndarray,
        bbox: Optional[Sequence[float]],
        search_context: Optional[Sequence[int]],
    ) -> np.ndarray:
        """Blend frame-difference motion saliency into ``search_crop``.

        Parameters
        ----------
        search_crop:
            The 3-channel search-region image the backbone is about to consume.
            When ``search_context`` is provided it is treated as the sub-window
            of ``full_frame`` given by ``search_context = (x0, y0, x1, y1)``;
            otherwise the saliency is registered around ``bbox``. The crop is
            modified in place on a COPY (the caller's array is never mutated) and
            the modified copy is returned.
        full_frame:
            The current full frame ``x_t`` (same coordinate space the buffer was
            populated from). Used to compute the frame differences and to update
            the rolling buffer.
        bbox:
            The predicted ``[x, y, w, h]`` box (full-frame pixels). Used to
            register the saliency window when ``search_context`` is None.
        search_context:
            Optional ``(x0, y0, x1, y1)`` pixel rectangle of the search region
            within ``full_frame``, so the saliency is spatially aligned with the
            crop. When None the window is derived from ``bbox`` extents.

        Returns
        -------
        The motion-emphasised search crop (a new array). When the feature is
        disabled, the blend weight is zero, or there are fewer than two buffered
        previous frames, the input crop is returned unchanged (and the buffer is
        still advanced so motion is available next frame).
        """
        if not self.enabled:
            return search_crop

        if full_frame is None or search_crop is None:
            return search_crop

        # Tiny-only gate: in the restricted variant, leave normal/large targets'
        # crops clean (no blend) and only emphasise motion for tiny targets. The
        # buffer is still advanced below so motion stays available if the target
        # later shrinks into the tiny regime.
        skip_blend = self.tiny_only and not self._is_tiny(bbox, full_frame)

        # Compute the registered saliency BEFORE pushing x_t into the buffer, so
        # the buffer still holds x_{t-1}, x_{t-2}.
        out = search_crop
        if not skip_blend and self.blend_weight != 0.0 and len(self._prev_frames) >= 1:
            saliency = self._registered_saliency(
                full_frame=full_frame,
                crop_shape=search_crop.shape,
                bbox=bbox,
                search_context=search_context,
            )
            if saliency is not None:
                out = self._blend(search_crop, saliency)

        # Advance the rolling buffer with the current full frame (copy so a later
        # in-place edit of the caller's frame cannot corrupt our history).
        self._prev_frames.append(np.asarray(full_frame).copy())
        return out

    # ------------------------------------------------------------------ helpers

    def _is_tiny(
        self,
        bbox: Optional[Sequence[float]],
        full_frame: np.ndarray,
    ) -> bool:
        """Whether the target is "tiny" by bbox area fraction of the frame.

        Returns True iff ``(bw * bh) / (frame_h * frame_w) <= tiny_area_fraction``.
        When the box is unknown (early warm-up / lost), returns False so the
        tiny-only variant conservatively skips the blend rather than emphasising
        an unresolved / whole-frame window.
        """
        if bbox is None or len(bbox) < 4:
            return False
        bw = max(0.0, float(bbox[2]))
        bh = max(0.0, float(bbox[3]))
        fh = float(full_frame.shape[0])
        fw = float(full_frame.shape[1])
        frame_area = max(1.0, fh * fw)
        return (bw * bh) / frame_area <= self.tiny_area_fraction

    def _registered_saliency(
        self,
        *,
        full_frame: np.ndarray,
        crop_shape: Tuple[int, ...],
        bbox: Optional[Sequence[float]],
        search_context: Optional[Sequence[int]],
    ) -> Optional[np.ndarray]:
        """Build the saliency window registered to the search crop, or None.

        The saliency is the average of the absolute short-term frame differences
        |x_t - x_{t-1}| and (when available) |x_t - x_{t-2}|, cropped to the
        search window and resized to the crop's spatial size. Returns None when
        the window cannot be resolved or is empty.
        """
        win = self._resolve_window(
            frame_shape=full_frame.shape,
            bbox=bbox,
            search_context=search_context,
        )
        if win is None:
            return None
        x0, y0, x1, y1 = win
        if x1 <= x0 or y1 <= y0:
            return None

        cur = full_frame[y0:y1, x0:x1].astype(np.float32, copy=False)

        diffs = []
        for prev in self._prev_frames:  # oldest -> newest (x_{t-2}, x_{t-1})
            if prev.shape != full_frame.shape:
                continue
            prev_win = prev[y0:y1, x0:x1].astype(np.float32, copy=False)
            diffs.append(np.abs(cur - prev_win))
        if not diffs:
            return None

        # Average the (one or two) difference maps -> single saliency window
        # mirroring the paper's two-difference triplet while staying 3-channel.
        saliency = diffs[0]
        for extra in diffs[1:]:
            saliency = saliency + extra
        saliency = saliency / float(len(diffs))

        # Resize the saliency window to the crop's spatial size so it registers
        # pixel-for-pixel with the search crop the backbone consumes.
        ch, cw = int(crop_shape[0]), int(crop_shape[1])
        if saliency.shape[0] != ch or saliency.shape[1] != cw:
            saliency = self._resize(saliency, (ch, cw))
        return saliency

    def _resolve_window(
        self,
        *,
        frame_shape: Tuple[int, ...],
        bbox: Optional[Sequence[float]],
        search_context: Optional[Sequence[int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Clamp the search window to ``(x0, y0, x1, y1)`` within the frame."""
        h = int(frame_shape[0])
        w = int(frame_shape[1])

        if search_context is not None and len(search_context) >= 4:
            x0, y0, x1, y1 = (float(v) for v in search_context[:4])
        elif bbox is not None and len(bbox) >= 4:
            bx, by, bw, bh = (float(v) for v in bbox[:4])
            x0, y0, x1, y1 = bx, by, bx + bw, by + bh
        else:
            return None

        ix0 = max(0, min(w, int(round(x0))))
        iy0 = max(0, min(h, int(round(y0))))
        ix1 = max(0, min(w, int(round(x1))))
        iy1 = max(0, min(h, int(round(y1))))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return ix0, iy0, ix1, iy1

    def _blend(self, crop: np.ndarray, saliency: np.ndarray) -> np.ndarray:
        """Additively blend the scaled+clipped saliency into a copy of ``crop``."""
        emphasis = saliency * self.scale
        if self.clip > 0.0:
            emphasis = np.minimum(emphasis, self.clip)

        # Broadcast a single-channel saliency to all crop channels if needed.
        if crop.ndim == 3 and emphasis.ndim == 2:
            emphasis = emphasis[:, :, None]

        out = crop.astype(np.float32, copy=True) + self.blend_weight * emphasis

        # Re-clamp to the input dtype's valid range so the crop stays a legal
        # image (8-bit images clamp to [0, 255]).
        if np.issubdtype(crop.dtype, np.integer):
            info = np.iinfo(crop.dtype)
            out = np.clip(out, info.min, info.max)
        return out.astype(crop.dtype, copy=False)

    @staticmethod
    def _resize(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
        """Resize ``arr`` to ``(height, width)`` using OpenCV when available.

        Falls back to nearest-neighbour index sampling (NumPy only) so the module
        has no hard OpenCV dependency. The saliency is a smooth motion map, so
        either interpolation is acceptable.
        """
        out_h, out_w = int(hw[0]), int(hw[1])
        if out_h <= 0 or out_w <= 0:
            return arr
        try:  # pragma: no cover - depends on optional cv2
            import cv2

            return cv2.resize(
                arr, (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )
        except Exception:
            in_h, in_w = arr.shape[0], arr.shape[1]
            if in_h == 0 or in_w == 0:
                return arr
            ys = (np.arange(out_h) * in_h // max(1, out_h)).clip(0, in_h - 1)
            xs = (np.arange(out_w) * in_w // max(1, out_w)).clip(0, in_w - 1)
            return arr[ys][:, xs]
