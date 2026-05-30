"""
Adaptive DRM acceptance margin for SiamRAM (``drm_margin: "auto"``).

The occlusion-recovery DRM gate accepts a re-acquisition candidate when its
composite score exceeds ``margin`` (see ``AppearanceMemory.drm_match``). A single
fixed ``margin`` does not generalise across videos: a low-texture / small / fast
target genuinely scores lower than a distinctive one, so a margin tuned for one
sequence either rejects valid re-acquisitions or admits distractors on another.

When ``drm_margin`` is the string ``"auto"`` the tracker measures, during healthy
tracking, how well the *genuine* target scores against its own DRM anchors using
the exact same composite formula the matcher uses (so the number is on the same
scale as ``margin``). An EMA of that signal, minus a small ``delta`` (sit just
below where the true target lives, so real re-acquisitions still clear the bar),
clamped to ``[min_margin, max_margin]``, becomes the live margin.

This class is intentionally free of tracker/numpy-array coupling: it only consumes
scalar scores and a frame index, which keeps the legacy numeric path (a plain
float ``drm_margin``) completely untouched -- the tracker never constructs this
object in that case.
"""

from __future__ import annotations

from typing import Optional


class AutoDrmMargin:
    """EMA-based per-video estimator for the DRM acceptance margin.

    ``value`` is always a valid float inside ``[min_margin, max_margin]``. It
    starts at ``warmup`` and only begins adapting once at least ``min_samples``
    observations have been collected, so an unwarmed estimator reproduces the
    behaviour of a fixed ``warmup`` margin.
    """

    def __init__(
        self,
        *,
        min_margin: float = 0.55,
        max_margin: float = 0.80,
        delta: float = 0.10,
        ema_alpha: float = 0.2,
        n_frames: int = 10,
        warmup: float = 0.70,
        min_samples: int = 5,
    ) -> None:
        lo, hi = float(min_margin), float(max_margin)
        if lo > hi:
            lo, hi = hi, lo
        self.min_margin = lo
        self.max_margin = hi
        self.delta = max(0.0, float(delta))
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.n_frames = max(1, int(n_frames))
        self.min_samples = max(0, int(min_samples))
        # Keep the warm-up default inside the clamp range so `value` is always
        # a legal margin even before any sample arrives.
        self.warmup = min(self.max_margin, max(self.min_margin, float(warmup)))

        self._ema: Optional[float] = None
        self._samples: int = 0
        self._last_frame: Optional[int] = None
        self.value: float = self.warmup

    def due(self, frame_idx: int) -> bool:
        """Whether enough frames have elapsed to take a fresh sample."""
        return (
            self._last_frame is None
            or (int(frame_idx) - self._last_frame) >= self.n_frames
        )

    def observe(self, self_score: float, frame_idx: int) -> float:
        """Fold one target self-score into the EMA and refresh ``value``.

        Returns the (possibly unchanged) current margin.
        """
        self._last_frame = int(frame_idx)
        score = float(self_score)
        if self._ema is None:
            self._ema = score
        else:
            a = self.ema_alpha
            self._ema = (1.0 - a) * self._ema + a * score
        self._samples += 1

        if self._samples >= self.min_samples:
            target = self._ema - self.delta
            self.value = min(self.max_margin, max(self.min_margin, target))
        return self.value

    def reset(self) -> None:
        """Forget accumulated statistics (e.g. on tracker re-init)."""
        self._ema = None
        self._samples = 0
        self._last_frame = None
        self.value = self.warmup
