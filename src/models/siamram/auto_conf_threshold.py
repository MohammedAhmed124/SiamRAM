"""
Adaptive occlusion-entry confidence threshold for SiamRAM.

When ``conf_threshold`` is the string ``"auto"``, the tracker measures healthy
tracking scores with an EMA and maps that EMA from an input score range onto an
output threshold range. Numeric thresholds never construct this estimator.
"""

from __future__ import annotations

from typing import Optional


class AutoConfThreshold:
    """EMA-to-range estimator for confidence thresholds."""

    def __init__(
        self,
        *,
        in_min: float = 0.60,
        in_max: float = 0.90,
        out_min: float = 0.30,
        out_max: float = 0.50,
        ema_alpha: float = 0.2,
        warmup: float = 0.50,
        n_frames: int = 1,
        min_samples: int = 1,
    ) -> None:
        in_lo, in_hi = float(in_min), float(in_max)
        if in_lo > in_hi:
            in_lo, in_hi = in_hi, in_lo
        self.in_min = in_lo
        self.in_max = in_hi

        out_lo, out_hi = float(out_min), float(out_max)
        if out_lo > out_hi:
            out_lo, out_hi = out_hi, out_lo
        self.out_min = out_lo
        self.out_max = out_hi

        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.n_frames = max(1, int(n_frames))
        self.min_samples = max(0, int(min_samples))
        self.warmup = min(self.out_max, max(self.out_min, float(warmup)))

        self._ema: Optional[float] = None
        self._samples = 0
        self._last_frame: Optional[int] = None
        self.value = self.warmup

    @property
    def ema(self) -> Optional[float]:
        return self._ema

    @property
    def samples(self) -> int:
        return self._samples

    def _map(self, ema: float) -> float:
        span = self.in_max - self.in_min
        if span <= 1e-9:
            frac = 1.0 if ema >= self.in_max else 0.0
        else:
            frac = (ema - self.in_min) / span
            frac = min(1.0, max(0.0, frac))
        return self.out_min + frac * (self.out_max - self.out_min)

    def due(self, frame_idx: int) -> bool:
        return (
            self._last_frame is None
            or (int(frame_idx) - self._last_frame) >= self.n_frames
        )

    def observe(self, score: float, frame_idx: int) -> float:
        self._last_frame = int(frame_idx)
        s = float(score)
        if self._ema is None:
            self._ema = s
        else:
            a = self.ema_alpha
            self._ema = (1.0 - a) * self._ema + a * s
        self._samples += 1

        if self._samples >= self.min_samples:
            self.value = self._map(self._ema)
        return self.value

    def reset(self) -> None:
        self._ema = None
        self._samples = 0
        self._last_frame = None
        self.value = self.warmup
