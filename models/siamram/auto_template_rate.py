"""
Adaptive dynamic-template update rate for SiamRAM (``template_rate_auto``).

The inner SiamABC tracker refreshes its dynamic template every ``N`` frames
(the ``select_representatives`` cadence) over a rolling ``memory_window_size``
window. A single fixed pair does not fit every clip: when the camera is steady
and the target moves slowly, the appearance is stable and the template should
update *slowly* (large N / large window = lots of averaging, very stable). When
either the camera or the target moves fast, the appearance changes quickly and
the template should update *fast* (small N / small window = snap to the latest
look).

When ``template_rate_auto_enabled`` is true the tracker measures, every frame, a
0..1 "fastness" from camera motion and target motion (combined with ``max`` so
either one being fast drives fast updating), smooths it with an EMA, then
linearly interpolates N and ``memory_window_size`` between the configured
slow-end and fast-end values:

    motion = 0  (slow / steady) -> N_slow,  window_slow   (end of scale)
    motion = 1  (fast)          -> N_fast,  window_fast   (start of scale)

This class is deliberately free of tracker/OpenCV coupling: it consumes a single
scalar ``motion`` in [0, 1] and emits ``(N, window)``. The normalisation of the
raw camera/target signals into that scalar lives in the tracker, where the
bbox-diagonal reference and velocity are available. When the mode is off the
tracker never constructs this object, so the legacy fixed-rate / binary
camera-motion-adapt paths are untouched.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _lerp(slow: float, fast: float, t: float) -> float:
    """Interpolate slow->fast as t goes 0->1."""
    return slow + (fast - slow) * t


class AutoTemplateRate:
    """EMA-smoothed motion -> (N, memory_window_size) controller.

    ``N`` and ``window`` are always valid (>= 1) and start at the slow-end
    values, so before any observation the behaviour matches a fixed slow rate.
    """

    def __init__(
        self,
        *,
        n_fast: int = 3,
        n_slow: int = 40,
        window_fast: int = 8,
        window_slow: int = 40,
        ema_alpha: float = 0.2,
    ) -> None:
        self.n_fast = max(1, int(n_fast))
        self.n_slow = max(1, int(n_slow))
        self.window_fast = max(1, int(window_fast))
        self.window_slow = max(1, int(window_slow))
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))

        self._ema: Optional[float] = None
        # Current smoothed motion and the resulting cadence/window. Start at the
        # slow end so an unobserved controller reproduces a fixed slow rate.
        self.motion: float = 0.0
        self.N: int = self.n_slow
        self.window: int = self.window_slow

    def observe(self, motion_raw: float) -> Tuple[int, int]:
        """Fold one motion sample (0..1) into the EMA and refresh N / window.

        Returns the current ``(N, window)``.
        """
        m = min(1.0, max(0.0, float(motion_raw)))
        if self._ema is None:
            self._ema = m
        else:
            a = self.ema_alpha
            self._ema = (1.0 - a) * self._ema + a * m
        self.motion = self._ema

        self.N = max(1, int(round(_lerp(self.n_slow, self.n_fast, self._ema))))
        self.window = max(
            1, int(round(_lerp(self.window_slow, self.window_fast, self._ema)))
        )
        return self.N, self.window

    def reset(self) -> None:
        """Forget accumulated smoothing (e.g. on tracker re-init)."""
        self._ema = None
        self.motion = 0.0
        self.N = self.n_slow
        self.window = self.window_slow
