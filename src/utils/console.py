"""Small runtime-console helpers for clean SiamRAM terminal output."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sys
import warnings
from collections.abc import Iterator

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_NOISY_LOGGERS = (
    "torch_tensorrt",
    "torch_tensorrt.dynamo",
    "torch_tensorrt.dynamo.conversion",
    "torch_tensorrt.dynamo.conversion.aten_ops_converters",
    "torch._dynamo",
    "torch._inductor",
    "torch.fx",
    "torch.compile",
    "tensorrt",
    "py.warnings",
)


def _color_enabled() -> bool:
    value = os.environ.get("SIAMRAM_COLOR", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return "NO_COLOR" not in os.environ and sys.stdout.isatty()


def _paint(text: str, color: str) -> str:
    if not _color_enabled():
        return text
    return f"{color}{text}{_RESET}"


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _fit_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def silence_noisy_libraries() -> None:
    """Turn down harmless third-party warnings that otherwise drown our logs."""
    if os.environ.get("TORCH_LOGS", None) == "":
        os.environ.pop("TORCH_LOGS", None)
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/siamram_mpl_cache")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="albumentations.*")
    # torchvision emits two UserWarnings when a backbone is built with the
    # legacy `pretrained=...` argument; they are harmless and only clutter the
    # weight-loading step, so silence them by both module and message.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision(\..*)?")
    warnings.filterwarnings("ignore", message=r".*'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*Arguments other than a weight enum.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*tensorrt.plugin module is experimental.*")
    warnings.filterwarnings("ignore", message=".*TRTLLM_PLUGIN_PATH.*")
    warnings.filterwarnings("ignore", message=".*modelopt.*quant.*")

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)


@contextlib.contextmanager
def quiet_external_logs(
    *,
    stdout: bool = True,
    stderr: bool = True,
) -> Iterator[None]:
    """
    Temporarily silence stdout/stderr at both Python and file-descriptor level.

    TensorRT / Torch-TensorRT emit some messages from native code before Python
    logging filters can catch them, so this redirects fd 1/2 as well.
    """
    silence_noisy_libraries()
    saved_levels = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)

    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_stdout_fd = os.dup(1) if stdout else None
    saved_stderr_fd = os.dup(2) if stderr else None

    try:
        if stdout:
            sys.stdout = devnull
            os.dup2(devnull.fileno(), 1)
        if stderr:
            sys.stderr = devnull
            os.dup2(devnull.fileno(), 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        if stdout and saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
            sys.stdout = saved_stdout
        if stderr and saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            sys.stderr = saved_stderr
        devnull.close()
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


def siamram_log(
    message: str,
    *,
    phase: str = "RUN",
    status: str = "info",
    indent: int = 0,
    label: str | None = None,
    blank_before: bool = False,
) -> None:
    """Print one compact, colored SiamRAM status line.

    `status` picks the accent color (and the default 5-char label); pass
    `label` to override just the label text while keeping the status color.
    Set `blank_before` to emit a leading newline so a block visually detaches
    from the line above it.
    """
    labels = {
        "info": ("info", _BLUE),
        "load": ("load", _CYAN),
        "build": ("build", _YELLOW),
        "ready": ("ready", _GREEN),
        "done": ("done", _GREEN),
        "warn": ("warn", _YELLOW),
        "error": ("error", _RED),
    }
    default_label, color = labels.get(status, labels["info"])
    text = (label if label is not None else default_label)[:5].ljust(5)
    prefix = (
        f"{_paint('[SiamRAM]', _BOLD + _CYAN)}"
        f"{_paint(f'[{phase}]', _DIM)}"
    )
    pad = "  " * max(0, int(indent))
    lead = "\n" if blank_before else ""
    print(f"{lead}{prefix} {_paint(text, color)} {pad}{message}", flush=True)


class SiamRAMProgressLine:
    """Single terminal line that accumulates per-frame tracker states."""

    _MODE_COLORS = {
        "tracking": _GREEN,
        "occluded": _RED,
        "distractor": _YELLOW,
    }

    def __init__(
        self,
        name: str,
        *,
        total_frames: int = 0,
        phase: str = "RUN",
        label: str = "state",
    ) -> None:
        """Create a bounded progress line for one sequence."""
        self.name = str(name or "video")
        self.total_frames = max(0, int(total_frames or 0))
        self.phase = phase
        self.label = label[:5].ljust(5)
        self._modes: list[str] = []
        self._counts = {"tracking": 0, "occluded": 0, "distractor": 0}
        self._interactive = sys.stdout.isatty()
        self._closed = False

    def update(self, mode: str) -> None:
        """Append one frame state and redraw the line when stdout is a TTY."""
        if self._closed:
            return
        canonical = self._canonical_mode(mode)
        self._modes.append(canonical)
        self._counts[canonical] += 1
        if self._interactive:
            self._write()

    def finish(self) -> None:
        """Render the final line and terminate it with one newline."""
        if self._closed:
            return
        self._write()
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._closed = True

    def _canonical_mode(self, mode: str) -> str:
        value = str(mode or "").strip().lower()
        if value == "distractor":
            return "distractor"
        if value in {"occluded", "occlusion"}:
            return "occluded"
        return "tracking"

    def _write(self) -> None:
        line = self._line()
        if self._interactive:
            sys.stdout.write("\r" + line + "\033[K")
        else:
            sys.stdout.write(line)
        sys.stdout.flush()

    def _line(self) -> str:
        term_width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
        prefix = (
            f"{_paint('[SiamRAM]', _BOLD + _CYAN)}"
            f"{_paint(f'[{self.phase}]', _DIM)}"
        )
        lead = f"{prefix} {_paint(self.label, _BLUE)} "
        frame_count = len(self._modes)
        if self.total_frames > 0:
            progress = f"{frame_count}/{self.total_frames}"
        else:
            progress = f"{frame_count}f"
        summary = (
            f" {progress}  "
            f"n={self._counts['tracking']} "
            f"o={self._counts['occluded']} "
            f"d={self._counts['distractor']}"
        )

        # Keep the rendered line inside the terminal width, even for long
        # sequence names and narrow consoles.
        min_bar = 1
        fixed_len = _visible_len(lead) + _visible_len(summary) + min_bar + 3
        name_width = max(0, term_width - fixed_len)
        name = _fit_text(self.name, min(24, name_width))
        name_gap = 1 if name else 0
        bar_width = max(
            min_bar,
            term_width
            - _visible_len(lead)
            - len(name)
            - name_gap
            - _visible_len(summary)
            - 2,
        )
        bar = self._render_bar(bar_width)
        name_part = f"{name} " if name else ""
        return f"{lead}{name_part}[{bar}]{summary}"

    def _render_bar(self, width: int) -> str:
        width = max(1, int(width))
        if not self._modes:
            return " " * width

        if self.total_frames > 0:
            slots = [" "] * width
            denom = max(1, self.total_frames - 1)
            for frame_idx, mode in enumerate(self._modes[: self.total_frames]):
                slot = min(width - 1, int(round(frame_idx * (width - 1) / denom)))
                slots[slot] = self._dot(mode)
            return "".join(slots)

        if len(self._modes) <= width:
            dots = "".join(self._dot(mode) for mode in self._modes)
            return dots + (" " * (width - len(self._modes)))

        dots = []
        n_modes = len(self._modes)
        for slot in range(width):
            idx = min(n_modes - 1, int((slot + 1) * n_modes / width) - 1)
            dots.append(self._dot(self._modes[idx]))
        return "".join(dots)

    def _dot(self, mode: str) -> str:
        return _paint(".", self._MODE_COLORS.get(mode, _GREEN))


class _CompileProgress:
    """Process-wide single progress line for the whole TensorRT compile phase.

    A *session* spans every engine compiled up front — SiamABC (backbone,
    attention, connect) and, when enabled, the OSNet descriptor — so the user
    sees ONE bar from 0 % to 100 %, not one per subsystem. The line matches the
    inference progress style: a wide, thin row of dots inside brackets that fills
    left-to-right, with a percentage and the current stage text.

        [SiamRAM][TRT] build [............              ]  50% · compiling connect engines

    On a TTY the line is rewritten in place (``\\r``); when stdout is not a TTY
    (e.g. redirected to a log file) each stage prints one plain line instead.

    Lifecycle:
        begin(total)   open a session sized to ``total`` stages (no-op if one is
                       already open, so a caller that pre-sizes the whole phase
                       wins over a subsystem that would otherwise self-size).
        stage(message) announce the stage about to compile (auto-opens a 1-stage
                       session if none is active, e.g. a lone OSNet build).
        complete()     mark the current stage done; the final stage paints the
                       bar green and prints "compiled successfully".
    """

    _DOT = "."

    def __init__(self) -> None:
        self.phase = "TRT"
        self._total = 0
        self._done = 0
        self._active = False
        self._message = ""
        self._interactive = sys.stdout.isatty()

    def begin(self, total: int, *, phase: str = "TRT") -> bool:
        """Open a session of ``total`` stages. Returns False if one was open."""
        if self._active:
            return False
        self.phase = phase
        self._total = max(1, int(total))
        self._done = 0
        self._message = ""
        self._active = True
        # No initial draw: the first stage() renders the 0% frame, which keeps
        # non-TTY logs free of an empty leading line.
        return True

    def stage(self, message: str) -> None:
        """Announce the stage currently compiling and redraw the bar."""
        if not self._active:
            self.begin(1)
        self._message = str(message or "")
        self._draw("build", _YELLOW)

    def complete(self) -> None:
        """Mark one stage finished; close the session on the last stage."""
        if not self._active:
            return
        self._done = min(self._total, self._done + 1)
        if self._done >= self._total:
            self._message = "compiled successfully"
            self._draw("done", _GREEN)
            self._close()
        elif self._interactive:
            # On a TTY, advance the fill in place. In a log file the per-stage
            # stage() line already recorded progress, so skip the extra line.
            self._draw("build", _YELLOW)

    def finish(self, message: str = "compiled successfully") -> None:
        """Force the bar to 100 % and close it (idempotent safety net)."""
        if not self._active:
            return
        self._done = self._total
        self._message = str(message or "")
        self._draw("done", _GREEN)
        self._close()

    def _close(self) -> None:
        if self._interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._active = False
        self._done = 0
        self._total = 0

    def _draw(self, label: str, color: str) -> None:
        prefix = (
            f"{_paint('[SiamRAM]', _BOLD + _CYAN)}"
            f"{_paint(f'[{self.phase}]', _DIM)}"
        )
        accent = _paint(label[:5].ljust(5), color)
        pct = int(round(100 * self._done / max(1, self._total)))
        pct_text = f"{pct:3d}%"
        if not self._interactive:
            # Non-TTY (log file): one clean plain line per stage, no carriage
            # returns. Keeps captured logs readable.
            print(f"{prefix} {accent} {pct_text} · {self._message}", flush=True)
            return
        term_width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
        message = _fit_text(self._message, 40)
        # "<prefix> <accent> [<bar>] <pct> · <message>" — the bar fills the slack.
        fixed = _visible_len(prefix) + 1 + 5 + 1 + 1 + 1 + len(pct_text) + 3 + len(message)
        bar_width = max(8, term_width - fixed)
        filled = int(round(bar_width * self._done / max(1, self._total)))
        bar = _paint(self._DOT * filled, color) + " " * (bar_width - filled)
        line = f"{prefix} {accent} [{bar}] {pct_text} · {message}"
        sys.stdout.write("\r" + line + "\033[K")
        sys.stdout.flush()


_COMPILE_PROGRESS = _CompileProgress()


def compile_progress() -> _CompileProgress:
    """Return the process-wide TensorRT compile progress bar (one session)."""
    return _COMPILE_PROGRESS


def siamram_latency(
    rows,
    *,
    header: str | None = "latency report",
    occlusion_pct: float | None = None,
    phase: str = "PERF",
) -> None:
    """Render a latency report as a small block of colored SiamRAM lines.

    `rows` is a list of ``(label, stats)`` pairs where ``stats`` is either
    ``None`` (rendered as a dim "no data") or a dict with the keys
    ``n, mean, med, p95, p99, min, max, fps`` (all numeric, times in ms).
    The group label drives a subtle accent: all=cyan, track=green, occl=amber.
    `occlusion_pct`, if given, is appended to the ``occl`` row.
    """
    accent = {
        "all": "load",
        "track": "ready",
        "occl": "build",
    }
    if header:
        siamram_log(_paint(header, _DIM), phase=phase, status="info", label="perf",
                    blank_before=True)
    for name, stats in rows:
        status = accent.get(name, "info")
        if not stats:
            siamram_log(_paint("no data", _DIM), phase=phase, status=status, label=name)
            continue
        metrics = _paint(
            f"mean {stats['mean']:6.1f}  med {stats['med']:6.1f}  "
            f"p95 {stats['p95']:6.1f}  p99 {stats['p99']:6.1f}  "
            f"min {stats['min']:6.1f}  max {stats['max']:6.1f} ms",
            _DIM,
        )
        frames = _paint(f"{int(stats['n']):>4} frames", _DIM)
        fps = _paint(f"{stats['fps']:>4.0f} fps", _BOLD + _GREEN)
        message = f"{frames}  {metrics}   {fps}"
        if name == "occl" and occlusion_pct is not None:
            message += _paint(f"   ·  {occlusion_pct:.1f}% of frames", _DIM)
        siamram_log(message, phase=phase, status=status, label=name)
