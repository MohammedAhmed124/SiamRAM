"""Small runtime-console helpers for clean SiamRAM terminal output."""

from __future__ import annotations

import contextlib
import logging
import os
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
