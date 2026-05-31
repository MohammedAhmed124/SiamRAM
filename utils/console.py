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
) -> None:
    """Print one compact, colored SiamRAM status line."""
    labels = {
        "info": ("info", _BLUE),
        "load": ("load", _CYAN),
        "build": ("build", _YELLOW),
        "ready": ("ready", _GREEN),
        "done": ("done", _GREEN),
        "warn": ("warn", _YELLOW),
        "error": ("error", _RED),
    }
    label, color = labels.get(status, labels["info"])
    prefix = (
        f"{_paint('[SiamRAM]', _BOLD + _CYAN)}"
        f"{_paint(f'[{phase}]', _DIM)}"
    )
    pad = "  " * max(0, int(indent))
    print(f"{prefix} {_paint(label.ljust(5), color)} {pad}{message}", flush=True)
