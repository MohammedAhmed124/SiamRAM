"""
Predictor Module — SiamRAM Competition Adapter.

Bridges the MTC-AIC4 Phase 2 inference loop (inference.py) and the SiamRAM
tracking codebase housed under src/.

Public API
----------
load_model(device)   → SiamRAMExperimentTracker
run_tracker(model, video_path, init_box_path) → List[dict]
"""

from __future__ import annotations

import functools
import logging
import os
import queue
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import cv2
import torch

# ---------------------------------------------------------------------------
# 1.  Inject src/ onto sys.path so all internal imports (models.*, utils.*)
#     resolve correctly without touching any source file.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# 2.  Silence noisy third-party log chatter before any heavy imports.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
logging.basicConfig(level=logging.ERROR)
for _noisy in ("torch", "torch_tensorrt", "torch._inductor", "py.warnings", "ultralytics"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

os.environ["TORCHDYNAMO_VERBOSE"] = "0"
if os.environ.get("TORCH_LOGS") == "":
    os.environ.pop("TORCH_LOGS", None)

# ---------------------------------------------------------------------------
# 3.  Now import from the src/ tree and from the top-level download helper.
# ---------------------------------------------------------------------------
from omegaconf import OmegaConf  # noqa: E402

from models.SiamABC.tracker.tracker_setup import (  # noqa: E402
    get_tracker,
    normalize_tracker_config_aliases,
)
from models.siamram.config import (  # noqa: E402
    OSNET_CHECKPOINT_CHOICES,
    flatten_ram_tracker_config,
)
from models.siamram.tracker import SiamRAMExperimentTracker  # noqa: E402

from download import download_all_checkpoints  # noqa: E402


# Master switch for Jetson NVDEC decode.  Set by load_model() from the
# runtime.use_nvdec config key.  Default False restores the old serial
# software-decode behaviour on any non-Jetson machine.
_USE_NVDEC: bool = False

# ---------------------------------------------------------------------------
# Fixed paths (relative to the submission root where inference.py lives).
# ---------------------------------------------------------------------------
_CHECKPOINTS_DIR = _HERE / "checkpoints"
_CONFIG_PATH     = _SRC / "config" / "inference_config_944f2b8.yaml"

_SIAMABC_WEIGHTS = str(_CHECKPOINTS_DIR / "model.pth")
_YOLO_WEIGHTS    = str(_CHECKPOINTS_DIR / "yolo11n.pt")
_OSNET_WEIGHTS   = str(_CHECKPOINTS_DIR / "osnet_x0_25_imagenet.pth")


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def load_model(device: str = "cuda") -> SiamRAMExperimentTracker:
    """
    Download checkpoints (if missing), build the SiamRAM tracker, and return it.

    Called ONCE before the evaluation loop.  Returns the fully initialised
    SiamRAMExperimentTracker with eval-mode SiamABC backbone.
    """
    # -- 3a.  Ensure all checkpoints are present ----------------------------
    download_all_checkpoints(checkpoint_dir=str(_CHECKPOINTS_DIR))

    # -- 3b.  Load the 944f2b8-parity inference config ----------------------
    config = OmegaConf.load(str(_CONFIG_PATH))
    normalize_tracker_config_aliases(config)

    # Override the absolute weights_path baked into the YAML so that the
    # submission works on any machine without touching the config file.
    config.weights_path = _SIAMABC_WEIGHTS

    # Override YOLO path inside the ram_tracker.yolo block.
    if OmegaConf.select(config, "ram_tracker.yolo") is not None:
        OmegaConf.update(config, "ram_tracker.yolo.yolo_weights", _YOLO_WEIGHTS)

    # Override OSNet checkpoint: use 'custom' so torchreid loads from disk.
    if OmegaConf.select(config, "ram_tracker.descriptor") is not None:
        OmegaConf.update(
            config, "ram_tracker.descriptor.osnet_pretrained_checkpoint", "custom"
        )
        OmegaConf.update(
            config, "ram_tracker.descriptor.osnet_model_path", _OSNET_WEIGHTS
        )

    # -- 3c.  Apply runtime settings from config ----------------------------
    global _USE_NVDEC
    runtime_cfg = config.get("runtime", {}) or {}
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
    _USE_NVDEC = bool(runtime_cfg.get("use_nvdec", False))

    # -- 3d.  Flatten ram_tracker config to kwargs dict ---------------------
    ram_tracker_kwargs = flatten_ram_tracker_config(config)

    # Inject the corrected paths (config overrides above are reflected here
    # because flatten_ram_tracker_config re-reads the live OmegaConf object).
    ram_tracker_kwargs["yolo_weights"]                 = _YOLO_WEIGHTS
    ram_tracker_kwargs["osnet_pretrained_checkpoint"]  = "custom"
    ram_tracker_kwargs["osnet_model_path"]             = _OSNET_WEIGHTS

    # Disable TRT compilation — eval sandbox may not have TensorRT.
    ram_tracker_kwargs["trt_compile_osnet"]  = False
    ram_tracker_kwargs["trt_compile_siamabc"] = False
    ram_tracker_kwargs["trt_cache_dir"]      = ""
    ram_tracker_kwargs["trt_rebuild_cache"]  = False

    # -- 3e.  Build the SiamABC backbone (standard PyTorch mode) ------------
    siam_tracker = get_tracker(
        config=config,
        weights_path=_SIAMABC_WEIGHTS,
        lambda_tta=float(
            (config.get("trt_engine") or {}).get("lambda_tta", 0.1)
        ),
        continuous=False,
    )

    # -- 3f.  Validate the osnet_pretrained_checkpoint value ----------------
    osnet_ckpt = str(
        ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
    ).strip()
    if osnet_ckpt and osnet_ckpt not in OSNET_CHECKPOINT_CHOICES:
        raise ValueError(
            f"Unsupported osnet_pretrained_checkpoint='{osnet_ckpt}'. "
            f"Expected one of: {', '.join(sorted(OSNET_CHECKPOINT_CHOICES))}."
        )

    # -- 3g.  Wrap in SiamRAMExperimentTracker ------------------------------
    tracker = SiamRAMExperimentTracker(
        siam_tracker=siam_tracker,
        **ram_tracker_kwargs,
    )

    print("[predictor] SiamRAM tracker ready.")
    return tracker


# ---------------------------------------------------------------------------
# Helper: read first-frame bounding box
# ---------------------------------------------------------------------------

def read_init_box(path: str) -> list[float]:
    """
    Read the initial bounding box from an annotation file.

    Accepts comma-separated or whitespace-separated values on the first
    non-empty line.  Returns [x, y, w, h] as floats.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid bounding-box line in '{path}': {line!r}"
                )
            return [float(parts[0]), float(parts[1]),
                    float(parts[2]), float(parts[3])]
    raise ValueError(f"Annotation file is empty: '{path}'")


# ---------------------------------------------------------------------------
# Video capture helpers — NVDEC on Jetson, CPU decode everywhere else
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _nvv4l2decoder_available() -> bool:
    """
    Return True if the Jetson hardware H.264 decoder GStreamer plugin is
    present and loadable.  Result is cached so the subprocess runs only once
    per process, not once per video.
    """
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "nvv4l2decoder"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def _open_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file for reading.

    When runtime.use_nvdec is true AND nvv4l2decoder is installed (Jetson),
    routes through GStreamer hardware NVDEC (<1 ms/frame).  Otherwise falls
    back to OpenCV's default software decoder — identical behaviour to the
    original code on x86 / CI / eval machines.
    """
    if _USE_NVDEC and _nvv4l2decoder_available():
        gst = (
            f"filesrc location={path} ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )
        try:
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(path)


# ---------------------------------------------------------------------------
# run_tracker
# ---------------------------------------------------------------------------

def run_tracker(
    model: SiamRAMExperimentTracker,
    video_path: str,
    init_box_path: str,
) -> list[dict]:
    """
    Run SiamRAM on one video sequence and return per-frame bounding boxes.

    Args:
        model:         Tracker returned by load_model().
        video_path:    Path to the video file (.mp4 or similar).
        init_box_path: Path to the first-frame annotation (x,y,w,h).

    Returns:
        List of dicts with keys: frame_idx, x, y, w, h.
        The first entry is always the ground-truth init_box.

    Performance notes
    -----------------
    * A background reader thread decodes the next frame while the GPU tracks
      the current one, hiding H.264 decode latency (~8-15 ms on CPU, <1 ms
      with Jetson NVDEC) behind compute.
    * On Jetson Orin Nano, _open_video() routes through GStreamer nvv4l2decoder
      (hardware NVDEC) when available, falling back to software decode silently.
    * Frame order is preserved by the FIFO queue — tracker output is identical
      to a serial loop.
    """
    init_box = read_init_box(init_box_path)

    cap = _open_video(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: '{video_path}'")

    # ------------------------------------------------------------------
    # Frame 0: read synchronously and initialise the tracker.
    # The reader thread starts after this so it never races on frame 0.
    # ------------------------------------------------------------------
    success, first_frame = cap.read()
    if not success or first_frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame from: '{video_path}'")

    model.initialize(first_frame, init_box)

    x0, y0, w0, h0 = init_box
    predictions: list[dict] = [
        {"frame_idx": 0, "x": x0, "y": y0, "w": w0, "h": h0}
    ]

    # ------------------------------------------------------------------
    # Frames 1..N: reader thread decodes ahead while GPU tracks.
    # Queue depth 4 keeps the GPU fed without excessive memory use.
    # ------------------------------------------------------------------
    frame_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    reader_exc: list = []

    def _frame_reader() -> None:
        try:
            while not stop_event.is_set():
                ret, fr = cap.read()
                if not ret or fr is None:
                    break
                # Block until there is room, but wake periodically so a
                # stop request (consumer error/done) never deadlocks us.
                while not stop_event.is_set():
                    try:
                        frame_q.put(fr, timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            reader_exc.append(exc)
        finally:
            # Deliver end-of-stream sentinel; skip if consumer already stopped.
            while not stop_event.is_set():
                try:
                    frame_q.put(None, timeout=0.25)
                    break
                except queue.Full:
                    continue

    reader = threading.Thread(target=_frame_reader, name="frame-prefetch", daemon=True)
    reader.start()

    frame_idx = 1
    try:
        while True:
            frame = frame_q.get()
            if frame is None:
                break

            bbox_arr, _score, _in_occ, _yolo = model.update(frame)
            x, y, w, h = (float(v) for v in bbox_arr[:4])
            predictions.append(
                {"frame_idx": frame_idx, "x": x, "y": y, "w": w, "h": h}
            )
            frame_idx += 1

    finally:
        # Signal reader to stop, drain any buffered frames so it can unblock
        # from a put(), then join before releasing the capture handle.
        stop_event.set()
        try:
            while True:
                frame_q.get_nowait()
        except queue.Empty:
            pass
        reader.join(timeout=2.0)
        cap.release()

    if reader_exc:
        raise reader_exc[0]

    return predictions
