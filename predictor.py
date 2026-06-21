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

import logging
import os
import sys
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

    # -- 3c.  Apply cuDNN benchmark setting from config ---------------------
    runtime_cfg = config.get("runtime", {}) or {}
    cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.benchmark = cudnn_benchmark

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
    """
    import numpy as np  # lazy import — ensures it's available at call time

    init_box = read_init_box(init_box_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{video_path}'")

    predictions: list[dict] = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx == 0:
            # ----------------------------------------------------------------
            # Frame 0: initialise the tracker with the ground-truth bbox.
            # initialize() expects a full-resolution BGR numpy array and
            # [x, y, w, h] in pixel coordinates.
            # ----------------------------------------------------------------
            model.initialize(frame, init_box)
            pred_box = init_box  # first prediction is the given init box
        else:
            # ----------------------------------------------------------------
            # Subsequent frames: call update() which returns
            #   (bbox_array, score, in_occlusion, yolo_detections)
            # bbox_array is a numpy array [x, y, w, h] in full-frame pixels.
            # ----------------------------------------------------------------
            bbox_arr, _score, _in_occ, _yolo = model.update(frame)
            pred_box = [float(v) for v in bbox_arr[:4]]

        x, y, w, h = pred_box
        predictions.append(
            {
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
        frame_idx += 1

    cap.release()
    return predictions
