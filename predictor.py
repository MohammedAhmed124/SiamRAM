"""
Predictor for the MTC-AIC4 Phase 2 submission.

This is the one file the competition asks us to write. The organisers' runner
(inference.py) imports two functions from here:

    load_model(device)                      -> builds and returns our tracker
    run_tracker(model, video, init_box)     -> tracks one video, returns boxes

Everything else (the SiamRAM tracker itself, the SiamABC backbone, the utils)
lives unchanged under src/. This file is just the glue that wires our code into
the format the runner expects.

How the settings are decided:
We keep all of the tunable settings in the YAML config under
src/config/. This file does NOT hardcode tracker behaviour or overwrite config
values. It loads the config, resolves the checkpoint paths to absolute paths so
the submission runs from any working directory, and hands the values straight to
the tracker. If you want to change how the tracker behaves, edit the config, not
this file.
"""

from __future__ import annotations

import functools
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import cv2
import torch
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omegaconf import OmegaConf  # noqa: E402

from utils.console import siamram_log  # noqa: E402

from download import download_all_checkpoints  # noqa: E402
from models.SiamABC.tracker.tracker_setup import (  # noqa: E402
    get_tracker, normalize_tracker_config_aliases)
from models.siamram.config import (OSNET_CHECKPOINT_CHOICES,  # noqa: E402
                                   flatten_ram_tracker_config)
from models.siamram.tracker import SiamRAMExperimentTracker  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
logging.basicConfig(level=logging.ERROR)
for _noisy in ("torch", "torch_tensorrt", "torch._inductor", "py.warnings", "ultralytics"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

os.environ["TORCHDYNAMO_VERBOSE"] = "0"
if os.environ.get("TORCH_LOGS") == "":
    os.environ.pop("TORCH_LOGS", None)





_USE_NVDEC: bool = False

_CHECKPOINTS_DIR = _HERE / "checkpoints"

# Which inference config to load. Defaults to the standard (Small) config so
# existing behaviour is unchanged. Set SIAMRAM_INFERENCE_CONFIG to a path (or a
# bare filename resolved against src/config) to run a different model, e.g.
# SIAMRAM_INFERENCE_CONFIG=inference_config_tiny.yaml for SiamABC-Tiny.
_DEFAULT_CONFIG_PATH = _SRC / "config" / "inference_config.yaml"
_config_override = os.environ.get("SIAMRAM_INFERENCE_CONFIG", "").strip()
if _config_override:
    _override_path = Path(_config_override).expanduser()
    _CONFIG_PATH = _override_path if _override_path.is_absolute() else (_SRC / "config" / _override_path)
else:
    _CONFIG_PATH = _DEFAULT_CONFIG_PATH


def _resolve_checkpoint_path(path_value: str) -> str:
    """
    Turn a checkpoint path from the config into a usable absolute path.

    Paths in the config are written relative to the submission root (for example
    "checkpoints/model.pth") so they stay portable and contain no machine-specific
    absolute paths. Here we expand them against this file's directory. If the
    config already gives an absolute path we leave it alone, so you can still
    point at a checkpoint somewhere else on disk if you want.
    """
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_HERE / path).resolve())



def load_model(device: str = "cuda") -> SiamRAMExperimentTracker:
    """
    Build the SiamRAM tracker and return it. The runner calls this once, before
    it starts looping over the videos.

    The steps are:
      1. Download the checkpoints if they are not already on disk.
      2. Load the YAML config and let it drive every setting.
      3. Resolve the checkpoint paths from the config to absolute paths.
      4. Build the SiamABC backbone and wrap it in the SiamRAM tracker.

    We deliberately do not override any behavioural setting here. Whatever the
    config says is what the tracker gets, so the config file is the single place
    to change anything.
    """
    download_all_checkpoints(checkpoint_dir=str(_CHECKPOINTS_DIR))

    config = OmegaConf.load(str(_CONFIG_PATH))
    normalize_tracker_config_aliases(config)

    global _USE_NVDEC
    runtime_cfg = config.get("runtime", {}) or {}
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
    _USE_NVDEC = bool(runtime_cfg.get("use_nvdec", False))

    ram_tracker_kwargs = flatten_ram_tracker_config(config)

    if ram_tracker_kwargs.get("yolo_weights"):
        ram_tracker_kwargs["yolo_weights"] = _resolve_checkpoint_path(
            ram_tracker_kwargs["yolo_weights"]
        )
    if ram_tracker_kwargs.get("osnet_model_path"):
        ram_tracker_kwargs["osnet_model_path"] = _resolve_checkpoint_path(
            ram_tracker_kwargs["osnet_model_path"]
        )

    siam_weights_path = _resolve_checkpoint_path(config.get("weights_path", ""))
    trt_cfg = config.get("trt_engine") or {}
    lambda_tta = float(trt_cfg.get("lambda_tta", 0.1))

    osnet_trt_cache_dir = str(trt_cfg.get("trt_cache_dir", "") or "").strip()
    if osnet_trt_cache_dir:
        _osnet_cache_path = Path(osnet_trt_cache_dir).expanduser()
        if not _osnet_cache_path.is_absolute():
            _osnet_cache_path = _HERE / _osnet_cache_path
        _osnet_cache_path.mkdir(parents=True, exist_ok=True)
        osnet_trt_cache_dir = str(_osnet_cache_path)
    ram_tracker_kwargs["trt_compile_osnet"] = bool(trt_cfg.get("trt_compile_osnet", False))
    ram_tracker_kwargs["osnet_fp16"] = bool(trt_cfg.get("osnet_fp16", False))
    ram_tracker_kwargs["osnet_async_overlap"] = bool(trt_cfg.get("osnet_async_overlap", False))
    ram_tracker_kwargs["trt_cache_dir"] = osnet_trt_cache_dir
    ram_tracker_kwargs["trt_rebuild_cache"] = bool(trt_cfg.get("rebuild_trt_cache", False))

    ram_tracker_kwargs["trt_compile_yolo"] = bool(trt_cfg.get("trt_compile_yolo", False))
    ram_tracker_kwargs["yolo_fp16"] = bool(trt_cfg.get("yolo_fp16", False))
    ram_tracker_kwargs["cuda_id"] = int(trt_cfg.get("cuda_id", 0))

    if bool(trt_cfg.get("trt_compile_siamabc", False)):
        from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker

        engine_cache_dir = str(trt_cfg.get("trt_cache_dir", "") or "").strip()
        if engine_cache_dir:
            engine_cache_path = Path(engine_cache_dir).expanduser()
            if not engine_cache_path.is_absolute():
                engine_cache_path = _HERE / engine_cache_path
            engine_cache_path.mkdir(parents=True, exist_ok=True)
            engine_cache_dir = str(engine_cache_path)

        siam_tracker = get_trt_tracker(
            config=config,
            weights_path=siam_weights_path,
            lambda_tta=lambda_tta,
            fp16=bool(trt_cfg.get("fp16", True)),
            cuda_id=int(trt_cfg.get("cuda_id", 0)),
            backbone_mode=str(trt_cfg.get("backbone_mode", "")),
            disable_tf32=bool(trt_cfg.get("backbone_disable_tf32", False)),
            engine_cache_dir=engine_cache_dir,
            rebuild_cache=bool(trt_cfg.get("rebuild_trt_cache", False)),
        )
    else:
        siam_tracker = get_tracker(
            config=config,
            weights_path=siam_weights_path,
            lambda_tta=lambda_tta,
            continuous=False,
        )

    osnet_ckpt = str(
        ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
    ).strip()
    if osnet_ckpt and osnet_ckpt not in OSNET_CHECKPOINT_CHOICES:
        raise ValueError(
            f"Unsupported osnet_pretrained_checkpoint='{osnet_ckpt}'. "
            f"Expected one of: {', '.join(sorted(OSNET_CHECKPOINT_CHOICES))}."
        )

    tracker = SiamRAMExperimentTracker(
        siam_tracker=siam_tracker,
        **ram_tracker_kwargs,
    )

    siamram_log("SiamRAM tracker ready", phase="INIT", status="ready")
    return tracker



def read_init_box(path: str) -> list[float]:
    """
    Read the starting bounding box for a video from its annotation file.

    The file's first non-empty line holds the box as four numbers: x, y, width,
    height. We accept either commas or spaces between them so it does not matter
    which the annotation uses. Returns the four numbers as floats.
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



@functools.lru_cache(maxsize=1)
def _nvv4l2decoder_available() -> bool:
    """
    Check whether the Jetson hardware H.264 decoder (the nvv4l2decoder GStreamer
    plugin) is installed and loadable.

    We ask gst-inspect once and cache the answer, so we do not pay the cost of
    spawning a subprocess for every single video. If gst-inspect is missing or
    errors out we just treat the plugin as unavailable.
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


def _open_video(path: str):
    """
    Open a video file for reading.

    runtime.use_nvdec is the single master switch for hardware H.264 decode.
    When it is on we pick the best NVDEC backend available for this machine and
    fall back to the next option whenever one is missing, so turning it on never
    crashes the run — it only loses the speed-up:

      1. Jetson: if the nvv4l2decoder GStreamer plugin is present we decode on
         the Jetson's NVDEC block through a GStreamer pipeline (original branch).
      2. x86 + CUDA: otherwise we try PyNvVideoCodec (NvdecVideoCapture), which
         decodes on the desktop/server GPU's NVDEC block. It is API-compatible
         with the subset of cv2.VideoCapture this module uses, so run_tracker
         consumes it unchanged.
      3. CPU: if neither hardware decoder opens the file we fall back to the
         plain OpenCV reader, which is exactly the original behaviour.

    The returned object is always either a cv2.VideoCapture or an
    NvdecVideoCapture; both expose isOpened(), read(), get() and release(), the
    only capture methods run_tracker calls.
    """
    if _USE_NVDEC and _nvv4l2decoder_available():
        gst = (
            f"filesrc location={path} ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false drop=false max-buffers=4"
        )
        try:
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    if _USE_NVDEC:
        try:
            from utils.nvdec_reader import NvdecVideoCapture

            nvdec_cap = NvdecVideoCapture(path)
            if nvdec_cap.isOpened():
                return nvdec_cap
            nvdec_cap.release()
        except Exception:
            pass

    return cv2.VideoCapture(path)



def _resolve_input_path(path_value: str) -> str:
    """
    Resolve a manifest path (video or annotation) to a file that exists.
    """
    if not path_value:
        return path_value
    given = Path(path_value).expanduser()
    if given.is_absolute():
        return str(given)
    for candidate in (Path(path_value), _HERE / path_value, _HERE / "data" / path_value):
        if candidate.exists():
            return str(candidate)
    return path_value



def run_tracker(
    model: SiamRAMExperimentTracker,
    video_path: str,
    init_box_path: str,
) -> list[dict]:
    """
    Track the target through one video and return its box on every frame.

    Args:
        model:         the tracker returned by load_model().
        video_path:    path to the video file.
        init_box_path: path to the first-frame annotation holding x, y, w, h.

    Returns:
        A list of dicts, one per frame, each with keys frame_idx, x, y, w, h.
        The first entry is the ground-truth box we were given to start with.

    A note on speed:
    We read the video on a background thread. While the GPU is busy tracking the
    current frame, that thread is already decoding the next one, so the decode
    time hides behind the compute time instead of adding to it. The frames go
    through a small FIFO queue, so their order is preserved and the result is
    exactly the same as a plain one-frame-at-a-time loop.
    """
    init_box_path = _resolve_input_path(init_box_path)
    video_path = _resolve_input_path(video_path)

    init_box = read_init_box(init_box_path)

    cap = _open_video(video_path)
    video_name = Path(video_path).stem
    reader: threading.Thread | None = None
    frame_q: queue.Queue | None = None
    stop_event: threading.Event | None = None

    decode_time = 0.0      # seconds spent inside cap.read(), all frames
    decode_frames = 0      # number of frames successfully decoded

    reader_exc: list = []
    reader_stats = {"time": 0.0, "frames": 0}

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: '{video_path}'")

    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq_start = time.perf_counter()
    try:
        _t = time.perf_counter()
        success, first_frame = cap.read()
        decode_time += time.perf_counter() - _t
        if not success or first_frame is None:
            raise RuntimeError(f"Cannot read first frame from: '{video_path}'")
        decode_frames += 1

        model.begin_sequence(video_name, total_frames_hint)
        model.initialize(first_frame, init_box)

        x0, y0, w0, h0 = init_box
        predictions: list[dict] = [
            {"frame_idx": 0, "x": x0, "y": y0, "w": w0, "h": h0}
        ]

        frame_q = queue.Queue(maxsize=4)
        stop_event = threading.Event()

        def _frame_reader() -> None:
            try:
                while not stop_event.is_set():
                    _t = time.perf_counter()
                    ret, fr = cap.read()
                    reader_stats["time"] += time.perf_counter() - _t
                    if not ret or fr is None:
                        break
                    reader_stats["frames"] += 1
                    while not stop_event.is_set():
                        try:
                            frame_q.put(fr, timeout=0.25)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                reader_exc.append(exc)
            finally:
                while not stop_event.is_set():
                    try:
                        frame_q.put(None, timeout=0.25)
                        break
                    except queue.Full:
                        continue

        reader = threading.Thread(
            target=_frame_reader, name="frame-prefetch", daemon=True
        )
        reader.start()

        frame_idx = 1
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
        if stop_event is not None:
            stop_event.set()
        if frame_q is not None:
            try:
                while True:
                    frame_q.get_nowait()
            except queue.Empty:
                pass
        if reader is not None and reader.is_alive():
            reader.join(timeout=2.0)
        cap.release()

    if reader_exc:
        raise reader_exc[0]

    seq_elapsed = time.perf_counter() - seq_start

    decode_time += reader_stats["time"]
    decode_frames += reader_stats["frames"]

    model.end_sequence(
        elapsed_s=seq_elapsed,
        decode_time_s=decode_time,
        decode_frames=decode_frames,
    )

    return predictions
