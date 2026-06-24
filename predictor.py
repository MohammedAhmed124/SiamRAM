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
# ---------------------------------------------------------------------------
# Imports from our own code (under src/) and the checkpoint downloader.
# These come after the sys.path edit above, hence the noqa comments.
# ---------------------------------------------------------------------------
from omegaconf import OmegaConf  # noqa: E402

from download import download_all_checkpoints  # noqa: E402
from models.SiamABC.tracker.tracker_setup import (  # noqa: E402
    get_tracker, normalize_tracker_config_aliases)
from models.siamram.config import (OSNET_CHECKPOINT_CHOICES,  # noqa: E402
                                   flatten_ram_tracker_config)
from models.siamram.tracker import SiamRAMExperimentTracker  # noqa: E402

# ---------------------------------------------------------------------------
# Put src/ on the import path.
# Our actual code (models.*, utils.*) lives under src/, but this file sits one
# level above it. Adding src/ to sys.path lets those imports resolve without us
# having to edit any file inside src/.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Quieten the third-party libraries.
# Torch, TensorRT and Ultralytics print a lot of info/warning noise that makes
# the real output hard to read. We turn their log levels down before importing
# anything heavy so the startup stays clean.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
logging.basicConfig(level=logging.ERROR)
for _noisy in ("torch", "torch_tensorrt", "torch._inductor", "py.warnings", "ultralytics"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

os.environ["TORCHDYNAMO_VERBOSE"] = "0"
if os.environ.get("TORCH_LOGS") == "":
    os.environ.pop("TORCH_LOGS", None)





# When this is True we decode video on a Jetson's NVDEC hardware block instead
# of the CPU. load_model() sets it from the config key runtime.use_nvdec. We
# keep it False by default so on any normal (non-Jetson) machine we use the
# plain OpenCV decoder, which is the original behaviour.
_USE_NVDEC: bool = False

# The folder where the checkpoints get downloaded. We build it from this file's
# location rather than hardcoding an absolute path, so it follows the submission
# wherever it is unzipped.
_CHECKPOINTS_DIR = _HERE / "checkpoints"

# The config we run with. The trailing hash in the name marks which commit's
# behaviour this config reproduces.
_CONFIG_PATH = _SRC / "config" / "inference_config_944f2b8.yaml"


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


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

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
    # 1. Make sure the three checkpoint files are present (model, YOLO, OSNet).
    download_all_checkpoints(checkpoint_dir=str(_CHECKPOINTS_DIR))

    # 2. Load the config and fill in the legacy SiamABC key names the tracker
    #    still reads internally (this only renames keys, it changes no values).
    config = OmegaConf.load(str(_CONFIG_PATH))
    normalize_tracker_config_aliases(config)

    # 3. Apply the process-level runtime switches the config asks for.
    global _USE_NVDEC
    runtime_cfg = config.get("runtime", {}) or {}
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
    _USE_NVDEC = bool(runtime_cfg.get("use_nvdec", False))

    # 4. Flatten the grouped ram_tracker config into the flat keyword arguments
    #    the tracker constructor expects.
    ram_tracker_kwargs = flatten_ram_tracker_config(config)

    # 5. The YOLO and OSNet weight paths come from the config as relative paths.
    #    Resolve them to absolute so they load no matter where we run from. This
    #    only fixes up paths; it does not change any tracker setting.
    if ram_tracker_kwargs.get("yolo_weights"):
        ram_tracker_kwargs["yolo_weights"] = _resolve_checkpoint_path(
            ram_tracker_kwargs["yolo_weights"]
        )
    if ram_tracker_kwargs.get("osnet_model_path"):
        ram_tracker_kwargs["osnet_model_path"] = _resolve_checkpoint_path(
            ram_tracker_kwargs["osnet_model_path"]
        )

    # 6. Build the SiamABC backbone. The trt_engine.trt_compile_siamabc switch
    #    selects the backend: standard PyTorch (get_tracker) or the TensorRT-
    #    compiled backbone (get_trt_tracker). Both return a SiamABCTracker, so the
    #    rest of load_model is identical either way. The TRT factory is imported
    #    lazily so the standard path never requires tensorrt / torch_tensorrt. The
    #    SiamABC checkpoint path and every TRT option come from the config.
    siam_weights_path = _resolve_checkpoint_path(config.get("weights_path", ""))
    trt_cfg = config.get("trt_engine") or {}
    lambda_tta = float(trt_cfg.get("lambda_tta", 0.1))

    # The OSNet descriptor TRT options are declared under trt_engine, but the
    # tracker only receives the flattened ram_tracker kwargs — flatten_ram_tracker_config
    # never reads trt_engine. Bridge them across here so trt_compile_osnet (and the
    # related FP16 / cache switches) actually reach configure_descriptor_backend;
    # without this they silently stay at their False/"" defaults and OSNet never
    # compiles. trt_cache_dir is resolved to an absolute path (like the SiamABC
    # engines) so the cache lands under the repo root regardless of the CWD.
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

    # The YOLO re-detector compiles through Ultralytics' native engine export,
    # but its switches also live under trt_engine, so bridge them the same way
    # as the OSNet ones. The engine is cached under the shared trt_cache_dir
    # (resolved above) and rebuilt when rebuild_trt_cache is set.
    ram_tracker_kwargs["trt_compile_yolo"] = bool(trt_cfg.get("trt_compile_yolo", False))
    ram_tracker_kwargs["yolo_fp16"] = bool(trt_cfg.get("yolo_fp16", False))
    ram_tracker_kwargs["cuda_id"] = int(trt_cfg.get("cuda_id", 0))

    if bool(trt_cfg.get("trt_compile_siamabc", False)):
        from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker

        # Cache the compiled engines so reruns skip the multi-minute build. The
        # cache key is derived from the config/weights/shapes, so a stale entry is
        # never reused for a different setup. Empty dir = caching disabled.
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

    # 7. Sanity-check the OSNet checkpoint choice before building the tracker so
    #    a typo in the config gives a clear error instead of a confusing one
    #    later from torchreid.
    osnet_ckpt = str(
        ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
    ).strip()
    if osnet_ckpt and osnet_ckpt not in OSNET_CHECKPOINT_CHOICES:
        raise ValueError(
            f"Unsupported osnet_pretrained_checkpoint='{osnet_ckpt}'. "
            f"Expected one of: {', '.join(sorted(OSNET_CHECKPOINT_CHOICES))}."
        )

    # 8. Wrap the backbone in the full SiamRAM tracker and hand it back.
    tracker = SiamRAMExperimentTracker(
        siam_tracker=siam_tracker,
        **ram_tracker_kwargs,
    )

    print("[predictor] SiamRAM tracker ready.")
    return tracker


# ---------------------------------------------------------------------------
# Reading the first-frame bounding box
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Opening a video: Jetson hardware decode when asked, CPU decode otherwise
# ---------------------------------------------------------------------------

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


def _open_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file for reading.

    If the config turned on runtime.use_nvdec AND we are actually on a Jetson
    (the nvv4l2decoder plugin is present), we build a GStreamer pipeline that
    decodes the video on the Jetson's NVDEC hardware. That keeps the CPU free for
    the rest of the pipeline and is far faster per frame than software decoding.

    On anything else (an x86 machine, CI, the eval server) we fall back to the
    normal OpenCV reader, which is exactly the original behaviour. We also fall
    back if the GStreamer pipeline refuses to open for any reason, so choosing
    the wrong option never crashes the run, it only loses the hardware speed-up.
    """
    if _USE_NVDEC and _nvv4l2decoder_available():
        # A few appsink options worth explaining:
        #   sync=false    : deliver frames as fast as the decoder can produce
        #                   them instead of pacing to the video's real frame
        #                   rate, otherwise we would be capped at e.g. 30 FPS.
        #   drop=false    : never throw frames away. Tracking needs one output
        #                   row per frame, so a dropped frame would shift every
        #                   later frame's box in the CSV.
        #   max-buffers=4 : let at most four decoded frames wait in the queue.
        #                   When it is full the decoder pauses, which stops the
        #                   pipeline from racing ahead and eating memory.
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
    init_box = read_init_box(init_box_path)

    cap = _open_video(video_path)
    reader: threading.Thread | None = None
    frame_q: queue.Queue | None = None
    stop_event: threading.Event | None = None

    # ------------------------------------------------------------------
    # Latency accounting. Decode (cap.read) runs on the reader thread and
    # overlaps with tracking (model.update) on this thread, so the two totals
    # below add up to MORE than the wall-clock time. We report them per frame
    # so each component's true cost is visible regardless of the overlap.
    # ------------------------------------------------------------------
    decode_time = 0.0      # seconds spent inside cap.read(), all frames
    decode_frames = 0      # number of frames successfully decoded
    track_time = 0.0       # seconds spent inside model.update(), frames 1..N

    reader_exc: list = []
    # The reader thread owns these; we read them only after join(), so no lock
    # is needed. {"time": seconds in cap.read(), "frames": frames decoded}.
    reader_stats = {"time": 0.0, "frames": 0}

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: '{video_path}'")

        # --------------------------------------------------------------
        # Frame 0: read it directly and initialise the tracker on it. This
        # must stay inside the cleanup scope because initialization can fail
        # after the decoder has already allocated file or hardware resources.
        # --------------------------------------------------------------
        _t = time.perf_counter()
        success, first_frame = cap.read()
        decode_time += time.perf_counter() - _t
        if not success or first_frame is None:
            raise RuntimeError(f"Cannot read first frame from: '{video_path}'")
        decode_frames += 1

        model.initialize(first_frame, init_box)

        x0, y0, w0, h0 = init_box
        predictions: list[dict] = [
            {"frame_idx": 0, "x": x0, "y": y0, "w": w0, "h": h0}
        ]

        # --------------------------------------------------------------
        # Frames 1..N: the reader thread decodes ahead while the GPU tracks.
        # A queue depth of 4 keeps the GPU fed without buffering the video.
        # --------------------------------------------------------------
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
                    # Wait for room in the queue, but wake up every so often
                    # so a stop request can never leave us blocked forever.
                    while not stop_event.is_set():
                        try:
                            frame_q.put(fr, timeout=0.25)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                reader_exc.append(exc)
            finally:
                # Push a None to mark end-of-stream, unless the consumer
                # already told us to stop.
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

            _t = time.perf_counter()
            bbox_arr, _score, _in_occ, _yolo = model.update(frame)
            track_time += time.perf_counter() - _t
            x, y, w, h = (float(v) for v in bbox_arr[:4])
            predictions.append(
                {"frame_idx": frame_idx, "x": x, "y": y, "w": w, "h": h}
            )
            frame_idx += 1

    finally:
        # Cleanup covers every failure after opening the capture, including
        # first-frame decoding and model initialization.
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

    # ------------------------------------------------------------------
    # Per-sequence latency report. reader_stats is final now that the reader
    # thread has joined, so we fold its decode total into ours and print the
    # decode (CPU cap.read) and tracker (model.update) costs side by side.
    # ------------------------------------------------------------------
    decode_time += reader_stats["time"]
    decode_frames += reader_stats["frames"]
    track_frames = frame_idx - 1

    decode_ms = 1000.0 * decode_time / decode_frames if decode_frames else 0.0
    track_ms = 1000.0 * track_time / track_frames if track_frames else 0.0
    decode_fps = decode_frames / decode_time if decode_time > 0 else 0.0
    track_fps = track_frames / track_time if track_time > 0 else 0.0

    print(
        f"[latency] decode (CPU): {decode_ms:.2f} ms/frame "
        f"({decode_fps:.1f} FPS, {decode_time:.2f} s over {decode_frames} frames)"
    )
    print(
        f"[latency] tracker:      {track_ms:.2f} ms/frame "
        f"({track_fps:.1f} FPS, {track_time:.2f} s over {track_frames} frames)"
    )

    # Per-component breakdown of the tracker cost, if the tracker recorded it.
    # siamabc / yolo / osnet run *inside* track_update; prescale / homography /
    # motion_kf are separate pipeline stages. The components do not sum to the
    # tracker total (track_update also holds bookkeeping; with GPU stream
    # overlap the model timings are launch-side unless SIAMRAM_PROFILE_SYNC=1).
    comp_times = getattr(model, "_comp_times", None) or {}
    comp_counts = getattr(model, "_comp_counts", None) or {}
    if comp_times:
        order = [
            "prescale", "homography", "motion_kf", "track_update",
            "siamabc", "yolo", "osnet",
        ]
        names = [n for n in order if n in comp_times]
        names += [n for n in comp_times if n not in order]
        print(
            "[latency] tracker components "
            "(siamabc/yolo/osnet are inside track_update):"
        )
        for n in names:
            total = comp_times[n]
            calls = comp_counts.get(n, 0)
            per = 1000.0 * total / calls if calls else 0.0
            print(
                f"[latency]   {n:<12}: {per:7.2f} ms/call "
                f"({total:6.2f} s over {calls} calls)"
            )

    return predictions
