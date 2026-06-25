"""
SiamRAM GFLOPs + latency profiler.

Loads the tracker, runs it against a video (default: RcCar4), measures
per-component GFLOPs and latency, then prints two summary sections in
SiamRAM console style:

  1. GFLOPs breakdown — per sub-module GFLOPs, average calls/frame, and
     weighted GFLOPs/frame averaged over the profiled video.
  2. Latency breakdown — per-component average ms and fps at the granularity
     the deployed backend exposes (backbone+neck are fused inside
     get_features(); attention+connector are fused inside track()).

Usage (from repo root):
    python src/profiler.py
    python src/profiler.py --video data/dataset2/RcCar4/RcCar4_24.mp4 \\
                           --annot data/dataset2/RcCar4/annotation.txt
    python src/profiler.py --frames 200   # fewer frames for call-count stats
    python src/profiler.py --warmup 10 --reps 50  # faster (less precise) timing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — mirrors predictor.py so src/ imports always resolve
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # src/
_REPO = _HERE.parent                     # repo root
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils.console as _con
_con.silence_noisy_libraries()

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.console import (
    _BOLD, _CYAN, _DIM, _GREEN, _BLUE, _YELLOW, _RED, _RESET,
    _paint, siamram_log,
)

_CONFIG_PATH  = _HERE / "config" / "inference_config.yaml"
_DEFAULT_VIDEO = _REPO / "data" / "dataset2" / "RcCar4" / "RcCar4_24.mp4"
_DEFAULT_ANNOT = _REPO / "data" / "dataset2" / "RcCar4" / "annotation.txt"
_PHASE = "PROF"

try:
    from torch.utils.flop_counter import FlopCounterMode as _FlopCounterMode
    _HAS_FLOP_COUNTER = True
except ImportError:
    _HAS_FLOP_COUNTER = False


# ---------------------------------------------------------------------------
# Low-level measurement helpers
# ---------------------------------------------------------------------------

def _count_gflops(module: nn.Module, *args, **kwargs) -> float:
    """One forward pass through module(*args, **kwargs); return GFLOPs.

    Uses PyTorch FlopCounterMode. Divides by 2e9 because FlopCounterMode counts
    multiply and add as separate raw FLOPs (2 per MAC), while the standard ML
    convention treats 1 MAC = 1 FLOP. Result matches thop / published papers.
    Returns 0.0 on failure.
    """
    if not _HAS_FLOP_COUNTER:
        return 0.0
    try:
        with _FlopCounterMode(display=False) as fcm:
            with torch.no_grad():
                module(*args, **kwargs)
        return fcm.get_total_flops() / 2e9  # raw_FLOPs / 2 = GFLOPs (ML convention)
    except Exception:
        return 0.0


def _time_ms(fn, *args, warmup: int = 20, reps: int = 100, **kwargs) -> float:
    """Time fn(*args, **kwargs) with CUDA sync; return mean latency in ms."""
    use_cuda = torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args, **kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args, **kwargs)
        if use_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


# ---------------------------------------------------------------------------
# SiamABC: fresh PyTorch net for GFLOPs, deployed net for timing
# ---------------------------------------------------------------------------

def _siamabc_sizes(config) -> Tuple[int, int]:
    """(template_size, search_size) read from config with safe fallbacks."""
    try:
        t = int(config.tracker.architectural_config.template_crop_size)
        s = int(config.tracker.architectural_config.search_crop_size)
        return t, s
    except Exception:
        pass
    try:
        t = int(config.tracker.get("template_size") or 128)
        s = int(config.tracker.get("instance_size") or 256)
        return t, s
    except Exception:
        return 128, 256


def _build_fresh_net(config) -> nn.Module:
    """Instantiate SiamABCNet from config (no weight download, just architecture)."""
    from hydra.utils import instantiate
    net = instantiate(config.model, pretrained=False, inference_mode=True)
    if torch.cuda.is_available():
        net = net.cuda()
    return net.eval()


def _profile_siamabc_gflops(fresh_net: nn.Module, t_sz: int, s_sz: int) -> Dict[str, float]:
    """GFLOPs for each SiamABC sub-component on synthetic inputs at the right sizes."""
    dev = next(fresh_net.parameters()).device
    t_crop = torch.zeros(1, 3, t_sz, t_sz, device=dev)
    s_crop = torch.zeros(1, 3, s_sz, s_sz, device=dev)

    with torch.no_grad():
        t_enc  = fresh_net.encoder(t_crop)
        s_enc  = fresh_net.encoder(s_crop)
        t_feat = fresh_net.neck(t_enc)
        s_feat = fresh_net.neck(s_enc)
        t_comb = torch.cat([t_feat, t_feat], dim=1)
        s_comb = torch.cat([s_feat, s_feat], dim=1)
        t_psa  = fresh_net.polarized_self_attention(t_comb)
        s_psa  = fresh_net.polarized_self_attention(s_comb)
        t_mix  = fresh_net.attention_neck(t_psa)
        s_mix  = fresh_net.attention_neck(s_psa)
        lam    = torch.zeros(1, device=dev)

    return {
        "backbone/template":    _count_gflops(fresh_net.encoder, t_crop),
        "neck/template":        _count_gflops(fresh_net.neck,    t_enc),
        "backbone/search":      _count_gflops(fresh_net.encoder, s_crop),
        "neck/search":          _count_gflops(fresh_net.neck,    s_enc),
        "attention/template":   _count_gflops(fresh_net.polarized_self_attention, t_comb),
        "attn-neck/template":   _count_gflops(fresh_net.attention_neck,           t_psa),
        "attention/search":     _count_gflops(fresh_net.polarized_self_attention, s_comb),
        "attn-neck/search":     _count_gflops(fresh_net.attention_neck,           s_psa),
        "connector":            _count_gflops(
            fresh_net.connect_model,
            search_org=s_feat, search=s_mix, kernel=t_mix, lam=lam,
        ),
    }


def _profile_siamabc_timing(
    net,
    t_sz: int,
    s_sz: int,
    warmup: int,
    reps: int,
) -> Dict[str, float]:
    """Time SiamABC through the actual deployed backend (TRT or eager).

    Returns ms for:
      get_features/template — backbone + neck on the template crop
      get_features/search   — backbone + neck on the search crop
      track                 — attention×2 + attn-neck×2 + connector
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_feat = getattr(net, "get_features", None)
    if get_feat is None:
        return {}

    t_crop = torch.zeros(1, 3, t_sz, t_sz, device=dev)
    s_crop = torch.zeros(1, 3, s_sz, s_sz, device=dev)

    result: Dict[str, float] = {}
    result["get_features/template"] = _time_ms(get_feat, t_crop, warmup=warmup, reps=reps)
    result["get_features/search"]   = _time_ms(get_feat, s_crop, warmup=warmup, reps=reps)

    track_fn = getattr(net, "track", None)
    if track_fn is not None:
        with torch.no_grad():
            tf  = get_feat(t_crop)
            dtf = get_feat(t_crop)
            sf  = get_feat(s_crop)
            dsf = get_feat(s_crop)
            lam = torch.zeros(1, device=dev)
        result["track"] = _time_ms(track_fn, sf, dsf, tf, dtf, lam, warmup=warmup, reps=reps)

    return result


# ---------------------------------------------------------------------------
# OSNet profiling
# ---------------------------------------------------------------------------

def _profile_osnet(extractor, warmup: int, reps: int) -> Tuple[float, float]:
    """GFLOPs and ms/call for the OSNet descriptor backbone (one 256×128 crop)."""
    if extractor is None:
        return 0.0, 0.0
    model = extractor._model
    h, w  = extractor._INPUT_H, extractor._INPUT_W
    dev   = extractor._device
    dtype = extractor._dtype
    dummy_t  = torch.zeros(1, 3, h, w, device=dev, dtype=dtype)
    gf = _count_gflops(model, dummy_t)
    dummy_np = np.zeros((h, w, 3), dtype=np.uint8)
    ms = _time_ms(extractor.extract_batch, [dummy_np], warmup=warmup, reps=reps)
    return gf, ms


# ---------------------------------------------------------------------------
# YOLO profiling
# ---------------------------------------------------------------------------

def _yolo_gflops(config, imgsz: int) -> float:
    """GFLOPs for YOLO via FlopCounterMode on a fresh PyTorch model.

    Loads the same weights used at runtime (no TRT), extracts the inner
    DetectionModel nn.Module, and measures with the same _count_gflops()
    path used for all SiamABC components — same convention, no surprises.
    """
    try:
        yolo_weights = str(config.ram_tracker.yolo.yolo_weights)
    except Exception:
        yolo_weights = "checkpoints/yolo11n.pt"
    weights_path = _REPO / yolo_weights
    if not weights_path.exists():
        weights_path = Path(yolo_weights)

    try:
        from ultralytics import YOLO
        from utils.console import quiet_external_logs
        with quiet_external_logs():
            fresh = YOLO(str(weights_path))

        # fresh.model is DetectionModel — a proper nn.Module with forward().
        # fresh.model.model is an nn.ModuleList (not callable), so stop here.
        inner: nn.Module = fresh.model.eval()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inner = inner.to(dev)
        dummy = torch.rand(1, 3, imgsz, imgsz, device=dev)
        return _count_gflops(inner, dummy)
    except Exception:
        return 0.0


def _profile_yolo(tracker, config, imgsz: int, warmup: int, reps: int) -> Tuple[float, float]:
    """GFLOPs and ms/call for YOLO at imgsz×imgsz."""
    gf = _yolo_gflops(config, imgsz)

    dummy_bgr = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        ms = _time_ms(tracker._predict_yolo, dummy_bgr, warmup=warmup, reps=reps)
    except Exception:
        ms = 0.0
    return gf, ms


# ---------------------------------------------------------------------------
# Video run — realistic call-count statistics
# ---------------------------------------------------------------------------

def _run_video_counts(tracker, video_path: str, annot_path: str, max_frames: int) -> Dict[str, int]:
    """Run the tracker on one video and count per-component neural-net calls.

    Patched counters (restored on exit):
      total_frames — frames fed to tracker.update() (excludes frame 0)
      yolo         — YOLO predict() invocations
      osnet        — OSNet/descriptor extract_batch invocations
    """
    counts: Dict[str, int] = defaultdict(int)

    # Wrap YOLO
    _orig_yolo = tracker._predict_yolo
    def _yolo_counted(image, conf=None):
        counts["yolo"] += 1
        return _orig_yolo(image, conf)
    tracker._predict_yolo = _yolo_counted

    # Wrap descriptor extraction
    from utils import utils as _utils_mod
    _orig_desc = _utils_mod._extract_descriptor
    def _desc_counted(frame, bbox, **kw):
        counts["osnet"] += 1
        return _orig_desc(frame, bbox, **kw)
    _utils_mod._extract_descriptor = _desc_counted

    try:
        # Read first-frame bounding box
        init_box: List[float] = []
        with open(annot_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    init_box = [float(v) for v in line.replace(",", " ").split()[:4]]
                    break
        if not init_box:
            raise ValueError(f"No bounding box found in {annot_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        run_n = min(total_in_file, max_frames) if max_frames > 0 else total_in_file
        siamram_log(
            f"{Path(video_path).name}  ·  {run_n}/{total_in_file} frames",
            phase=_PHASE, status="load",
        )

        tracker.begin_sequence(Path(video_path).stem, run_n)
        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        tracker.initialize(frame0, init_box)

        while True:
            if max_frames > 0 and counts["total_frames"] >= max_frames - 1:
                break
            ret, frame = cap.read()
            if not ret:
                break
            tracker.update(frame)
            counts["total_frames"] += 1

        cap.release()

        # Harvest OSNet count from tracker's own _comp_counts as a fallback
        # (covers async paths that bypass the patched _extract_descriptor)
        comp_counts = getattr(tracker, "_comp_counts", {})
        if counts["osnet"] == 0 and "osnet" in comp_counts:
            counts["osnet"] = comp_counts["osnet"]

    finally:
        tracker._predict_yolo = _orig_yolo
        _utils_mod._extract_descriptor = _orig_desc

    return dict(counts)


# ---------------------------------------------------------------------------
# Console table (SiamRAM style)
# ---------------------------------------------------------------------------

_SEP = "─"
# GFLOPs table column widths: component | GFLOPs | calls/fr | GF/fr | avg ms | fps
_GFC = (30, 8, 9, 9, 8, 7)
# Latency table column widths: component | avg ms | fps
_LTC = (38, 10, 10)


def _pf() -> str:
    return (
        f"{_paint('[SiamRAM]', _BOLD + _CYAN)}"
        f"{_paint(f'[{_PHASE}]', _DIM)} "
    )


def _hline(width: int = 80) -> None:
    print(_pf() + _paint(_SEP * width, _DIM), flush=True)


def _pad_color(text: str, width: int, color: str) -> str:
    """Right-pad `text` to `width` chars (based on visible length), then colorize."""
    pad = max(0, width - len(text))
    return " " * pad + _paint(text, color)


def _gf_section_header(title: str) -> None:
    _hline()
    siamram_log(_paint(title, _DIM), phase=_PHASE, status="info", label="head ")
    _hline()
    cols = ("component", "GFLOPs", "calls/fr", "GFLOPs/fr", "avg ms", "fps")
    head = (
        cols[0].ljust(_GFC[0]) + "  "
        + "  ".join(c.rjust(_GFC[i + 1]) for i, c in enumerate(cols[1:]))
    )
    print(_pf() + _paint("cols ", _DIM) + _paint(head, _BOLD), flush=True)
    _hline()


def _lat_section_header(title: str) -> None:
    _hline()
    siamram_log(_paint(title, _DIM), phase=_PHASE, status="info", label="head ")
    _hline()
    cols = ("component", "avg ms", "fps")
    head = (
        cols[0].ljust(_LTC[0]) + "  "
        + "  ".join(c.rjust(_LTC[i + 1]) for i, c in enumerate(cols[1:]))
    )
    print(_pf() + _paint("cols ", _DIM) + _paint(head, _BOLD), flush=True)
    _hline()


def _gf_row(
    name: str,
    gflops: Optional[float],
    cpf: Optional[float],
    label: str = "info",
    color: str = _BLUE,
) -> float:
    """Print one GFLOPs table row; return the GF/frame contribution."""
    na = "—"
    gf_fr = (gflops * cpf) if (gflops is not None and cpf is not None) else None

    gf_s  = na if gflops is None else f"{gflops:.3f}"
    cpf_s = na if cpf    is None else f"{cpf:.3f}"
    gff_s = na if gf_fr  is None else f"{gf_fr:.3f}"

    # Colorize GF/frame value without breaking right-alignment
    gff_col = (
        _pad_color(gff_s, _GFC[3], _GREEN)
        if gf_fr is not None
        else gff_s.rjust(_GFC[3])
    )

    row = (
        name.ljust(_GFC[0]) + "  "
        + gf_s.rjust(_GFC[1]) + "  "
        + cpf_s.rjust(_GFC[2]) + "  "
        + gff_col + "  "
        + na.rjust(_GFC[4]) + "  "
        + na.rjust(_GFC[5])
    )
    print(_pf() + _paint(label[:5].ljust(5), color) + " " + row, flush=True)
    return gf_fr or 0.0


def _lat_row(
    name: str,
    avg_ms: Optional[float],
    note: str = "",
    label: str = "info",
    color: str = _BLUE,
) -> None:
    """Print one latency table row (name | avg ms | fps) with optional note."""
    fps   = (1000.0 / avg_ms) if (avg_ms and avg_ms > 0) else None
    ms_s  = "—" if avg_ms is None else f"{avg_ms:.2f} ms"
    fps_s = "—" if fps    is None else f"{fps:.1f} fps"
    note_s = f"   {_paint(note, _DIM)}" if note else ""
    row = (
        name.ljust(_LTC[0]) + "  "
        + _paint(ms_s.rjust(_LTC[1]),  _GREEN if fps else _DIM) + "  "
        + _paint(fps_s.rjust(_LTC[2]), _BOLD + _GREEN if fps else _DIM)
        + note_s
    )
    print(_pf() + _paint(label[:5].ljust(5), color) + " " + row, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SiamRAM GFLOPs + latency profiler")
    ap.add_argument("--video",   default=str(_DEFAULT_VIDEO), help="video file path")
    ap.add_argument("--annot",   default=str(_DEFAULT_ANNOT), help="annotation .txt path")
    ap.add_argument("--frames",  type=int, default=300,
                    help="max frames for call-count stats (0 = full video)")
    ap.add_argument("--warmup",  type=int, default=20,  help="timing warmup iterations")
    ap.add_argument("--reps",    type=int, default=100, help="timing measurement iterations")
    args = ap.parse_args()

    os.chdir(_REPO)  # so config-relative checkpoint paths resolve correctly

    # ── 1. config ──────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from models.SiamABC.tracker.tracker_setup import normalize_tracker_config_aliases

    config = OmegaConf.load(str(_CONFIG_PATH))
    normalize_tracker_config_aliases(config)

    # ── 2. load tracker (may compile TRT engines on first run) ─────────────
    siamram_log("loading tracker", phase=_PHASE, status="load", blank_before=True)
    from predictor import load_model
    tracker = load_model(device="cuda" if torch.cuda.is_available() else "cpu")
    siam_net = tracker.tracker.net

    t_sz, s_sz = _siamabc_sizes(config)
    siamram_log(
        f"backbone  template={t_sz}px  search={s_sz}px  |  "
        f"YOLO imgsz={tracker.yolo_imgsz}px",
        phase=_PHASE, status="info",
    )

    # ── 3. GFLOPs on fresh PyTorch model ───────────────────────────────────
    siamram_log("measuring GFLOPs (fresh PyTorch model, no TRT)", phase=_PHASE, status="build")
    if not _HAS_FLOP_COUNTER:
        siamram_log(
            "torch.utils.flop_counter unavailable — GFLOPs will show as 0",
            phase=_PHASE, status="warn",
        )
    fresh_net = _build_fresh_net(config)
    gf = _profile_siamabc_gflops(fresh_net, t_sz, s_sz)
    del fresh_net   # free memory before loading OSNet timing

    # ── 4. timing on deployed backend ──────────────────────────────────────
    siamram_log("timing SiamABC (deployed backend)", phase=_PHASE, status="build")
    ms = _profile_siamabc_timing(siam_net, t_sz, s_sz, args.warmup, args.reps)

    # ── 5. OSNet ──────────────────────────────────────────────────────────
    siamram_log("profiling OSNet", phase=_PHASE, status="build")
    from utils.utils import _DESCRIPTOR_BACKEND, _get_osnet_extractor
    osnet_ext = _get_osnet_extractor() if _DESCRIPTOR_BACKEND == "osnet" else None
    gf_osnet, ms_osnet = _profile_osnet(osnet_ext, args.warmup, args.reps)

    # ── 6. YOLO ───────────────────────────────────────────────────────────
    siamram_log(
        f"profiling YOLO {tracker.yolo_imgsz}×{tracker.yolo_imgsz} (FP16)",
        phase=_PHASE, status="build",
    )
    gf_yolo, ms_yolo = _profile_yolo(tracker, config, tracker.yolo_imgsz, args.warmup, args.reps)

    # ── 7. run video for call-count statistics ─────────────────────────────
    siamram_log("running video — collecting call counts", phase=_PHASE, status="load")
    counts = _run_video_counts(tracker, args.video, args.annot, args.frames)
    total_frames = max(1, counts.get("total_frames", 1))
    yolo_cpf  = counts.get("yolo",  0) / total_frames
    osnet_cpf = counts.get("osnet", 0) / total_frames

    # ── 8. call-frequency estimates from config ────────────────────────────
    try:
        update_interval = int(config.tracker.dynamic_template.update_interval)
    except Exception:
        update_interval = 40
    try:
        descriptor_stride = int(
            config.ram_tracker.descriptor.descriptor_stride
        )
    except Exception:
        descriptor_stride = 1

    bb_t_cpf   = 1.0 / max(1, update_interval)  # template backbone every N frames
    bb_s_cpf   = 1.0                              # search backbone every frame
    # PSA is called for both template and search branches each frame
    att_t_cpf  = 1.0
    att_s_cpf  = 1.0
    an_t_cpf   = 1.0   # attn-neck/template
    an_s_cpf   = 1.0   # attn-neck/search
    conn_cpf   = 1.0

    # OSNet/YOLO: use measured call counts from video; fall back to config
    if osnet_cpf == 0.0:
        osnet_cpf = 1.0 / max(1, descriptor_stride)

    # ── 9. GFLOPs table ───────────────────────────────────────────────────
    print("", flush=True)
    siamram_log(
        _paint(
            f"GFLOPs + Latency Report  ·  {Path(args.video).name}"
            f"  ·  {total_frames} frames profiled",
            _BOLD,
        ),
        phase=_PHASE, status="info", label="perf ",
    )

    _gf_section_header(
        f"GFLOPs breakdown  (PyTorch arch, 1 FLOP = 1 MAC · FlopCounterMode / 2)"
        f"  —  update_interval={update_interval}  descriptor_stride={descriptor_stride}"
    )

    normal_gf_fr = 0.0
    occ_gf_fr    = 0.0

    normal_gf_fr += _gf_row("siamabc · backbone / template",  gf.get("backbone/template"),  bb_t_cpf,  "load", _CYAN)
    normal_gf_fr += _gf_row("siamabc · neck / template",      gf.get("neck/template"),       bb_t_cpf,  "load", _CYAN)
    normal_gf_fr += _gf_row("siamabc · backbone / search",    gf.get("backbone/search"),     bb_s_cpf,  "load", _CYAN)
    normal_gf_fr += _gf_row("siamabc · neck / search",        gf.get("neck/search"),         bb_s_cpf,  "load", _CYAN)
    normal_gf_fr += _gf_row("siamabc · attention / template", gf.get("attention/template"),  att_t_cpf, "info", _BLUE)
    normal_gf_fr += _gf_row("siamabc · attn-neck / template", gf.get("attn-neck/template"),  an_t_cpf,  "info", _BLUE)
    normal_gf_fr += _gf_row("siamabc · attention / search",   gf.get("attention/search"),    att_s_cpf, "info", _BLUE)
    normal_gf_fr += _gf_row("siamabc · attn-neck / search",   gf.get("attn-neck/search"),    an_s_cpf,  "info", _BLUE)
    normal_gf_fr += _gf_row("siamabc · connector",            gf.get("connector"),           conn_cpf,  "info", _BLUE)
    normal_gf_fr += _gf_row("osnet · descriptor (256×128)",   gf_osnet or None,              osnet_cpf, "ready",_GREEN)
    occ_gf_fr    += _gf_row(
        f"yolo · {tracker.yolo_imgsz}px FP16  [half-res]",
        gf_yolo or None,
        yolo_cpf,
        "build", _YELLOW,
    )

    total_gf_fr = normal_gf_fr + occ_gf_fr

    _hline()

    def _summary_row(name: str, value: float, tag: str, color: str) -> None:
        """One summary row: fixed name + GFLOPs/frame value, rest are dashes."""
        col = _pad_color(f"{value:.3f}", _GFC[3], color)
        print(
            _pf() + _paint(tag[:5].ljust(5), color)
            + " " + _paint(name.ljust(_GFC[0]), _BOLD)
            + "  " + "—".rjust(_GFC[1])
            + "  " + "—".rjust(_GFC[2])
            + "  " + col
            + "  " + "—".rjust(_GFC[4])
            + "  " + "—".rjust(_GFC[5]),
            flush=True,
        )

    _summary_row("avg GFLOPs / frame",  total_gf_fr,  "done ",  _BOLD + _GREEN)
    _summary_row("├─ normal tracking",  normal_gf_fr, "     ",  _GREEN)
    _summary_row("└─ occlusion mode",   occ_gf_fr,    "     ",  _RED)
    _hline()

    # ── 10. latency table ─────────────────────────────────────────────────
    print("", flush=True)
    _lat_section_header(
        "Latency breakdown  (deployed backend — TRT if configured)"
        f"  —  {args.warmup} warmup + {args.reps} reps, CUDA-synced"
    )

    _lat_row(
        "siamabc · backbone+neck / template",
        ms.get("get_features/template"),
        note="backbone+neck fused inside get_features()",
        label="load", color=_CYAN,
    )
    _lat_row(
        "siamabc · backbone+neck / search",
        ms.get("get_features/search"),
        note="backbone+neck fused inside get_features()",
        label="load", color=_CYAN,
    )
    _lat_row(
        "siamabc · track forward",
        ms.get("track"),
        note="attention×2 + attn-neck×2 + connector fused in track()",
        label="info", color=_BLUE,
    )
    _lat_row(
        "osnet · descriptor  (1 crop, 256×128)",
        ms_osnet if ms_osnet > 0 else None,
        label="ready", color=_GREEN,
    )
    _lat_row(
        f"yolo · {tracker.yolo_imgsz}px FP16  (1 frame, [half-res])",
        ms_yolo if ms_yolo > 0 else None,
        note="called only during occlusion recovery",
        label="build", color=_YELLOW,
    )

    _hline()
    # End-to-end estimate for a typical normal-tracking frame
    bb_t_ms  = ms.get("get_features/template", 0.0)
    bb_s_ms  = ms.get("get_features/search",   0.0)
    track_ms = ms.get("track",                 0.0)
    ee = bb_s_ms + track_ms + bb_t_ms / max(1, update_interval)
    if ms_osnet > 0:
        ee += ms_osnet / max(1, descriptor_stride)
    _lat_row(
        "estimated end-to-end  (tracking mode)",
        ee if ee > 0 else None,
        note=f"bb_search + track + bb_template/{update_interval} + osnet/{descriptor_stride}",
        label="done", color=_GREEN,
    )
    _hline()
    print("", flush=True)


if __name__ == "__main__":
    main()
