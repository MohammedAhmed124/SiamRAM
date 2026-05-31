"""
Profile SiamRAM inference components on one video.

Example:
    .venv/bin/python profile_components.py \
      --video_key dataset5/person19_3 \
      --yaml_config_path config/inference_config_experimental.yaml \
      --frames 300 \
      --warmup_frames 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

import run_inference as inference
import models.siamram.tracker as siamram_tracker_module
import utils.utils as utils_module
from models.SiamABC.tracker.SiamABC_Tracker import SiamABCTracker
from models.siamram.config import (
    OSNET_CHECKPOINT_CHOICES,
    flatten_ram_tracker_config,
)
from models.siamram.tracker import SiamRAMExperimentTracker
from utils.console import siamram_log, silence_noisy_libraries
from utils.utils import _OSNetDescriptorExtractor, _SiameseDescriptorExtractor

silence_noisy_libraries()


class ComponentProfiler:
    def __init__(self) -> None:
        self.enabled = False
        self.times: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name].append((time.perf_counter() - t0) * 1000.0)

    def record(self, name: str, ms: float) -> None:
        if self.enabled:
            self.times[name].append(float(ms))

    def reset(self) -> None:
        self.times.clear()


def _patch_method(
    cls,
    method_name: str,
    bucket_name: str,
    profiler: ComponentProfiler,
) -> Callable:
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        with profiler.span(bucket_name):
            return original(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)
    return original


def install_profile_patches(profiler: ComponentProfiler) -> list[tuple[object, str, Callable]]:
    patches: list[tuple[object, str, Callable]] = []

    for cls, method_name, bucket_name in [
        (SiamRAMExperimentTracker, "update", "siamram_update_total"),
        (SiamRAMExperimentTracker, "_prescale_frame", "frame_prescale"),
        (SiamRAMExperimentTracker, "_estimate_homography", "camera_motion_homography"),
        (SiamRAMExperimentTracker, "_normal_update", "normal_update_total"),
        (SiamRAMExperimentTracker, "_occlusion_update", "occlusion_update_total"),
        (SiamRAMExperimentTracker, "_maybe_run_class_warmup", "yolo_class_warmup"),
        (SiamRAMExperimentTracker, "_maybe_run_detectability_probe", "yolo_detectability_probe"),
        (SiamRAMExperimentTracker, "_yolo_detect_in_roi", "yolo_detect_in_roi"),
        (SiamRAMExperimentTracker, "_yolo_detect", "yolo_detect_main"),
        (SiamRAMExperimentTracker, "_compute_effective_threshold", "effective_threshold"),
        (SiamABCTracker, "update", "siamabc_update"),
        (_OSNetDescriptorExtractor, "extract_batch", "osnet_extract_batch"),
        (_SiameseDescriptorExtractor, "extract_batch", "siamese_descriptor_extract_batch"),
    ]:
        original = _patch_method(cls, method_name, bucket_name, profiler)
        patches.append((cls, method_name, original))

    original_extract_descriptor = utils_module._extract_descriptor

    def timed_extract_descriptor(*args, **kwargs):
        with profiler.span("descriptor_total"):
            return original_extract_descriptor(*args, **kwargs)

    utils_module._extract_descriptor = timed_extract_descriptor
    siamram_tracker_module._extract_descriptor = timed_extract_descriptor
    patches.append((utils_module, "_extract_descriptor", original_extract_descriptor))
    patches.append((siamram_tracker_module, "_extract_descriptor", original_extract_descriptor))
    return patches


def restore_profile_patches(patches: list[tuple[object, str, Callable]]) -> None:
    for owner, attr, original in reversed(patches):
        setattr(owner, attr, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile component timings for one SiamRAM inference video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_key", required=True, help="Manifest video key to profile.")
    parser.add_argument(
        "--run_split",
        default="public_lb",
        choices=["public_lb", "train", "all"],
        help="Manifest split to read.",
    )
    parser.add_argument(
        "--yaml_config_path",
        default=str(inference.BASE_DIR / "config" / "inference_config_experimental.yaml"),
        help="Inference config YAML path.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(inference.BASE_DIR / "data"),
        help="Root data directory.",
    )
    parser.add_argument(
        "--manifest_path",
        default=str(inference.BASE_DIR / "data" / "metadata" / "contestant_manifest.json"),
        help="Competition manifest JSON path.",
    )
    parser.add_argument(
        "--weights_path",
        default=str(inference.BASE_DIR / "checkpoints" / "inference_checkpoint.pth"),
        help="SiamABC checkpoint path.",
    )
    parser.add_argument(
        "--yolo_weights_path",
        default=None,
        help="YOLO checkpoint path. Defaults to the path from the inference config.",
    )
    parser.add_argument("--model_size", default="M", choices=["S", "M", "L"])
    parser.add_argument("--lambda_tta", type=float, default=0.1)
    parser.add_argument(
        "--disable_siamabc_trt",
        action="store_true",
        help="Profiler-only fallback: force the SiamABC tracker to run without TensorRT.",
    )
    parser.add_argument(
        "--disable_osnet_trt",
        action="store_true",
        help="Profiler-only fallback: force the OSNet descriptor to run without TensorRT.",
    )
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warm-up.")
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=30,
        help="Frames to run before collecting profile timings.",
    )
    return parser.parse_args()


def _build_entries(args: argparse.Namespace) -> dict[str, dict]:
    data_dir = inference._resolve_data_dir(args.data_dir)
    manifest_path = inference._resolve_manifest_path(args.manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_datasets = {
        p.name for p in Path(data_dir).iterdir() if p.is_dir() and p.name != "metadata"
    }

    if args.run_split in {"public_lb", "train"}:
        entries = inference._build_manifest_entries(
            manifest=manifest,
            target_datasets=target_datasets,
            split_name=args.run_split,
            output_layout="dataset",
        )
    else:
        entries = {}
        entries.update(
            inference._build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="train",
                output_layout="dataset",
                key_prefix="train/",
            )
        )
        entries.update(
            inference._build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="public_lb",
                output_layout="dataset",
                key_prefix="test/",
            )
        )

    entries = dict(
        sorted(entries.items(), key=lambda item: (str(item[1].get("dataset", "")), item[0]))
    )
    return inference._filter_run_entries_by_video_key(entries, args.video_key)


def _build_tracker(args: argparse.Namespace):
    config = OmegaConf.load(args.yaml_config_path)
    inference._normalize_tracker_config_aliases(config)
    ram_tracker_kwargs = flatten_ram_tracker_config(config)

    resolved_weights_path = inference._resolve_weights_path(args.weights_path)
    yolo_weights_cfg = str(
        args.yolo_weights_path
        or ram_tracker_kwargs.get("yolo_weights", inference.CHECKPOINTS_DIR / "yolo11n.pt")
    )
    resolved_yolo_path = inference._resolve_weights_path(yolo_weights_cfg)
    resolved_weights_path, resolved_yolo_path = inference._ensure_required_checkpoints(
        resolved_weights_path, resolved_yolo_path
    )

    ram_tracker_kwargs["yolo_weights"] = str(resolved_yolo_path)
    config.model.model_size = args.model_size

    trt_cfg = config.get("trt_engine", {}) or {}
    compile_siamabc = bool(
        trt_cfg.get(
            "trt_compile_siamabc",
            config.get("trt_compile_siamabc", config.get("make_trt_engine", False)),
        )
    )
    if args.disable_siamabc_trt:
        compile_siamabc = False
    ram_tracker_kwargs["osnet_fp16"] = bool(
        trt_cfg.get("osnet_fp16", ram_tracker_kwargs.get("osnet_fp16", False))
    )
    ram_tracker_kwargs["trt_compile_osnet"] = bool(
        trt_cfg.get(
            "trt_compile_osnet", ram_tracker_kwargs.get("trt_compile_osnet", False)
        )
    )
    if args.disable_osnet_trt:
        ram_tracker_kwargs["trt_compile_osnet"] = False

    if compile_siamabc:
        if inference.get_trt_tracker is None:
            raise ModuleNotFoundError(
                "TensorRT tracker dependencies are missing. "
                "Set trt_engine.trt_compile_siamabc: False in your config."
            ) from inference._trt_import_error
        wrapped = inference.get_trt_tracker(
            config=config,
            weights_path=str(resolved_weights_path),
            lambda_tta=float(trt_cfg.get("lambda_tta", args.lambda_tta)),
            fp16=bool(trt_cfg.get("fp16", True)),
            cuda_id=int(trt_cfg.get("cuda_id", 0)),
        )
    else:
        wrapped = inference.get_tracker(
            config=config,
            weights_path=str(resolved_weights_path),
            lambda_tta=args.lambda_tta,
            continuous=False,
        )

    osnet_ckpt = str(
        ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
    ).strip()
    if osnet_ckpt and osnet_ckpt not in OSNET_CHECKPOINT_CHOICES:
        choices = ", ".join(sorted(OSNET_CHECKPOINT_CHOICES))
        raise ValueError(
            f"Unsupported osnet_pretrained_checkpoint='{osnet_ckpt}'. "
            f"Expected one of: {choices}."
        )

    osnet_model_path_raw = str(ram_tracker_kwargs.get("osnet_model_path", "")).strip()
    if osnet_model_path_raw:
        ram_tracker_kwargs["osnet_model_path"] = str(
            inference._resolve_weights_path(osnet_model_path_raw)
        )

    return SiamRAMExperimentTracker(siam_tracker=wrapped, **ram_tracker_kwargs)


def _read_first_frame(video_path: str):
    if os.path.isdir(video_path):
        frame_paths = sorted(
            str(Path(video_path) / name)
            for name in os.listdir(video_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        if not frame_paths:
            raise RuntimeError(f"No image frames found in directory: {video_path}")
        first = cv2.imread(frame_paths[0])
        return first, frame_paths, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, first = cap.read()
    if not ret or first is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    return first, [], cap


def _frame_reader(frame_paths: list[str], cap):
    idx = 1
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame
            continue

        if idx >= len(frame_paths):
            break
        frame = cv2.imread(frame_paths[idx])
        idx += 1
        if frame is not None:
            yield frame


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "total": float(arr.sum()),
    }


def _print_report(
    profiler: ComponentProfiler,
    *,
    video_key: str,
    warmup_frames: int,
    measured_frames: int,
    normal_frames: int,
    occlusion_frames: int,
) -> None:
    frame_mean = (
        np.mean(profiler.times["frame_update_plus_cuda_drain"])
        if profiler.times.get("frame_update_plus_cuda_drain")
        else 0.0
    )
    order = [
        "frame_update_plus_cuda_drain",
        "siamram_update_total",
        "frame_prescale",
        "camera_motion_homography",
        "normal_update_total",
        "siamabc_update",
        "descriptor_total",
        "osnet_extract_batch",
        "siamese_descriptor_extract_batch",
        "yolo_detectability_probe",
        "yolo_detect_in_roi",
        "yolo_detect_main",
        "yolo_class_warmup",
        "effective_threshold",
        "occlusion_update_total",
        "cuda_drain_after_update",
    ]

    print("\n" + "─" * 78)
    print(f"{'SIAMRAM COMPONENT PROFILE':^78}")
    print("─" * 78)
    print(f"video_key       : {video_key}")
    print(f"warmup frames   : {warmup_frames}")
    print(f"measured frames : {measured_frames}")
    print(f"normal/occlusion: {normal_frames}/{occlusion_frames}")
    print("\nInclusive timings. Nested buckets do not sum to total.")
    print("Forced .cpu().numpy() waits are included inside SiamABC/OSNet/YOLO buckets.")
    print("\n  Component                          n     mean    median      p95   % frame")
    print(f"  {'─' * 32} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for name in order:
        values = profiler.times.get(name, [])
        if not values:
            continue
        s = _stats(values)
        pct = (s["mean"] / frame_mean * 100.0) if frame_mean > 0 else 0.0
        print(
            f"  {name:<32} {int(s['n']):5d} "
            f"{s['mean']:8.2f} {s['median']:8.2f} {s['p95']:8.2f} {pct:8.1f}"
        )

    if frame_mean > 0:
        print(f"\n  Effective profiled FPS: {1000.0 / frame_mean:.2f}")
    print("─" * 78 + "\n")


def main() -> None:
    args = parse_args()
    profiler = ComponentProfiler()
    patches = install_profile_patches(profiler)

    try:
        entries = _build_entries(args)
        if len(entries) != 1:
            raise RuntimeError(f"Expected one selected entry, got {len(entries)}.")
        video_key, entry = next(iter(entries.items()))

        video_path = inference._resolve_data_asset_path(entry["video_path"], args.data_dir)
        ann_path = inference._resolve_data_asset_path(entry["annotation_path"], args.data_dir)
        init_bbox = inference._load_initial_bbox(ann_path)

        siamram_log(f"Building tracker for {video_key}", phase="PROFILE", status="load")
        tracker = _build_tracker(args)

        first_frame, frame_paths, cap = _read_first_frame(video_path)
        siamram_log(f"Initializing tracker on {video_path}", phase="PROFILE", status="load")
        t_init = time.perf_counter()
        tracker.initialize(first_frame, init_bbox)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        init_ms = (time.perf_counter() - t_init) * 1000.0
        siamram_log(
            f"Tracker initialize: {init_ms:.1f} ms",
            phase="PROFILE",
            status="ready",
        )

        warmup_target = max(0, int(args.warmup_frames))
        profile_target = max(1, int(args.frames))
        total_target = warmup_target + profile_target

        measured = 0
        normal_frames = 0
        occlusion_frames = 0

        try:
            for frame_idx, frame in enumerate(_frame_reader(frame_paths, cap), start=1):
                profiling_this_frame = frame_idx > warmup_target
                profiler.enabled = profiling_this_frame

                t0 = time.perf_counter()
                result = tracker.update(frame)
                if torch.cuda.is_available():
                    t_sync = time.perf_counter()
                    torch.cuda.synchronize()
                    profiler.record(
                        "cuda_drain_after_update",
                        (time.perf_counter() - t_sync) * 1000.0,
                    )
                profiler.record(
                    "frame_update_plus_cuda_drain",
                    (time.perf_counter() - t0) * 1000.0,
                )

                if profiling_this_frame:
                    measured += 1
                    in_occlusion = bool(result[2]) if len(result) > 2 else False
                    if in_occlusion:
                        occlusion_frames += 1
                    else:
                        normal_frames += 1

                if frame_idx >= total_target:
                    break
        finally:
            profiler.enabled = False
            if cap is not None:
                cap.release()

        _print_report(
            profiler,
            video_key=video_key,
            warmup_frames=min(warmup_target, warmup_target + measured),
            measured_frames=measured,
            normal_frames=normal_frames,
            occlusion_frames=occlusion_frames,
        )
    finally:
        restore_profile_patches(patches)


if __name__ == "__main__":
    main()
