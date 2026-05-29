"""A/B test for Telescope-style tiny-object single-object tracking."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tiny_objects.foveation import FoveationConfig
from experiments.tiny_objects.tiny_tracker import TinyObjectRecord, TinyObjectTracker, clip_bbox_to_frame
from models.SiamABC.tracker.tracker_setup import get_tracker
from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
from models.SiamRAM import SiamRAMTracker
from run_inference import _ensure_required_checkpoints, _resolve_weights_path
from utils.logger import create_logger


logger = create_logger(__name__)

DEFAULT_SEQUENCE_DIR = Path(r"E:\Coding projects\AIC-4-Hackathon\contest_release\dataset3\air_conditioning_box2")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "tiny_objects"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "inference_config.yaml"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "inference_checkpoint.pth"
TINY_VARIANT_NAME = "telescope_tiny_objects"


@dataclass
class RunArtifacts:
    run_name: str
    output_dir: Path
    bbox_file: Path
    metrics_file: Path
    summary_file: Path
    diagnostic_log: Optional[Path]
    annotated_video: Optional[Path]


TINY_LOG_KEYS = [field.name for field in fields(TinyObjectRecord)]


def parse_bbox_line(line: str) -> list[float]:
    parts = [float(part.strip()) for part in line.replace("\t", ",").split(",") if part.strip()]
    if len(parts) < 4:
        raise ValueError(f"Expected bbox with four values, got {line!r}")
    return parts[:4]


def load_annotations(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(parse_bbox_line(line))
    if not rows:
        raise ValueError(f"No annotations found in {path}")
    return np.asarray(rows, dtype=np.float32)


def find_video(sequence_dir: Path) -> Path:
    videos = sorted([p for p in sequence_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    if not videos:
        raise FileNotFoundError(f"No video found in {sequence_dir}")
    return videos[0]


def compute_iou(gt_box: Any, pred_box: Any) -> float:
    gx, gy, gw, gh = [float(v) for v in np.asarray(gt_box).reshape(-1)[:4]]
    px, py, pw, ph = [float(v) for v in np.asarray(pred_box).reshape(-1)[:4]]
    if gw <= 0 or gh <= 0 or pw <= 0 or ph <= 0:
        return 0.0
    gx2, gy2 = gx + gw, gy + gh
    px2, py2 = px + pw, py + ph
    ix1, iy1 = max(gx, px), max(gy, py)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = gw * gh + pw * ph - inter
    return float(inter / union) if union > 0 else 0.0


def center_error(gt_box: Any, pred_box: Any) -> float:
    gx, gy, gw, gh = [float(v) for v in np.asarray(gt_box).reshape(-1)[:4]]
    px, py, pw, ph = [float(v) for v in np.asarray(pred_box).reshape(-1)[:4]]
    gcx, gcy = gx + gw / 2.0, gy + gh / 2.0
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    return float(np.hypot(gcx - pcx, gcy - pcy))


def normalized_center_error(gt_box: Any, pred_box: Any) -> Optional[float]:
    gx, gy, gw, gh = [float(v) for v in np.asarray(gt_box).reshape(-1)[:4]]
    diag = float(np.hypot(gw, gh))
    if diag <= 0:
        return None
    return center_error(gt_box, pred_box) / diag


def success_auc(ious: list[float]) -> float:
    if not ious:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 21)
    values = np.asarray(ious, dtype=np.float32)
    return float(np.mean([(values >= threshold).mean() for threshold in thresholds]))


def _valid_pred(row: dict[str, Any]) -> bool:
    return float(row["pred_w"]) > 0 and float(row["pred_h"]) > 0


def _center_step_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    prev: Optional[tuple[float, float]] = None
    for row in rows:
        if not _valid_pred(row):
            prev = None
            continue
        center = (float(row["pred_x"]) + float(row["pred_w"]) / 2.0, float(row["pred_y"]) + float(row["pred_h"]) / 2.0)
        if prev is not None:
            values.append(float(np.hypot(center[0] - prev[0], center[1] - prev[1])))
        prev = center
    return values


def _association_state(row: dict[str, Any], threshold: float = 0.2) -> bool:
    return _valid_pred(row) and float(row["iou"]) >= threshold


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["iou"]) for row in rows]
    center_errors = [float(row["center_error"]) for row in rows]
    norm_errors = [float(row["norm_center_error"]) for row in rows if row["norm_center_error"] is not None]
    center_steps = _center_step_values(rows)
    associated = [_association_state(row) for row in rows]
    lost_segments = 0
    recoveries = 0
    prev_state: Optional[bool] = None
    for state in associated:
        if prev_state is None:
            lost_segments += int(not state)
        else:
            lost_segments += int(prev_state and not state)
            recoveries += int((not prev_state) and state)
        prev_state = state
    return {
        "frames": len(rows),
        "success_auc": success_auc(ious),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "association_quality_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "association_quality_iou_ge_0_5": float(np.mean(np.asarray(ious) >= 0.5)) if ious else 0.0,
        "precision_20px": float(np.mean(np.asarray(center_errors) <= 20.0)) if center_errors else 0.0,
        "normalized_precision_0_5": float(np.mean(np.asarray(norm_errors) <= 0.5)) if norm_errors else 0.0,
        "mean_center_error": float(np.mean(center_errors)) if center_errors else 0.0,
        "median_center_error": float(np.median(center_errors)) if center_errors else 0.0,
        "tracking_stability_center_step_mean": float(np.mean(center_steps)) if center_steps else 0.0,
        "tracking_stability_center_step_std": float(np.std(center_steps)) if center_steps else 0.0,
        "tracking_stability_large_jump_count": int(np.sum(np.asarray(center_steps) > 50.0)) if center_steps else 0,
        "id_switches_proxy": int(recoveries),
        "id_switch_proxy_definition": "single-object proxy: count transitions from IoU<0.2/lost to IoU>=0.2/associated",
        "lost_segments_proxy": int(lost_segments),
        "false_positive_proxy_iou_lt_0_1": int(np.sum(np.asarray(ious) < 0.1)) if ious else 0,
        "false_negative_proxy_iou_lt_0_1": int(np.sum(np.asarray(ious) < 0.1)) if ious else 0,
        "failure_count_iou_0": int(np.sum(np.asarray(ious) <= 0.0)) if ious else 0,
        "mean_runtime_ms": float(np.mean([float(row["runtime_ms"]) for row in rows])) if rows else 0.0,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value)


def _draw_bbox(frame: np.ndarray, bbox: Any, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x, y, w, h = [int(v) for v in np.asarray(bbox).reshape(-1)[:4]]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(frame, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    line_h = 18
    pad = 8
    width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, 1)
        width = max(width, tw)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, width + 2 * pad + 8), 8 + len(lines) * line_h + 2 * pad), (8, 18, 22), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = 8 + pad + 5
    for line in lines:
        cv2.putText(frame, line, (16, y), font, scale, (235, 245, 240), 1, cv2.LINE_AA)
        y += line_h


def _clone_config_with_resolved_paths(config_path: Path, weights_path: Path, force_no_trt: bool) -> Any:
    config = OmegaConf.load(config_path)
    if force_no_trt:
        config.make_trt_engine = False
    if "ram_tracker" in config:
        config.ram_tracker.disable_camera_motion = True
        config.ram_tracker.yolo_weights = str(_resolve_weights_path(str(config.ram_tracker.yolo_weights)))
        osnet_model_path_raw = str(getattr(config.ram_tracker, "osnet_model_path", "")).strip()
        if osnet_model_path_raw:
            config.ram_tracker.osnet_model_path = str(_resolve_weights_path(osnet_model_path_raw))
    return config


def build_experiment_tracker(
    *,
    config_path: Path,
    weights_path: Path,
    model_size: str,
    lambda_tta: float,
    tracker_kind: str,
    force_no_trt: bool,
) -> Any:
    config = _clone_config_with_resolved_paths(config_path, weights_path, force_no_trt)
    config.model.model_size = model_size
    if bool(config.make_trt_engine):
        siam_tracker = get_trt_tracker(config=config, weights_path=str(weights_path), **config.trt_engine)
    else:
        siam_tracker = get_tracker(config=config, weights_path=str(weights_path), lambda_tta=lambda_tta, continuous=False)
    if tracker_kind == "siamabc":
        return siam_tracker
    if tracker_kind == "siamram":
        tracker = SiamRAMTracker(siam_tracker=siam_tracker, **config.ram_tracker)
        if torch.cuda.is_available() and hasattr(tracker, "yolo") and hasattr(tracker.yolo, "to"):
            tracker.yolo.to("cuda")
        return tracker
    raise ValueError(f"Unknown tracker kind: {tracker_kind}")


def ensure_gpu_or_fail(allow_cpu: bool) -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device 0: %s", device_name)
        return
    if allow_cpu:
        logger.warning("CUDA is not available; continuing on CPU because --allow-cpu was provided.")
        return
    raise RuntimeError(
        "CUDA is not available. This experiment is configured to require GPU execution. "
        "Install/use a CUDA PyTorch environment or pass --allow-cpu only for debugging."
    )


def _draw_foveation_guides(frame: np.ndarray, tracker: Optional[TinyObjectTracker]) -> None:
    if tracker is None or tracker.last_state is None:
        return
    center = tuple(np.round(tracker.last_state.center_xy).astype(int))
    radius = int(round(tracker.last_state.radius_px))
    cv2.circle(frame, center, max(2, radius), (255, 80, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, center, max(2, radius // 2), (255, 80, 220), 1, cv2.LINE_AA)
    cv2.drawMarker(frame, center, (255, 80, 220), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1, line_type=cv2.LINE_AA)
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        end = (int(round(center[0] + radius * np.cos(angle))), int(round(center[1] + radius * np.sin(angle))))
        cv2.line(frame, center, end, (120, 60, 120), 1, cv2.LINE_AA)


def _draw_foveated_inset(frame: np.ndarray, tracker: Optional[TinyObjectTracker]) -> None:
    if tracker is None or tracker.last_foveated_frame is None or tracker.last_state is None:
        return

    height, width = frame.shape[:2]
    inset_w = int(np.clip(width * 0.24, 180, 300))
    inset_h = int(np.clip(height * 0.24, 140, 240))
    x0 = max(8, width - inset_w - 10)
    y0 = 10

    cx, cy = int(tracker.last_state.center_xy[0]), int(tracker.last_state.center_xy[1])
    r = int(tracker.last_state.radius_px * 1.5)
    r = max(r, 60)

    crop_x1, crop_y1 = max(0, cx - r), max(0, cy - r)
    crop_x2, crop_y2 = min(width, cx + r), min(height, cy + r)
    bubble_crop = tracker.last_foveated_frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if bubble_crop.size == 0:
        return

    inset = cv2.resize(bubble_crop, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    scale_x = inset_w / max(crop_x2 - crop_x1, 1)
    scale_y = inset_h / max(crop_y2 - crop_y1, 1)

    def scaled_box(bbox: Any) -> np.ndarray:
        x, y, w, h = np.asarray(bbox, dtype=float).reshape(-1)[:4]
        return np.asarray([(x - crop_x1) * scale_x, (y - crop_y1) * scale_y, w * scale_x, h * scale_y], dtype=float)

    if tracker.last_foveated_prior is not None:
        _draw_bbox(inset, scaled_box(tracker.last_foveated_prior), (0, 190, 255), "search", 2)
    if tracker.last_foveated_pred is not None:
        _draw_bbox(inset, scaled_box(tracker.last_foveated_pred), (0, 230, 80), "pred", 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + inset_w + 4, y0 + inset_h + 22), (8, 18, 22), -1)
    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
    frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
    cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (255, 80, 220), 1, cv2.LINE_AA)
    gain = "?"
    if tracker.records:
        gain = f"{float(tracker.records[-1].tiny_predicted_gain):.1f}"
    elif tracker.last_state is not None:
        gain = f"{tracker.last_state.alpha:.1f}"
    cv2.putText(
        frame,
        f"FOVEATED ZOOM ({gain}x)",
        (x0, y0 + inset_h + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 235, 250),
        1,
        cv2.LINE_AA,
    )


def _crop_dynamic_template(dynamic_image: np.ndarray, dynamic_bbox: Any) -> Optional[np.ndarray]:
    if dynamic_image is None or dynamic_bbox is None:
        return None
    image = np.asarray(dynamic_image)
    if image.ndim != 3 or image.size == 0:
        return None
    x, y, w, h = np.asarray(dynamic_bbox, dtype=float).reshape(-1)[:4]
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        return None
    height, width = image.shape[:2]
    pad = max(float(w), float(h), 8.0) * 1.5
    cx, cy = x + w / 2.0, y + h / 2.0
    x1 = int(max(0, np.floor(cx - pad)))
    y1 = int(max(0, np.floor(cy - pad)))
    x2 = int(min(width, np.ceil(cx + pad)))
    y2 = int(min(height, np.ceil(cy + pad)))
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def _draw_dynamic_template_thumbnail(frame: np.ndarray, tracker: Optional[TinyObjectTracker]) -> None:
    if tracker is None or not tracker.last_dynamic_template:
        return
    dyn = tracker.last_dynamic_template
    crop = _crop_dynamic_template(dyn.get("image"), dyn.get("bbox"))
    if crop is None:
        return

    height, width = frame.shape[:2]
    thumb_w = int(np.clip(width * 0.18, 150, 240))
    thumb_h = int(np.clip(height * 0.16, 96, 160))
    x0 = max(8, width - thumb_w - 10)
    foveated_inset_h = int(np.clip(height * 0.24, 140, 240))
    y0 = 10 + foveated_inset_h + 34
    if y0 + thumb_h + 24 > height - 8:
        y0 = max(10, height - thumb_h - 32)

    thumb = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    refreshed = False
    if tracker.records:
        refreshed = bool(tracker.records[-1].dynamic_template_refreshed)
    border = (70, 255, 120) if refreshed else (255, 210, 70)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + thumb_w + 4, y0 + thumb_h + 22), (8, 18, 22), -1)
    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
    frame[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    cv2.rectangle(frame, (x0, y0), (x0 + thumb_w, y0 + thumb_h), border, 2, cv2.LINE_AA)
    label = "DYN TEMPLATE"
    if refreshed:
        label += " UPDATED"
    cv2.putText(
        frame,
        label,
        (x0, y0 + thumb_h + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 235, 250),
        1,
        cv2.LINE_AA,
    )


def _render_frame(
    frame: np.ndarray,
    *,
    run_name: str,
    frame_idx: int,
    gt_bbox: np.ndarray,
    pred_bbox: np.ndarray,
    score: float,
    iou: float,
    tiny_tracker: Optional[TinyObjectTracker],
) -> np.ndarray:
    canvas = frame.copy()
    _draw_bbox(canvas, gt_bbox, (255, 150, 0), "GT", 2)
    _draw_bbox(canvas, pred_bbox, (0, 230, 80) if iou >= 0.5 else (0, 90, 255), "PRED", 2)
    lines = [f"{run_name} | frame={frame_idx:04d} score={score:.3f} iou={iou:.3f}"]
    if tiny_tracker is not None:
        _draw_foveation_guides(canvas, tiny_tracker)
        if tiny_tracker.last_focus_bbox is not None:
            _draw_bbox(canvas, tiny_tracker.last_focus_bbox, (0, 190, 255), "focus", 1)
        log = tiny_tracker.records_as_dicts()[-1] if tiny_tracker.records else {}
        if tiny_tracker.last_state is not None:
            dyn = tiny_tracker.last_dynamic_template or {}
            dyn_best = dyn.get("best_score")
            dyn_threshold = dyn.get("threshold")
            dyn_line = (
                "dyn_template "
                f"enabled={int(bool(dyn.get('enabled', False)))} "
                f"refresh={int(bool(log.get('dynamic_template_refreshed', False)))} "
                f"mem={int(dyn.get('memory_size') or 0)}"
            )
            if dyn_best is not None and dyn_threshold is not None:
                dyn_line += f" best={float(dyn_best):.3f}/{float(dyn_threshold):.3f}"
            lines.extend(
                [
                    f"tiny foveation R={tiny_tracker.last_state.radius_px:.1f}px alpha={tiny_tracker.last_state.alpha:.2f} gain={float(log.get('tiny_predicted_gain', 0.0) or 0.0):.2f}",
                    f"inverse err={float(log.get('tiny_roundtrip_error_px', 0.0) or 0.0):.4f}px search_set={int(bool(log.get('tiny_prior_set', False)))}",
                    dyn_line,
                    "legend: GT=orange PRED=green/red focus/search=yellow fovea=magenta",
                ]
            )
        _draw_foveated_inset(canvas, tiny_tracker)
        _draw_dynamic_template_thumbnail(canvas, tiny_tracker)
    _draw_hud(canvas, lines)
    return canvas


def _tracker_score(result: Any) -> float:
    return float(result[1]) if hasattr(result, "__len__") and len(result) > 1 else 0.0


def run_tracker_once(
    *,
    run_name: str,
    tracker: Any,
    video_path: Path,
    annotations: np.ndarray,
    output_dir: Path,
    max_frames: int,
    output_video: bool,
    progress_every: int,
) -> tuple[dict[str, Any], RunArtifacts]:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tiny_tracker = tracker if isinstance(tracker, TinyObjectTracker) else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    initial_bbox = clip_bbox_to_frame(annotations[0], first_frame.shape[:2])
    tracker.initialize(first_frame, initial_bbox)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(annotations)
    planned_frames = min(total_frames, max_frames) if max_frames else total_frames
    logger.info("[%s] starting | planned_frames=%s | output_video=%s", run_name, planned_frames, bool(output_video))

    writer: Optional[cv2.VideoWriter] = None
    annotated_video: Optional[Path] = None
    if output_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        height, width = first_frame.shape[:2]
        annotated_video = run_dir / f"{video_path.stem}_{run_name}_annotated.mp4"
        writer = cv2.VideoWriter(str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated video: {annotated_video}")

    rows: list[dict[str, Any]] = []
    preds: list[np.ndarray] = []

    def append_row(frame_idx: int, pred_bbox: np.ndarray, score: float, runtime_ms: float) -> dict[str, Any]:
        gt = annotations[frame_idx] if frame_idx < len(annotations) else annotations[-1]
        row: dict[str, Any] = {
            "frame": frame_idx,
            "gt_x": float(gt[0]),
            "gt_y": float(gt[1]),
            "gt_w": float(gt[2]),
            "gt_h": float(gt[3]),
            "pred_x": float(pred_bbox[0]),
            "pred_y": float(pred_bbox[1]),
            "pred_w": float(pred_bbox[2]),
            "pred_h": float(pred_bbox[3]),
            "score": float(score),
            "iou": compute_iou(gt, pred_bbox),
            "center_error": center_error(gt, pred_bbox),
            "norm_center_error": normalized_center_error(gt, pred_bbox),
            "runtime_ms": float(runtime_ms),
        }
        if tiny_tracker is not None:
            log = tiny_tracker.records_as_dicts()[-1] if tiny_tracker.records else {}
            for key in TINY_LOG_KEYS:
                value = log.get(key)
                row_key = f"tiny_log_{key}" if key in {"frame", "pred_bbox", "score", "runtime_ms"} else key
                row[row_key] = _json_or_none(value) if isinstance(value, (list, tuple, dict)) else value
        rows.append(row)
        preds.append(pred_bbox.copy())
        return row

    first_row = append_row(0, initial_bbox, 1.0, 0.0)
    if writer is not None:
        writer.write(
            _render_frame(
                first_frame,
                run_name=run_name,
                frame_idx=0,
                gt_bbox=annotations[0],
                pred_bbox=initial_bbox,
                score=1.0,
                iou=float(first_row["iou"]),
                tiny_tracker=tiny_tracker,
            )
        )

    frame_idx = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames and frame_idx >= max_frames:
                break
            start = time.perf_counter()
            result = tracker.update(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            pred_bbox = clip_bbox_to_frame(result[0], frame.shape[:2])
            score = _tracker_score(result)
            row = append_row(frame_idx, pred_bbox, score, elapsed_ms)
            if progress_every > 0 and (frame_idx % progress_every == 0 or frame_idx == planned_frames - 1):
                msg = ""
                if tiny_tracker is not None and tiny_tracker.records:
                    log = tiny_tracker.records_as_dicts()[-1]
                    msg = (
                        f" | gain={float(log.get('tiny_predicted_gain') or 0.0):.2f}"
                        f" R={float(log.get('tiny_radius_px') or 0.0):.1f}px"
                        f" accepted={int(bool(log.get('tiny_update_accepted')))}"
                    )
                logger.info(
                    "[%s] frame=%04d/%04d score=%.3f iou=%.3f err=%.1fpx latency=%.1fms%s",
                    run_name,
                    frame_idx,
                    max(planned_frames - 1, 0),
                    score,
                    float(row["iou"]),
                    float(row["center_error"]),
                    elapsed_ms,
                    msg,
                )
            if writer is not None:
                gt = annotations[frame_idx] if frame_idx < len(annotations) else annotations[-1]
                writer.write(
                    _render_frame(
                        frame,
                        run_name=run_name,
                        frame_idx=frame_idx,
                        gt_bbox=gt,
                        pred_bbox=pred_bbox,
                        score=score,
                        iou=float(row["iou"]),
                        tiny_tracker=tiny_tracker,
                    )
                )
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    bbox_file = run_dir / f"{video_path.stem}.txt"
    np.savetxt(bbox_file, np.round(np.asarray(preds, dtype=float)).astype(int), fmt="%d", delimiter=" ")
    metrics_file = run_dir / "tracking_metrics_by_frame.csv"
    write_rows(metrics_file, rows)
    diagnostic_log = None
    if tiny_tracker is not None:
        diagnostic_log = run_dir / "tiny_telescope_log.csv"
        write_rows(diagnostic_log, tiny_tracker.records_as_dicts())
        (run_dir / "tiny_telescope_log.json").write_text(json.dumps(tiny_tracker.records_as_dicts(), indent=2), encoding="utf-8")

    summary = summarise_rows(rows)
    summary.update(
        {
            "run_name": run_name,
            "video_path": str(video_path),
            "bbox_file": str(bbox_file),
            "metrics_file": str(metrics_file),
            "diagnostic_log": str(diagnostic_log) if diagnostic_log else None,
            "annotated_video": str(annotated_video) if annotated_video else None,
        }
    )
    if tiny_tracker is not None:
        logs = tiny_tracker.records_as_dicts()
        summary.update(
            {
                "tiny_foveation_frames": len(logs),
                "tiny_update_acceptance_ratio": float(np.mean([bool(row["tiny_update_accepted"]) for row in logs])) if logs else 0.0,
                "tiny_mean_foveation_gain": float(np.mean([float(row["tiny_predicted_gain"]) for row in logs])) if logs else 0.0,
                "tiny_median_radius_px": float(np.median([float(row["tiny_radius_px"]) for row in logs])) if logs else 0.0,
                "tiny_max_roundtrip_error_px": float(np.max([float(row["tiny_roundtrip_error_px"]) for row in logs])) if logs else 0.0,
            }
        )
    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, RunArtifacts(run_name, run_dir, bbox_file, metrics_file, summary_file, diagnostic_log, annotated_video)


def delta_dict(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "success_auc",
        "mean_iou",
        "median_iou",
        "association_quality_iou_ge_0_5",
        "precision_20px",
        "normalized_precision_0_5",
        "mean_center_error",
        "tracking_stability_center_step_std",
        "tracking_stability_large_jump_count",
        "id_switches_proxy",
        "false_positive_proxy_iou_lt_0_1",
        "false_negative_proxy_iou_lt_0_1",
        "failure_count_iou_0",
        "mean_runtime_ms",
    ]
    return {key: variant.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def frame_delta_summary(baseline_metrics: Path, variant_metrics: Path) -> dict[str, Any]:
    def load_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    baseline_rows = load_csv(baseline_metrics)
    variant_rows = load_csv(variant_metrics)
    n = min(len(baseline_rows), len(variant_rows))
    deltas = [float(variant_rows[idx]["iou"]) - float(baseline_rows[idx]["iou"]) for idx in range(n)]
    return {
        "frames_iou_improved_by_0_05": int(sum(delta >= 0.05 for delta in deltas)),
        "frames_iou_regressed_by_0_05": int(sum(delta <= -0.05 for delta in deltas)),
        "mean_frame_iou_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def write_report(path: Path, comparison: dict[str, Any]) -> None:
    baseline = comparison["baseline"]
    variant = comparison[TINY_VARIANT_NAME]
    delta = comparison["delta"]
    frame_delta = comparison["frame_delta"]
    lines = [
        "# Tiny Objects Telescope A/B Report",
        "",
        f"Sequence: `{comparison['sequence_dir']}`",
        f"Video: `{comparison['video_path']}`",
        f"Tracker: `{comparison['tracker_kind']}`",
        "",
        "| Metric | Baseline | Telescope tiny objects | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "success_auc",
        "mean_iou",
        "association_quality_iou_ge_0_5",
        "precision_20px",
        "mean_center_error",
        "tracking_stability_center_step_std",
        "tracking_stability_large_jump_count",
        "failure_count_iou_0",
        "mean_runtime_ms",
    ]:
        lines.append(f"| `{key}` | {baseline.get(key, 0):.4f} | {variant.get(key, 0):.4f} | {delta.get(key, 0):+.4f} |")
    lines.extend(
        [
            "",
            "## Tiny-Object Module",
            "",
            "- The variant uses a Telescope hyperbolic foveation frontend/backend around the previous accepted target box.",
            "- The tracker receives foveated frames and predicts in foveated coordinates; the wrapper maps predictions back to Euclidean image coordinates before scoring.",
            f"- Mean foveation gain: `{variant.get('tiny_mean_foveation_gain', 0):.4f}`",
            f"- Median foveation radius: `{variant.get('tiny_median_radius_px', 0):.2f}px`",
            f"- Update acceptance ratio: `{variant.get('tiny_update_acceptance_ratio', 0):.4f}`",
            f"- Max inverse roundtrip error: `{variant.get('tiny_max_roundtrip_error_px', 0):.6f}px`",
            "",
            "## Frame Delta",
            "",
            f"- Frames improved by IoU >= 0.05: `{frame_delta['frames_iou_improved_by_0_05']}`",
            f"- Frames regressed by IoU <= -0.05: `{frame_delta['frames_iou_regressed_by_0_05']}`",
            f"- Mean per-frame IoU delta: `{frame_delta['mean_frame_iou_delta']:.4f}`",
            "",
            "Annotated variant video legend: GT=orange, predicted box=green/red, focus box=yellow, foveation rings=magenta, and the top-right inset is the actual foveated frame sent to the tracker.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline vs Telescope tiny-object SOT A/B test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR), help="Sequence directory with annotation.txt and a video.")
    parser.add_argument("--video", default=None, help="Optional explicit video path.")
    parser.add_argument("--annotation", default=None, help="Optional explicit annotation path.")
    parser.add_argument("--output-dir", default=None, help="Base output directory for A/B artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="SiamRAM inference YAML config.")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH), help="SiamABC checkpoint path.")
    parser.add_argument("--model-size", default="M", choices=["S", "M", "L"], help="SiamABC model size.")
    parser.add_argument("--lambda-tta", type=float, default=0.1, help="TTA lambda for the base SiamABC tracker.")
    parser.add_argument("--tracker-kind", choices=["siamram", "siamabc"], default="siamram", help="Tracker stack to compare.")
    parser.add_argument("--no-trt", action="store_true", help="Force PyTorch tracker path even if config enables TensorRT.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for smoke tests. 0 means full video.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log progress every N frames. 0 disables progress logs.")
    parser.add_argument("--output-video", action="store_true", help="Write annotated MP4 videos for baseline and variant.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution. By default this experiment requires CUDA.")
    parser.add_argument("--foveation-alpha", type=float, default=3.5, help="Hyperbolic contraction/expansion strength.")
    parser.add_argument("--foveation-p", type=float, default=2.0, help="Radial interpolation blend exponent.")
    parser.add_argument("--foveation-beta", type=float, default=5.0, help="Radius multiplier over max target side.")
    parser.add_argument("--foveation-min-radius-px", type=float, default=16.0, help="Minimum foveation radius in pixels.")
    parser.add_argument("--foveation-max-radius-norm", type=float, default=1.25, help="Maximum normalized foveation radius.")
    parser.add_argument("--foveation-inverse-iters", type=int, default=8, help="Fixed-point inverse iterations.")
    parser.add_argument("--foveation-radius-ema-alpha", type=float, default=0.2, help="EMA update factor for foveation radius smoothing.")
    parser.add_argument("--tiny-min-update-score", type=float, default=0.55, help="Minimum wrapped tracker score required before a prediction can seed the next foveation focus.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_gpu_or_fail(args.allow_cpu)
    sequence_dir = Path(args.sequence_dir)
    video_path = Path(args.video) if args.video else find_video(sequence_dir)
    annotation_path = Path(args.annotation) if args.annotation else sequence_dir / "annotation.txt"
    annotations = load_annotations(annotation_path)

    config_path = Path(args.config)
    weights_path = _resolve_weights_path(args.weights_path)
    config_for_checkpoints = OmegaConf.load(config_path)
    yolo_path = _resolve_weights_path(str(config_for_checkpoints.ram_tracker.yolo_weights))
    weights_path, _ = _ensure_required_checkpoints(weights_path, yolo_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / f"{sequence_dir.name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing experiment artifacts to %s", output_dir)

    shared_tracker_kwargs = {
        "config_path": config_path,
        "weights_path": weights_path,
        "model_size": args.model_size,
        "lambda_tta": args.lambda_tta,
        "tracker_kind": args.tracker_kind,
        "force_no_trt": args.no_trt,
    }
    run_kwargs = {
        "video_path": video_path,
        "annotations": annotations,
        "output_dir": output_dir,
        "max_frames": args.max_frames,
        "output_video": args.output_video,
        "progress_every": args.progress_every,
    }

    logger.info("Running baseline tracker")
    baseline_tracker = build_experiment_tracker(**shared_tracker_kwargs)
    baseline, baseline_artifacts = run_tracker_once(run_name="baseline", tracker=baseline_tracker, **run_kwargs)

    logger.info("Running Telescope tiny-object tracker")
    variant_core = build_experiment_tracker(**shared_tracker_kwargs)
    safe_eta = min(0.8 / args.foveation_alpha, 0.45)
    variant_tracker = TinyObjectTracker(
        variant_core,
        foveation_config=FoveationConfig(
            alpha=args.foveation_alpha,
            blend_exponent=args.foveation_p,
            radius_beta=args.foveation_beta,
            min_radius_px=args.foveation_min_radius_px,
            max_radius_norm=args.foveation_max_radius_norm,
            inverse_iters=args.foveation_inverse_iters,
            inverse_eta=safe_eta,
        ),
        radius_ema_alpha=args.foveation_radius_ema_alpha,
        min_update_score=args.tiny_min_update_score,
    )
    variant, variant_artifacts = run_tracker_once(run_name=TINY_VARIANT_NAME, tracker=variant_tracker, **run_kwargs)

    comparison = {
        "sequence_dir": str(sequence_dir),
        "video_path": str(video_path),
        "annotation_path": str(annotation_path),
        "output_dir": str(output_dir),
        "tracker_kind": args.tracker_kind,
        "baseline": baseline,
        TINY_VARIANT_NAME: variant,
        "delta": delta_dict(baseline, variant),
        "frame_delta": frame_delta_summary(baseline_artifacts.metrics_file, variant_artifacts.metrics_file),
        "artifacts": {
            "baseline": asdict(baseline_artifacts),
            TINY_VARIANT_NAME: asdict(variant_artifacts),
        },
        "foveation_config": {
            "alpha": args.foveation_alpha,
            "blend_exponent": args.foveation_p,
            "radius_beta": args.foveation_beta,
            "min_radius_px": args.foveation_min_radius_px,
            "max_radius_norm": args.foveation_max_radius_norm,
            "inverse_iters": args.foveation_inverse_iters,
            "inverse_eta": safe_eta,
            "radius_ema_alpha": args.foveation_radius_ema_alpha,
            "min_update_score": args.tiny_min_update_score,
        },
    }
    comparison_path = output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    report_path = output_dir / "comparison_report.md"
    write_report(report_path, comparison)
    logger.info("Wrote comparison report to %s", report_path)
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
