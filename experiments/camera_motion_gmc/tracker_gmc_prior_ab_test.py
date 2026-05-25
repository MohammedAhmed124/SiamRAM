"""SiamRAM A/B test for Philo's valid-only GMC search prior.

This experiment compares the same tracker on the same annotated sequence in two
modes:

1. baseline: no Philo camera-motion prior
2. gmc_prior_valid_only: wraps the tracker with GMCPriorTracker, which shifts the
   visual search prior using CameraMotionEstimator.transform_original only when
   the source confidence gates accept the motion estimate.

The ported camera-motion estimator and prior adapter live beside this script so
this experiment remains isolated from production inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Allow `python experiments/camera_motion_gmc/tracker_gmc_prior_ab_test.py` from
# the repository root without requiring installation as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.camera_motion_gmc.gmc_prior_adapter import GMCPriorTracker, MotionConfig
from models.SiamABC.tracker.tracker_setup import get_tracker
from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
from models.SiamRAM import SiamRAMTracker
from run_inference import _ensure_required_checkpoints, _resolve_weights_path
from utils.logger import create_logger

logger = create_logger(__name__)

DEFAULT_SEQUENCE_DIR = Path(r"E:\Coding projects\AIC-4-Hackathon\contest_release\dataset3\air_conditioning_box2")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "camera_motion_gmc"


@dataclass
class RunArtifacts:
    run_name: str
    output_dir: Path
    bbox_file: Path
    metrics_file: Path
    summary_file: Path
    gmc_prior_log: Optional[Path]
    annotated_video: Optional[Path]


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


def compute_iou(gt_box: np.ndarray, pred_box: np.ndarray) -> float:
    gx, gy, gw, gh = [float(v) for v in gt_box[:4]]
    px, py, pw, ph = [float(v) for v in pred_box[:4]]
    if gw <= 0 or gh <= 0 or pw <= 0 or ph <= 0:
        return 0.0
    gx2, gy2 = gx + gw, gy + gh
    px2, py2 = px + pw, py + ph
    ix1, iy1 = max(gx, px), max(gy, py)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = gw * gh + pw * ph - inter
    return float(inter / union) if union > 0 else 0.0


def center_error(gt_box: np.ndarray, pred_box: np.ndarray) -> float:
    gx, gy, gw, gh = [float(v) for v in gt_box[:4]]
    px, py, pw, ph = [float(v) for v in pred_box[:4]]
    gcx, gcy = gx + gw / 2.0, gy + gh / 2.0
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    return float(np.hypot(gcx - pcx, gcy - pcy))


def normalized_center_error(gt_box: np.ndarray, pred_box: np.ndarray) -> Optional[float]:
    gx, gy, gw, gh = [float(v) for v in gt_box[:4]]
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _clip_bbox_to_frame(bbox: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(bbox, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")
    x, y, w, h = [float(v) for v in arr[:4]]
    height, width = frame_shape
    w = max(1.0, min(w, float(width)))
    h = max(1.0, min(h, float(height)))
    x = min(max(0.0, x), max(0.0, float(width) - w))
    y = min(max(0.0, y), max(0.0, float(height) - h))
    return np.array([round(x), round(y), round(w), round(h)], dtype=int)


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


def _draw_klt_tracks(frame: np.ndarray, gmc_adapter: GMCPriorTracker) -> None:
    motion = gmc_adapter.last_motion
    if motion is None:
        return
    prev_points = np.asarray(motion.prev_points, dtype=float).reshape(-1, 2)
    curr_points = np.asarray(motion.curr_points, dtype=float).reshape(-1, 2)
    inliers = np.asarray(motion.inliers, dtype=bool).reshape(-1)
    if len(prev_points) == 0 or len(curr_points) == 0:
        return

    work_scale = max(float(gmc_adapter.motion_config.work_scale), 1e-6)
    stride = max(1, len(curr_points) // 350)
    for i in range(0, min(len(prev_points), len(curr_points)), stride):
        p0 = tuple(np.round(prev_points[i] / work_scale).astype(int))
        p1 = tuple(np.round(curr_points[i] / work_scale).astype(int))
        is_inlier = i < len(inliers) and bool(inliers[i])
        color = (80, 235, 80) if is_inlier else (60, 80, 255)
        cv2.circle(frame, p1, 2, color, -1, cv2.LINE_AA)
        if is_inlier:
            cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)


def _render_tracking_debug_frame(
    frame: np.ndarray,
    *,
    run_name: str,
    frame_idx: int,
    gt_bbox: np.ndarray,
    pred_bbox: np.ndarray,
    score: float,
    iou: float,
    gmc_adapter: Optional[GMCPriorTracker],
) -> np.ndarray:
    canvas = frame.copy()
    _draw_bbox(canvas, gt_bbox, (255, 150, 0), "GT", 2)
    _draw_bbox(canvas, pred_bbox, (0, 230, 80) if iou >= 0.5 else (0, 90, 255), "PRED", 2)
    lines = [f"{run_name} | frame={frame_idx:04d} score={score:.3f} iou={iou:.3f}"]
    if gmc_adapter is not None and gmc_adapter.last_motion is not None:
        motion = gmc_adapter.last_motion
        stats = motion.applied_stats_original
        _draw_klt_tracks(canvas, gmc_adapter)
        if gmc_adapter.last_search_prior is not None:
            _draw_bbox(canvas, gmc_adapter.last_search_prior, (0, 190, 255), "GMC prior", 1)
        lines.extend(
            [
                f"GMC used={int(gmc_adapter.last_used_camera_prior)} valid={int(motion.confidence.valid)} reason={motion.confidence.reason}",
                f"inliers={motion.confidence.num_inliers}/{motion.confidence.num_tracked} ratio={motion.confidence.inlier_ratio:.2f} residual={motion.confidence.median_residual:.2f}",
                f"camera dx={stats.dx:.1f} dy={stats.dy:.1f} scale={stats.scale:.4f} rot={stats.rotation_deg:.2f}deg",
                "KLT tracks: green=inlier background flow, red=outlier/rejected flow",
            ]
        )
    _draw_hud(canvas, lines)
    return canvas


def _clone_config_with_resolved_paths(config_path: Path, weights_path: Path, force_no_trt: bool, disable_internal_camera_motion: bool):
    config = OmegaConf.load(config_path)
    if force_no_trt:
        config.make_trt_engine = False
    if "ram_tracker" in config:
        config.ram_tracker.disable_camera_motion = bool(disable_internal_camera_motion)
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
    disable_internal_camera_motion: bool,
):
    config = _clone_config_with_resolved_paths(config_path, weights_path, force_no_trt, disable_internal_camera_motion)
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


def run_tracker_once(
    *,
    run_name: str,
    use_gmc_prior: bool,
    config_path: Path,
    weights_path: Path,
    video_path: Path,
    annotations: np.ndarray,
    output_dir: Path,
    model_size: str,
    lambda_tta: float,
    tracker_kind: str,
    force_no_trt: bool,
    disable_internal_camera_motion: bool,
    max_frames: int,
    progress_every: int,
    motion_config: MotionConfig,
    gmc_prior_min_inlier_ratio: float,
    gmc_prior_min_coverage_cells: int,
    gmc_prior_use_fallback: bool,
    output_video: bool,
) -> tuple[dict[str, Any], RunArtifacts]:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_experiment_tracker(
        config_path=config_path,
        weights_path=weights_path,
        model_size=model_size,
        lambda_tta=lambda_tta,
        tracker_kind=tracker_kind,
        force_no_trt=force_no_trt,
        disable_internal_camera_motion=disable_internal_camera_motion,
    )
    gmc_adapter: Optional[GMCPriorTracker] = None
    if use_gmc_prior:
        gmc_adapter = GMCPriorTracker(
            tracker,
            motion_config=deepcopy(motion_config),
            min_inlier_ratio=gmc_prior_min_inlier_ratio,
            min_coverage_cells=gmc_prior_min_coverage_cells,
            use_fallback_prior=gmc_prior_use_fallback,
        )
        tracker = gmc_adapter

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    initial_bbox = _clip_bbox_to_frame(annotations[0], first_frame.shape[:2])
    tracker.initialize(first_frame, initial_bbox)

    tracked_bboxes: list[np.ndarray] = [initial_bbox]
    frame_runtime_ms: list[float] = [0.0]
    rows: list[dict[str, Any]] = []
    writer: Optional[cv2.VideoWriter] = None
    annotated_video_path: Optional[Path] = None
    if output_video:
        annotated_video_path = run_dir / f"{video_path.stem}_{run_name}_annotated.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated video: {annotated_video_path}")

    def append_metric_row(frame_idx: int, pred_bbox: np.ndarray, runtime_ms: float) -> None:
        gt = annotations[frame_idx] if frame_idx < len(annotations) else annotations[-1]
        norm_err = normalized_center_error(gt, pred_bbox)
        row = {
            "frame": frame_idx,
            "gt_x": float(gt[0]),
            "gt_y": float(gt[1]),
            "gt_w": float(gt[2]),
            "gt_h": float(gt[3]),
            "pred_x": float(pred_bbox[0]),
            "pred_y": float(pred_bbox[1]),
            "pred_w": float(pred_bbox[2]),
            "pred_h": float(pred_bbox[3]),
            "iou": compute_iou(gt, pred_bbox),
            "center_error": center_error(gt, pred_bbox),
            "norm_center_error": norm_err,
            "runtime_ms": float(runtime_ms),
        }
        if gmc_adapter is not None:
            gmc_row = gmc_adapter.records_as_dicts()[-1] if gmc_adapter.records else {}
            row.update(
                {
                    "gmc_used_camera_prior": gmc_row.get("used_camera_prior"),
                    "gmc_reason": gmc_row.get("reason"),
                    "gmc_motion_valid": gmc_row.get("motion_valid"),
                    "gmc_motion_used_fallback": gmc_row.get("motion_used_fallback"),
                    "gmc_inlier_ratio": gmc_row.get("inlier_ratio"),
                    "gmc_median_residual": gmc_row.get("median_residual"),
                    "gmc_coverage_cells": gmc_row.get("coverage_cells"),
                    "gmc_applied_dx": gmc_row.get("applied_dx"),
                    "gmc_applied_dy": gmc_row.get("applied_dy"),
                    "gmc_applied_scale": gmc_row.get("applied_scale"),
                    "gmc_applied_rotation_deg": gmc_row.get("applied_rotation_deg"),
                    "gmc_applied_max_corner_displacement": gmc_row.get("applied_max_corner_displacement"),
                    "gmc_motion_runtime_ms": gmc_row.get("motion_runtime_ms"),
                    "gmc_prev_bbox": json.dumps(gmc_row.get("prev_bbox")) if gmc_row else None,
                    "gmc_search_prior_bbox": json.dumps(gmc_row.get("search_prior_bbox")) if gmc_row else None,
                }
            )
        rows.append(row)

    append_metric_row(0, initial_bbox, 0.0)
    if writer is not None:
        writer.write(
            _render_tracking_debug_frame(
                first_frame,
                run_name=run_name,
                frame_idx=0,
                gt_bbox=annotations[0],
                pred_bbox=initial_bbox,
                score=1.0,
                iou=1.0,
                gmc_adapter=gmc_adapter,
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
            t0 = time.perf_counter()
            result = tracker.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            pred_bbox = _clip_bbox_to_frame(result[0], frame.shape[:2])
            gt_bbox = annotations[frame_idx] if frame_idx < len(annotations) else annotations[-1]
            frame_iou = compute_iou(gt_bbox, pred_bbox)
            tracked_bboxes.append(pred_bbox)
            frame_runtime_ms.append(elapsed_ms)
            append_metric_row(frame_idx, pred_bbox, elapsed_ms)
            if writer is not None:
                writer.write(
                    _render_tracking_debug_frame(
                        frame,
                        run_name=run_name,
                        frame_idx=frame_idx,
                        gt_bbox=gt_bbox,
                        pred_bbox=pred_bbox,
                        score=float(result[1]) if len(result) > 1 else 0.0,
                        iou=frame_iou,
                        gmc_adapter=gmc_adapter,
                    )
                )
            if progress_every > 0 and frame_idx % progress_every == 0:
                score = float(result[1]) if len(result) > 1 else 0.0
                gmc_msg = ""
                if gmc_adapter is not None and gmc_adapter.last_motion is not None:
                    motion = gmc_adapter.last_motion
                    stats = motion.applied_stats_original
                    gmc_msg = (
                        f" | GMC used={int(gmc_adapter.last_used_camera_prior)}"
                        f" valid={int(motion.confidence.valid)}"
                        f" fb={int(motion.used_fallback)}"
                        f" inl={motion.confidence.num_inliers}/{motion.confidence.num_tracked}"
                        f" ratio={motion.confidence.inlier_ratio:.2f}"
                        f" dx={stats.dx:.1f} dy={stats.dy:.1f}"
                        f" gmc={motion.runtime_ms:.1f}ms"
                    )
                logger.info("[%s] frame=%04d score=%.3f latency=%.1fms%s", run_name, frame_idx, score, elapsed_ms, gmc_msg)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    bbox_file = run_dir / f"{video_path.stem}.txt"
    np.savetxt(bbox_file, np.asarray(tracked_bboxes, dtype=int), fmt="%d", delimiter=" ")
    metrics_file = run_dir / "tracking_metrics_by_frame.csv"
    write_rows(metrics_file, rows)

    summary = summarise_rows(rows)
    summary.update(
        {
            "run_name": run_name,
            "use_gmc_prior": use_gmc_prior,
            "tracker_kind": tracker_kind,
            "disable_internal_camera_motion": disable_internal_camera_motion,
            "video_path": str(video_path),
            "bbox_file": str(bbox_file),
            "metrics_file": str(metrics_file),
            "annotated_video": str(annotated_video_path) if annotated_video_path is not None else None,
        }
    )
    gmc_log_path = None
    if gmc_adapter is not None:
        gmc_rows = gmc_adapter.records_as_dicts()
        gmc_log_path = run_dir / "gmc_prior_log.csv"
        write_rows(gmc_log_path, gmc_rows)
        (run_dir / "gmc_prior_log.json").write_text(json.dumps(gmc_rows, indent=2), encoding="utf-8")
        summary["gmc_prior_frames"] = len(gmc_rows)
        summary["gmc_prior_used_ratio"] = float(np.mean([bool(row["used_camera_prior"]) for row in gmc_rows])) if gmc_rows else 0.0
        summary["gmc_valid_ratio"] = float(np.mean([bool(row["motion_valid"]) for row in gmc_rows])) if gmc_rows else 0.0
        summary["gmc_fallback_ratio"] = float(np.mean([bool(row["motion_used_fallback"]) for row in gmc_rows])) if gmc_rows else 0.0
        summary["gmc_mean_motion_runtime_ms"] = float(np.mean([float(row["motion_runtime_ms"]) for row in gmc_rows])) if gmc_rows else 0.0
        summary["gmc_median_abs_dx"] = float(np.median([abs(float(row["applied_dx"])) for row in gmc_rows])) if gmc_rows else 0.0
        summary["gmc_median_abs_dy"] = float(np.median([abs(float(row["applied_dy"])) for row in gmc_rows])) if gmc_rows else 0.0

    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, RunArtifacts(
        run_name=run_name,
        output_dir=run_dir,
        bbox_file=bbox_file,
        metrics_file=metrics_file,
        summary_file=summary_file,
        gmc_prior_log=gmc_log_path,
        annotated_video=annotated_video_path,
    )


def _delta_dict(baseline: dict[str, Any], gmc: dict[str, Any]) -> dict[str, Any]:
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
    return {key: gmc.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def _motion_specific_delta(baseline_metrics: Path, gmc_metrics: Path) -> dict[str, Any]:
    def load_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    base_rows = load_csv(baseline_metrics)
    gmc_rows = load_csv(gmc_metrics)
    n = min(len(base_rows), len(gmc_rows))
    improvements = 0
    regressions = 0
    deltas_used: list[float] = []
    for idx in range(n):
        delta_iou = float(gmc_rows[idx]["iou"]) - float(base_rows[idx]["iou"])
        used = str(gmc_rows[idx].get("gmc_used_camera_prior", "")).lower() == "true"
        if delta_iou >= 0.05:
            improvements += 1
        elif delta_iou <= -0.05:
            regressions += 1
        if used:
            deltas_used.append(delta_iou)
    return {
        "frames_iou_improved_by_0_05": improvements,
        "frames_iou_regressed_by_0_05": regressions,
        "mean_iou_delta_on_gmc_used_frames": float(np.mean(deltas_used)) if deltas_used else 0.0,
        "gmc_used_frame_count": len(deltas_used),
    }


def write_markdown_report(path: Path, comparison: dict[str, Any]) -> None:
    baseline = comparison["baseline"]
    gmc = comparison["gmc_prior_valid_only"]
    delta = comparison["delta"]
    motion = comparison["motion_specific"]
    lines = [
        "# Camera Motion GMC A/B Report",
        "",
        f"Sequence: `{comparison['sequence_dir']}`",
        f"Video: `{comparison['video_path']}`",
        f"Tracker: `{comparison['tracker_kind']}`",
        "",
        "| Metric | Baseline | GMC prior | Delta |",
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
        "id_switches_proxy",
        "false_positive_proxy_iou_lt_0_1",
        "false_negative_proxy_iou_lt_0_1",
        "mean_runtime_ms",
    ]:
        lines.append(f"| `{key}` | {baseline.get(key, 0):.4f} | {gmc.get(key, 0):.4f} | {delta.get(key, 0):+.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Camera motion compensation estimates how the background moved between frames, "
                "warps the previous target box by that transform, and uses the warped box as the next search prior "
                "only when the motion estimate passes the original validity gates."
            ),
            (
                f"- On this run, the GMC prior changed mean IoU by `{delta.get('mean_iou', 0):+.4f}`, "
                f"changed 20px precision by `{delta.get('precision_20px', 0):+.4f}`, "
                f"and changed mean center error by `{delta.get('mean_center_error', 0):+.2f}px`."
            ),
            (
                f"- Motion-specific effect: `{motion['frames_iou_improved_by_0_05']}` frames improved by at least 0.05 IoU, "
                f"`{motion['frames_iou_regressed_by_0_05']}` frames regressed by at least 0.05 IoU, "
                f"with mean IoU delta `{motion['mean_iou_delta_on_gmc_used_frames']:+.4f}` on frames where the GMC prior was used."
            ),
            "",
            "## Motion Compensation",
            "",
            f"- GMC valid ratio: `{gmc.get('gmc_valid_ratio', 0):.4f}`",
            f"- GMC prior used ratio: `{gmc.get('gmc_prior_used_ratio', 0):.4f}`",
            f"- GMC fallback ratio: `{gmc.get('gmc_fallback_ratio', 0):.4f}`",
            f"- Mean GMC runtime: `{gmc.get('gmc_mean_motion_runtime_ms', 0):.2f} ms`",
            f"- Frames improved by IoU >= 0.05: `{motion['frames_iou_improved_by_0_05']}`",
            f"- Frames regressed by IoU <= -0.05: `{motion['frames_iou_regressed_by_0_05']}`",
            f"- Mean IoU delta on GMC-used frames: `{motion['mean_iou_delta_on_gmc_used_frames']:.4f}`",
            "",
            "Note: `id_switches_proxy`, false positives, and false negatives are single-object proxies derived from IoU because this dataset provides one GT box per frame, not multi-object IDs/detections.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SiamRAM baseline vs Philo valid-only GMC-prior A/B test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR), help="Sequence directory with annotation.txt and a video.")
    parser.add_argument("--video", default=None, help="Optional explicit video path.")
    parser.add_argument("--annotation", default=None, help="Optional explicit annotation path.")
    parser.add_argument("--output-dir", default=None, help="Base output directory for A/B artifacts.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "inference_config.yaml"), help="SiamRAM inference YAML config.")
    parser.add_argument("--weights-path", default=str(REPO_ROOT / "checkpoints" / "inference_checkpoint.pth"), help="SiamABC checkpoint path.")
    parser.add_argument("--model-size", default="M", choices=["S", "M", "L"], help="SiamABC model size.")
    parser.add_argument("--lambda-tta", type=float, default=0.1, help="TTA lambda for the base SiamABC tracker.")
    parser.add_argument("--tracker-kind", choices=["siamram", "siamabc"], default="siamram", help="Tracker stack to compare.")
    parser.add_argument("--no-trt", action="store_true", help="Force PyTorch tracker path even if config enables TensorRT.")
    parser.add_argument("--keep-internal-camera-motion", action="store_true", help="Do not disable SiamRAM's legacy internal homography block.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for smoke tests. 0 means full video.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log progress every N frames. 0 disables progress logs.")
    parser.add_argument("--output-video", action="store_true", help="Write annotated MP4 videos for baseline and GMC runs.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution. By default this experiment requires CUDA.")
    parser.add_argument("--gmc-min-inlier-ratio", type=float, default=0.35, help="Adapter-side minimum inlier ratio gate from Philo A/B script.")
    parser.add_argument("--gmc-min-coverage-cells", type=int, default=6, help="Adapter-side minimum coverage cells gate from Philo A/B script.")
    parser.add_argument("--gmc-use-fallback-prior", action="store_true", help="Allow fallback transform to seed the prior; default preserves valid-only behavior.")
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

    motion_config = MotionConfig(model="partial_affine")
    shared = {
        "config_path": config_path,
        "weights_path": weights_path,
        "video_path": video_path,
        "annotations": annotations,
        "output_dir": output_dir,
        "model_size": args.model_size,
        "lambda_tta": args.lambda_tta,
        "tracker_kind": args.tracker_kind,
        "force_no_trt": args.no_trt,
        "disable_internal_camera_motion": not args.keep_internal_camera_motion,
        "max_frames": args.max_frames,
        "progress_every": args.progress_every,
        "motion_config": motion_config,
        "gmc_prior_min_inlier_ratio": args.gmc_min_inlier_ratio,
        "gmc_prior_min_coverage_cells": args.gmc_min_coverage_cells,
        "gmc_prior_use_fallback": args.gmc_use_fallback_prior,
        "output_video": args.output_video,
    }

    logger.info("Running baseline without Philo camera-motion prior")
    baseline, baseline_artifacts = run_tracker_once(run_name="baseline", use_gmc_prior=False, **shared)
    logger.info("Running GMC valid-only prior")
    gmc, gmc_artifacts = run_tracker_once(run_name="gmc_prior_valid_only", use_gmc_prior=True, **shared)

    comparison = {
        "sequence_dir": str(sequence_dir),
        "video_path": str(video_path),
        "annotation_path": str(annotation_path),
        "output_dir": str(output_dir),
        "tracker_kind": args.tracker_kind,
        "baseline": baseline,
        "gmc_prior_valid_only": gmc,
        "delta": _delta_dict(baseline, gmc),
        "motion_specific": _motion_specific_delta(baseline_artifacts.metrics_file, gmc_artifacts.metrics_file),
        "artifacts": {
            "baseline": asdict(baseline_artifacts),
            "gmc_prior_valid_only": asdict(gmc_artifacts),
        },
    }
    comparison_path = output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    report_path = output_dir / "comparison_report.md"
    write_markdown_report(report_path, comparison)
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
