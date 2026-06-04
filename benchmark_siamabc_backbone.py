"""Measure SiamABC backbone parity and latency against eager FP32."""

from __future__ import annotations

import argparse
import gc
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from models.SiamABC.tracker.tracker_setup import get_tracker
from run_inference import _normalize_tracker_config_aliases
from utils.utils import extend_bbox, get_extended_crop

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS = (
    "data/dataset1/Car_video/Car_video.mp4",
    "data/dataset1/basketball/basketball.mp4",
    "data/dataset1/surfer_1/surfer_1.mp4",
    "data/dataset5/car10/car10_96.mp4",
    "data/dataset5/uav7/uav7_30.mp4",
    "data/dataset5/person20/person20_96.mp4",
)
MODE_SPECS = {
    "dynamic_fp16": ("dynamic_fp16", False),
    "static_fp16": ("static_fp16", False),
    "static_fp16_fp32_neck": ("static_fp16_fp32_neck", False),
    "dynamic_fp32": ("dynamic_fp32", False),
    "dynamic_fp32_no_tf32": ("dynamic_fp32", True),
}
SCORE_THRESHOLDS = (0.55, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark SiamABC backbone drift on real tracker crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "inference_config_experimental.yaml"),
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "checkpoints" / "inference_checkpoint.pth"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "checkpoints" / "trt_engines"),
    )
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--feature-stride", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_SPECS),
        default=list(MODE_SPECS),
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=list(DEFAULT_VIDEOS),
        help="Video paths. Each annotation.txt is read from the video's directory.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--json-output",
        default=str(ROOT / "outputs" / "siamabc_backbone_benchmark.json"),
    )
    return parser.parse_args()


def _load_config(path: str):
    config = OmegaConf.load(path)
    _normalize_tracker_config_aliases(config)
    return config


def _read_bbox(video_path: Path) -> np.ndarray:
    line = (video_path.parent / "annotation.txt").read_text(encoding="utf-8").splitlines()[0]
    values = [int(float(value)) for value in line.replace(",", " ").split()[:4]]
    if len(values) != 4:
        raise RuntimeError(f"Invalid initial bbox beside {video_path}")
    return np.asarray(values, dtype=int)


def _read_video(video_path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")
    return frames


def _template_crop(tracker, frame: np.ndarray, bbox: np.ndarray) -> torch.Tensor:
    context = extend_bbox(
        bbox,
        offset=tracker.tracking_config["template_bbox_offset"],
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )
    crop, _, _ = get_extended_crop(
        image=frame,
        bbox=bbox,
        context=context,
        crop_size=tracker.tracking_config["template_size"],
    )
    return tracker._preprocess_image(crop, tracker._template_transform)


def _search_crop(tracker, frame: np.ndarray, bbox: np.ndarray) -> torch.Tensor:
    context = extend_bbox(
        bbox,
        offset=tracker.tracking_config["search_context"],
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )
    crop, _, _ = get_extended_crop(
        image=frame,
        bbox=bbox,
        context=context,
        crop_size=tracker.tracking_config["instance_size"],
        padding_value=tracker.tracking_state.mean_color,
    )
    return tracker._preprocess_image(crop, tracker._search_transform)


def _collect_reference(
    tracker,
    videos: list[Path],
    max_frames: int,
    feature_stride: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    decisions: dict[str, list[dict[str, Any]]] = {}
    samples: list[dict[str, Any]] = []
    for video_path in videos:
        frames = _read_video(video_path, max_frames)
        initial_bbox = _read_bbox(video_path)
        tracker.initialize(frames[0], initial_bbox)

        template = _template_crop(tracker, frames[0], initial_bbox)
        template_features = tracker.net.get_features(template)
        samples.append(
            {
                "video": str(video_path),
                "frame": 0,
                "kind": "template",
                "crop": template.detach().cpu(),
                "reference": template_features.detach().float().cpu(),
            }
        )

        video_decisions: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames[1:], start=1):
            if frame_index % max(1, feature_stride) == 0:
                bbox_before = np.asarray(tracker.tracking_state.bbox).copy()
                crop = _search_crop(tracker, frame, bbox_before)
                features = tracker.net.get_features(crop)
                samples.append(
                    {
                        "video": str(video_path),
                        "frame": frame_index,
                        "kind": "search",
                        "crop": crop.detach().cpu(),
                        "reference": features.detach().float().cpu(),
                    }
                )
            bbox, score, _ = tracker.update(frame)
            video_decisions.append(
                {
                    "bbox": np.asarray(bbox, dtype=float).tolist(),
                    "score": float(score),
                }
            )
        decisions[str(video_path)] = video_decisions
    return decisions, samples


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _measure_features(
    run_features: Callable[[torch.Tensor], torch.Tensor],
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    by_size: dict[int, torch.Tensor] = {}
    for sample in samples:
        crop = sample["crop"].to(device)
        by_size[int(crop.shape[-1])] = crop
    with torch.no_grad():
        for crop in by_size.values():
            for _ in range(5):
                run_features(crop)
        torch.cuda.synchronize(device)

    latencies: list[float] = []
    # Deltas are scored against two references when both are attached to the
    # samples: "fp32" (the eager runtime backbone, always present) and "fp16"
    # (eager FP16 weights run with PyTorch kernels). The fp16 reference shares
    # byte-identical weights/precision with the TRT FP16 engines, so any drift
    # beyond it is the non-precision "something else" (TRT kernels / fp16
    # accumulation / layer fusion / dynamic-shape tactic).
    mean_abs: dict[str, list[float]] = {"fp32": [], "fp16": []}
    rmse: dict[str, list[float]] = {"fp32": [], "fp16": []}
    max_abs: dict[str, list[float]] = {"fp32": [], "fp16": []}
    cosine: dict[str, list[float]] = {"fp32": [], "fp16": []}
    # Per-crop-size mean-abs vs the FP32 reference. The dynamic-shape tactic
    # penalty only surfaces at the larger search size, so keep the sizes split.
    by_size_abs: dict[int, list[float]] = {}
    with torch.no_grad():
        for sample in samples:
            crop = sample["crop"].to(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = run_features(crop).float()
            end.record()
            end.synchronize()
            latencies.append(float(start.elapsed_time(end)))

            refs = {"fp32": sample["reference"]}
            if "reference_fp16" in sample:
                refs["fp16"] = sample["reference_fp16"]
            for key, ref_cpu in refs.items():
                reference = ref_cpu.to(device)
                delta = output - reference
                mean_abs[key].append(float(delta.abs().mean().item()))
                rmse[key].append(float(delta.square().mean().sqrt().item()))
                max_abs[key].append(float(delta.abs().max().item()))
                cosine[key].append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            output.flatten(), reference.flatten(), dim=0
                        ).item()
                    )
                )
            by_size_abs.setdefault(int(crop.shape[-1]), []).append(mean_abs["fp32"][-1])

    metrics: dict[str, float] = {
        "feature_mean_abs": float(np.mean(mean_abs["fp32"])),
        "feature_p95_abs": _percentile(mean_abs["fp32"], 95),
        "feature_worst_abs": float(max(max_abs["fp32"])),
        "feature_mean_rmse": float(np.mean(rmse["fp32"])),
        "feature_mean_cosine": float(np.mean(cosine["fp32"])),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": _percentile(latencies, 95),
    }
    if mean_abs["fp16"]:
        metrics.update(
            {
                "feature_mean_abs_vs_fp16": float(np.mean(mean_abs["fp16"])),
                "feature_worst_abs_vs_fp16": float(max(max_abs["fp16"])),
                "feature_mean_rmse_vs_fp16": float(np.mean(rmse["fp16"])),
                "feature_mean_cosine_vs_fp16": float(np.mean(cosine["fp16"])),
            }
        )
    for size, values in sorted(by_size_abs.items()):
        metrics[f"feature_mean_abs_hw{size}"] = float(np.mean(values))
    return metrics


def _run_eager_fp16(module: torch.nn.Module, crop: torch.Tensor) -> torch.Tensor:
    return module(crop.half()).float()


def _bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def _measure_decisions(
    tracker,
    videos: list[Path],
    reference: dict[str, list[dict[str, Any]]],
    max_frames: int,
) -> dict[str, float]:
    ious: list[float] = []
    score_errors: list[float] = []
    exact = 0
    high_agreement = 0
    flips = 0
    flip_opportunities = 0
    for video_path in videos:
        frames = _read_video(video_path, max_frames)
        tracker.initialize(frames[0], _read_bbox(video_path))
        expected = reference[str(video_path)]
        for frame, target in zip(frames[1:], expected):
            bbox, score, _ = tracker.update(frame)
            bbox_list = np.asarray(bbox, dtype=float).tolist()
            iou = _bbox_iou(bbox_list, target["bbox"])
            ious.append(iou)
            exact += int(bbox_list == target["bbox"])
            high_agreement += int(iou >= 0.95)
            score_errors.append(abs(float(score) - target["score"]))
            for threshold in SCORE_THRESHOLDS:
                flips += int((float(score) >= threshold) != (target["score"] >= threshold))
                flip_opportunities += 1
    count = max(1, len(ious))
    return {
        "decision_frames": float(len(ious)),
        "threshold_flip_rate": flips / max(1, flip_opportunities),
        "bbox_exact_rate": exact / count,
        "bbox_iou95_rate": high_agreement / count,
        "bbox_median_iou": float(np.median(ious)),
        "bbox_mean_iou": float(np.mean(ious)),
        "score_mean_abs": float(np.mean(score_errors)),
    }


def _build_eager(config_path: str, weights_path: str, cuda_id: int):
    config = _load_config(config_path)
    config.tracker.cuda_id = cuda_id
    return get_tracker(
        config=config,
        weights_path=weights_path,
        lambda_tta=0.1,
        continuous=False,
    )


def _build_trt(
    config_path: str,
    weights_path: str,
    cache_dir: str,
    cuda_id: int,
    mode: str,
    disable_tf32: bool,
    rebuild_cache: bool,
):
    from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker

    config = _load_config(config_path)
    trt_config = config.get("trt_engine", {}) or {}
    return get_trt_tracker(
        config=config,
        weights_path=weights_path,
        lambda_tta=float(trt_config.get("lambda_tta", 0.1)),
        fp16=mode != "dynamic_fp32",
        cuda_id=cuda_id,
        backbone_mode=mode,
        disable_tf32=disable_tf32,
        engine_cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )


def _select_mode(results: dict[str, dict[str, float]]) -> str:
    baseline = results["dynamic_fp16"]
    eligible: list[str] = []
    for name, metrics in results.items():
        if name == "eager_fp32" or name == "eager_fp16":
            continue
        latency_ok = (
            metrics["latency_mean_ms"] <= baseline["latency_mean_ms"] * 1.03
            and metrics["latency_p95_ms"] <= baseline["latency_p95_ms"] * 1.05
        )
        parity_not_worse = (
            metrics["threshold_flip_rate"] <= baseline["threshold_flip_rate"]
            and metrics["bbox_median_iou"] >= baseline["bbox_median_iou"]
        )
        parity_improved = (
            metrics["threshold_flip_rate"] < baseline["threshold_flip_rate"]
            or metrics["bbox_median_iou"] > baseline["bbox_median_iou"]
            or metrics["feature_mean_abs"] < baseline["feature_mean_abs"]
        )
        if latency_ok and parity_not_worse and parity_improved:
            eligible.append(name)
    if not eligible:
        return "dynamic_fp16"
    return min(
        eligible,
        key=lambda name: (
            results[name]["threshold_flip_rate"],
            -results[name]["bbox_median_iou"],
            results[name]["feature_mean_abs"],
            results[name]["latency_mean_ms"],
        ),
    )


def _print_result(name: str, metrics: dict[str, float]) -> None:
    line = (
        f"{name:24s} "
        f"feat_abs={metrics['feature_mean_abs']:.6g} "
        f"cos={metrics['feature_mean_cosine']:.7f} "
    )
    if "feature_mean_abs_vs_fp16" in metrics:
        line += f"abs_vs_fp16={metrics['feature_mean_abs_vs_fp16']:.6g} "
    line += f"mean/p95={metrics['latency_mean_ms']:.3f}/{metrics['latency_p95_ms']:.3f} ms"
    print(line, end="")
    if "threshold_flip_rate" in metrics:
        print(
            f" flips={metrics['threshold_flip_rate']:.4%} "
            f"median_iou={metrics['bbox_median_iou']:.6f} "
            f"iou95={metrics['bbox_iou95_rate']:.4%}"
        )
    else:
        print()


def _decomposition(results: dict[str, dict[str, float]]) -> dict[str, float | None]:
    """Attribute the backbone drift to independent causes from the per-mode deltas."""

    def vs_fp32(name: str) -> float | None:
        metrics = results.get(name)
        return metrics["feature_mean_abs"] if metrics else None

    def vs_fp16(name: str) -> float | None:
        metrics = results.get(name)
        if not metrics or "feature_mean_abs_vs_fp16" not in metrics:
            return None
        return metrics["feature_mean_abs_vs_fp16"]

    def diff(a: float | None, b: float | None) -> float | None:
        return (a - b) if (a is not None and b is not None) else None

    return {
        # Pure FP16 precision effect (PyTorch on both sides).
        "pure_precision_fp16_vs_fp32": vs_fp32("eager_fp16"),
        # The non-precision "something else": TRT FP16 kernels vs eager FP16,
        # static-shape so the dynamic-profile tactic is not a confound.
        "trt_impl_fp16": vs_fp16("static_fp16"),
        # Extra drift the dynamic-shape profile adds over exact-shape engines.
        "dynamic_shape_tactic": diff(vs_fp16("dynamic_fp16"), vs_fp16("static_fp16")),
        # Cost of running the neck in FP16 instead of eager FP32.
        "neck_in_fp16": diff(vs_fp32("static_fp16"), vs_fp32("static_fp16_fp32_neck")),
        # TRT FP32 kernels vs eager FP32 (fusion/tracing, precision matched).
        "trt_impl_fp32": vs_fp32("dynamic_fp32"),
        # TF32 contribution within the FP32 path.
        "tf32": diff(vs_fp32("dynamic_fp32"), vs_fp32("dynamic_fp32_no_tf32")),
    }


def _print_decomposition(results: dict[str, dict[str, float]]) -> dict[str, float | None]:
    decomp = _decomposition(results)
    labels = {
        "pure_precision_fp16_vs_fp32": "precision: eager fp16 vs fp32",
        "trt_impl_fp16": "TRT impl @ fp16 (static_fp16 vs eager_fp16)  <- 'something else'",
        "dynamic_shape_tactic": "dynamic-shape tactic (dynamic - static, vs fp16)",
        "neck_in_fp16": "neck in fp16 (static_fp16 - fp32_neck, vs fp32)",
        "trt_impl_fp32": "TRT impl @ fp32 (dynamic_fp32 vs eager_fp32)",
        "tf32": "tf32 within fp32 path (dynamic_fp32 - no_tf32)",
    }
    print("\nbackbone drift decomposition (mean abs feature delta):")
    for key, label in labels.items():
        value = decomp.get(key)
        shown = f"{value:.6g}" if value is not None else "n/a (mode not run)"
        print(f"  {label:60s} {shown}")
    return decomp


def main() -> None:
    """Run feature and free-running decision parity measurements."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and TensorRT.")
    torch.cuda.set_device(args.cuda_id)
    device = torch.device(f"cuda:{args.cuda_id}")
    videos = [
        (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
        for path in args.videos
    ]

    eager = _build_eager(args.config, args.weights, args.cuda_id)
    reference, samples = _collect_reference(
        eager,
        videos,
        max_frames=max(2, args.max_frames),
        feature_stride=max(1, args.feature_stride),
    )

    from models.SiamABC.tracker.trt_engine.trt_utils import _FeatureExtractorModule

    eager_fp16 = _FeatureExtractorModule(eager.net).eval().to(device).half()
    run_eager_fp16 = partial(_run_eager_fp16, eager_fp16)
    # Attach the same-precision (FP16 weights, PyTorch kernels) reference to every
    # sample so each TRT FP16 mode can be scored against it. Drift beyond this
    # reference is the non-precision "something else" the diagnosis is hunting.
    with torch.no_grad():
        for sample in samples:
            crop = sample["crop"].to(device)
            sample["reference_fp16"] = run_eager_fp16(crop).detach().float().cpu()

    results: dict[str, dict[str, float]] = {}
    results["eager_fp32"] = _measure_features(eager.net.get_features, samples, device)
    _print_result("eager_fp32", results["eager_fp32"])

    results["eager_fp16"] = _measure_features(run_eager_fp16, samples, device)
    _print_result("eager_fp16", results["eager_fp16"])
    del run_eager_fp16, eager_fp16
    torch.cuda.empty_cache()

    for label in args.modes:
        mode, disable_tf32 = MODE_SPECS[label]
        tracker = _build_trt(
            args.config,
            args.weights,
            args.cache_dir,
            args.cuda_id,
            mode,
            disable_tf32,
            args.rebuild_cache,
        )
        metrics = _measure_features(tracker.net.get_features, samples, device)
        metrics.update(_measure_decisions(tracker, videos, reference, args.max_frames))
        results[label] = metrics
        _print_result(label, metrics)
        del tracker
        gc.collect()
        torch.cuda.empty_cache()

    if "dynamic_fp16" not in results:
        raise RuntimeError("dynamic_fp16 must be included so latency eligibility can be evaluated")
    decomposition = _print_decomposition(results)
    selected = _select_mode(results)
    selected_mode, selected_disable_tf32 = MODE_SPECS[selected]
    report = {
        "selected_mode": selected,
        "selected_config": {
            "backbone_mode": selected_mode,
            "backbone_disable_tf32": selected_disable_tf32,
        },
        "latency_budget": {"mean_multiplier": 1.03, "p95_multiplier": 1.05},
        "videos": [str(path) for path in videos],
        "max_frames": int(args.max_frames),
        "feature_stride": int(args.feature_stride),
        "decomposition": decomposition,
        "results": results,
    }
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRecommended production mode: {selected}")
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
