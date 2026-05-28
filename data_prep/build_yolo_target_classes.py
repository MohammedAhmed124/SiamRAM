"""
Annotate sequence-index CSV rows with YOLO target class hints.

Why this exists
---------------
Hard negative mining is stronger when we know each sequence's semantic class
from a detector rather than relying only on folder names. This script scans a
few frames per sequence, matches YOLO detections to the ground-truth box by
IoU, and writes `yolo_target_class` back into a new CSV.

Usage
-----
python data_prep/build_yolo_target_classes.py \
  --input data/train_dataframe.csv \
  --output data/train_dataframe_with_yolo.csv \
  --weights checkpoints/yolo11n.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset import TrackingSequence


def _xywh_to_xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _sample_indices(start_idx: int, end_idx: int, count: int) -> List[int]:
    if end_idx < start_idx:
        return []
    total = end_idx - start_idx + 1
    n = max(1, min(count, total))
    idxs = np.linspace(start_idx, end_idx, num=n, dtype=np.int64)
    return sorted({int(v) for v in idxs.tolist()})


def _load_image_rgb(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _best_detection_class(
    model,
    image_rgb: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    conf_thr: float,
    nms_iou_thr: float,
    match_iou_thr: float,
) -> Optional[int]:
    gt_xyxy = _xywh_to_xyxy(gt_bbox_xywh)

    result = model.predict(
        image_rgb,
        conf=conf_thr,
        iou=nms_iou_thr,
        verbose=False,
    )[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    det_xyxy = boxes.xyxy.detach().cpu().numpy()
    det_cls = boxes.cls.detach().cpu().numpy().astype(np.int64)

    best_iou = 0.0
    best_cls: Optional[int] = None
    for det_box, cls_id in zip(det_xyxy, det_cls):
        iou = _iou_xyxy(det_box, gt_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_cls = int(cls_id)

    if best_cls is None or best_iou < match_iou_thr:
        return None
    return best_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO class hints for sequence CSV")
    parser.add_argument("--input", type=Path, required=True, help="Input sequence CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>_with_yolo.csv)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("checkpoints/yolo11n.pt"),
        help="Path to YOLO weights",
    )
    parser.add_argument("--frames-per-seq", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.25,
        help="Minimum IoU between detection and GT box to accept class vote",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    # Allow Ultralytics to auto-download known checkpoint names (e.g. yolo11l.pt).
    # We only hard-fail when a non-current-directory path is explicitly provided
    # but does not exist.
    if not args.weights.exists() and args.weights.parent not in {Path("."), Path("")}:
        raise FileNotFoundError(
            f"YOLO weights not found at explicit path: {args.weights}. "
            "Either provide an existing local path or use a model name like yolo11l.pt."
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required for this script. Install it in your environment."
        ) from exc

    df = pd.read_csv(args.input)
    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_with_yolo{args.input.suffix}"
    )

    if not args.weights.exists():
        print(
            f"[info] Weights '{args.weights}' not found locally. "
            "Ultralytics will try to auto-download it."
        )
    model = YOLO(str(args.weights))
    names = model.names

    yolo_cls_ids: List[Optional[int]] = []
    yolo_cls_names: List[str] = []
    yolo_vote_counts: List[int] = []

    matched = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="YOLO class hints"):
        seq = TrackingSequence(row.to_dict())
        votes: List[int] = []

        frame_indices = _sample_indices(
            start_idx=int(row["start_idx"]),
            end_idx=int(row["end_idx"]),
            count=int(args.frames_per_seq),
        )

        for frame_idx in frame_indices:
            bbox = seq.get_bbox(frame_idx)
            if not seq.is_valid_bbox(bbox):
                continue

            frame_path = seq.frame_path(frame_idx)
            img = _load_image_rgb(frame_path)
            if img is None:
                continue

            cls_id = _best_detection_class(
                model=model,
                image_rgb=img,
                gt_bbox_xywh=bbox,
                conf_thr=float(args.conf),
                nms_iou_thr=float(args.nms_iou),
                match_iou_thr=float(args.match_iou),
            )
            if cls_id is not None:
                votes.append(cls_id)

        if votes:
            cls_id = Counter(votes).most_common(1)[0][0]
            cls_name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
            matched += 1
            yolo_cls_ids.append(cls_id)
            yolo_cls_names.append(cls_name)
            yolo_vote_counts.append(len(votes))
        else:
            fallback = row.get("class", "")
            yolo_cls_ids.append(None)
            yolo_cls_names.append(str(fallback) if not pd.isna(fallback) else "")
            yolo_vote_counts.append(0)

    out_df = df.copy()
    out_df["yolo_target_class_id"] = yolo_cls_ids
    out_df["yolo_target_class"] = yolo_cls_names
    out_df["yolo_target_votes"] = yolo_vote_counts
    out_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Matched by YOLO: {matched}/{len(df)} ({(100.0 * matched / max(1, len(df))):.1f}%)")


if __name__ == "__main__":
    main()
