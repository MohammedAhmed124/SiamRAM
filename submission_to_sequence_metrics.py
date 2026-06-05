"""
Evaluate a frame-level submission CSV against ground truth from video-links.csv.

For each sequence in video-links.csv, ground truth is loaded from
``aic_annotation_path`` when present, otherwise ``local_annotation_path``.

Usage:
    uv run python submission_to_sequence_metrics.py \
        --submission submission.csv \
        --video-links video-links.csv \
        --output sequence_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

W1, W2 = 0.6, 0.4
SCORED_DATASETS = {"dataset2", "dataset3", "dataset4", "dataset5"}
REQUIRED_SUBMISSION_COLS = ("id", "x", "y", "w", "h")


# ── I/O helpers ───────────────────────────────────────────────────────────────


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Read CSV, auto-detecting UTF-16 (common for Excel exports on Windows)."""
    path = Path(path)
    raw = path.read_bytes()[:4]
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return pd.read_csv(path, encoding="utf-16")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-16")


def _nonempty(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(str(value).strip())


def resolve_annotation_path(row: pd.Series) -> tuple[str | None, str]:
    """Return (path, source) where source is 'aic', 'local', or 'none'."""
    aic = row.get("aic_annotation_path")
    if _nonempty(aic):
        return str(aic).strip(), "aic"
    local = row.get("local_annotation_path")
    if _nonempty(local):
        return str(local).strip(), "local"
    return None, "none"


def load_annotation_bboxes(path: str | Path) -> np.ndarray:
    """
    Load x,y,w,h per line from an annotation file.
    Unparseable lines become NaN rows (skipped during metric computation).
    """
    boxes: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p]
            if len(parts) < 4:
                boxes.append([np.nan, np.nan, np.nan, np.nan])
                continue
            try:
                boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                boxes.append([np.nan, np.nan, np.nan, np.nan])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(boxes, dtype=np.float64)


def parse_submission_id(sample_id: str) -> tuple[str, str, int]:
    """Parse ids produced by run_inference.py: ``{dataset}/{seq_name}_{frame}``."""
    if "/" not in sample_id:
        raise ValueError(f"Invalid submission id (expected dataset/seq_frame): {sample_id!r}")
    dataset, rest = sample_id.split("/", 1)
    seq_name, frame_s = rest.rsplit("_", 1)
    if not frame_s.isdigit():
        raise ValueError(f"Invalid submission id (frame suffix not numeric): {sample_id!r}")
    return dataset, seq_name, int(frame_s)


def index_submission(submission: pd.DataFrame) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    """Map (dataset, seq_name) -> {frame_idx: [x,y,w,h]}."""
    missing = [c for c in REQUIRED_SUBMISSION_COLS if c not in submission.columns]
    if missing:
        raise ValueError(f"Submission CSV missing columns: {missing}")

    preds: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    duplicates = 0
    parse_errors = 0

    for row in submission.itertuples(index=False):
        try:
            dataset, seq_name, frame_idx = parse_submission_id(str(row.id))
        except ValueError:
            parse_errors += 1
            continue
        key = (dataset, seq_name)
        if frame_idx in preds[key]:
            duplicates += 1
        preds[key][frame_idx] = np.array(
            [float(row.x), float(row.y), float(row.w), float(row.h)],
            dtype=np.float64,
        )

    if parse_errors:
        print(f"Warning: skipped {parse_errors} submission rows with invalid ids.", file=sys.stderr)
    if duplicates:
        print(f"Warning: {duplicates} duplicate frame predictions overwritten.", file=sys.stderr)
    return preds


# ── Metric helpers (match notebook eval code) ───────────────────────────────


def _valid_box(box: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(box)) and box[2] > 0 and box[3] > 0)


def _iou(gt: Iterable[float], pred: Iterable[float]) -> float:
    gx, gy, gw, gh = gt
    px, py, pw, ph = pred
    if gw <= 0 or gh <= 0 or pw <= 0 or ph <= 0:
        return 0.0
    inter_x = max(0.0, min(gx + gw, px + pw) - max(gx, px))
    inter_y = max(0.0, min(gy + gh, py + ph) - max(gy, py))
    inter = inter_x * inter_y
    union = gw * gh + pw * ph - inter
    return inter / union if union > 0 else 0.0


def _norm_center_error(gt: Iterable[float], pred: Iterable[float]) -> float | None:
    gx, gy, gw, gh = gt
    px, py, pw, ph = pred
    if gw <= 0 or gh <= 0:
        return None
    dist = np.hypot(gx + gw / 2 - px - pw / 2, gy + gh / 2 - py - ph / 2)
    diag = np.hypot(gw, gh)
    return float(dist / diag) if diag > 0 else 0.0


def compute_auc(gt_arr: np.ndarray, pred_arr: np.ndarray, thresholds: np.ndarray | None = None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    ious = np.array([_iou(g, p) for g, p in zip(gt_arr, pred_arr)])
    return float(np.mean([(ious >= t).mean() for t in thresholds]))


def compute_norm_precision(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    mode: str = "auc",
) -> float:
    errors = [_norm_center_error(g, p) for g, p in zip(gt_arr, pred_arr)]
    errors = np.array([e for e in errors if e is not None])
    if len(errors) == 0:
        return 0.0
    if mode == "auc":
        thresholds = np.linspace(0, 0.5, 21)
        return float(np.mean([(errors < t).mean() for t in thresholds]))
    return float((errors < 0.5).mean())


def align_sequence_boxes(
    gt_boxes: np.ndarray,
    pred_by_frame: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, str | None]:
    """
    Align GT and predictions on frame indices 0 .. n_frames-1.
    Missing predictions are treated as zero boxes (target lost).
    """
    if len(gt_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 4)), 0, "empty ground-truth annotation"

    n_frames = len(gt_boxes)
    max_pred_frame = max(pred_by_frame) if pred_by_frame else -1
    if max_pred_frame >= n_frames:
        n_frames = max_pred_frame + 1
        if n_frames > len(gt_boxes):
            pad = np.full((n_frames - len(gt_boxes), 4), np.nan)
            gt_boxes = np.vstack([gt_boxes, pad])

    gt_out = gt_boxes[:n_frames]
    pred_out = np.zeros((n_frames, 4), dtype=np.float64)
    for frame_idx in range(n_frames):
        if frame_idx in pred_by_frame:
            pred_out[frame_idx] = pred_by_frame[frame_idx]

    extra = max(0, max_pred_frame - (len(gt_boxes) - 1)) if pred_by_frame else 0
    warning = None
    if extra > 0:
        warning = f"{extra} prediction frame(s) beyond annotation length"
    return gt_out, pred_out, n_frames, warning


def gt_has_annotation(gt_boxes: np.ndarray) -> bool:
    if len(gt_boxes) == 0:
        return False
    valid = [box for box in gt_boxes if _valid_box(box)]
    if not valid:
        return False
    valid_arr = np.asarray(valid)
    return not np.all(valid_arr == 0)


def metric_frames_mask(gt_boxes: np.ndarray) -> np.ndarray:
    """Frames used for AUC / norm precision (valid, non-degenerate GT)."""
    mask = np.array([_valid_box(box) for box in gt_boxes], dtype=bool)
    return mask & ~(np.all(gt_boxes == 0, axis=1))


# ── Main ──────────────────────────────────────────────────────────────────────


def evaluate_sequence(
    dataset: str,
    seq_name: str,
    link_row: pd.Series,
    pred_by_frame: dict[int, np.ndarray] | None,
) -> dict:
    sequence_id = f"{dataset}__{seq_name}"
    key = f"{dataset}/{seq_name}"
    scored = dataset in SCORED_DATASETS

    ann_path, ann_source = resolve_annotation_path(link_row)
    video_path = link_row.get("aic_video_path", np.nan)
    match_type = link_row.get("match_type", np.nan)
    local_label = link_row.get("local_label", np.nan)

    base = {
        "sequence_id": sequence_id,
        "key": key,
        "dataset": dataset,
        "seq_name": seq_name,
        "annotation_source": ann_source,
        "video_path": video_path,
        "annotation_path": ann_path if ann_path else np.nan,
        "match_type": match_type,
        "local_label": local_label,
        "evaluation_mode": "unscored",
        "frame_count": 0,
        "ground_truth_frames": 0,
        "scored_frames": 0,
        "matched_prediction_frames": 0,
        "auc": np.nan,
        "norm_precision": np.nan,
        "accuracy_score": np.nan,
        "total_lost_frames": 0,
        "error": np.nan,
        "warning": np.nan,
    }

    if pred_by_frame is None:
        return base

    if ann_path is None:
        base["matched_prediction_frames"] = len(pred_by_frame)
        base["frame_count"] = len(pred_by_frame)
        base["total_lost_frames"] = int(
            sum(1 for box in pred_by_frame.values() if np.all(box == 0))
        )
        return base

    ann_file = Path(ann_path)
    if not ann_file.is_file():
        base["error"] = f"annotation file not found: {ann_path}"
        base["matched_prediction_frames"] = len(pred_by_frame)
        return base

    try:
        gt_boxes = load_annotation_bboxes(ann_file)
    except OSError as exc:
        base["error"] = f"failed to read annotation: {exc}"
        return base

    gt_boxes, pred_boxes, n_frames, align_warning = align_sequence_boxes(gt_boxes, pred_by_frame)
    has_gt = gt_has_annotation(gt_boxes)
    mask = metric_frames_mask(gt_boxes)

    base["frame_count"] = n_frames
    base["ground_truth_frames"] = int(mask.sum()) if has_gt else 0
    base["matched_prediction_frames"] = len(pred_by_frame)
    base["total_lost_frames"] = int(np.all(pred_boxes == 0, axis=1).sum())
    if align_warning:
        base["warning"] = align_warning

    if scored and has_gt and mask.any():
        gt_scored = gt_boxes[mask]
        pred_scored = pred_boxes[mask]
        auc = compute_auc(gt_scored, pred_scored)
        norm_prec = compute_norm_precision(gt_scored, pred_scored, mode="auc")
        base.update(
            {
                "evaluation_mode": "scored",
                "scored_frames": int(mask.sum()),
                "auc": auc,
                "norm_precision": norm_prec,
                "accuracy_score": W1 * auc + W2 * norm_prec,
            }
        )
    elif not scored:
        base["evaluation_mode"] = "unscored"
    elif not has_gt:
        base["evaluation_mode"] = "unscored"
        base["error"] = "ground truth is empty or all-zero"
    else:
        base["evaluation_mode"] = "unscored"
        base["error"] = "no valid ground-truth frames for scoring"

    return base


def main(
    submission_path: str,
    video_links_path: str,
    output_path: str,
    *,
    only_submitted: bool = False,
) -> pd.DataFrame:
    submission = read_csv_robust(submission_path)
    links = read_csv_robust(video_links_path)

    required_link_cols = {"aic_dataset", "aic_sequence"}
    missing_link = required_link_cols - set(links.columns)
    if missing_link:
        raise ValueError(f"video-links CSV missing columns: {sorted(missing_link)}")

    pred_index = index_submission(submission)
    submitted_keys = set(pred_index)

    rows: list[dict] = []
    for row_index, link_row in links.iterrows():
        dataset = str(link_row["aic_dataset"]).strip()
        seq_name = str(link_row["aic_sequence"]).strip()
        key = (dataset, seq_name)
        if only_submitted and key not in submitted_keys:
            continue
        result = evaluate_sequence(
            dataset,
            seq_name,
            link_row,
            pred_index.get(key),
        )
        result["row_index"] = row_index
        rows.append(result)

    out = pd.DataFrame(rows)
    col_order = [
        "row_index",
        "sequence_id",
        "key",
        "dataset",
        "seq_name",
        "annotation_source",
        "evaluation_mode",
        "frame_count",
        "ground_truth_frames",
        "scored_frames",
        "matched_prediction_frames",
        "auc",
        "norm_precision",
        "accuracy_score",
        "total_lost_frames",
        "video_path",
        "annotation_path",
        "match_type",
        "local_label",
        "warning",
        "error",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Written {len(out)} rows to {out_path}")

    submitted_rows = out[out["matched_prediction_frames"] > 0]
    print(f"Sequences with submission rows: {len(submitted_rows)}")
    if len(submitted_rows) < len(submitted_keys):
        print(
            f"Warning: {len(submitted_keys) - len(submitted_rows)} submitted sequence(s) "
            "not found in video-links.",
            file=sys.stderr,
        )

    scored = out[out["evaluation_mode"] == "scored"]
    if len(scored):
        mean_auc = scored["auc"].mean()
        mean_prec = scored["norm_precision"].mean()
        print(f"Scored sequences:         {len(scored)}")
        print(f"Mean AUC (dataset2-5):    {mean_auc:.4f}")
        print(f"Mean NormPrec (dataset2-5): {mean_prec:.4f}")
        print(f"Final score (W1*AUC+W2*NP): {W1 * mean_auc + W2 * mean_prec:.4f}")
    else:
        print("No scored sequences (need dataset2-5 with valid annotations and predictions).")

    errors = out[out["error"].notna() & (out["error"].astype(str).str.strip() != "")]
    if len(errors):
        print(f"Sequences with errors:    {len(errors)}", file=sys.stderr)
        for _, err_row in errors.head(5).iterrows():
            print(f"  - {err_row['key']}: {err_row['error']}", file=sys.stderr)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more", file=sys.stderr)

    missing_submission = int((out["matched_prediction_frames"] == 0).sum())
    if missing_submission and not only_submitted:
        print(f"Sequences without submission: {missing_submission}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate submission.csv against ground truth from video-links.csv.",
    )
    parser.add_argument("--submission", default="submission.csv", help="Frame-level submission CSV.")
    parser.add_argument(
        "--video-links",
        default="video-links.csv",
        help="Sequence metadata CSV (aic/local annotation paths).",
    )
    parser.add_argument(
        "--output",
        default="sequence_metrics.csv",
        help="Per-sequence metrics output CSV.",
    )
    parser.add_argument(
        "--only-submitted",
        action="store_true",
        help="Only output rows for sequences present in the submission CSV.",
    )
    args = parser.parse_args()
    main(
        args.submission,
        args.video_links,
        args.output,
        only_submitted=args.only_submitted,
    )
