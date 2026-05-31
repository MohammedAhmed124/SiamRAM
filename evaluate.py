"""
Evaluate public-LB tracking predictions against generated labels.

Usage:
    python evaluate.py submission_experimental_public_lb.csv
    python evaluate.py submission.csv --gt_csv /path/to/generated_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GT_CSV = BASE_DIR / "data" / "metadata" / "generated_labels.csv"
SUBMISSION_COLUMNS = ["id", "x", "y", "w", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate public-LB tracking predictions against generated labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pred_csv",
        help="Prediction CSV with columns: id,x,y,w,h.",
    )
    parser.add_argument(
        "--gt_csv",
        default=str(DEFAULT_GT_CSV),
        help="Generated label CSV with columns: id,x,y,w,h,key.",
    )
    parser.add_argument(
        "--norm_precision_mode",
        choices=["auc", "single"],
        default="auc",
        help="Use normalized-precision AUC over 0..threshold, or one threshold.",
    )
    parser.add_argument(
        "--norm_precision_threshold",
        type=float,
        default=0.5,
        help="Normalized precision threshold/range end.",
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=0.5,
        help="Final-score weight for mean AUC.",
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=0.5,
        help="Final-score weight for mean normalized precision.",
    )
    parser.add_argument(
        "--scored_datasets",
        nargs="+",
        default=["dataset2", "dataset3", "dataset4", "dataset5"],
        help="Datasets included in the final score.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required: list[str], csv_path: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path!r} is missing required columns: {missing}")


def _load_inputs(gt_csv: str, pred_csv: str) -> pd.DataFrame:
    gt_df = pd.read_csv(gt_csv, dtype={"id": str, "key": str})
    pred_df = pd.read_csv(pred_csv, dtype={"id": str})

    _require_columns(gt_df, SUBMISSION_COLUMNS + ["key"], gt_csv)
    _require_columns(pred_df, SUBMISSION_COLUMNS, pred_csv)

    if gt_df["id"].duplicated().any():
        examples = gt_df.loc[gt_df["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Ground-truth CSV has duplicate ids, e.g. {examples}")
    if pred_df["id"].duplicated().any():
        examples = pred_df.loc[pred_df["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Prediction CSV has duplicate ids, e.g. {examples}")

    gt_ids = set(gt_df["id"])
    pred_ids = set(pred_df["id"])
    missing_pred = sorted(gt_ids - pred_ids)
    extra_pred = sorted(pred_ids - gt_ids)
    if missing_pred:
        raise ValueError(
            "Prediction CSV is missing ids from the labels. "
            f"Missing count: {len(missing_pred)}. Examples: {missing_pred[:10]}"
        )
    if extra_pred:
        print(
            "Warning: prediction CSV has ids not present in labels; "
            f"ignoring {len(extra_pred)} extra row(s). Examples: {extra_pred[:10]}"
        )

    gt_eval = gt_df[["id", "key", "x", "y", "w", "h"]].rename(
        columns={"key": "gt_key", "x": "gx", "y": "gy", "w": "gw", "h": "gh"}
    )
    pred_eval = pred_df[SUBMISSION_COLUMNS].rename(
        columns={"x": "px", "y": "py", "w": "pw", "h": "ph"}
    )

    merged = gt_eval.merge(pred_eval, on="id", how="inner", validate="one_to_one")
    for column in ["gx", "gy", "gw", "gh", "px", "py", "pw", "ph"]:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
    merged["dataset"] = merged["gt_key"].str.split("/", n=1).str[0]
    return merged


def _valid_box_mask(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)


def _box_iou_xywh(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    gt = gt_boxes.astype(np.float64, copy=False)
    pred = pred_boxes.astype(np.float64, copy=False)

    gt_valid = _valid_box_mask(gt)
    pred_valid = _valid_box_mask(pred)

    gt_x2 = gt[:, 0] + np.maximum(gt[:, 2], 0.0)
    gt_y2 = gt[:, 1] + np.maximum(gt[:, 3], 0.0)
    pred_x2 = pred[:, 0] + np.maximum(pred[:, 2], 0.0)
    pred_y2 = pred[:, 1] + np.maximum(pred[:, 3], 0.0)

    inter_w = np.maximum(0.0, np.minimum(gt_x2, pred_x2) - np.maximum(gt[:, 0], pred[:, 0]))
    inter_h = np.maximum(0.0, np.minimum(gt_y2, pred_y2) - np.maximum(gt[:, 1], pred[:, 1]))
    inter = inter_w * inter_h

    gt_area = np.maximum(gt[:, 2], 0.0) * np.maximum(gt[:, 3], 0.0)
    pred_area = np.maximum(pred[:, 2], 0.0) * np.maximum(pred[:, 3], 0.0)
    union = gt_area + pred_area - inter

    iou = np.zeros(len(gt), dtype=np.float64)
    valid_union = union > 0.0
    iou[valid_union] = inter[valid_union] / union[valid_union]

    # If both boxes are absent/empty, count that as a correct absent prediction.
    both_absent = ~gt_valid & ~pred_valid
    iou[both_absent] = 1.0
    return np.clip(iou, 0.0, 1.0)


def sequence_auc(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> float:
    ious = _box_iou_xywh(gt_boxes, pred_boxes)
    thresholds = np.linspace(0.0, 1.0, 101)
    success = (ious[:, None] >= thresholds[None, :]).mean(axis=0)
    return float(success.mean())


def _normalized_center_errors(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    gt = gt_boxes.astype(np.float64, copy=False)
    pred = pred_boxes.astype(np.float64, copy=False)

    gt_valid = _valid_box_mask(gt)
    pred_valid = _valid_box_mask(pred)

    gt_cx = gt[:, 0] + gt[:, 2] / 2.0
    gt_cy = gt[:, 1] + gt[:, 3] / 2.0
    pred_cx = pred[:, 0] + pred[:, 2] / 2.0
    pred_cy = pred[:, 1] + pred[:, 3] / 2.0

    errors = np.full(len(gt), np.inf, dtype=np.float64)
    valid = gt_valid
    errors[valid] = np.sqrt(
        ((pred_cx[valid] - gt_cx[valid]) / gt[valid, 2]) ** 2
        + ((pred_cy[valid] - gt_cy[valid]) / gt[valid, 3]) ** 2
    )

    # If both boxes are absent/empty, count that as zero center error.
    errors[~gt_valid & ~pred_valid] = 0.0
    return errors


def sequence_norm_precision(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    mode: str = "auc",
    threshold: float = 0.5,
) -> float:
    errors = _normalized_center_errors(gt_boxes, pred_boxes)
    if len(errors) == 0:
        return 0.0

    if mode == "single":
        return float((errors <= threshold).mean())

    thresholds = np.linspace(0.0, threshold, 101)
    precision = (errors[:, None] <= thresholds[None, :]).mean(axis=0)
    return float(precision.mean())


def evaluate_tracking(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    norm_precision_mode: str = "auc",
    norm_precision_threshold: float = 0.5,
    scored_datasets: list[str] | None = None,
    w1: float = 0.5,
    w2: float = 0.5,
) -> tuple[dict[str, dict[str, float]], float]:
    if scored_datasets is None:
        scored_datasets = ["dataset2", "dataset3", "dataset4", "dataset5"]

    gt_eval = gt_df[["id", "key", "x", "y", "w", "h"]].rename(
        columns={"key": "gt_key", "x": "gx", "y": "gy", "w": "gw", "h": "gh"}
    )
    pred_eval = pred_df[SUBMISSION_COLUMNS].rename(
        columns={"x": "px", "y": "py", "w": "pw", "h": "ph"}
    )
    merged = gt_eval.merge(pred_eval, on="id", how="inner", validate="one_to_one")
    for column in ["gx", "gy", "gw", "gh", "px", "py", "pw", "ph"]:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
    merged["dataset"] = merged["gt_key"].str.split("/", n=1).str[0]
    return evaluate_merged_tracking(
        merged=merged,
        norm_precision_mode=norm_precision_mode,
        norm_precision_threshold=norm_precision_threshold,
        scored_datasets=scored_datasets,
        w1=w1,
        w2=w2,
    )


def evaluate_merged_tracking(
    merged: pd.DataFrame,
    norm_precision_mode: str = "auc",
    norm_precision_threshold: float = 0.5,
    scored_datasets: list[str] | None = None,
    w1: float = 0.5,
    w2: float = 0.5,
) -> tuple[dict[str, dict[str, float]], float]:
    if scored_datasets is None:
        scored_datasets = ["dataset2", "dataset3", "dataset4", "dataset5"]

    all_datasets = sorted(merged["dataset"].dropna().unique())
    results: dict[str, dict[str, float]] = {}
    sep = "─" * 55

    print(sep)
    print(f"{'TRACKING EVALUATION REPORT':^55}")
    print(sep)

    for dataset_name in all_datasets:
        ds_data = merged[merged["dataset"] == dataset_name]
        sequences = ds_data["gt_key"].unique()

        print(f"\n📂  {dataset_name.upper()}  ({len(sequences)} sequences)")
        print(f"  {'Sequence':<35} {'AUC':>7}  {'NormPrec':>9}")
        print(f"  {'─' * 35} {'─' * 7}  {'─' * 9}")

        ds_aucs: list[float] = []
        ds_precs: list[float] = []

        for seq_key in sorted(sequences):
            seq_data = ds_data[ds_data["gt_key"] == seq_key].sort_values("id")

            gt_boxes = seq_data[["gx", "gy", "gw", "gh"]].values
            pred_boxes = seq_data[["px", "py", "pw", "ph"]].values

            auc = sequence_auc(gt_boxes, pred_boxes)
            prec = sequence_norm_precision(
                gt_boxes,
                pred_boxes,
                mode=norm_precision_mode,
                threshold=norm_precision_threshold,
            )

            results[seq_key] = {"auc": auc, "norm_precision": prec}
            ds_aucs.append(auc)
            ds_precs.append(prec)

            seq_label = seq_key.split("/", 1)[-1]
            print(f"  {seq_label:<35} {auc:>7.4f}  {prec:>9.4f}")

        ds_mean_auc = np.mean(ds_aucs) if ds_aucs else 0.0
        ds_mean_prec = np.mean(ds_precs) if ds_precs else 0.0
        print(f"  {'─' * 35} {'─' * 7}  {'─' * 9}")
        print(f"  {'DATASET MEAN':<35} {ds_mean_auc:>7.4f}  {ds_mean_prec:>9.4f}")

    print(f"\n{sep}")
    print(f"{'FINAL SCORE  (dataset2 → dataset5)':^55}")
    print(sep)

    scored_keys = [k for k in results if k.split("/")[0] in scored_datasets]
    all_aucs = [results[k]["auc"] for k in scored_keys]
    all_precs = [results[k]["norm_precision"] for k in scored_keys]

    mean_auc = np.mean(all_aucs) if all_aucs else 0.0
    mean_prec = np.mean(all_precs) if all_precs else 0.0
    final_score = w1 * mean_auc + w2 * mean_prec

    scored_label = _format_scored_datasets(scored_datasets)
    print(f"  Mean AUC           ({scored_label}) : {mean_auc:.4f}")
    print(f"  Mean NormPrecision ({scored_label}) : {mean_prec:.4f}")
    print(f"\n  S_acc = {w1} × {mean_auc:.4f}  +  {w2} × {mean_prec:.4f}")
    print(f"\n  ★  Final Score  =  {final_score:.4f}")
    print(sep)

    return results, final_score


def _format_scored_datasets(scored_datasets: list[str]) -> str:
    if scored_datasets == ["dataset2", "dataset3", "dataset4", "dataset5"]:
        return "dataset2–5"
    return ",".join(scored_datasets)


def main() -> None:
    args = parse_args()

    print("\n>>> MODE: AUC over thresholds 0→0.5\n" if args.norm_precision_mode == "auc" else "\n>>> MODE: single threshold (0.5)\n")

    merged = _load_inputs(args.gt_csv, args.pred_csv)
    _, score = evaluate_merged_tracking(
        merged=merged,
        norm_precision_mode=args.norm_precision_mode,
        norm_precision_threshold=args.norm_precision_threshold,
        scored_datasets=args.scored_datasets,
        w1=args.w1,
        w2=args.w2,
    )

    if args.norm_precision_mode == "auc":
        print(f"  auc    : {score:.4f}")
    else:
        print(f"  single : {score:.4f}")


if __name__ == "__main__":
    main()
