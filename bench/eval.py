"""Standard OPE metrics (Success AUC, Precision@20px, Normalized Precision) for tracking benchmarks.

    python bench/eval.py --results results --dataset dtb70 --data-root D:/data/DTB70 \
        --trackers siamram,siamabc --out results/dtb70.csv --protocol-check
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.datasets import load_sequences, read_boxes  # noqa: E402
from bench.splits import load_manifest, overlap  # noqa: E402

SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 21)      # 0:0.05:1
NORM_PREC_THRESHOLDS = np.linspace(0.0, 0.5, 11)    # 0:0.05:0.5
PRECISION_THRESHOLD = 20.0                          # pixels


def iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame intersection over union of two Nx4 x,y,w,h box arrays."""
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    gt = np.atleast_2d(np.asarray(gt, dtype=float))
    pw, ph = np.maximum(pred[:, 2], 0), np.maximum(pred[:, 3], 0)
    gw, gh = np.maximum(gt[:, 2], 0), np.maximum(gt[:, 3], 0)
    iw = np.maximum(np.minimum(pred[:, 0] + pw, gt[:, 0] + gw) - np.maximum(pred[:, 0], gt[:, 0]), 0)
    ih = np.maximum(np.minimum(pred[:, 1] + ph, gt[:, 1] + gh) - np.maximum(pred[:, 1], gt[:, 1]), 0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean distance in pixels between box centres."""
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    gt = np.atleast_2d(np.asarray(gt, dtype=float))
    d = (pred[:, :2] + pred[:, 2:4] / 2.0) - (gt[:, :2] + gt[:, 2:4] / 2.0)
    return np.sqrt((d ** 2).sum(axis=1))


def norm_center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame centre error normalised by the ground-truth box size."""
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    gt = np.atleast_2d(np.asarray(gt, dtype=float))
    d = (pred[:, :2] + pred[:, 2:4] / 2.0) - (gt[:, :2] + gt[:, 2:4] / 2.0)
    return np.sqrt((d ** 2 / np.maximum(gt[:, 2:4], 1e-12) ** 2).sum(axis=1))


def success_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Area under the success plot: mean over IoU thresholds of the fraction of frames above each.

    Strict `>` over the 21 thresholds including 1.0, as in OTB/pysot/pytracking, so a
    perfect tracker scores 20/21 = 0.952 rather than 1.0.
    """
    overlaps = iou(pred, gt)
    return float(np.mean([(overlaps > t).mean() for t in SUCCESS_THRESHOLDS]))


def precision(pred: np.ndarray, gt: np.ndarray, threshold: float = PRECISION_THRESHOLD) -> float:
    """Fraction of frames whose centre error is within the threshold in pixels."""
    return float((center_error(pred, gt) <= threshold).mean())


def norm_precision_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Area under the normalized precision plot over 0:0.05:0.5."""
    errors = norm_center_error(pred, gt)
    return float(np.mean([(errors <= t).mean() for t in NORM_PREC_THRESHOLDS]))


def evaluate_sequence(pred: np.ndarray, gt: np.ndarray) -> dict:
    """AUC / Precision@20px / Normalized Precision over the frames with present ground truth."""
    pred = np.atleast_2d(np.asarray(pred, dtype=float))
    gt = np.atleast_2d(np.asarray(gt, dtype=float))
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    keep = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    pred, gt = pred[keep], gt[keep]
    if len(gt) == 0:
        return {"frames": 0, "auc": float("nan"), "prec20": float("nan"), "norm_prec": float("nan")}
    return {
        "frames": len(gt),
        "auc": success_auc(pred, gt),
        "prec20": precision(pred, gt),
        "norm_prec": norm_precision_auc(pred, gt),
    }


def _tracker_dirs(results: Path, trackers: list[str] | None) -> dict[str, Path]:
    if trackers:
        return {t: (results / t if (results / t).is_dir() else results) for t in trackers}
    subdirs = [p for p in sorted(results.iterdir()) if p.is_dir()]
    return {p.name: p for p in subdirs} if subdirs else {results.name: results}


def _protocol_check(dataset: str, sequences, tracker_dirs: dict[str, Path]) -> None:
    absent = sum(0 if s.gt is None else int((~np.isfinite(s.gt).all(axis=1)).sum()) for s in sequences)
    no_gt = [s.name for s in sequences if s.gt is None]
    print(f"\n## Protocol check - {dataset}")
    print(f"sequences: {len(sequences)} | frames: {sum(len(s.frames) for s in sequences)} | "
          f"absent-GT frames: {absent}")
    if no_gt:
        print(f"sequences with NO ground truth (not evaluable): {len(no_gt)} - {', '.join(no_gt)}")
    problems = 0
    for tracker, d in tracker_dirs.items():
        for s in sequences:
            f = d / f"{s.name}.txt"
            if not f.is_file():
                print(f"  !! MISSING  {tracker}/{s.name}.txt")
                problems += 1
                continue
            n = len(read_boxes(f))
            if n != len(s.frames):
                print(f"  !! LENGTH MISMATCH  {tracker}/{s.name}: {n} result rows vs "
                      f"{len(s.frames)} frames")
                problems += 1
    print("all result files present and frame-aligned" if problems == 0
          else f"{problems} problem(s) - these invalidate the numbers below")

    leaked = overlap(dataset, [s.name for s in sequences])
    if leaked:
        print(f"  !! LEAKAGE  {len(leaked)} of {len(sequences)} test sequences are in the "
              f"training index (splits/manifest.json): {', '.join(leaked[:8])}"
              f"{' ...' if len(leaked) > 8 else ''}")
        print("  these scores are diagnostic, not held-out evidence")
    elif load_manifest():
        print("no test sequence appears in the training index")


def main():
    ap = argparse.ArgumentParser(description="OPE evaluation for tracking benchmarks.")
    ap.add_argument("--results", required=True, help="Directory holding <tracker>/<sequence>.txt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True, help="Directory holding the official benchmark")
    ap.add_argument("--trackers", default=None, help="Comma-separated tracker names (default: subdirs)")
    ap.add_argument("--out", default=None, help="CSV output path")
    ap.add_argument("--protocol-check", action="store_true", help="Report sequence/frame/GT sanity first")
    args = ap.parse_args()

    results = Path(args.results)
    trackers = [t.strip() for t in args.trackers.split(",") if t.strip()] if args.trackers else None
    tracker_dirs = _tracker_dirs(results, trackers)
    sequences = load_sequences(args.dataset, args.data_root)

    if args.protocol_check:
        _protocol_check(args.dataset, sequences, tracker_dirs)

    rows, summary = [], []
    for tracker, d in tracker_dirs.items():
        per_seq = []
        fps_frames, fps_time = 0, 0.0
        for s in sequences:
            f = d / f"{s.name}.txt"
            if s.gt is None or not f.is_file():
                continue
            m = evaluate_sequence(read_boxes(f), s.gt)
            if m["frames"] == 0:
                continue
            per_seq.append(m)
            rows.append({"tracker": tracker, "sequence": s.name, **m})
            time_file = d / f"{s.name}_time.txt"
            if time_file.is_file():
                times = [float(v) for v in time_file.read_text().split() if v]
                fps_frames += len(times)
                fps_time += sum(times)
        if not per_seq:
            print(f"[warn] {tracker}: no evaluable sequences under {d}")
            continue
        overall = {
            "tracker": tracker,
            "sequence": "OVERALL",
            "frames": sum(m["frames"] for m in per_seq),
            "auc": float(np.mean([m["auc"] for m in per_seq])),
            "prec20": float(np.mean([m["prec20"] for m in per_seq])),
            "norm_prec": float(np.mean([m["norm_prec"] for m in per_seq])),
        }
        rows.append(overall)
        summary.append((tracker, len(per_seq), overall,
                        fps_frames / fps_time if fps_time > 0 else float("nan")))

    print(f"\n## {args.dataset} - OPE ({len(sequences)} sequences)\n")
    print("| Tracker | Seqs | Success AUC | Precision@20px | Norm. Precision | FPS |")
    print("|---|---:|---:|---:|---:|---:|")
    for tracker, n, o, fps in summary:
        fps_str = f"{fps:.1f}" if np.isfinite(fps) else "-"
        print(f"| {tracker} | {n} | {o['auc']:.3f} | {o['prec20']:.3f} | "
              f"{o['norm_prec']:.3f} | {fps_str} |")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["tracker", "sequence", "frames",
                                                    "auc", "prec20", "norm_prec"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
