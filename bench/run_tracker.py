"""Run SiamRAM over a benchmark under the one-pass (OPE) protocol.

Reads original JPEG frames, initialises on frame 0 and never re-initialises.
Writes <out>/<sequence>.txt (comma-separated x,y,w,h per frame) and
<out>/<sequence>_time.txt (per-frame tracker seconds).

    python bench/run_tracker.py --dataset dtb70 --data-root D:/data/DTB70 \
        --config inference_config.yaml --out results/siamram/dtb70
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench.datasets import Sequence, load_sequences  # noqa: E402


def track_sequence(model, seq: Sequence) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """One-pass track over a frame directory. Returns per-frame boxes and tracker times."""
    first = cv2.imread(str(seq.frames[0]))
    if first is None:
        raise RuntimeError(f"cannot read frame {seq.frames[0]}")

    model.begin_sequence(seq.name, len(seq.frames))
    t0 = time.perf_counter()
    model.initialize(first, list(seq.init_box))
    boxes = [tuple(float(v) for v in seq.init_box)]
    times = [time.perf_counter() - t0]

    for path in seq.frames[1:]:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read frame {path}")
        t0 = time.perf_counter()
        bbox, _score, _in_occ, _yolo = model.update(frame)
        times.append(time.perf_counter() - t0)
        boxes.append(tuple(float(v) for v in bbox[:4]))

    model.end_sequence(elapsed_s=sum(times))
    return boxes, times


def main():
    ap = argparse.ArgumentParser(description="One-pass benchmark run for SiamRAM.")
    ap.add_argument("--dataset", required=True, help="Benchmark name (see bench/datasets.py LOADERS)")
    ap.add_argument("--data-root", required=True, help="Directory holding the official benchmark")
    ap.add_argument("--config", default=None,
                    help="Inference config: bare filename resolved under src/config, or a path")
    ap.add_argument("--out", required=True, help="Output directory for the raw prediction files")
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N sequences")
    ap.add_argument("--seq", default=None, help="Only run this sequence")
    args = ap.parse_args()

    import predictor
    import torch

    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = _REPO_ROOT / "src" / "config" / cfg_path
        if not cfg_path.exists():
            raise SystemExit(f"config not found: {cfg_path}")
        predictor._CONFIG_PATH = cfg_path

    sequences = load_sequences(args.dataset, args.data_root)
    if args.seq:
        sequences = [s for s in sequences if s.name == args.seq]
        if not sequences:
            raise SystemExit(f"sequence {args.seq!r} not found in {args.dataset}")
    if args.limit:
        sequences = sequences[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | config: {Path(predictor._CONFIG_PATH).name} | "
          f"dataset: {args.dataset} | sequences: {len(sequences)}")
    model = predictor.load_model(device=device)

    total_frames, total_time, failed = 0, 0.0, []
    for i, seq in enumerate(sequences, 1):
        print(f"[{i}/{len(sequences)}] {seq.name} ({len(seq.frames)} frames)")
        try:
            boxes, times = track_sequence(model, seq)
        except (RuntimeError, ValueError) as exc:
            print(f"  [skip] {seq.name}: {exc}")
            failed.append(seq.name)
            continue
        (out_dir / f"{seq.name}.txt").write_text(
            "".join(f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n" for x, y, w, h in boxes))
        (out_dir / f"{seq.name}_time.txt").write_text(
            "".join(f"{t:.6f}\n" for t in times))
        total_frames += len(boxes)
        total_time += sum(times)

    fps = total_frames / total_time if total_time > 0 else 0.0
    print(f"\n{total_frames} frames | {total_time:.1f}s tracker time | {fps:.1f} FPS | results in {out_dir}")
    if failed:
        print(f"Failed {len(failed)}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
