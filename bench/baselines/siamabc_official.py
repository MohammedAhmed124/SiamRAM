"""Run the upstream (wvuvl/SiamABC) tracker over a benchmark under the one-pass (OPE) protocol.

Clones github.com/wvuvl/SiamABC at a pinned commit, drives its own SiamABCTracker and
writes the same files as bench/run_tracker.py so bench/eval.py scores both side by side.
The official weights ship inside that repo under assets/, so the clone is the download.

    python bench/baselines/siamabc_official.py --dataset lasot --data-root /data/lasot \
        --out results/lasot/siamabc_official_tiny --variant tiny
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.datasets import Sequence, load_sequences  # noqa: E402

UPSTREAM_URL = "https://github.com/wvuvl/SiamABC.git"
UPSTREAM_COMMIT = "b1c94e06fdf2dd3cb14ed07b05e38aa4601ece03"  # master, 2025-06-23

# model_size 'S' is the FBNetV2 backbone (paper: S-Tiny), 'M' is ResNet-50 (S-Small).
VARIANTS = {
    "tiny": ("S", "assets/S_Tiny/model_S_Tiny_v1.pt"),
    "small": ("M", "assets/S_Small/model_S_Small_v1.pt"),
}

# core/config/tracker/siam_tracker.yaml ships N=150 with the dynamic update and the
# cosine-window smoothing off; the paper's protocol is N=60 with both on.
PAPER_TRACKER_CONFIG = {"N": 60, "dynamic_update": True, "smooth": True}
DTTA_LAMBDA = 0.1  # paper lambda_BN


def clone_upstream(work_dir: Path) -> Path:
    """Upstream repo (code plus the official weights) checked out at the pinned commit."""
    repo = work_dir / "SiamABC"
    if (repo / "SiamABC_tracker.py").is_file():
        return repo
    repo.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(a, cwd=repo, check=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", UPSTREAM_URL)
    run("git", "fetch", "-q", "--depth", "1", "origin", UPSTREAM_COMMIT)
    run("git", "checkout", "-q", "FETCH_HEAD")
    return repo


def build_tracker(repo: Path, variant: str, weights: Path, dtta: bool, overrides: dict):
    """Upstream SiamABCTracker with the official weights loaded."""
    import torch

    model_size, _ = VARIANTS[variant]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)  # core.utils.hydra resolves the config path relative to the cwd

    from hydra.utils import instantiate

    from core.models.custom_bn import replace_layers
    from core.utils.hydra import load_hydra_config_from_path
    from realtime_test import load_model

    cfg = load_hydra_config_from_path(config_path="core/config", config_name="SiamABC_tracker")
    cfg["model"]["model_size"] = model_size
    cfg["tracker"].update(PAPER_TRACKER_CONFIG)
    cfg["tracker"].update(overrides)
    print(f"tracker config: {cfg['tracker']}")

    model = instantiate(cfg["model"])
    if dtta:
        for head in (model.connect_model.cls_dw, model.connect_model.reg_dw,
                     model.connect_model.bbox_tower, model.connect_model.cls_tower):
            replace_layers(head, DTTA_LAMBDA, False)
    # core.utils.utils.to_device indexes GPUs by int and falls back to CPU on its own.
    cuda_id = 0 if torch.cuda.is_available() else "cpu"
    model = load_model(model, str(weights), map_location=cuda_id, strict=False)
    model = (model.cuda() if torch.cuda.is_available() else model).eval()
    return instantiate(cfg["tracker"], model=model, cuda_id=cuda_id)


def track_sequence(tracker, seq: Sequence) -> tuple[list[tuple[float, ...]], list[float]]:
    """One-pass track over a frame directory. Returns per-frame boxes and tracker times."""
    def read(path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cannot read frame {path}")
        # Upstream reads frames with PIL, so the tracker expects RGB, not cv2's BGR.
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    tracker.initialize(read(seq.frames[0]), np.array(seq.init_box).astype(int))
    boxes = [tuple(float(v) for v in seq.init_box)]
    times = [time.perf_counter() - t0]

    for path in seq.frames[1:]:
        frame = read(path)
        t0 = time.perf_counter()
        bbox, _score = tracker.update(frame)
        times.append(time.perf_counter() - t0)
        boxes.append(tuple(float(v) for v in bbox[:4]))
    return boxes, times


def main():
    ap = argparse.ArgumentParser(description="One-pass benchmark run for the upstream SiamABC.")
    ap.add_argument("--dataset", required=True, help="Benchmark name (see bench/datasets.py LOADERS)")
    ap.add_argument("--data-root", help="Directory holding the official benchmark")
    ap.add_argument("--out", help="Output directory for the raw prediction files")
    ap.add_argument("--variant", default="tiny", choices=sorted(VARIANTS))
    ap.add_argument("--weights", default=None, help="Checkpoint path (default: the repo's v1 asset)")
    ap.add_argument("--work-dir", default=str(_REPO_ROOT / "third_party"),
                    help="Where the upstream repo and its weights are cloned")
    ap.add_argument("--fetch-only", action="store_true", help="Clone the repo and weights, then exit")
    ap.add_argument("--no-dtta", action="store_true", help="Skip the test-time BN adaptation")
    ap.add_argument("--tracker-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a tracker config entry, e.g. --tracker-opt window_influence=0.4")
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N sequences")
    ap.add_argument("--seq", default=None, help="Only run this sequence")
    args = ap.parse_args()

    repo = clone_upstream(Path(args.work_dir).expanduser().resolve())
    if args.fetch_only:
        print(f"upstream repo and weights at {repo}")
        return
    if not args.data_root or not args.out:
        raise SystemExit("--data-root and --out are required unless --fetch-only is given")

    import torch
    import yaml

    overrides = {k: yaml.safe_load(v) for k, v in
                 (o.split("=", 1) for o in args.tracker_opt)}
    weights = Path(args.weights).expanduser().resolve() if args.weights \
        else repo / VARIANTS[args.variant][1]
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")

    sequences = load_sequences(args.dataset, args.data_root)
    if args.seq:
        sequences = [s for s in sequences if s.name == args.seq]
        if not sequences:
            raise SystemExit(f"sequence {args.seq!r} not found in {args.dataset}")
    if args.limit:
        sequences = sequences[: args.limit]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | variant: {args.variant} | weights: {weights.name} | "
          f"DTTA: {not args.no_dtta} | dataset: {args.dataset} | sequences: {len(sequences)}")
    tracker = build_tracker(repo, args.variant, weights, not args.no_dtta, overrides)

    total_frames, total_time, failed = 0, 0.0, []
    for i, seq in enumerate(sequences, 1):
        print(f"[{i}/{len(sequences)}] {seq.name} ({len(seq.frames)} frames)", flush=True)
        try:
            boxes, times = track_sequence(tracker, seq)
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
    print(f"\n{total_frames} frames | {total_time:.1f}s tracker time | {fps:.1f} FPS | "
          f"results in {out_dir}")
    if failed:
        print(f"Failed {len(failed)}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
