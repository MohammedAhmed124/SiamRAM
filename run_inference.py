"""
Run SiamRAM inference on a competition dataset.

Expected data layout (AIC-4 competition structure):

    dataset/
    ├── metadata/
    │   └── contestant_manifest.json
    └── <dataset_name>/
        ├── <video_id>/
        │   └── video.mp4          # (or frames — see manifest video_path)
        └── annotations/
            └── <video_id>.txt     # single-line: x,y,w,h init bbox

All default paths are resolved relative to the repository root (the directory
that contains this script), so the script works regardless of the working
directory from which it is invoked.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from models.SiamABC.tracker.tracker_setup import get_tracker
from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
from models.SiamRAM import SiamRAMTracker
from vis.test_model import run_inference

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*tensorrt.plugin module is experimental.*")

os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DOWNLOADER = CHECKPOINTS_DIR / "download_checkpoints.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SiamRAM inference on a competition manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=str(BASE_DIR / "data"),
        help="Root directory containing video files and annotation sub-folders.",
    )
    parser.add_argument(
        "--outputs_dir",
        default=str(BASE_DIR / "outputs" / "SiamRAM"),
        help="Directory where per-video bounding-box predictions are written.",
    )
    parser.add_argument(
        "--manifest_path",
        default=str(BASE_DIR / "data" / "metadata" / "contestant_manifest.json"),
        help="Path to the competition manifest JSON file.",
    )
    parser.add_argument(
        "--weights_path",
        default=str(BASE_DIR / "checkpoints" / "head_epoch_000.pth"),
        help="Path to the SiamABC checkpoint (.pth file).",
    )
    parser.add_argument(
        "--yaml_config_path",
        default=str(BASE_DIR / "config" / "inference_config.yaml"),
        help="Path to the inference config YAML file.",
    )
    parser.add_argument(
        "--model_size",
        default="M",
        choices=["S", "M", "L"],
        help="SiamABC model size.",
    )
    parser.add_argument(
        "--lambda_tta",
        type=float,
        default=0.1,
        help="TTA lambda for the base tracker.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset names to include from the manifest public_lb split. "
            "Defaults to all sub-directories found inside --data_dir (excluding 'metadata')."
        ),
    )
    parser.add_argument(
        "--submission_csv",
        default=str(BASE_DIR / "submission.csv"),
        help="Output path for the submission CSV file.",
    )
    return parser.parse_args()


def _resolve_weights_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        if not path.exists():
            checkpoint_candidate = CHECKPOINTS_DIR / path.name
            if checkpoint_candidate.exists():
                return checkpoint_candidate.resolve()
        return path
    if path.exists():
        return path.resolve()
    if len(path.parts) == 1:
        return (CHECKPOINTS_DIR / path.name).resolve()
    return (BASE_DIR / path).resolve()


def _is_valid_checkpoint_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False
    with path.open("rb") as f:
        head = f.read(512).lstrip().lower()
    return not (head.startswith(b"<!doctype html") or head.startswith(b"<html"))


def _resolve_data_dir(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidate = (BASE_DIR / path).resolve()
    if candidate.exists():
        return candidate
    if path.name == "dataset":
        alt = (BASE_DIR / "data").resolve()
        if alt.exists():
            return alt
    if path.name == "data":
        alt = (BASE_DIR / "dataset").resolve()
        if alt.exists():
            return alt
    return candidate


def _resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((BASE_DIR / path).resolve())
    candidates.append((BASE_DIR / "data" / "metadata" / "contestant_manifest.json").resolve())
    candidates.append((BASE_DIR / "dataset" / "metadata" / "contestant_manifest.json").resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_required_checkpoints(weights_path: Path, yolo_path: Path) -> tuple[Path, Path]:
    invalid_or_missing = [
        p for p in (weights_path, yolo_path) if not _is_valid_checkpoint_file(p)
    ]
    if not invalid_or_missing:
        return weights_path, yolo_path

    if not CHECKPOINT_DOWNLOADER.exists():
        missing_str = ", ".join(str(p) for p in invalid_or_missing)
        raise FileNotFoundError(
            f"Missing checkpoints: {missing_str}. Also missing downloader script: {CHECKPOINT_DOWNLOADER}"
        )

    print("Missing or invalid checkpoints detected. Downloading required models...")
    subprocess.run(
        [sys.executable, str(CHECKPOINT_DOWNLOADER), "--force"],
        check=True,
        cwd=str(BASE_DIR),
    )

    resolved_weights = _resolve_weights_path(str(weights_path))
    resolved_yolo = _resolve_weights_path(str(yolo_path))
    still_missing = [
        p for p in (resolved_weights, resolved_yolo) if not _is_valid_checkpoint_file(p)
    ]
    if still_missing:
        missing_str = ", ".join(str(p) for p in still_missing)
        raise FileNotFoundError(f"Checkpoint download finished, but files are still missing: {missing_str}")

    return resolved_weights, resolved_yolo


def main():
    args = parse_args()
    args.data_dir = str(_resolve_data_dir(args.data_dir))
    args.manifest_path = str(_resolve_manifest_path(args.manifest_path))

    config = OmegaConf.load(args.yaml_config_path)
    resolved_weights_path = _resolve_weights_path(args.weights_path)
    resolved_yolo_path = _resolve_weights_path(str(config.ram_tracker.yolo_weights))
    resolved_weights_path, resolved_yolo_path = _ensure_required_checkpoints(
        resolved_weights_path, resolved_yolo_path
    )

    args.weights_path = str(resolved_weights_path)
    config.ram_tracker.yolo_weights = str(resolved_yolo_path)
    config.model.model_size = args.model_size
    if config.make_trt_engine:
        wrapped = get_trt_tracker(
            config=config, weights_path=args.weights_path, **config.trt_engine
        )
    else:
        wrapped = get_tracker(
            config=config,
            weights_path=args.weights_path,
            lambda_tta=args.lambda_tta,
            continuous=False,
        )

    tracker = SiamRAMTracker(siam_tracker=wrapped, **config.ram_tracker)

    with open(args.manifest_path, "r") as f:
        manifest = json.load(f)

    if args.datasets is not None:
        target_datasets = set(args.datasets)
    else:
        data_root = Path(args.data_dir)
        target_datasets = {
            p.name for p in data_root.iterdir() if p.is_dir() and p.name != "metadata"
        }
        print(f"Auto-discovered datasets: {sorted(target_datasets)}")

    test_public_lb = {
        k: v
        for k, v in manifest["public_lb"].items()
        if v["dataset"] in target_datasets
    }

    print(f"Starting inference on {len(test_public_lb)} videos...")
    for i, (key, value) in enumerate(test_public_lb.items()):
        video_path = os.path.join(args.data_dir, value["video_path"])
        ann_path = os.path.join(args.data_dir, value["annotation_path"])
        output_path = os.path.join(args.outputs_dir, value["video_path"])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"[{i + 1}/{len(test_public_lb)}] Processing: {value['video_path']}")

        init_bbox = np.loadtxt(ann_path, delimiter=",", dtype=np.float32).tolist()
        if isinstance(init_bbox[0], list):
            init_bbox = init_bbox[0]

        run_inference(
            video_path=video_path,
            initial_bbox=init_bbox,
            tracker=tracker,
            output_path=output_path,
            output_video=False,
        )

    print("\nCompiling submission CSV...")
    submission_df = defaultdict(list)

    for key, value in test_public_lb.items():
        head, tail = os.path.split(os.path.join(args.outputs_dir, value["video_path"]))
        bbox_file = os.path.join(head, "bboxes", os.path.splitext(tail)[0] + ".txt")

        if os.path.exists(bbox_file):
            with open(bbox_file, "r") as f:
                lines = f.read().strip().split("\n")
                for frame_idx, line in enumerate(lines):
                    x, y, w, h = line.strip().split()
                    submission_df["id"].append(f"{key}_{frame_idx}")
                    submission_df["x"].append(float(x))
                    submission_df["y"].append(float(y))
                    submission_df["w"].append(float(w))
                    submission_df["h"].append(float(h))
        else:
            print(f"Warning: bbox file missing for {key}")

    submission = pd.DataFrame(submission_df)
    submission.to_csv(args.submission_csv, index=False)

    print(f"\nSubmission written to {args.submission_csv}")
    print(submission.head())
    print(f"Total rows: {len(submission)}")


if __name__ == "__main__":
    main()
