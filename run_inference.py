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
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from models.SiamABC.tracker.tracker_setup import get_tracker
from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
from models.SiamRAM import SiamRAMTracker
from vis.test_model import run_inference

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SiamRAM inference on a competition manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=str(BASE_DIR / "dataset"),
        help="Root directory containing video files and annotation sub-folders.",
    )
    parser.add_argument(
        "--outputs_dir",
        default=str(BASE_DIR / "outputs" / "SiamRAM"),
        help="Directory where per-video bounding-box predictions are written.",
    )
    parser.add_argument(
        "--manifest_path",
        default=str(BASE_DIR / "dataset" / "metadata" / "contestant_manifest.json"),
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


def main():
    args = parse_args()

    config = OmegaConf.load(args.yaml_config_path)
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
