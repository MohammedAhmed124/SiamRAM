"""
Smoke test for the `siamese` descriptor backend.

Builds the same tracker the regression harness uses, but flips
`descriptor_backend` to "siamese" so the SiamABC encoder serves as the
appearance backbone. Runs the first N frames of the reference video and
asserts:

  - The tracker constructs and initializes without raising.
  - tracker.update() returns a sane (bbox, score, in_occlusion) tuple every
    frame.
  - The tracked bbox stays within frame bounds (i.e. the descriptor backend
    swap did not corrupt downstream gating).

This is intentionally NOT a behavior-preservation regression — switching the
descriptor backend WILL change the bbox/score numbers, just not in a way that
breaks the tracker contract. Pair this with the OSNet regression harness:
together they cover "did the abstraction land cleanly".
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the CPU shim + builder from the main harness.
from omegaconf import OmegaConf  # noqa: E402

from models.SiamABC.tracker.tracker_setup import get_tracker  # noqa: E402
from models.siamram.tracker import SiamRAMExperimentTracker  # noqa: E402
from tests.test_regression import (  # noqa: E402
    REPO_ROOT as REGRESSION_REPO_ROOT,
)
from tests.test_regression import (
    VIDEO_PATH,
    _read_initial_bbox,
    _set_deterministic_seeds,
)

NUM_FRAMES = 25


def _build_tracker_with_siamese() -> SiamRAMExperimentTracker:
    """
    Mirror tests.test_regression._build_tracker but force the siamese backend.
    """
    config = OmegaConf.load(
        str(REGRESSION_REPO_ROOT / "config" / "inference_config_experimental.yaml")
    )
    weights_path = str(
        REGRESSION_REPO_ROOT / "checkpoints" / "inference_checkpoint.pth"
    )
    config.ram_tracker.yolo_weights = str(
        REGRESSION_REPO_ROOT / "checkpoints" / "yolo11n.pt"
    )
    config.ram_tracker.descriptor_backend = "siamese"

    wrapped = get_tracker(
        config=config,
        weights_path=weights_path,
        lambda_tta=0.1,
        continuous=False,
    )

    ram_kwargs = dict(OmegaConf.to_container(config.ram_tracker, resolve=True))
    exp_cfg = OmegaConf.to_container(
        getattr(config, "ram_tracker_experiment", {}), resolve=True
    )
    if isinstance(exp_cfg, dict):
        ram_kwargs.update(exp_cfg)
    ram_kwargs["descriptor_backend"] = "siamese"

    return SiamRAMExperimentTracker(siam_tracker=wrapped, **ram_kwargs)


def main() -> int:
    if not VIDEO_PATH.exists():
        print(f"[smoke] reference video missing: {VIDEO_PATH}", file=sys.stderr)
        return 2

    _set_deterministic_seeds()
    tracker = _build_tracker_with_siamese()
    initial_bbox = _read_initial_bbox()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[smoke] cannot open video: {VIDEO_PATH}", file=sys.stderr)
        return 2

    try:
        ok, first = cap.read()
        if not ok or first is None:
            print("[smoke] cannot read first frame", file=sys.stderr)
            return 2
        tracker.initialize(first, initial_bbox)
        h_fr, w_fr = first.shape[:2]

        for i in range(NUM_FRAMES - 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bbox, score, in_occlusion, _yolo = tracker.update(frame)
            bbox = np.asarray(bbox, dtype=int)
            if bbox.shape != (4,):
                print(f"[smoke] frame {i+1}: bbox has shape {bbox.shape}", file=sys.stderr)
                return 1
            x, y, w, h = bbox.tolist()
            if w <= 0 or h <= 0:
                print(f"[smoke] frame {i+1}: degenerate bbox {bbox}", file=sys.stderr)
                return 1
            if x + w <= 0 or y + h <= 0 or x >= w_fr or y >= h_fr:
                print(
                    f"[smoke] frame {i+1}: bbox {bbox} is entirely outside frame "
                    f"{w_fr}x{h_fr}",
                    file=sys.stderr,
                )
                return 1
            if not (0.0 <= float(score) <= 1.0):
                print(
                    f"[smoke] frame {i+1}: score {score} outside [0, 1]",
                    file=sys.stderr,
                )
                return 1
    finally:
        cap.release()

    print(f"[smoke] OK: tracker ran {NUM_FRAMES} frames cleanly on the siamese backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
