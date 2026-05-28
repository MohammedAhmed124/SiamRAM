"""
Behavior-preservation regression harness for SiamRAMExperimentTracker.

Runs the tracker on a fixed reference video for a fixed number of frames and
asserts that the per-frame `(bbox, score, in_occlusion)` outputs match a
recorded golden snapshot byte-for-byte.

Usage:

    # First-time setup: record the current behavior as the golden baseline.
    python -m tests.test_regression --record

    # After a refactor: re-run and assert no drift.
    python -m tests.test_regression

The harness is intentionally minimal: one short reference video, a small
number of frames, deterministic seeds, single-process. If you change the
reference video, frame count, or config, you must re-record the golden file.

The golden file is a JSON dict written to tests/golden_*.json, holding:
- meta (config path, video path, frame count, tracker class)
- per_frame: list of {bbox: [x,y,w,h], score: float, in_occlusion: bool}

Floating-point comparisons use a generous tolerance (1e-6) on `score` and
exact equality on `bbox` and `in_occlusion`, matching the fact that the
tracker quantises its bbox output to int and its mode flag is boolean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch


def _patch_torch_for_cpu_if_needed() -> None:
    """
    The production tracker hard-codes a few `.cuda()` and `torch.zeros(
    device='cuda:0')` calls. When this test runs on a CPU-only host we
    re-route those onto CPU so the harness still executes. This is contained
    to the test process and does not touch the production code paths.
    """
    if torch.cuda.is_available():
        return

    def _module_cuda_noop(self, device=None):  # type: ignore[no-redef]
        return self

    torch.nn.Module.cuda = _module_cuda_noop  # type: ignore[assignment]

    def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=None):
        return self

    torch.Tensor.cuda = _tensor_cuda_noop  # type: ignore[assignment]

    def _looks_like_cuda_device(value) -> bool:
        # Exclude bool first: in Python `isinstance(False, int) == True`,
        # and `.to(device, dtype, non_blocking)` passes non_blocking as a
        # bool positional arg — we must not mistake that for a cuda id.
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            # PyTorch 2.11+: bare ints to .to() / device= are interpreted as
            # an accelerator ordinal, which probes torch.accelerator and
            # raises on CPU-only builds.
            return True
        if isinstance(value, torch.device):
            return value.type == "cuda"
        if isinstance(value, str):
            return value.startswith("cuda")
        return False

    _original_to = torch.Tensor.to

    def _tensor_to_cpu(self, *args, **kwargs):
        new_args = []
        for a in args:
            if _looks_like_cuda_device(a):
                new_args.append("cpu")
            else:
                new_args.append(a)
        if "device" in kwargs and _looks_like_cuda_device(kwargs["device"]):
            kwargs["device"] = "cpu"
        return _original_to(self, *new_args, **kwargs)

    torch.Tensor.to = _tensor_to_cpu  # type: ignore[assignment]

    _original_module_to = torch.nn.Module.to

    def _module_to_cpu(self, *args, **kwargs):
        new_args = []
        for a in args:
            if _looks_like_cuda_device(a):
                new_args.append("cpu")
            else:
                new_args.append(a)
        if "device" in kwargs and _looks_like_cuda_device(kwargs["device"]):
            kwargs["device"] = "cpu"
        return _original_module_to(self, *new_args, **kwargs)

    torch.nn.Module.to = _module_to_cpu  # type: ignore[assignment]

    _original_zeros = torch.zeros

    def _zeros_cpu(*args, **kwargs):
        dev = kwargs.get("device")
        if dev is not None and str(dev).startswith("cuda"):
            kwargs["device"] = "cpu"
        return _original_zeros(*args, **kwargs)

    torch.zeros = _zeros_cpu  # type: ignore[assignment]

    _original_tensor = torch.tensor

    def _tensor_cpu(*args, **kwargs):
        dev = kwargs.get("device")
        if dev is not None and str(dev).startswith("cuda"):
            kwargs["device"] = "cpu"
        return _original_tensor(*args, **kwargs)

    torch.tensor = _tensor_cpu  # type: ignore[assignment]


_patch_torch_for_cpu_if_needed()

# Make repo root importable so `models.*` resolves.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from models.SiamABC.tracker.tracker_setup import get_tracker
from models.siamram.config import flatten_subsystem_overrides
from models.siamram.tracker import SiamRAMExperimentTracker

CONFIG_PATH = REPO_ROOT / "config" / "inference_config_experimental.yaml"
VIDEO_PATH = REPO_ROOT / "data" / "dataset1" / "Car_video" / "Car_video.mp4"
ANNOTATION_PATH = REPO_ROOT / "data" / "dataset1" / "Car_video" / "annotation.txt"
GOLDEN_PATH = REPO_ROOT / "tests" / "golden_dataset1_car_video.json"

# Hold the frame count short so CPU runs finish in a reasonable time. Bumping
# this raises confidence at the cost of golden-record latency.
NUM_FRAMES = 25

SCORE_TOL = 1e-6


def _read_initial_bbox() -> np.ndarray:
    line = ANNOTATION_PATH.read_text(encoding="utf-8").splitlines()[0]
    parts = [int(p) for p in line.replace(",", " ").split()]
    if len(parts) < 4:
        raise RuntimeError(f"Expected 4 comma-separated ints, got: {line!r}")
    return np.array(parts[:4], dtype=int)


def _set_deterministic_seeds() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    cv2.setRNGSeed(0)


def _build_tracker() -> SiamRAMExperimentTracker:
    config = OmegaConf.load(str(CONFIG_PATH))
    weights_path = str(REPO_ROOT / "checkpoints" / "inference_checkpoint.pth")
    config.ram_tracker.yolo_weights = str(
        REPO_ROOT / "checkpoints" / "yolo11n.pt"
    )

    # Force the imagenet preset for OSNet so the harness does not depend on a
    # pre-staged ReID checkpoint that may have restrictive permissions on
    # disposable test hosts. Same OSNet between record and validate runs is
    # what matters for behavior preservation — the absolute weights only need
    # to be deterministic, not "production".
    config.ram_tracker.osnet_pretrained_checkpoint = "imagenet"
    config.ram_tracker.osnet_model_path = ""

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
    sub_ram, sub_exp = flatten_subsystem_overrides(config)
    ram_kwargs.update(sub_ram)
    ram_kwargs.update(sub_exp)
    ram_kwargs["osnet_pretrained_checkpoint"] = "imagenet"
    ram_kwargs["osnet_model_path"] = ""

    return SiamRAMExperimentTracker(siam_tracker=wrapped, **ram_kwargs)


def _run_tracker(num_frames: int) -> List[Dict[str, Any]]:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Reference video missing: {VIDEO_PATH}")

    _set_deterministic_seeds()
    tracker = _build_tracker()
    initial_bbox = _read_initial_bbox()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    try:
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError("Cannot read first frame")
        tracker.initialize(first, initial_bbox)

        per_frame: List[Dict[str, Any]] = [
            {
                "bbox": initial_bbox.astype(int).tolist(),
                "score": 1.0,
                "in_occlusion": False,
            }
        ]

        for _ in range(num_frames - 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bbox, score, in_occlusion, _yolo = tracker.update(frame)
            per_frame.append(
                {
                    "bbox": np.asarray(bbox, dtype=int).tolist(),
                    "score": float(score),
                    "in_occlusion": bool(in_occlusion),
                }
            )
    finally:
        cap.release()

    return per_frame


def record_golden(num_frames: int = NUM_FRAMES) -> None:
    per_frame = _run_tracker(num_frames)
    payload = {
        "meta": {
            "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "video_path": str(VIDEO_PATH.relative_to(REPO_ROOT)),
            "num_frames": len(per_frame),
            "tracker_class": "SiamRAMExperimentTracker",
        },
        "per_frame": per_frame,
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[record] wrote {len(per_frame)} frames to {GOLDEN_PATH}")


def check_against_golden() -> int:
    if not GOLDEN_PATH.exists():
        print(
            f"[check] no golden at {GOLDEN_PATH} — run with --record first.",
            file=sys.stderr,
        )
        return 2
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["per_frame"]
    actual = _run_tracker(len(expected))

    if len(expected) != len(actual):
        print(
            f"[check] FAIL: golden has {len(expected)} frames, run produced {len(actual)}",
            file=sys.stderr,
        )
        return 1

    mismatches = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e["bbox"] != a["bbox"]:
            mismatches.append(f"frame {i}: bbox {e['bbox']} != {a['bbox']}")
        if abs(float(e["score"]) - float(a["score"])) > SCORE_TOL:
            mismatches.append(
                f"frame {i}: score {e['score']:.6f} != {a['score']:.6f}"
            )
        if bool(e["in_occlusion"]) != bool(a["in_occlusion"]):
            mismatches.append(
                f"frame {i}: in_occlusion {e['in_occlusion']} != {a['in_occlusion']}"
            )

    if mismatches:
        print(f"[check] FAIL: {len(mismatches)} mismatches", file=sys.stderr)
        for m in mismatches[:10]:
            print("  " + m, file=sys.stderr)
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more", file=sys.stderr)
        return 1

    print(f"[check] OK: all {len(actual)} frames match golden")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--record",
        action="store_true",
        help="Re-record the golden snapshot from the current code.",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=NUM_FRAMES,
        help="Number of frames to run (only honoured with --record).",
    )
    args = ap.parse_args()

    if args.record:
        record_golden(args.frames)
        return 0
    return check_against_golden()


if __name__ == "__main__":
    sys.exit(main())
