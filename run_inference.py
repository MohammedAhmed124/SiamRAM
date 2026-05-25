"""
SiamRAM Inference Script — Competition Submission Pipeline
==========================================================

What does this script do?
--------------------------
This script runs the full SiamRAM object-tracking pipeline on a competition
video dataset and produces a submission CSV file that you can upload directly.

Here's the big picture of what happens, in plain English:

  1. You point it at a folder of competition videos and a manifest JSON that
     tells it which videos exist and where their annotation files live.

  2. For each video, the script reads the *initial bounding box* from the
     annotation file — that's the one box that tells the tracker "this is
     the object you need to follow, starting here".

  3. It runs SiamRAM frame-by-frame on every video, predicting where the
     target object is in each frame, and saves those bounding-box predictions
     to disk.

  4. Finally, it gathers all those per-video prediction files and stitches
     them together into a single submission.csv that the competition platform
     expects.

What is SiamRAM?
----------------
SiamRAM is a Siamese-network-based visual object tracker. It wraps a core
SiamABC tracker (which does the heavy Siamese matching) with a RAM (Re-detection
and Attention Module) layer that uses YOLO to recover the target if it gets
lost. Think of SiamABC as the fast, accurate follower and the RAM layer as the
safety net that kicks in when things go wrong.

Expected data layout on disk:
------------------------------
    data/
    ├── metadata/
    │   └── contestant_manifest.json   ← the index of all competition videos
    └── <dataset_name>/
        ├── <video_id>/
        │   └── video.mp4              ← the actual video (or frame images)
        └── annotations/
            └── <video_id>.txt         ← one line: x,y,w,h of the initial bbox

All paths default to locations relative to this script's own directory, so you
can run it from anywhere without worrying about your current working directory.

Typical usage:
--------------
    python run_inference.py                         # use all defaults
    python run_inference.py --model_size L          # use the large model
    python run_inference.py --datasets dataset_A    # only process one dataset
"""

import argparse
import csv
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
from models.SiamRAM import SiamRAMTracker
from models.SiamRAM_experiment import SiamRAMExperimentTracker
from vis.test_model import run_inference

try:
    from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
    _trt_import_error: Exception | None = None
except ModuleNotFoundError as exc:
    get_trt_tracker = None  # type: ignore[assignment]
    _trt_import_error = exc

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings that clutter the terminal output.
# These are all harmless deprecation notices from TensorRT, PyTorch, and
# OmegaConf — they don't affect results, they just make the output messy.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Anchor paths — everything is resolved relative to the directory where this
# script lives, so the script is portable regardless of where you run it from.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DOWNLOADER = CHECKPOINTS_DIR / "download_checkpoints.py"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """
    Define and parse all command-line arguments for the inference run.

    Most arguments have sensible defaults so you can just run the script
    with no flags at all. The key things you might want to override:

      --data_dir        : where your competition videos live
      --outputs_dir     : where per-video bbox predictions will be written
      --manifest_path   : the competition's JSON index of videos
      --weights_path    : the trained SiamABC checkpoint (.pth file)
      --yaml_config_path: the inference config (model params, YOLO path, etc.)
      --model_size      : S / M / L — larger = more accurate but slower
      --lambda_tta      : test-time augmentation strength for the base tracker
      --datasets        : optionally restrict to specific dataset sub-folders
      --video_key       : run only one manifest/sequence entry by key
      --submission_csv  : where to write the final submission CSV
      --output_video    : if set, writes annotated debug videos to outputs_dir

    Returns
    -------
    argparse.Namespace
        Parsed arguments with all defaults filled in.
    """
    parser = argparse.ArgumentParser(
        description="Run SiamRAM inference on a competition manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "test", "all"],
        help=(
            "Convenience split selector. "
            "'test' maps to manifest public_lb, "
            "'train' maps to manifest train, "
            "'all' runs both train and test in one pass."
        ),
    )
    parser.add_argument(
        "--run_split",
        default="public_lb",
        choices=["public_lb", "train", "train_csv", "all"],
        help=(
            "Which split to run. "
            "'public_lb' reads manifest public_lb entries (test), "
            "'train' reads manifest train entries, "
            "'all' runs both manifest train + public_lb, "
            "'train_csv' reads --train_csv_path sequences."
        ),
    )
    parser.add_argument(
        "--train_csv_path",
        default=str(BASE_DIR / "data" / "train_dataframe.csv"),
        help=(
            "Training dataframe CSV path used when --run_split train_csv. "
            "Expected columns include img_path/seq_path and annot_path."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default=str(BASE_DIR / "data"),
        help="Root directory containing video files and annotation sub-folders.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Optional cap on number of sequences to process (0 means no cap).",
    )
    parser.add_argument(
        "--id_prefix",
        default="pred",
        help="Prefix for generated frame IDs in the output CSV (train split).",
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
        default=str(BASE_DIR / "checkpoints" / "inference_checkpoint.pth"),
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
        "--video_key",
        default=None,
        help=(
            "Optional single-sequence selector. "
            "Matches manifest key/source_key (for example: dataset1/volleyball). "
            "When used with --run_split all, the matching entry is selected from the merged map."
        ),
    )
    parser.add_argument(
        "--submission_csv",
        default=str(BASE_DIR / "submission.csv"),
        help="Output path for the submission CSV file.",
    )
    parser.add_argument(
        "--output_video",
        "--output_videos",
        action="store_true",
        help=(
            "Write annotated debug videos (slower). "
            "When not set, runs in fast-inference mode and writes bbox files only."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_weights_path(path_value: str) -> Path:
    """
    Figure out where the model checkpoint actually lives.

    This handles a few common scenarios so you don't have to be super precise
    with your paths:

      - If you give an absolute path and it exists → use it as-is.
      - If you give an absolute path but it's missing → check whether a file
        with the same name is sitting in the default checkpoints/ folder.
      - If you give just a filename (no directory part) → look for it in
        checkpoints/ automatically.
      - Otherwise, treat the path as relative to BASE_DIR.

    Parameters
    ----------
    path_value : str
        Raw path string from the command line or config.

    Returns
    -------
    Path
        Resolved (possibly remapped) absolute path to the checkpoint file.
    """
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
        # Bare filename like "inference_checkpoint.pth" → look in checkpoints/
        return (CHECKPOINTS_DIR / path.name).resolve()
    return (BASE_DIR / path).resolve()


def _is_valid_checkpoint_file(path: Path) -> bool:
    """
    Check whether a checkpoint file actually looks like a real model weight file.

    Two things can go wrong:
      1. The file doesn't exist at all, or is suspiciously tiny (< 1 MB).
      2. The file is an HTML error page — this happens when a download URL
         redirected to a 404 page and the "file" you have is actually HTML.

    We do a quick binary sniff of the first 512 bytes to catch case 2.

    Parameters
    ----------
    path : Path
        Path to the file to check.

    Returns
    -------
    bool
        True if the file looks like a genuine checkpoint, False otherwise.
    """
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False
    with path.open("rb") as f:
        head = f.read(512).lstrip().lower()
    return not (head.startswith(b"<!doctype html") or head.startswith(b"<html"))


def _resolve_data_dir(path_value: str) -> Path:
    """
    Find the root data directory even if the caller used a slightly different name.

    Competition setups sometimes call the data folder "data/" and sometimes
    "dataset/". This function tries the given path first, then falls back to
    the alternate name if the given one doesn't exist.

    Parameters
    ----------
    path_value : str
        Raw path string (absolute or relative) provided by the user.

    Returns
    -------
    Path
        Resolved path to the data root directory.
    """
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
    """
    Locate the competition manifest JSON, trying a handful of common locations.

    The manifest is the JSON file that lists every video in the competition,
    its path, its annotation path, and which split (public_lb, private_lb, etc.)
    it belongs to. We try the user-specified path first, then fall back to
    the two standard locations under data/ and dataset/.

    Parameters
    ----------
    path_value : str
        Raw path string provided by the user.

    Returns
    -------
    Path
        First path in the candidate list that actually exists on disk,
        or the first candidate if none exist (so the error message is clear).
    """
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


def _resolve_data_asset_path(
    path_value: str,
    data_dir: str,
) -> str:
    """
    Resolve sequence/annotation paths from either manifest-relative or CSV rows.

    Supports stale Docker-prefixed paths such as /app/... by remapping to BASE_DIR.
    """
    raw = str(path_value or "").strip()
    if not raw:
        return raw

    path = Path(raw).expanduser()
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((Path(data_dir) / path).resolve())
        candidates.append((BASE_DIR / path).resolve())

    norm = raw.replace("\\", "/")
    if norm.startswith("/app/"):
        candidates.append((BASE_DIR / norm[len("/app/"):]).resolve())
    elif "/app/" in norm:
        candidates.append((BASE_DIR / norm.split("/app/", 1)[1]).resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0]) if candidates else raw


def _load_initial_bbox(
    annotation_path: str,
) -> list[float]:
    """
    Load initial bbox from the first non-empty line of annotation txt.
    Accepts comma or whitespace separators.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid bbox line in {annotation_path}: {line!r}"
                )
            return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
    raise ValueError(f"Annotation file is empty: {annotation_path}")


def _build_manifest_entries(
    manifest: dict,
    target_datasets: set[str],
    split_name: str,
    key_prefix: str = "",
    output_prefix: str = "",
) -> dict[str, dict]:
    """
    Build a uniform entry dictionary from a manifest split.
    """
    split_entries = manifest.get(split_name, {})
    entries: dict[str, dict] = {}
    for key, value in split_entries.items():
        if value["dataset"] not in target_datasets:
            continue
        out_rel_path = value["video_path"]
        if output_prefix:
            out_rel_path = os.path.join(output_prefix, out_rel_path)
        out_key = f"{key_prefix}{key}"
        entries[out_key] = {
            "dataset": value["dataset"],
            "video_path": value["video_path"],
            "annotation_path": value["annotation_path"],
            "output_rel_path": out_rel_path,
            "split": split_name,
            "source_key": key,
        }
    return entries


def _build_train_entries_from_csv(
    train_csv_path: str,
    data_dir: str,
    target_datasets: set[str],
    max_sequences: int = 0,
) -> dict[str, dict]:
    """
    Build inference entries from training dataframe CSV rows.
    """
    entries: dict[str, dict] = {}
    dropped_missing_path = 0
    with open(train_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            dataset_name = str(row.get("dataset", "")).strip() or "unknown"
            if dataset_name not in target_datasets:
                continue

            seq_raw = row.get("img_path") or row.get("seq_path") or ""
            ann_raw = row.get("annot_path") or ""
            seq_path = _resolve_data_asset_path(seq_raw, data_dir)
            ann_path = _resolve_data_asset_path(ann_raw, data_dir)

            if not os.path.exists(seq_path) or not os.path.exists(ann_path):
                dropped_missing_path += 1
                continue

            seq_leaf = Path(seq_path).name
            seq_name = Path(seq_path).parent.name if seq_leaf == "img" else seq_leaf
            base_key = f"train_{dataset_name}__{seq_name}"
            key = base_key
            suffix = 1
            while key in entries:
                suffix += 1
                key = f"{base_key}_{suffix:03d}"

            output_rel_path = os.path.join(dataset_name, f"{key}.mp4")
            entries[key] = {
                "dataset": dataset_name,
                "video_path": seq_path,
                "annotation_path": ann_path,
                "output_rel_path": output_rel_path,
                "row_idx": row_idx,
            }
            if max_sequences > 0 and len(entries) >= max_sequences:
                break

    if dropped_missing_path > 0:
        print(
            f"[train-csv] Skipped {dropped_missing_path} rows with missing seq/annotation paths."
        )
    return entries


def _filter_run_entries_by_video_key(
    run_entries: dict[str, dict],
    video_key: str | None,
) -> dict[str, dict]:
    """
    Filter run entries down to one selected sequence key.

    Matching order:
      1) exact key / source_key / key without train|test prefix
      2) case-insensitive substring on the same fields
    """
    if not video_key:
        return run_entries

    needle = str(video_key).strip()
    if not needle:
        return run_entries
    needle_l = needle.lower()

    def _norm_key(
        key: str,
    ) -> str:
        if key.startswith("train/"):
            return key[len("train/"):]
        if key.startswith("test/"):
            return key[len("test/"):]
        return key

    exact: dict[str, dict] = {}
    fuzzy: dict[str, dict] = {}
    for key, value in run_entries.items():
        source_key = str(value.get("source_key", key))
        key_norm = _norm_key(key)
        fields = [
            key,
            source_key,
            key_norm,
            str(value.get("video_path", "")),
            str(value.get("annotation_path", "")),
        ]
        fields_l = [f.lower() for f in fields]

        if needle in fields or needle_l in fields_l:
            exact[key] = value
            continue

        if any(needle_l in f for f in fields_l):
            fuzzy[key] = value

    chosen = exact if exact else fuzzy
    if not chosen:
        sample = list(run_entries.keys())[:20]
        raise KeyError(
            f"--video_key '{needle}' did not match any entry. "
            f"Sample available keys: {sample}"
        )

    if len(chosen) > 1:
        raise KeyError(
            f"--video_key '{needle}' is ambiguous. Matched keys: {list(chosen.keys())}"
        )
    return chosen


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _ensure_required_checkpoints(weights_path: Path, yolo_path: Path) -> tuple[Path, Path]:
    """
    Make sure both required checkpoint files are present and valid before we
    start any inference. If either is missing or corrupt, we trigger an
    automatic download.

    The two checkpoints we need are:
      - weights_path : the SiamABC tracker head (the Siamese network weights)
      - yolo_path    : the YOLO detector used by the RAM re-detection module

    If the download script itself is missing, we raise immediately with a clear
    error rather than letting things fail mysteriously later.

    Parameters
    ----------
    weights_path : Path
        Expected location of the SiamABC checkpoint.
    yolo_path : Path
        Expected location of the YOLO checkpoint.

    Returns
    -------
    tuple[Path, Path]
        Re-resolved (weights_path, yolo_path) after any download that occurred.
        The caller should use these returned paths because the download may have
        placed files in slightly different locations.

    Raises
    ------
    FileNotFoundError
        If the downloader script is missing, or if files are still missing
        after the download completes (which would indicate a download failure).
    """
    invalid_or_missing = [
        p for p in (weights_path, yolo_path) if not _is_valid_checkpoint_file(p)
    ]
    if not invalid_or_missing:
        # Both files look good — nothing to do.
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

    # Re-resolve after download in case paths shifted slightly.
    resolved_weights = _resolve_weights_path(str(weights_path))
    resolved_yolo = _resolve_weights_path(str(yolo_path))
    still_missing = [
        p for p in (resolved_weights, resolved_yolo) if not _is_valid_checkpoint_file(p)
    ]
    if still_missing:
        missing_str = ", ".join(str(p) for p in still_missing)
        raise FileNotFoundError(f"Checkpoint download finished, but files are still missing: {missing_str}")

    return resolved_weights, resolved_yolo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrate the full end-to-end inference pipeline. Here's exactly what
    happens, step by step:

    Step 1 — Parse arguments and resolve all paths
    -----------------------------------------------
    We read the CLI flags (or use defaults) and normalise every path so that
    relative paths, typos like "dataset" vs "data", and bare filenames all
    just work without the user having to be exact.

    Step 2 — Load config and ensure checkpoints exist
    --------------------------------------------------
    We load the YAML inference config (model hyperparams, YOLO weights path,
    TRT engine settings, etc.) using OmegaConf. Then we verify that both the
    SiamABC checkpoint and the YOLO checkpoint are present and valid. If either
    is missing or looks like a corrupt HTML file, we automatically invoke the
    download script before continuing.

    Step 3 — Build the tracker
    --------------------------
    We construct the tracker in one of two modes:

      a) TensorRT mode (config.make_trt_engine = True):
         Compiles the model to a TRT engine for maximum GPU throughput.
         This is slower to initialise but much faster per frame — ideal for
         large datasets or tight time budgets.

      b) Standard PyTorch mode (config.make_trt_engine = False):
         Loads the model normally with optional test-time augmentation (TTA)
         controlled by lambda_tta. Easier to set up and still fast.

    The raw SiamABC tracker is then wrapped inside SiamRAMTracker, which adds
    the YOLO-based re-detection logic. If SiamABC ever loses the target (e.g.
    due to heavy occlusion or the object leaving the frame), the RAM layer
    fires up YOLO to scan the whole frame and hand back a fresh crop for
    SiamABC to re-lock onto.

    Step 4 — Load the manifest and filter to target datasets
    ---------------------------------------------------------
    The manifest JSON has a "public_lb" section listing every competition video
    with its video path and annotation path. We filter this down to only the
    datasets the user cares about (either from --datasets or auto-discovered
    from the data directory).

    Step 5 — Run inference video by video
    --------------------------------------
    For each video in the filtered list we:
      a) Read the initial bounding box from the annotation .txt file.
         The format is a single line: x, y, width, height (top-left corner + size).
      b) Call run_inference(), which opens the video, initialises the tracker
         on frame 0 using the given bbox, then tracks the target through every
         subsequent frame, writing predictions to a bboxes/ sub-folder next to
         the video output path.

    Step 6 — Compile the submission CSV
    ------------------------------------
    After all videos are processed, we walk through every video's bboxes .txt
    file and flatten the per-frame predictions into a single DataFrame with
    columns: id, x, y, w, h. The "id" column is "<video_key>_<frame_index>",
    which is the format the competition scorer expects. We then save this as
    submission.csv and print a preview.
    """

    # ------------------------------------------------------------------
    # Parse and normalise all input paths
    # ------------------------------------------------------------------
    args = parse_args()
    if args.split is not None:
        args.run_split = "public_lb" if args.split == "test" else args.split

    args.data_dir = str(_resolve_data_dir(args.data_dir))
    if args.run_split in {"public_lb", "train", "all"}:
        args.manifest_path = str(_resolve_manifest_path(args.manifest_path))
    else:
        args.train_csv_path = _resolve_data_asset_path(
            args.train_csv_path, args.data_dir
        )

    # ------------------------------------------------------------------
    # Load YAML config, resolve checkpoint paths, auto-download
    #         if anything is missing or corrupted.
    # ------------------------------------------------------------------
    config = OmegaConf.load(args.yaml_config_path)
    resolved_weights_path = _resolve_weights_path(args.weights_path)
    resolved_yolo_path = _resolve_weights_path(str(config.ram_tracker.yolo_weights))
    resolved_weights_path, resolved_yolo_path = _ensure_required_checkpoints(
        resolved_weights_path, resolved_yolo_path
    )

    # Write the resolved paths back so the rest of the code uses them.
    args.weights_path = str(resolved_weights_path)
    config.ram_tracker.yolo_weights = str(resolved_yolo_path)
    osnet_model_path_raw = str(getattr(config.ram_tracker, "osnet_model_path", "")).strip()
    if osnet_model_path_raw:
        config.ram_tracker.osnet_model_path = str(
            _resolve_weights_path(osnet_model_path_raw)
        )
    config.model.model_size = args.model_size

    # ------------------------------------------------------------------
    # Build the tracker.
    #
    # Two paths:
    #   - TRT (TensorRT): faster inference, higher setup cost, GPU-only.
    #   - Standard PyTorch: flexible, works on CPU too, supports TTA.
    #
    # Either way, we wrap the result in SiamRAMTracker to get the
    # YOLO-powered re-detection safety net on top.
    # ------------------------------------------------------------------
    if config.make_trt_engine:
        if get_trt_tracker is None:
            raise ModuleNotFoundError(
                "TensorRT tracker dependencies are missing. "
                "Set make_trt_engine: False in your config or install torch_tensorrt."
            ) from _trt_import_error
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

    ram_impl = str(getattr(config, "ram_tracker_impl", "base")).strip().lower()
    if ram_impl in {"base", "siamram"}:
        tracker_cls = SiamRAMTracker
    elif ram_impl in {"experiment", "exp", "siamram_experiment"}:
        tracker_cls = SiamRAMExperimentTracker
    else:
        raise ValueError(
            f"Unsupported ram_tracker_impl='{ram_impl}'. "
            "Expected one of: base, experiment."
        )

    ram_tracker_kwargs_obj = OmegaConf.to_container(config.ram_tracker, resolve=True)
    assert isinstance(ram_tracker_kwargs_obj, dict)
    ram_tracker_kwargs = dict(ram_tracker_kwargs_obj)

    if tracker_cls is SiamRAMExperimentTracker:
        exp_cfg_obj = OmegaConf.to_container(
            getattr(config, "ram_tracker_experiment", {}), resolve=True
        )
        if isinstance(exp_cfg_obj, dict):
            ram_tracker_kwargs.update(exp_cfg_obj)

    tracker = tracker_cls(siam_tracker=wrapped, **ram_tracker_kwargs)
    print(f"Using tracker implementation: {tracker_cls.__name__}")

    if args.datasets is not None:
        # The user explicitly named which datasets they want.
        target_datasets = set(args.datasets)
    else:
        # Auto-discover: every sub-folder in data_dir that isn't "metadata".
        data_root = Path(args.data_dir)
        target_datasets = {
            p.name for p in data_root.iterdir() if p.is_dir() and p.name != "metadata"
        }
        print(f"Auto-discovered datasets: {sorted(target_datasets)}")

    # Build a unified entry map regardless of source split.
    if args.run_split in {"public_lb", "train"}:
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        run_entries = _build_manifest_entries(
            manifest=manifest,
            target_datasets=target_datasets,
            split_name=args.run_split,
        )
    elif args.run_split == "all":
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        run_entries = {}
        run_entries.update(
            _build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="train",
                key_prefix="train/",
                output_prefix="train",
            )
        )
        run_entries.update(
            _build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="public_lb",
                key_prefix="test/",
                output_prefix="test",
            )
        )
    else:  # train_csv
        if not os.path.exists(args.train_csv_path):
            raise FileNotFoundError(
                f"Training CSV not found: {args.train_csv_path}"
            )
        run_entries = _build_train_entries_from_csv(
            train_csv_path=args.train_csv_path,
            data_dir=args.data_dir,
            target_datasets=target_datasets,
            max_sequences=int(args.max_sequences),
        )

    run_entries = _filter_run_entries_by_video_key(
        run_entries=run_entries,
        video_key=args.video_key,
    )

    if args.max_sequences > 0 and len(run_entries) > args.max_sequences:
        run_entries = dict(list(run_entries.items())[: args.max_sequences])

    if not run_entries:
        raise RuntimeError(
            f"No entries selected for split={args.run_split} and datasets={sorted(target_datasets)}"
        )

    # ------------------------------------------------------------------
    # Run frame-by-frame inference on every video.
    #
    # For each video we:
    #   a) Build the full path to the video file and its annotation.
    #   b) Create the output directory if it doesn't exist yet.
    #   c) Load the initial bounding box (x, y, w, h) from the annotation.
    #   d) Hand everything off to run_inference(), which handles the actual
    #      tracking loop and writes per-frame bbox predictions to disk.
    # ------------------------------------------------------------------
    print(f"Starting inference on {len(run_entries)} entries (split={args.run_split})...")
    for i, (key, value) in enumerate(run_entries.items()):
        video_path = _resolve_data_asset_path(value["video_path"], args.data_dir)
        ann_path = _resolve_data_asset_path(value["annotation_path"], args.data_dir)
        output_path = os.path.join(args.outputs_dir, value["output_rel_path"])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"[{i + 1}/{len(run_entries)}] Processing: {video_path}")

        # Load the initial bbox. The annotation file is a single line like:
        #   320.5,140.0,64.0,80.0   (x, y, width, height in pixels)
        # If the file has multiple rows for some reason, we take the first.
        init_bbox = _load_initial_bbox(ann_path)

        run_inference(
            video_path=video_path,
            initial_bbox=init_bbox,
            tracker=tracker,
            output_path=output_path,
            output_video=args.output_video,
        )

    # ------------------------------------------------------------------
    #  Stitch all per-video bbox files into one submission CSV.
    #
    # run_inference() writes a text file at:
    #   <outputs_dir>/<video_path_stem>/bboxes/<video_id>.txt
    #
    # Each line in that file is one frame's prediction: "x y w h".
    # We assign each line an id of "<video_key>_<frame_index>" and collect
    # everything into a flat DataFrame that the competition scorer expects.
    # ------------------------------------------------------------------
    print("\nCompiling submission CSV...")
    submission_df = defaultdict(list)

    for key, value in run_entries.items():
        head, tail = os.path.split(os.path.join(args.outputs_dir, value["output_rel_path"]))
        bbox_file = os.path.join(head, "bboxes", os.path.splitext(tail)[0] + ".txt")

        if os.path.exists(bbox_file):
            with open(bbox_file, "r") as f:
                lines = f.read().strip().split("\n")
                for frame_idx, line in enumerate(lines):
                    x, y, w, h = line.strip().split()
                    split_name = str(value.get("split", args.run_split))
                    if args.run_split == "train_csv" or (
                        args.run_split in {"train", "all"} and split_name == "train"
                    ):
                        sample_id = f"{args.id_prefix}_{key}_{frame_idx}"
                    else:
                        sample_id = f"{key}_{frame_idx}"
                    submission_df["id"].append(sample_id)
                    submission_df["x"].append(float(x))
                    submission_df["y"].append(float(y))
                    submission_df["w"].append(float(w))
                    submission_df["h"].append(float(h))
        else:
            # This shouldn't happen if inference completed successfully,
            # but we warn instead of crashing so one bad video doesn't
            # destroy the whole submission.
            print(f"Warning: bbox file missing for {key}")

    submission = pd.DataFrame(submission_df)
    submission.to_csv(args.submission_csv, index=False)

    print(f"\nSubmission written to {args.submission_csv}")
    print(submission.head())
    print(f"Total rows: {len(submission)}")


if __name__ == "__main__":
    main()
