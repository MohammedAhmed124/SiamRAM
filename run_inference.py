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
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from utils.console import (
    compile_progress,
    quiet_external_logs,
    siamram_log,
    silence_noisy_libraries,
)
from utils.utils import configure_descriptor_backend

silence_noisy_libraries()

import pandas as pd
from omegaconf import OmegaConf

from models.SiamABC.tracker.tracker_setup import get_tracker
from models.siamram.config import (
    OSNET_CHECKPOINT_CHOICES,
    flatten_ram_tracker_config,
)
from models.siamram.tracker import SiamRAMExperimentTracker
from vis.test_model import run_inference

try:
    with quiet_external_logs():
        from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
    _trt_import_error: Exception | None = None
except (ModuleNotFoundError, ImportError, AssertionError, RuntimeError) as exc:
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

if os.environ.get("TORCH_LOGS", None) == "":
    os.environ.pop("TORCH_LOGS", None)
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

class _SiamRAMArgumentFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if action.dest == "rebuild_trt_cache":
            return action.help
        return super()._get_help_string(action)


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
      --override_csv    : rebuild/overlay CSV rows from saved bbox outputs
      --output_video    : if set, writes annotated debug videos to outputs_dir

    Returns
    -------
    argparse.Namespace
        Parsed arguments with all defaults filled in.
    """
    parser = argparse.ArgumentParser(
        description="Run SiamRAM inference on a competition manifest.",
        formatter_class=_SiamRAMArgumentFormatter,
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
        "--output_layout",
        default="dataset",
        choices=["dataset", "video"],
        help=(
            "On-disk output folder layout under --outputs_dir. "
            "'dataset' (default): <dataset>/<video_name>/ — consistent across all "
            "datasets, with the video + bbox file together in each video folder. "
            "'video': <video_name>/ — drops the dataset folder (used by "
            "run_single_video for a flat, easy-to-reach single-video output)."
        ),
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
        default=str(BASE_DIR / "config" / "inference_config_experimental.yaml"),
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
        "--override_csv",
        "--override-csv",
        action="store_true",
        help=(
            "After this run, rebuild --submission_csv from all bbox .txt outputs "
            "found for the selected split under --outputs_dir. Useful for rerunning "
            "one video while keeping a full submission CSV built from saved outputs."
        ),
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
    parser.add_argument(
        "--reuse_tracker",
        action="store_true",
        help=(
            "Reuse one SiamRAM wrapper across all selected videos. Faster, but "
            "sequence-local state bugs can make a clip depend on previous clips. "
            "By default each video gets a fresh wrapper around the loaded SiamABC core."
        ),
    )
    parser.add_argument(
        "--trt_cache_dir",
        default=str(BASE_DIR / "checkpoints" / "trt_engines"),
        help="Directory for serialized TensorRT engine caches.",
    )
    trt_cache_mode = parser.add_mutually_exclusive_group()
    trt_cache_mode.add_argument(
        "--rebuild_trt_cache",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="rebuild_trt_cache",
        help=(
            "Ignore any existing TensorRT cache and rebuild/save engines once for "
            "this run."
        ),
    )
    trt_cache_mode.add_argument(
        "--use_existing_trt_cache",
        "--use-existing-trt-cache",
        action="store_false",
        default=argparse.SUPPRESS,
        dest="rebuild_trt_cache",
        help=(
            "Use already-saved TensorRT engines when available instead of forcing "
            "a rebuild at run startup. This is the default."
        ),
    )
    parser.set_defaults(rebuild_trt_cache=False)
    parser.add_argument(
        "--disable_trt_cache",
        action="store_true",
        help="Disable TensorRT engine load/save caching.",
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


def _output_rel_path(
    dataset: str,
    video_name: str,
    video_file: str,
    output_layout: str,
) -> str:
    """
    Compute the consistent on-disk output path for one video, relative to
    --outputs_dir.

    The structure is identical for every split and every dataset so the output
    tree is never messy:

      - "dataset" layout : <dataset>/<video_name>/<video_file>
      - "video"   layout : <video_name>/<video_file>

    The per-frame bbox .txt is written by run_inference() right beside the video
    file inside the same <video_name>/ folder (see vis/test_model.py).

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. "dataset1").
    video_name : str
        Per-video folder name (the sequence name, e.g. "Car_video_2").
    video_file : str
        Leaf filename for the (optional) annotated video, e.g. "Car_video_2.mp4".
    output_layout : str
        Either "dataset" (include the dataset folder) or "video" (drop it).

    Returns
    -------
    str
        Relative output path joined with the right number of folder levels.
    """
    if output_layout == "video":
        return os.path.join(video_name, video_file)
    return os.path.join(dataset, video_name, video_file)


def _build_manifest_entries(
    manifest: dict,
    target_datasets: set[str],
    split_name: str,
    output_layout: str = "dataset",
    key_prefix: str = "",
) -> dict[str, dict]:
    """
    Build a uniform entry dictionary from a manifest split.

    The on-disk output path is always <dataset>/<video_name>/... (or
    <video_name>/... for the "video" layout) regardless of split, so the train,
    public_lb and combined "all" runs all share one consistent folder tree. The
    run-entry *key* still carries any key_prefix (e.g. "train/"/"test/") so the
    submission CSV ids and single-video selection stay unambiguous.
    """
    split_entries = manifest.get(split_name, {})
    entries: dict[str, dict] = {}
    for key, value in split_entries.items():
        if value["dataset"] not in target_datasets:
            continue
        video_rel = Path(value["video_path"])
        out_rel_path = _output_rel_path(
            dataset=value["dataset"],
            video_name=video_rel.parent.name,
            video_file=video_rel.name,
            output_layout=output_layout,
        )
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
    output_layout: str = "dataset",
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

            output_rel_path = _output_rel_path(
                dataset=dataset_name,
                video_name=key,
                video_file=f"{key}.mp4",
                output_layout=output_layout,
            )
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
        siamram_log(
            f"skipped {dropped_missing_path} rows with missing seq/annotation paths",
            phase="DATA",
            status="warn",
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


TRACKER_CONFIG_ALIASES = {
    "score_window_type": "windowing",
    "score_grid_stride": "total_stride",
    "score_grid_size": "score_size",
    "template_update_enabled": "dynamic_update",
    "bbox_smoothing_enabled": "smooth",
    "template_context_padding": "template_bbox_offset",
    "search_context_scale": "search_context",
    "search_crop_size": "instance_size",
    "template_crop_size": "template_size",
    "template_update_interval": "N",
    "warmup_template_update_interval": "warm_up_n",
    "template_memory_window": "memory_window_size",
    "template_admit_conf_threshold": "dynamic_update_threshold",
    "running_confidence_floor": "running_confidence_floor_value",
    "template_admit_iou_threshold": "iou_threshold",
    "template_warmup_frames": "warmup_frames",
    "warmup_template_memory_window": "warmup_window_size",
}

HANNING_WINDOW_PENALTY_ALIASES = {
    "enabled": "hanning_window_enabled",
    "window_type": "windowing",
    "influence": "window_influence",
    "size_penalty_enabled": "size_penalty_enabled",
    "size_penalty_k": "penalty_k",
    "bbox_size_smoothing_enabled": "smooth",
    "bbox_size_smoothing_lr": "lr",
}

ARCHITECTURAL_CONFIG_ALIASES = {
    "score_grid_stride": "total_stride",
    "score_grid_size": "score_size",
    "template_context_padding": "template_bbox_offset",
    "search_context_scale": "search_context",
    "search_crop_size": "instance_size",
    "template_crop_size": "template_size",
}

DYNAMIC_TEMPLATE_ALIASES = {
    "enabled": "dynamic_update",
    "update_interval": "N",
    "warmup_update_interval": "warm_up_n",
    "memory_window": "memory_window_size",
    "admit_conf_threshold": "dynamic_update_threshold",
    "running_confidence_floor": "running_confidence_floor_value",
    "admit_iou_threshold": "iou_threshold",
    "warmup_frames": "warmup_frames",
    "warmup_memory_window": "warmup_window_size",
}


def _apply_tracker_aliases(tracker_cfg, source_cfg, aliases) -> None:
    for friendly_key, legacy_key in aliases.items():
        if friendly_key in source_cfg:
            tracker_cfg[legacy_key] = source_cfg[friendly_key]


def _normalize_tracker_config_aliases(config) -> None:
    """
    Let inference YAML use readable tracker keys while keeping SiamABC internals
    compatible with the original short/legacy names.
    """
    if "tracker" not in config:
        return
    tracker_cfg = config.tracker
    _apply_tracker_aliases(tracker_cfg, tracker_cfg, TRACKER_CONFIG_ALIASES)

    penalty_cfg = tracker_cfg.get("hanning_window_penalty", None)
    if penalty_cfg is not None:
        _apply_tracker_aliases(
            tracker_cfg, penalty_cfg, HANNING_WINDOW_PENALTY_ALIASES
        )

    architectural_cfg = tracker_cfg.get("architectural_config", None)
    if architectural_cfg is not None:
        _apply_tracker_aliases(
            tracker_cfg, architectural_cfg, ARCHITECTURAL_CONFIG_ALIASES
        )

    dynamic_template_cfg = tracker_cfg.get("dynamic_template", None)
    if dynamic_template_cfg is not None:
        _apply_tracker_aliases(
            tracker_cfg, dynamic_template_cfg, DYNAMIC_TEMPLATE_ALIASES
        )


SUBMISSION_COLUMNS = ["id", "x", "y", "w", "h"]


def _uses_train_submission_id(args, entry: dict) -> bool:
    split_name = str(entry.get("split", args.run_split))
    return args.run_split == "train_csv" or (
        args.run_split in {"train", "all"} and split_name == "train"
    )


def _build_submission_id(args, key: str, entry: dict, frame_idx: int) -> str:
    if _uses_train_submission_id(args, entry):
        return f"{args.id_prefix}_{key}_{frame_idx}"
    return f"{key}_{frame_idx}"


def _build_submission_id_prefix(args, key: str, entry: dict) -> str:
    if _uses_train_submission_id(args, entry):
        return f"{args.id_prefix}_{key}_"
    return f"{key}_"


def _submission_id_matches_prefix(sample_id: object, prefix: str) -> bool:
    sample_id = str(sample_id)
    if not sample_id.startswith(prefix):
        return False
    return sample_id[len(prefix) :].isdigit()


def _bbox_file_for_entry(outputs_dir: str, entry: dict) -> str:
    head, tail = os.path.split(os.path.join(outputs_dir, entry["output_rel_path"]))
    return os.path.join(head, os.path.splitext(tail)[0] + ".txt")


def _compile_submission_from_outputs(
    args,
    entries: dict[str, dict],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build a submission DataFrame from saved per-video bbox text files.

    In normal mode, entries are just the videos processed in this invocation.
    With --override_csv, entries are the full selected split, so the final CSV
    is rebuilt from the bbox outputs already present under --outputs_dir.
    """
    rows: list[dict[str, float | str]] = []
    missing_keys: list[str] = []
    present_prefixes: list[str] = []

    for key, entry in entries.items():
        bbox_file = _bbox_file_for_entry(args.outputs_dir, entry)

        if not os.path.exists(bbox_file):
            missing_keys.append(key)
            continue

        with open(bbox_file, "r", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]

        if not lines:
            raise ValueError(f"Bbox output file is empty: {bbox_file}")

        present_prefixes.append(_build_submission_id_prefix(args, key, entry))
        for frame_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid bbox line in {bbox_file!r} at frame {frame_idx}: "
                    f"expected 4 values, got {len(parts)} ({line!r})"
                )
            x, y, w, h = parts
            rows.append(
                {
                    "id": _build_submission_id(args, key, entry, frame_idx),
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                }
            )

    submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    _validate_submission(submission)

    return submission, missing_keys, present_prefixes


def _validate_submission(submission: pd.DataFrame) -> None:
    if submission["id"].duplicated().any():
        duplicate_ids = submission.loc[submission["id"].duplicated(), "id"].head(5)
        raise ValueError(
            "Compiled submission contains duplicate ids; refusing to write CSV. "
            f"Examples: {duplicate_ids.tolist()}"
        )


def _require_submission_columns(submission: pd.DataFrame, csv_path: str) -> None:
    missing_columns = [
        column for column in SUBMISSION_COLUMNS if column not in submission.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Cannot --override_csv because {csv_path!r} is missing "
            f"required column(s): {missing_columns}"
        )


def _merge_rebuilt_outputs_into_existing(
    existing_submission: pd.DataFrame,
    rebuilt_submission: pd.DataFrame,
    rebuilt_prefixes: list[str],
    csv_path: str,
) -> tuple[pd.DataFrame, int]:
    """
    Overlay rebuilt bbox-output rows onto an existing submission CSV.

    This is the safe one-video-rerun path: every saved output under
    --outputs_dir is trusted as fresh, while videos without a saved bbox file
    keep their previous CSV rows.
    """
    _require_submission_columns(existing_submission, csv_path)

    existing = existing_submission[SUBMISSION_COLUMNS].copy()
    rebuilt = rebuilt_submission[SUBMISSION_COLUMNS].copy()
    existing["id"] = existing["id"].astype(str)
    rebuilt["id"] = rebuilt["id"].astype(str)

    def matching_prefix(sample_id: object) -> str | None:
        for prefix in rebuilt_prefixes:
            if _submission_id_matches_prefix(sample_id, prefix):
                return prefix
        return None

    rebuilt_by_prefix = {
        prefix: rebuilt[
            rebuilt["id"].map(
                lambda sample_id, prefix=prefix: _submission_id_matches_prefix(
                    sample_id, prefix
                )
            )
        ]
        for prefix in rebuilt_prefixes
    }

    merged_rows: list[dict] = []
    inserted_prefixes: set[str] = set()
    replaced_existing_rows = 0

    for _, row in existing.iterrows():
        prefix = matching_prefix(row["id"])
        if prefix is None:
            merged_rows.append(row.to_dict())
            continue

        replaced_existing_rows += 1
        if prefix not in inserted_prefixes:
            merged_rows.extend(rebuilt_by_prefix[prefix].to_dict("records"))
            inserted_prefixes.add(prefix)

    for prefix in rebuilt_prefixes:
        if prefix not in inserted_prefixes:
            merged_rows.extend(rebuilt_by_prefix[prefix].to_dict("records"))

    merged = pd.DataFrame(merged_rows, columns=SUBMISSION_COLUMNS)
    _validate_submission(merged)
    return merged, replaced_existing_rows


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
    def _looks_like_ultralytics_yolo_asset(path: Path) -> bool:
        name = path.name.strip().lower()
        return name.startswith("yolo") and name.endswith(".pt")

    def _download_ultralytics_yolo_asset(destination: Path) -> Path:
        from ultralytics.utils.downloads import (
            attempt_download_asset,
            get_github_assets,
            safe_download,
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        name = destination.name
        errors: list[str] = []

        # Primary path: let ultralytics pick its default release behavior.
        try:
            downloaded = Path(
                attempt_download_asset(
                    file=str(destination),
                    repo="ultralytics/assets",
                )
            )
            if downloaded.exists():
                return downloaded.expanduser().resolve()
        except Exception as exc:
            errors.append(f"attempt_download_asset(default): {exc}")

        # Fallback: resolve the concrete latest tag then download directly.
        try:
            tag, assets = get_github_assets(
                repo="ultralytics/assets",
                version="latest",
                retry=True,
            )
            if tag and name in assets:
                safe_download(
                    url=(
                        f"https://github.com/ultralytics/assets/"
                        f"releases/download/{tag}/{name}"
                    ),
                    file=str(destination),
                    min_bytes=1e5,
                    unzip=False,
                )
                if destination.exists():
                    return destination.expanduser().resolve()
            errors.append(
                f"latest-tag lookup did not include '{name}' "
                f"(tag='{tag}', assets={len(assets)})"
            )
        except Exception as exc:
            errors.append(f"latest-tag direct download: {exc}")

        raise RuntimeError("; ".join(errors))

    def _try_download_missing_yolo_checkpoint(path: Path) -> Path:
        if _is_valid_checkpoint_file(path):
            return path
        if not _looks_like_ultralytics_yolo_asset(path):
            return path

        siamram_log(
            f"missing YOLO weights '{path.name}' · auto-downloading from "
            "ultralytics/assets",
            phase="CKPT",
            status="build",
        )
        try:
            downloaded = _download_ultralytics_yolo_asset(path)
        except Exception as exc:
            siamram_log(
                f"YOLO auto-download failed for {path.name}: {exc}",
                phase="CKPT",
                status="error",
            )
            return path

        if downloaded != path and _is_valid_checkpoint_file(downloaded):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, path)

        if _is_valid_checkpoint_file(path):
            siamram_log(f"YOLO weights ready · {path}", phase="CKPT", status="ready")
            return path
        if _is_valid_checkpoint_file(downloaded):
            return downloaded
        return path

    yolo_path = _try_download_missing_yolo_checkpoint(yolo_path)

    missing_weights = not _is_valid_checkpoint_file(weights_path)
    missing_yolo = not _is_valid_checkpoint_file(yolo_path)
    if not (missing_weights or missing_yolo):
        # Both files look good — nothing to do.
        return weights_path, yolo_path

    # The bundled downloader currently manages the SiamABC checkpoint and
    # the default YOLO weight file (yolo11n.pt). Other YOLO variants are
    # handled via the ultralytics/assets path above.
    needs_bundle_downloader = missing_weights or yolo_path.name.lower() == "yolo11n.pt"
    resolved_weights = weights_path
    resolved_yolo = yolo_path
    if needs_bundle_downloader:
        if not CHECKPOINT_DOWNLOADER.exists():
            missing_now: list[Path] = []
            if missing_weights:
                missing_now.append(weights_path)
            if missing_yolo:
                missing_now.append(yolo_path)
            missing_str = ", ".join(str(p) for p in missing_now)
            raise FileNotFoundError(
                f"Missing checkpoints: {missing_str}. "
                f"Also missing downloader script: {CHECKPOINT_DOWNLOADER}"
            )

        siamram_log(
            "missing or invalid checkpoints · downloading required models",
            phase="CKPT",
            status="build",
        )
        subprocess.run(
            [sys.executable, str(CHECKPOINT_DOWNLOADER), "--force"],
            check=True,
            cwd=str(BASE_DIR),
        )

        # Re-resolve after download in case paths shifted slightly.
        resolved_weights = _resolve_weights_path(str(weights_path))
        resolved_yolo = _resolve_weights_path(str(yolo_path))

    resolved_yolo = _try_download_missing_yolo_checkpoint(resolved_yolo)

    still_missing = [
        p
        for p in (resolved_weights, resolved_yolo)
        if not _is_valid_checkpoint_file(p)
    ]
    if still_missing:
        missing_str = ", ".join(str(p) for p in still_missing)
        raise FileNotFoundError(
            f"Checkpoint download finished, but files are still missing: {missing_str}"
        )

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

      a) TensorRT mode (config.trt_engine.trt_compile_siamabc = True):
         Compiles the model to a TRT engine for maximum GPU throughput.
         This is slower to initialise but much faster per frame — ideal for
         large datasets or tight time budgets.

      b) Standard PyTorch mode (config.trt_engine.trt_compile_siamabc = False):
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
         subsequent frame, writing the per-frame bbox predictions to a .txt
         file right beside the video inside its own <video_name>/ folder.

    Step 6 — Compile the submission CSV
    ------------------------------------
    After all videos are processed, we walk through every video's bbox .txt
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
    _normalize_tracker_config_aliases(config)
    ram_tracker_kwargs = flatten_ram_tracker_config(config)

    resolved_weights_path = _resolve_weights_path(args.weights_path)
    yolo_weights_cfg = str(
        ram_tracker_kwargs.get("yolo_weights", CHECKPOINTS_DIR / "yolo11n.pt")
    )
    resolved_yolo_path = _resolve_weights_path(yolo_weights_cfg)
    resolved_weights_path, resolved_yolo_path = _ensure_required_checkpoints(
        resolved_weights_path, resolved_yolo_path
    )

    # Write the resolved paths back so the rest of the code uses them.
    args.weights_path = str(resolved_weights_path)
    ram_tracker_kwargs["yolo_weights"] = str(resolved_yolo_path)
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
    # All TRT/FP16 settings now live under the `trt_engine` block. The SiamABC
    # toggle is `trt_engine.trt_compile_siamabc`; the legacy top-level
    # `trt_compile_siamabc` / `make_trt_engine` keys are honored as fallbacks so
    # older configs keep working.
    trt_cfg = config.get("trt_engine", {}) or {}
    _siam_flag = trt_cfg.get("trt_compile_siamabc", None)
    if _siam_flag is None:
        _siam_flag = config.get(
            "trt_compile_siamabc", config.get("make_trt_engine", False)
        )
    compile_siamabc = bool(_siam_flag)

    # OSNet FP16 / TRT also live under trt_engine; inject them into the tracker
    # kwargs (they are no longer in the ram_tracker.descriptor block). Fall back
    # to any legacy descriptor-block values, then to disabled.
    ram_tracker_kwargs["osnet_fp16"] = bool(
        trt_cfg.get("osnet_fp16", ram_tracker_kwargs.get("osnet_fp16", False))
    )
    ram_tracker_kwargs["trt_compile_osnet"] = bool(
        trt_cfg.get(
            "trt_compile_osnet", ram_tracker_kwargs.get("trt_compile_osnet", False)
        )
    )
    ram_tracker_kwargs["osnet_async_overlap"] = bool(
        trt_cfg.get(
            "osnet_async_overlap",
            ram_tracker_kwargs.get("osnet_async_overlap", False),
        )
    )
    trt_cache_dir = ""
    if not args.disable_trt_cache:
        trt_cache_path = Path(args.trt_cache_dir).expanduser()
        if not trt_cache_path.is_absolute():
            trt_cache_path = BASE_DIR / trt_cache_path
        trt_cache_path.mkdir(parents=True, exist_ok=True)
        trt_cache_dir = str(trt_cache_path.resolve())

    ram_tracker_kwargs["trt_cache_dir"] = trt_cache_dir
    ram_tracker_kwargs["trt_rebuild_cache"] = bool(args.rebuild_trt_cache)

    # One TensorRT compile progress bar for the whole setup phase: SiamABC
    # (backbone/attention/connect) plus, when enabled, the OSNet descriptor.
    # begin() pre-sizes the shared session so both subsystems fill a single
    # 0-100% bar; the up-front OSNet warm below advances and closes it.
    _descriptor_backend = str(
        ram_tracker_kwargs.get("descriptor_backend", "osnet")
    ).strip().lower()
    _osnet_will_compile = _descriptor_backend == "osnet" and bool(
        ram_tracker_kwargs.get("trt_compile_osnet", False)
    )
    _compile_stages = (3 if compile_siamabc else 0) + (1 if _osnet_will_compile else 0)
    if _compile_stages:
        compile_progress().begin(_compile_stages)

    if compile_siamabc:
        if get_trt_tracker is None:
            raise ModuleNotFoundError(
                "TensorRT tracker dependencies are missing. "
                "Set trt_engine.trt_compile_siamabc: False in your config or "
                "install torch_tensorrt."
            ) from _trt_import_error
        wrapped = get_trt_tracker(
            config=config,
            weights_path=args.weights_path,
            lambda_tta=float(trt_cfg.get("lambda_tta", args.lambda_tta)),
            fp16=bool(trt_cfg.get("fp16", True)),
            cuda_id=int(trt_cfg.get("cuda_id", 0)),
            backbone_mode=str(trt_cfg.get("backbone_mode", "")),
            disable_tf32=bool(trt_cfg.get("backbone_disable_tf32", False)),
            engine_cache_dir=trt_cache_dir,
            rebuild_cache=bool(args.rebuild_trt_cache),
        )
    else:
        wrapped = get_tracker(
            config=config,
            weights_path=args.weights_path,
            lambda_tta=args.lambda_tta,
            continuous=False,
        )

    # Compile the OSNet descriptor engine now, contiguously after SiamABC, so the
    # whole setup shares one progress bar. configure_descriptor_backend() builds
    # the engine eagerly at setup (see utils.utils); the per-clip trackers later
    # reconfigure with identical settings and reuse this warmed engine, so this
    # changes no tracking behavior.
    if _osnet_will_compile:
        configure_descriptor_backend(
            descriptor_backend=_descriptor_backend,
            osnet_model_name=str(ram_tracker_kwargs.get("osnet_model_name", "osnet_x1_0")),
            osnet_model_path=str(ram_tracker_kwargs.get("osnet_model_path", "")),
            osnet_pretrained_checkpoint=str(
                ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
            ),
            osnet_device=str(ram_tracker_kwargs.get("osnet_device", "auto")),
            siam_tracker=wrapped,
            siamese_feature_source=str(
                ram_tracker_kwargs.get("siamese_feature_source", "neck")
            ),
            siamese_comparison_mode=str(
                ram_tracker_kwargs.get("siamese_comparison_mode", "xcorr")
            ),
            osnet_fp16=bool(ram_tracker_kwargs.get("osnet_fp16", False)),
            trt_compile_osnet=bool(ram_tracker_kwargs.get("trt_compile_osnet", False)),
            osnet_max_candidate_batch=int(
                ram_tracker_kwargs.get("osnet_max_candidate_batch", 0)
            ),
            trt_cache_dir=str(ram_tracker_kwargs.get("trt_cache_dir", "")),
            trt_rebuild_cache=bool(ram_tracker_kwargs.get("trt_rebuild_cache", False)),
        )
    # Close the shared bar (idempotent if the last stage already closed it).
    compile_progress().finish()

    # ram_tracker_impl is kept in the config for backwards compatibility, but
    # the base tracker has been retired — SiamRAMExperimentTracker is now the
    # only supported implementation. We warn on legacy values rather than
    # erroring so older config files keep loading.
    ram_impl = str(getattr(config, "ram_tracker_impl", "experiment")).strip().lower()
    if ram_impl not in {"experiment", "exp", "siamram_experiment"}:
        siamram_log(
            f"config sets ram_tracker_impl='{ram_impl}' but the base tracker was "
            "removed; falling back to the experimental tracker",
            phase="INIT",
            status="warn",
        )
    tracker_cls = SiamRAMExperimentTracker

    osnet_ckpt = str(
        ram_tracker_kwargs.get("osnet_pretrained_checkpoint", "imagenet")
    ).strip()
    if osnet_ckpt:
        if osnet_ckpt not in OSNET_CHECKPOINT_CHOICES:
            choices = ", ".join(sorted(OSNET_CHECKPOINT_CHOICES))
            raise ValueError(
                f"Unsupported osnet_pretrained_checkpoint='{osnet_ckpt}'. "
                f"Expected one of: {choices}."
            )

    osnet_model_path_raw = str(ram_tracker_kwargs.get("osnet_model_path", "")).strip()
    if osnet_model_path_raw:
        ram_tracker_kwargs["osnet_model_path"] = str(
            _resolve_weights_path(osnet_model_path_raw)
        )

    shared_tracker = None
    if args.reuse_tracker:
        shared_tracker = tracker_cls(siam_tracker=wrapped, **ram_tracker_kwargs)
        siamram_log(
            f"tracker · {tracker_cls.__name__} · reused across clips",
            phase="INIT",
            status="ready",
        )
    else:
        siamram_log(
            f"tracker · {tracker_cls.__name__} · fresh wrapper per clip",
            phase="INIT",
            status="ready",
        )

    def _ram_kwargs_for_clip(clip_index: int) -> dict:
        kwargs = dict(ram_tracker_kwargs)
        # The OSNet descriptor engine (the only thing a fresh wrapper would TRT-
        # compile) is already built once up front during setup, honoring
        # --rebuild_trt_cache there. So per-clip wrappers must always reuse it —
        # forcing a rebuild here would recompile OSNet a second time.
        kwargs["trt_rebuild_cache"] = False
        return kwargs

    if args.datasets is not None:
        # The user explicitly named which datasets they want.
        target_datasets = set(args.datasets)
    else:
        # Auto-discover: every sub-folder in data_dir that isn't "metadata".
        data_root = Path(args.data_dir)
        target_datasets = {
            p.name for p in data_root.iterdir() if p.is_dir() and p.name != "metadata"
        }
        siamram_log(
            f"datasets · {', '.join(sorted(target_datasets))}",
            phase="DATA",
            status="info",
        )

    # Build a unified entry map regardless of source split.
    if args.run_split in {"public_lb", "train"}:
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        run_entries = _build_manifest_entries(
            manifest=manifest,
            target_datasets=target_datasets,
            split_name=args.run_split,
            output_layout=args.output_layout,
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
                output_layout=args.output_layout,
                key_prefix="train/",
            )
        )
        run_entries.update(
            _build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="public_lb",
                output_layout=args.output_layout,
                key_prefix="test/",
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
            output_layout=args.output_layout,
        )

    # Process the videos grouped dataset-by-dataset (then by key within each
    # dataset) so a full dataset is finished before the next one begins and the
    # console/output ordering is predictable — especially for --run_split all,
    # which would otherwise interleave the train and public_lb maps.
    run_entries = dict(
        sorted(
            run_entries.items(),
            key=lambda item: (str(item[1].get("dataset", "")), item[0]),
        )
    )
    csv_rebuild_entries = dict(run_entries)

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
    siamram_log(
        f"{len(run_entries)} clips queued · split={args.run_split}",
        phase="DATA",
        status="info",
    )
    for i, (key, value) in enumerate(run_entries.items()):
        video_path = _resolve_data_asset_path(value["video_path"], args.data_dir)
        ann_path = _resolve_data_asset_path(value["annotation_path"], args.data_dir)
        output_path = os.path.join(args.outputs_dir, value["output_rel_path"])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        siamram_log(
            f"[{i + 1}/{len(run_entries)}] {key}",
            phase="RUN",
            status="load",
            label="video",
            blank_before=True,
        )

        # Load the initial bbox. The annotation file is a single line like:
        #   320.5,140.0,64.0,80.0   (x, y, width, height in pixels)
        # If the file has multiple rows for some reason, we take the first.
        init_bbox = _load_initial_bbox(ann_path)

        tracker = shared_tracker or tracker_cls(
            siam_tracker=wrapped, **_ram_kwargs_for_clip(i)
        )
        try:
            run_inference(
                video_path=video_path,
                initial_bbox=init_bbox,
                tracker=tracker,
                output_path=output_path,
                output_video=args.output_video,
            )

            # Report the YOLO-detectability verdict the probe settled on for this
            # video (drives the occlusion-recovery strategy: detectable -> YOLO+DRM,
            # not -> SiamABC-alone). The verdict lives on the tracker after the run.
            enabled = bool(getattr(tracker, "_yolo_detectability_enabled", False))
            probe_done = bool(getattr(tracker, "_detectability_probe_done", False))
            detectable = bool(getattr(tracker, "_yolo_detectable", False))
            runs = int(getattr(tracker, "_detectability_runs", 0))
            hits = int(getattr(tracker, "_detectability_hits", 0))
            if not enabled:
                verdict, status = "probe disabled", "info"
            elif not probe_done:
                verdict, status = (
                    f"undetermined · {hits} hit(s) / {runs} run(s) "
                    "(clip too short or occluded before probe finished)",
                    "warn",
                )
            elif detectable:
                verdict, status = (
                    f"✓ detectable · {hits} hit(s) / {runs} run(s)",
                    "ready",
                )
            else:
                verdict, status = (
                    f"✗ not detectable · {hits} hit(s) / {runs} run(s)",
                    "warn",
                )
            siamram_log(verdict, phase="PROBE", status=status, label="yolo")
        finally:
            if shared_tracker is None and hasattr(tracker, "_discard_pending_descriptor"):
                tracker._discard_pending_descriptor()

    # ------------------------------------------------------------------
    #  Stitch all per-video bbox files into one submission CSV.
    #
    # run_inference() writes a text file at:
    #   <outputs_dir>/<dataset>/<video_name>/<video_name>.txt
    #   (or <outputs_dir>/<video_name>/<video_name>.txt for --output_layout video)
    #
    # Each line in that file is one frame's prediction: "x y w h".
    # We assign each line an id of "<video_key>_<frame_index>" and collect
    # everything into a flat DataFrame that the competition scorer expects.
    # ------------------------------------------------------------------
    if args.override_csv:
        siamram_log(
            f"Rebuilding submission CSV from saved outputs under {args.outputs_dir}",
            phase="CSV",
            status="build",
            blank_before=True,
        )
        submission, missing_bbox_keys, rebuilt_prefixes = _compile_submission_from_outputs(
            args=args,
            entries=csv_rebuild_entries,
        )
        if not rebuilt_prefixes:
            raise RuntimeError(
                "--override_csv found no bbox output files under "
                f"{args.outputs_dir}; refusing to modify the CSV."
            )
        if missing_bbox_keys:
            if not os.path.exists(args.submission_csv):
                sample_missing = missing_bbox_keys[:10]
                raise RuntimeError(
                    "--override_csv found missing bbox output files and cannot "
                    "preserve their previous rows because --submission_csv does "
                    f"not exist. Missing count: {len(missing_bbox_keys)}. "
                    f"Examples: {sample_missing}"
                )

            existing_submission = pd.read_csv(args.submission_csv, dtype={"id": str})
            original_rows = len(existing_submission)
            submission, replaced_existing_rows = _merge_rebuilt_outputs_into_existing(
                existing_submission=existing_submission,
                rebuilt_submission=submission,
                rebuilt_prefixes=rebuilt_prefixes,
                csv_path=args.submission_csv,
            )
            siamram_log(
                "override mode · rebuilt "
                f"{len(rebuilt_prefixes)} file(s), replaced "
                f"{replaced_existing_rows} row(s), preserved "
                f"{len(missing_bbox_keys)} missing file(s) "
                f"({original_rows} → {len(submission)} rows)",
                phase="CSV",
                status="done",
            )
        else:
            siamram_log(
                "override mode · rebuilt full CSV from "
                f"{len(csv_rebuild_entries)} saved output file(s)",
                phase="CSV",
                status="done",
            )
    else:
        siamram_log(
            "Compiling submission CSV", phase="CSV", status="build", blank_before=True
        )
        submission, missing_bbox_keys, _rebuilt_prefixes = _compile_submission_from_outputs(
            args=args,
            entries=run_entries,
        )
        for key in missing_bbox_keys:
            # This shouldn't happen if inference completed successfully,
            # but we warn instead of crashing so one bad video doesn't
            # destroy the whole submission.
            siamram_log(f"bbox file missing for {key}", phase="CSV", status="warn")

    submission.to_csv(args.submission_csv, index=False)

    siamram_log(
        f"Submission written · {args.submission_csv} · {len(submission)} rows",
        phase="CSV",
        status="done",
        blank_before=True,
    )
    print(submission.head())


if __name__ == "__main__":
    main()
