#!/usr/bin/env python3
"""
Rebuild a competition submission CSV from run_inference.py output folders.

The script mirrors run_inference.py's submission ID and bbox-file conventions:

  dataset layout: <outputs_dir>/<dataset>/<video_name>/<video_file_stem>.txt
  video layout  : <outputs_dir>/<video_name>/<video_file_stem>.txt

Each bbox text file must contain one "x y w h" row per frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_COLUMNS = ["id", "x", "y", "w", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a submission CSV from bbox .txt files already written by "
            "run_inference.py."
        )
    )
    parser.add_argument(
        "outputs_dir",
        nargs="?",
        help="Folder that contains run_inference.py outputs.",
    )
    parser.add_argument(
        "--outputs_dir",
        dest="outputs_dir_option",
        help="Same as the positional outputs_dir argument.",
    )
    parser.add_argument(
        "--submission_csv",
        default=str(BASE_DIR / "submission_from_outputs.csv"),
        help="Where to write the rebuilt CSV.",
    )
    parser.add_argument(
        "--run_split",
        default="public_lb",
        choices=["public_lb", "train", "train_csv", "all"],
        help="Which run_inference.py split convention to use for IDs.",
    )
    parser.add_argument(
        "--train_csv_path",
        default=str(BASE_DIR / "data" / "train_dataframe.csv"),
        help="Training dataframe CSV path used when --run_split train_csv.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(BASE_DIR / "data"),
        help="Root data directory used to auto-discover datasets.",
    )
    parser.add_argument(
        "--manifest_path",
        default=str(BASE_DIR / "data" / "metadata" / "contestant_manifest.json"),
        help="Competition manifest JSON path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to include. Defaults to all dataset folders.",
    )
    parser.add_argument(
        "--video_key",
        default=None,
        help=(
            "Optional single-sequence selector, e.g. dataset5/person19_3. "
            "Matches key/source_key/path exactly first, then by substring."
        ),
    )
    parser.add_argument(
        "--output_layout",
        default="dataset",
        choices=["dataset", "video", "auto"],
        help=(
            "Folder layout under outputs_dir. Use 'dataset' for normal "
            "run_inference.py output, 'video' for run_single_video.py-style "
            "output, or 'auto' to try dataset first then video."
        ),
    )
    parser.add_argument(
        "--id_prefix",
        default="pred",
        help="Prefix for generated train/train_csv frame IDs.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Optional cap on selected sequences after sorting (0 means no cap).",
    )
    parser.add_argument(
        "--allow_missing",
        "--allow-missing",
        action="store_true",
        help=(
            "Write a partial CSV from files that exist instead of failing when "
            "some selected bbox files are missing."
        ),
    )

    args = parser.parse_args()
    if args.outputs_dir_option:
        args.outputs_dir = args.outputs_dir_option
    if not args.outputs_dir:
        parser.error("outputs_dir is required, either positional or --outputs_dir")
    return args


def _output_rel_path(
    dataset: str,
    video_name: str,
    video_file: str,
    output_layout: str,
) -> str:
    if output_layout == "video":
        return os.path.join(video_name, video_file)
    return os.path.join(dataset, video_name, video_file)


def _candidate_output_rel_paths(entry: dict[str, Any], output_layout: str) -> list[str]:
    dataset = str(entry["dataset"])
    video_name = str(entry["video_name"])
    video_file = str(entry["video_file"])

    if output_layout == "auto":
        return [
            _output_rel_path(dataset, video_name, video_file, "dataset"),
            _output_rel_path(dataset, video_name, video_file, "video"),
        ]
    return [_output_rel_path(dataset, video_name, video_file, output_layout)]


def _bbox_candidates_for_entry(
    outputs_dir: str | Path,
    entry: dict[str, Any],
    output_layout: str,
) -> list[Path]:
    candidates: list[Path] = []
    for rel_path in _candidate_output_rel_paths(entry, output_layout):
        video_path = Path(outputs_dir) / rel_path
        candidates.append(video_path.with_suffix(".txt"))

    # Keep order stable while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            unique.append(candidate)
            seen.add(candidate_key)
    return unique


def _discover_datasets(data_dir: str | Path) -> set[str]:
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    return {
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and path.name != "metadata"
    }


def _build_manifest_entries(
    manifest: dict[str, Any],
    target_datasets: set[str],
    split_name: str,
    key_prefix: str = "",
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for key, value in manifest.get(split_name, {}).items():
        dataset = str(value["dataset"])
        if dataset not in target_datasets:
            continue

        video_rel = Path(value["video_path"])
        out_key = f"{key_prefix}{key}"
        entries[out_key] = {
            "dataset": dataset,
            "video_path": value["video_path"],
            "annotation_path": value["annotation_path"],
            "split": split_name,
            "source_key": key,
            "video_name": video_rel.parent.name,
            "video_file": video_rel.name,
        }
    return entries


def _build_train_entries_from_csv(
    train_csv_path: str | Path,
    target_datasets: set[str],
    max_sequences: int = 0,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with open(train_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            dataset = str(row.get("dataset", "")).strip() or "unknown"
            if dataset not in target_datasets:
                continue

            seq_raw = str(row.get("img_path") or row.get("seq_path") or "").strip()
            if not seq_raw:
                continue

            seq_path = Path(seq_raw)
            seq_leaf = seq_path.name
            seq_name = seq_path.parent.name if seq_leaf == "img" else seq_leaf
            base_key = f"train_{dataset}__{seq_name}"
            key = base_key
            suffix = 1
            while key in entries:
                suffix += 1
                key = f"{base_key}_{suffix:03d}"

            entries[key] = {
                "dataset": dataset,
                "video_path": seq_raw,
                "annotation_path": row.get("annot_path", ""),
                "split": "train_csv",
                "source_key": key,
                "row_idx": row_idx,
                "video_name": key,
                "video_file": f"{key}.mp4",
            }
            if max_sequences > 0 and len(entries) >= max_sequences:
                break
    return entries


def _norm_key(key: str) -> str:
    if key.startswith("train/"):
        return key[len("train/") :]
    if key.startswith("test/"):
        return key[len("test/") :]
    return key


def _filter_run_entries_by_video_key(
    run_entries: dict[str, dict[str, Any]],
    video_key: str | None,
) -> dict[str, dict[str, Any]]:
    if not video_key:
        return run_entries

    needle = str(video_key).strip()
    if not needle:
        return run_entries
    needle_l = needle.lower()

    exact: dict[str, dict[str, Any]] = {}
    fuzzy: dict[str, dict[str, Any]] = {}
    for key, value in run_entries.items():
        source_key = str(value.get("source_key", key))
        fields = [
            key,
            source_key,
            _norm_key(key),
            str(value.get("video_path", "")),
            str(value.get("annotation_path", "")),
        ]
        fields_l = [field.lower() for field in fields]

        if needle in fields or needle_l in fields_l:
            exact[key] = value
            continue

        if any(needle_l in field for field in fields_l):
            fuzzy[key] = value

    chosen = exact if exact else fuzzy
    if not chosen:
        sample = list(run_entries.keys())[:20]
        raise KeyError(
            f"--video_key {needle!r} did not match any entry. "
            f"Sample available keys: {sample}"
        )
    if len(chosen) > 1:
        raise KeyError(
            f"--video_key {needle!r} is ambiguous. Matched keys: {list(chosen)}"
        )
    return chosen


def _uses_train_submission_id(args: argparse.Namespace, entry: dict[str, Any]) -> bool:
    split_name = str(entry.get("split", args.run_split))
    return args.run_split == "train_csv" or (
        args.run_split in {"train", "all"} and split_name == "train"
    )


def _build_submission_id(
    args: argparse.Namespace,
    key: str,
    entry: dict[str, Any],
    frame_idx: int,
) -> str:
    if _uses_train_submission_id(args, entry):
        return f"{args.id_prefix}_{key}_{frame_idx}"
    return f"{key}_{frame_idx}"


def _validate_no_duplicate_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            duplicates.append(row_id)
            if len(duplicates) >= 5:
                break
        seen.add(row_id)
    if duplicates:
        raise ValueError(
            "Compiled submission contains duplicate ids; refusing to write CSV. "
            f"Examples: {duplicates}"
        )


def _read_bbox_lines(bbox_file: Path) -> list[str]:
    with open(bbox_file, "r", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Bbox output file is empty: {bbox_file}")
    return lines


def _compile_submission_from_outputs(
    args: argparse.Namespace,
    entries: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, list[Path]]], list[Path]]:
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, list[Path]]] = []
    present_files: list[Path] = []

    for key, entry in entries.items():
        candidates = _bbox_candidates_for_entry(
            outputs_dir=args.outputs_dir,
            entry=entry,
            output_layout=args.output_layout,
        )
        bbox_file = next((path for path in candidates if path.exists()), None)
        if bbox_file is None:
            missing.append((key, candidates))
            continue

        present_files.append(bbox_file)
        for frame_idx, line in enumerate(_read_bbox_lines(bbox_file)):
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid bbox line in {bbox_file!r} at frame {frame_idx}: "
                    f"expected 4 values, got {len(parts)} ({line!r})"
                )
            x, y, w, h = (float(part) for part in parts)
            rows.append(
                {
                    "id": _build_submission_id(args, key, entry, frame_idx),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    _validate_no_duplicate_ids(rows)
    return rows, missing, present_files


def _write_submission_csv(rows: list[dict[str, Any]], submission_csv: str | Path) -> None:
    output_path = Path(submission_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _load_entries(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    target_datasets = set(args.datasets) if args.datasets else _discover_datasets(args.data_dir)

    if args.run_split in {"public_lb", "train"}:
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = _build_manifest_entries(
            manifest=manifest,
            target_datasets=target_datasets,
            split_name=args.run_split,
        )
    elif args.run_split == "all":
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = {}
        entries.update(
            _build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="train",
                key_prefix="train/",
            )
        )
        entries.update(
            _build_manifest_entries(
                manifest=manifest,
                target_datasets=target_datasets,
                split_name="public_lb",
                key_prefix="test/",
            )
        )
    else:
        entries = _build_train_entries_from_csv(
            train_csv_path=args.train_csv_path,
            target_datasets=target_datasets,
            max_sequences=int(args.max_sequences),
        )

    entries = dict(
        sorted(
            entries.items(),
            key=lambda item: (str(item[1].get("dataset", "")), item[0]),
        )
    )
    entries = _filter_run_entries_by_video_key(entries, args.video_key)

    if args.max_sequences > 0 and args.run_split != "train_csv":
        entries = dict(list(entries.items())[: args.max_sequences])
    if not entries:
        raise RuntimeError(
            f"No entries selected for split={args.run_split} "
            f"and datasets={sorted(target_datasets)}"
        )
    return entries


def _print_missing_summary(missing: list[tuple[str, list[Path]]]) -> None:
    print(f"Missing bbox files: {len(missing)}")
    for key, candidates in missing[:10]:
        candidate_text = " | ".join(str(path) for path in candidates)
        print(f"  {key}: {candidate_text}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")


def main() -> None:
    args = parse_args()
    entries = _load_entries(args)
    rows, missing, present_files = _compile_submission_from_outputs(args, entries)

    if missing and not args.allow_missing:
        _print_missing_summary(missing)
        raise RuntimeError(
            "Some selected bbox files are missing, so the CSV was not written. "
            "Use --allow_missing if you intentionally want a partial CSV."
        )
    if not rows:
        raise RuntimeError(
            "No bbox rows were found. Check --outputs_dir, --run_split, "
            "--datasets, --video_key, and --output_layout."
        )

    _write_submission_csv(rows, args.submission_csv)

    print("Submission CSV rebuilt from saved outputs")
    print(f"  outputs_dir     : {Path(args.outputs_dir).resolve()}")
    print(f"  submission_csv  : {Path(args.submission_csv).resolve()}")
    print(f"  split           : {args.run_split}")
    print(f"  output_layout   : {args.output_layout}")
    print(f"  selected videos : {len(entries)}")
    print(f"  bbox files used : {len(present_files)}")
    print(f"  rows written    : {len(rows)}")
    if missing:
        print(f"  missing skipped : {len(missing)}")
    print("\nPreview:")
    for row in rows[:5]:
        print(f"  {row['id']},{row['x']},{row['y']},{row['w']},{row['h']}")


if __name__ == "__main__":
    main()
