#!/usr/bin/env python3
"""
Run SiamRAM inference for exactly one video, selected by numeric id or key.

This wrapper auto-builds a mapping table:
    video_id (int) <-> video_key (manifest key)

Then it resolves your input and runs `run_inference.py` with fixed defaults.

Usage:
    python run_single_video.py 42
    python run_single_video.py dataset5/person19_3
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = BASE_DIR / "config" / "inference_config_experimental.yaml"
DEFAULT_MANIFEST = BASE_DIR / "data" / "metadata" / "contestant_manifest.json"
DEFAULT_ID_MAP = BASE_DIR / "data" / "metadata" / "video_id_map.csv"


def _slugify_video_key(video_key: str) -> str:
    return video_key.strip().replace("\\", "_").replace("/", "_")


def _build_video_id_map(
    manifest_path: Path,
    id_map_path: Path,
) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key_to_splits: dict[str, set[str]] = {}

    for split_name in ("public_lb", "train"):
        split_entries = manifest.get(split_name, {})
        if not isinstance(split_entries, dict):
            continue
        for key in split_entries.keys():
            key_to_splits.setdefault(str(key), set()).add(split_name)

    rows: list[dict[str, str]] = []
    for idx, key in enumerate(sorted(key_to_splits.keys()), start=1):
        rows.append(
            {
                "video_id": str(idx),
                "video_key": key,
                "splits": ",".join(sorted(key_to_splits[key])),
            }
        )

    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    with id_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "video_key", "splits"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _resolve_video_key(
    video_ref: str,
    rows: list[dict[str, str]],
) -> str:
    ref = video_ref.strip()
    by_key = {row["video_key"]: row for row in rows}
    by_id = {row["video_id"]: row for row in rows}

    if ref.isdigit():
        ref = str(int(ref))

    if ref in by_id:
        return by_id[ref]["video_key"]

    if ref in by_key:
        return ref

    raise ValueError(
        f"Unknown video reference: {video_ref!r}. "
        "Pass either a numeric video_id from data/metadata/video_id_map.csv "
        "or an exact video_key."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one video (by numeric id or manifest key) with experimental SiamRAM config."
    )
    parser.add_argument(
        "video_ref",
        help="Numeric video_id (from data/metadata/video_id_map.csv) or a manifest key.",
    )
    args = parser.parse_args()

    rows = _build_video_id_map(DEFAULT_MANIFEST, DEFAULT_ID_MAP)
    video_key = _resolve_video_key(args.video_ref, rows)
    key_slug = _slugify_video_key(video_key)

    # Everything for single-video runs lands under one fixed, tidy folder:
    #   outputs/run_one_video/<video_name>/   ← annotated video + bbox .txt
    # The "video" output layout drops the dataset folder so the result is easy
    # to reach without clicking through nested dataset/split directories.
    outputs_dir = BASE_DIR / "outputs" / "run_one_video"
    submission_csv = outputs_dir / f"submission_{key_slug}.csv"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BASE_DIR / "run_inference.py"),
        "--yaml_config_path",
        str(DEFAULT_CFG),
        "--run_split",
        "all",
        "--video_key",
        video_key,
        "--output_video",
        "--outputs_dir",
        str(outputs_dir),
        "--output_layout",
        "video",
        "--submission_csv",
        str(submission_csv),
    ]

    print("Running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"Resolved video_key: {video_key}")
    print(f"Video id map: {DEFAULT_ID_MAP}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Submission CSV: {submission_csv}")
    completed = subprocess.run(cmd, cwd=str(BASE_DIR))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
