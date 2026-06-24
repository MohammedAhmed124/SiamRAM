"""
generate_manifest.py
====================
Scan a data directory for videos + their annotation .txt files and write a
manifest JSON in the exact format inference.py expects:

    {
      "<split>": {
        "<group>_<sequence>": {
          "video_path": "<group>/<sequence>/<sequence>.mp4",
          "annotation_path": "<group>/<sequence>/annotation_first_box.txt",
          "n_frames": 241
        },
        ...
      }
    }

Layout assumed (one video per sequence folder):

    <data_dir>/<group>/<sequence>/<sequence>.mp4
    <data_dir>/<group>/<sequence>/annotation_first_box.txt

The sequence key is the video's folder path relative to <data_dir> with the
separators turned into "_" (so "dataset1/basketball_4" -> "dataset1_basketball_4",
"hidden/1" -> "hidden_1"). That key is what inference.py uses as the submission-id
prefix ("<key>_<frame_idx>"), so it must match how you want the CSV ids to read.

inference.py opens video_path/annotation_path with OpenCV using whatever string
is in the manifest, so the paths must resolve from the directory you launch
inference from. By default paths are written relative to <data_dir> (run
inference from there); pass --absolute to write absolute paths that work from
anywhere.

Usage:
    python generate_manifest.py <data_dir> <output.json>
    python generate_manifest.py ./data manifest.json --split public_lb
    python generate_manifest.py ./data manifest.json --absolute --accurate-frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def find_annotation(folder: Path, preferred_name: str) -> Path | None:
    """Find the annotation .txt for a sequence folder.

    Tries the preferred filename first, then any ``annotation*.txt``, then a
    lone ``.txt`` if the folder has exactly one. Returns None if nothing fits.
    """
    exact = folder / preferred_name
    if exact.is_file():
        return exact
    annotations = sorted(folder.glob("annotation*.txt"))
    if annotations:
        return annotations[0]
    txts = sorted(folder.glob("*.txt"))
    if len(txts) == 1:
        return txts[0]
    return None


def count_frames(video_path: Path, accurate: bool) -> int:
    """Return the video's frame count.

    Uses OpenCV's metadata (fast) by default. With *accurate*, or when the
    metadata is unusable, it decodes the whole clip and counts. inference.py
    ignores n_frames (it reads until the stream ends), so the fast value is
    fine for picking a manifest; accuracy is only here for completeness.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    try:
        if not accurate:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n > 0:
                return n
        # Accurate (or metadata gave <= 0): decode and count.
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        return n
    finally:
        cap.release()


def build_manifest(
    data_dir: Path,
    annotation_name: str,
    absolute: bool,
    accurate_frames: bool,
) -> tuple[dict, list[str]]:
    """Walk *data_dir* and return (sequences_dict, warnings)."""
    sequences: dict[str, dict] = {}
    warnings: list[str] = []

    videos = sorted(
        p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS
    )

    for video in videos:
        folder = video.parent
        annotation = find_annotation(folder, annotation_name)
        if annotation is None:
            warnings.append(f"no annotation .txt next to {video} - skipped")
            continue

        # Key = video's folder relative to data_dir, separators -> "_".
        rel_folder = folder.relative_to(data_dir)
        key = "_".join(rel_folder.parts) if rel_folder.parts else video.stem
        if key in sequences:  # two videos share a folder — disambiguate
            key = f"{key}_{video.stem}"

        if absolute:
            video_str = str(video.resolve())
            annotation_str = str(annotation.resolve())
        else:
            video_str = video.relative_to(data_dir).as_posix()
            annotation_str = annotation.relative_to(data_dir).as_posix()

        sequences[key] = {
            "video_path": video_str,
            "annotation_path": annotation_str,
            "n_frames": count_frames(video, accurate_frames),
        }

    return dict(sorted(sequences.items())), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="directory to scan for videos")
    parser.add_argument("output", type=Path, help="manifest JSON to write")
    parser.add_argument("--split", default="public_lb", help="top-level split key (default: public_lb)")
    parser.add_argument(
        "--annotation-name",
        default="annotation_first_box.txt",
        help="preferred annotation filename (default: annotation_first_box.txt)",
    )
    parser.add_argument("--absolute", action="store_true", help="write absolute paths instead of paths relative to data_dir")
    parser.add_argument("--accurate-frames", action="store_true", help="decode every video to count frames exactly (slower)")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"data_dir not found or not a directory: {args.data_dir}")

    sequences, warnings = build_manifest(
        data_dir=args.data_dir,
        annotation_name=args.annotation_name,
        absolute=args.absolute,
        accurate_frames=args.accurate_frames,
    )

    for warning in warnings:
        print(f"  WARN: {warning}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({args.split: sequences}, f, indent=2)

    print(f"Wrote {len(sequences)} sequences to {args.output} under split '{args.split}'.")


if __name__ == "__main__":
    main()
