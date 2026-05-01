#!/usr/bin/env python3
"""Download SiamRAM checkpoints from Google Drive into this folder.

Usage:
  python checkpoints/download_checkpoints.py
  python checkpoints/download_checkpoints.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gdown

FILES: dict[str, str] = {
    "head_epoch_000.pth": "1VQdAZj0Mpf_ZMxvoZOaCRp3uo6wOPuDC",
    "yolo11n.pt": "1WUAArjVjMwrluMWBBlTqGO7NBkDy_CMv",
}
MIN_BYTES: dict[str, int] = {
    "head_epoch_000.pth": 1_000_000,
    "yolo11n.pt": 1_000_000,
}

CHUNK_SIZE = 1024 * 1024  # 1MB


def _looks_like_html(data: bytes) -> bool:
    prefix = data[:512].lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _is_valid_checkpoint(path: Path, min_bytes: int) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as f:
        head = f.read(512)
    return not _looks_like_html(head)


def _download_file(file_name: str, file_id: str, out_dir: Path, force: bool) -> None:
    destination = out_dir / file_name
    if destination.exists() and _is_valid_checkpoint(destination, MIN_BYTES[file_name]) and not force:
        print(f"[skip] {file_name} already exists")
        return

    print(f"[downloading] {file_name}")
    if destination.exists():
        destination.unlink()
    url = f"https://drive.google.com/uc?id={file_id}"
    output = gdown.download(url=url, output=str(destination), quiet=False)
    if not output:
        raise RuntimeError(f"gdown failed to download {file_name}.")

    if not _is_valid_checkpoint(destination, MIN_BYTES[file_name]):
        raise RuntimeError(
            f"Downloaded file looks invalid for {file_name} "
            f"(size={destination.stat().st_size} bytes; likely an HTML response)."
        )

    print(f"[done] {file_name} -> {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SiamRAM checkpoints")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(__file__).resolve().parent

    try:
        for file_name, file_id in FILES.items():
            _download_file(file_name, file_id, out_dir, force=args.force)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
