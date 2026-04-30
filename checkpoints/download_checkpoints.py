#!/usr/bin/env python3
"""Download SiamRAM checkpoints from Google Drive into this folder.

Usage:
  python checkpoints/download_checkpoints.py
  python checkpoints/download_checkpoints.py --force
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FILES: dict[str, str] = {
    "head_epoch_000.pth": "1ehwIYvwwIulWNAshYVWm5ashkMmRtcik",
    "yolo11n.pt": "1WUAArjVjMwrluMWBBlTqGO7NBkDy_CMv",
}

BASE_URL = "https://drive.google.com/uc"
CHUNK_SIZE = 1024 * 1024  # 1MB


def _extract_confirm_token(html: str) -> str | None:
    patterns = [
        r"confirm=([0-9A-Za-z_]+)",
        r'name="confirm"\s+value="([0-9A-Za-z_]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def _open_drive_stream(file_id: str):
    params = {"export": "download", "id": file_id}
    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(req, timeout=120)

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        return response

    html = response.read().decode("utf-8", errors="ignore")
    token = _extract_confirm_token(html)
    if not token:
        raise RuntimeError("Google Drive confirmation token was not found.")

    params["confirm"] = token
    confirm_url = f"{BASE_URL}?{urlencode(params)}"
    confirm_req = Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
    return urlopen(confirm_req, timeout=120)


def _download_file(file_name: str, file_id: str, out_dir: Path, force: bool) -> None:
    destination = out_dir / file_name
    if destination.exists() and not force:
        print(f"[skip] {file_name} already exists")
        return

    print(f"[downloading] {file_name}")
    with _open_drive_stream(file_id) as response, destination.open("wb") as f:
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            print(f"  {downloaded / (1024 * 1024):.1f} MB", end="\r", flush=True)

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
