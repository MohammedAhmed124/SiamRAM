"""
Download Module — SiamRAM Competition Submission.

Downloads all required model checkpoints from Google Drive:
  - model.pth              : SiamABC backbone checkpoint
  - yolo11n.pt             : YOLO re-detector weights
  - osnet_x0_25_imagenet.pth : OSNet ReID descriptor weights

All files are saved under ./checkpoints/.  Each download is skipped if a
valid (non-empty, non-HTML) file already exists at the destination.
"""

import os
import gdown


# ---------------------------------------------------------------------------
# Google Drive file IDs for each checkpoint
# ---------------------------------------------------------------------------
_GDRIVE_IDS = {
    "model.pth":                 "1VQdAZj0Mpf_ZMxvoZOaCRp3uo6wOPuDC",
    "yolo11n.pt":                "1WUAArjVjMwrluMWBBlTqGO7NBkDy_CMv",
    "osnet_x0_25_imagenet.pth":  "1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
}


def _is_valid_file(path: str) -> bool:
    """Return True if path exists, is at least 1 MB, and is not an HTML error page."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024 * 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return not (head.startswith(b"<!doctype html") or head.startswith(b"<html"))


def download_checkpoint(
    file_id: str,
    filename: str = "model.pth",
    checkpoint_dir: str = "./checkpoints",
) -> str:
    """
    Download a single checkpoint file from Google Drive if not already present.

    Args:
        file_id:        Google Drive file ID.
        filename:       Destination filename inside checkpoint_dir.
        checkpoint_dir: Directory where checkpoints are stored.

    Returns:
        Absolute path to the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    if _is_valid_file(checkpoint_path):
        print(f"[download] '{filename}' already exists — skipping.")
        return checkpoint_path

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] Downloading '{filename}' from Google Drive …")
    gdown.download(url, checkpoint_path, quiet=False)

    if not _is_valid_file(checkpoint_path):
        raise RuntimeError(
            f"Download of '{filename}' failed or produced an invalid file. "
            f"Check that the Google Drive link is publicly accessible."
        )

    print(f"[download] '{filename}' saved to '{checkpoint_path}'.")
    return checkpoint_path


def download_all_checkpoints(checkpoint_dir: str = "./checkpoints") -> dict:
    """
    Download all three required checkpoints.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.

    Returns:
        dict mapping filename → absolute path for each checkpoint.
    """
    paths = {}
    for filename, file_id in _GDRIVE_IDS.items():
        paths[filename] = download_checkpoint(
            file_id=file_id,
            filename=filename,
            checkpoint_dir=checkpoint_dir,
        )
    return paths


if __name__ == "__main__":
    download_all_checkpoints()
