"""
Checkpoint downloader for the SiamRAM submission.

The tracker needs three weight files to run, and we host them on Google Drive
so the submission stays small:
  - model.pth                : the SiamABC backbone weights
  - yolo11n.pt               : the YOLO re-detector weights
  - osnet_x0_25_imagenet.pth : the OSNet appearance-descriptor weights

Everything lands in ./checkpoints/. If a file is already there and looks valid
(big enough and not an HTML error page), we skip downloading it again, so
re-running the submission does not re-fetch gigabytes for no reason.
"""

import os

import gdown

# ---------------------------------------------------------------------------
# The Google Drive file ID for each checkpoint, keyed by the filename we save
# it as. To swap a checkpoint, change the ID here.
# ---------------------------------------------------------------------------
_GDRIVE_IDS = {
    "model.pth":                 "1VQdAZj0Mpf_ZMxvoZOaCRp3uo6wOPuDC",
    "yolo11n.pt":                "1WUAArjVjMwrluMWBBlTqGO7NBkDy_CMv",
    "osnet_x0_25_imagenet.pth":  "1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
}


def _is_valid_file(path: str) -> bool:
    """
    Decide whether a file on disk is a real checkpoint we can reuse.

    We check three things: the file exists, it is at least 1 MB (a truncated or
    empty download is smaller than any real checkpoint), and it does not start
    like an HTML page. That last check matters because when a Google Drive link
    is wrong or rate-limited, gdown sometimes saves the error web page instead of
    the actual file, and that page would otherwise look like a valid download.
    """
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
    Download one checkpoint from Google Drive, unless we already have it.

    Args:
        file_id:        the Google Drive file ID to fetch.
        filename:       what to name the file inside checkpoint_dir.
        checkpoint_dir: the folder to save it in.

    Returns:
        The path to the checkpoint file on disk.

    If a valid copy is already there we return its path straight away. Otherwise
    we download it and then check it really arrived; if the download produced an
    invalid file we raise, because it almost always means the Drive link is not
    public and silently carrying on would just fail later in a more confusing way.
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
    Download all three checkpoints the tracker needs.

    Args:
        checkpoint_dir: the folder to save the checkpoints in.

    Returns:
        A dict mapping each filename to its path on disk.
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
