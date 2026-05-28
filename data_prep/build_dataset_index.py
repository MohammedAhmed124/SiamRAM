"""
Generic dataset index builder for SiamRAM training.

This script discovers tracking sequences automatically from a data root,
extracts video frames alongside each sequence's video (under the data root by
default) when needed, then writes CSV indexes compatible with utils/dataset.py.

Frame extraction is skipped automatically for any sequence whose output
img/ folder already exists and contains JPEG frames.

Folder layout after extraction:
    data/
        dataset1/
            basketball/
                annotation.txt
                video.mp4
                img/
                    00000001.jpg
                    ...

CSV seq_path always points to the directory that contains the frame images
directly (i.e. the img/ folder itself), matching how TrackingSequence builds
frame paths via os.path.join(seq_path, frame_pattern.format(idx)).

Compatibility-required columns are guaranteed:
    seq_path, annot_path, start_idx, end_idx, n_frames, class, dataset

The CSV also includes an img_path column (same value as seq_path) and a
frame_pattern column so pattern auto-detection in TrackingSequence is skipped.

Typical usage:
    python data_prep/build_dataset_index.py
    python data_prep/build_dataset_index.py --data /path/to/data --imgs-root /path/to/data

Extra flags:
    --imgs-root /path   Where extracted frames are stored (default: --data root).
                        Pass an explicit path here to keep frames separate from
                        the source videos; otherwise frames land at
                        data/<dataset>/<seq>/img/.
    --skip-extraction   Skip video-to-frames step entirely
    --jpg-quality 95    JPEG quality for extracted frames (default: 95)
    --workers 4         Parallel workers for frame extraction (default: 4)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REQUIRED_COLUMNS = [
    "seq_path",
    "annot_path",
    "start_idx",
    "end_idx",
    "n_frames",
    "class",
    "dataset",
]


def _is_decodable_image(path: Path) -> bool:
    """
    Return True only if the image can be fully decoded.
    """
    try:
        with Image.open(path) as img:
            img.load()
            w, h = img.size
            return w > 0 and h > 0
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _partition_decodable_images(
    frames: List[Path],
) -> Tuple[List[Path], List[Path]]:
    good: List[Path] = []
    bad: List[Path] = []
    for frame in frames:
        if _is_decodable_image(frame):
            good.append(frame)
        else:
            bad.append(frame)
    return good, bad


def _cleanup_bad_images(
    bad_frames: List[Path],
) -> int:
    removed = 0
    for frame in bad_frames:
        try:
            frame.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _clear_frame_dir(
    img_dir: Path,
) -> None:
    """
    Remove extracted frame files (and temp leftovers) from img_dir.
    """
    if not img_dir.is_dir():
        return
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if p.suffix.lower() in IMAGE_EXTS or low.endswith(".tmp"):
            try:
                p.unlink()
            except OSError:
                continue


def _validate_frame_dir(
    img_dir: Path,
    delete_bad: bool = False,
) -> Tuple[int, int]:
    """
    Return (good_count, bad_count) for JPEG/PNG files in img_dir.
    Optionally deletes bad files.
    """
    frames = _image_files(img_dir)
    if not frames:
        return 0, 0
    good, bad = _partition_decodable_images(frames)
    if delete_bad and bad:
        _cleanup_bad_images(bad)
    return len(good), len(bad)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_data_dir(path_value: Optional[str]) -> Path:
    if not path_value:
        path_value = "data"

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


def _resolve_imgs_root(path_value: Optional[str], data_root: Path) -> Path:
    """
    Resolve the imgs root directory.
    Defaults to data_root itself so frames are extracted alongside their videos
    under data/<dataset>/<seq>/img/. Pass an explicit --imgs-root to redirect
    frames to a separate tree (legacy data_imgs/ layout).
    """
    if path_value:
        p = Path(path_value).expanduser()
        return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()
    return data_root


def _seq_img_dir(seq_dir: Path, dataset_root: Path, imgs_root: Path, dataset_name: str) -> Path:
    """
    Return the path where extracted frames for seq_dir should live inside imgs_root.

    Layout:  imgs_root / dataset_name / <relative path from dataset_root> / img
    Example: data / dataset1 / basketball / img
    """
    try:
        rel = seq_dir.relative_to(dataset_root)
    except ValueError:
        rel = Path(seq_dir.name)
    return imgs_root / dataset_name / rel / "img"


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------

def _find_video_file(seq_dir: Path) -> Optional[Path]:
    """Return the first video file found directly inside seq_dir."""
    for p in seq_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _probe_video(video_path: Path) -> bool:
    """
    Return True if ffprobe can successfully open and read the video file.
    Catches corrupt files (moov atom not found, truncated downloads, etc.)
    before ffmpeg wastes time on them.  Falls back to True if ffprobe is
    not installed so behaviour degrades gracefully.
    """
    if not shutil.which("ffprobe"):
        return True
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _extract_frames_ffmpeg(video_path: Path, img_dir: Path, jpg_quality: int) -> int:
    """
    Extract all frames from video_path into img_dir as zero-padded JPEGs.
    Returns the number of frames written, or 0 on failure.

    Assumes the caller has already validated the file via _probe_video —
    a single extraction attempt is therefore sufficient.
    """
    try:
        img_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  [warn] mkdir failed for {img_dir}: {exc}")
        return 0
    output_pattern = str(img_dir / "%08d.jpg")
    qscale = max(1, min(31, round(31 * (1.0 - (jpg_quality - 1) / 94.0))))

    cmd = [
        "ffmpeg", "-y",
        "-v", "error",
        "-fflags", "+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", str(video_path),
        "-pix_fmt", "yuvj420p",
        "-q:v", str(qscale),
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    extracted = len(list(img_dir.glob("*.jpg")))

    if extracted > 0:
        return extracted

    # ffmpeg wrote nothing — log the tail of stderr for diagnostics.
    print(
        f"  [warn] ffmpeg produced 0 frames from {video_path.name} "
        f"(rc={result.returncode}).\n"
        f"         {result.stderr.strip()[-300:]}"
    )
    return 0


def _extract_frames_cv2(video_path: Path, img_dir: Path, jpg_quality: int) -> int:
    """
    Fallback frame extractor using OpenCV.
    Returns the number of frames extracted, or 0 on failure.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        print("[error] Neither ffmpeg nor opencv-python is available.")
        return 0

    try:
        img_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  [warn] mkdir failed for {img_dir}: {exc}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] cv2 could not open {video_path.name}")
        return 0

    idx = 1
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = img_dir / f"{idx:08d}.jpg"
            tmp_path = img_dir / f".{idx:08d}.jpg.tmp"
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                print(f"  [warn] cv2 failed to encode frame {idx} for {video_path.name}")
                break
            try:
                with open(tmp_path, "wb") as f:
                    f.write(encoded.tobytes())
                tmp_path.replace(out_path)
            except OSError as exc:
                print(
                    f"  [warn] cv2 failed to atomically write frame {idx} "
                    f"for {video_path.name}: {exc}"
                )
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                print(f"  [warn] cv2 failed to write frame {idx} for {video_path.name}")
                break
            idx += 1
    except OSError as exc:
        # Common on unstable mounts / external disks / Docker bind mounts.
        print(f"  [warn] cv2 extraction I/O error for {video_path.name}: {exc}")
        return 0
    except Exception as exc:
        print(f"  [warn] cv2 extraction error for {video_path.name}: {exc}")
        return 0
    cap.release()

    extracted = idx - 1
    if extracted == 0:
        print(f"  [warn] cv2 opened {video_path.name} but read 0 frames.")
    return extracted


def extract_video_to_frames(
    seq_dir: Path,
    img_out_dir: Path,
    jpg_quality: int = 95,
    force_reextract: bool = False,
    validate_frames: bool = True,
) -> bool:
    """
    Extract frames from a video inside seq_dir into img_out_dir.

    Skips entirely if img_out_dir already contains JPEG frames.
    Validates the video with ffprobe before attempting extraction so corrupt
    files are rejected immediately with a single clean warning.

    Returns True if frames are available after the call.
    """
    if img_out_dir.is_dir():
        existing = _image_files(img_out_dir)
        if existing:
            if force_reextract:
                _clear_frame_dir(img_out_dir)
            elif validate_frames:
                good_count, bad_count = _validate_frame_dir(
                    img_out_dir, delete_bad=True
                )
                if bad_count > 0:
                    print(
                        f"  [warn] found {bad_count} corrupted extracted frames in "
                        f"{img_out_dir}; re-extracting sequence."
                    )
                    _clear_frame_dir(img_out_dir)
                elif good_count > 0:
                    return True
            else:
                return True

    video = _find_video_file(seq_dir)
    if video is None:
        return False

    try:
        if _ffmpeg_available():
            if not _probe_video(video):
                print(f"  [skip] {video.name}: corrupt or unreadable — skipping extraction")
                return False
            n = _extract_frames_ffmpeg(video, img_out_dir, jpg_quality)
        else:
            n = _extract_frames_cv2(video, img_out_dir, jpg_quality)
    except OSError as exc:
        print(f"  [warn] extraction I/O error for {video.name}: {exc}")
        return False
    except Exception as exc:
        print(f"  [warn] extraction failed for {video.name}: {exc}")
        return False

    if n <= 0:
        return False

    if validate_frames:
        good_count, bad_count = _validate_frame_dir(img_out_dir, delete_bad=True)
        if bad_count > 0:
            print(
                f"  [warn] removed {bad_count} corrupted frames after extraction "
                f"for {video.name}"
            )
        return good_count > 0

    return True


def extract_all_videos(
    dataset_root: Path,
    imgs_root: Path,
    dataset_name: str,
    jpg_quality: int = 95,
    workers: int = 4,
    force_reextract: bool = False,
    validate_frames: bool = True,
) -> None:
    """
    Walk dataset_root, find every sequence that contains a video whose frames
    have NOT yet been extracted to imgs_root, and extract them in parallel.
    """
    pending: List[Tuple[Path, Path]] = []

    for p in dataset_root.rglob("*"):
        if not p.is_dir():
            continue
        if _find_video_file(p) is None:
            continue

        img_out_dir = _seq_img_dir(p, dataset_root, imgs_root, dataset_name)

        if img_out_dir.is_dir() and not force_reextract:
            good_count, bad_count = _validate_frame_dir(
                img_out_dir, delete_bad=validate_frames
            )
            if good_count > 0 and bad_count == 0:
                continue

        pending.append((p, img_out_dir))

    if not pending:
        print(f"  [extract] All frames already present — skipping extraction for {dataset_name}")
        return

    backend = "ffmpeg" if _ffmpeg_available() else "cv2"
    print(
        f"  [extract] {len(pending)} sequences to extract "
        f"→ {imgs_root / dataset_name}  [{backend}, workers={workers}]"
    )

    failed: List[Path] = []

    def _worker(args: Tuple[Path, Path]) -> Tuple[Path, bool]:
        seq_dir, img_out_dir = args
        ok = extract_video_to_frames(
            seq_dir,
            img_out_dir,
            jpg_quality=jpg_quality,
            force_reextract=force_reextract,
            validate_frames=validate_frames,
        )
        return seq_dir, ok

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, item): item for item in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                seq_dir, ok = fut.result()
                if not ok:
                    failed.append(seq_dir)
            except OSError as exc:
                # Keep going when a worker fails due to storage/mount errors.
                seq_dir, _ = futures[fut]
                failed.append(seq_dir)
                print(f"  [warn] worker OSError on {seq_dir}: {exc}")
            except Exception as exc:
                seq_dir, _ = futures[fut]
                failed.append(seq_dir)
                print(f"  [warn] worker failed on {seq_dir}: {exc}")

    if failed:
        print(f"  [warn] Extraction failed or skipped for {len(failed)} sequences:")
        for d in failed[:10]:
            print(f"    {d}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


# ---------------------------------------------------------------------------
# Image / annotation helpers
# ---------------------------------------------------------------------------

def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _annotation_len(annot_path: Path) -> int:
    try:
        lines = [
            ln.strip()
            for ln in annot_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        ]
    except OSError:
        return 0
    return sum(1 for ln in lines if ln)


def _extract_numeric_stem(stem: str) -> Optional[int]:
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def _frame_indices(frames: Iterable[Path]) -> List[int]:
    out: List[int] = []
    for frame in frames:
        idx = _extract_numeric_stem(frame.stem)
        if idx is not None:
            out.append(idx)
    return sorted(out)


def _contiguous_end_index(
    frame_idxs: List[int],
    start_idx: int,
    max_end_idx: int,
) -> int:
    """
    Return the last contiguous frame index starting from start_idx, capped by max_end_idx.
    """
    frame_set = set(frame_idxs)
    end_idx = start_idx
    while (end_idx + 1) in frame_set and (end_idx + 1) <= max_end_idx:
        end_idx += 1
    return min(end_idx, max_end_idx)


def _image_size(first_image: Path) -> Tuple[int, int]:
    with Image.open(first_image) as img:
        return img.size  # (W, H)


def _first_valid_image_size(frames: List[Path]) -> Optional[Tuple[int, int]]:
    """
    Return size from the first decodable image in `frames`.
    Corrupt images are skipped.
    """
    for frame in frames:
        try:
            return _image_size(frame)
        except (UnidentifiedImageError, OSError):
            continue
    return None


def _infer_frame_pattern(img_dir: Path) -> Optional[str]:
    """
    Infer the zero-padded frame filename format from the files present in
    img_dir.  Returns a format string like ``{:08d}.jpg`` that is relative
    to img_dir itself — matching the contract that seq_path == img_dir in
    the output CSV, so TrackingSequence can do
    ``os.path.join(seq_path, pattern.format(idx))`` directly.
    """
    frames = _image_files(img_dir)
    if not frames:
        return None
    stem = frames[0].stem
    if not stem.isdigit():
        return None
    width = len(stem)
    ext = frames[0].suffix.lower()
    return f"{{:0{width}d}}{ext}"


def _clean_class_name(name: str) -> str:
    cls = re.sub(r"[-_]?\d+$", "", name)
    return cls if cls else name


def _find_annotation_files(dataset_root: Path) -> List[Path]:
    annots = []
    for p in dataset_root.rglob("*.txt"):
        low = p.name.lower()
        if low in {"readme.txt", "license.txt"}:
            continue
        annots.append(p)
    return sorted(annots)


def _candidate_seq_dirs_from_annot(annot_path: Path, dataset_root: Path) -> List[Path]:
    stem = annot_path.stem
    candidates: List[Path] = []

    for base in [annot_path.parent, annot_path.parent.parent, dataset_root]:
        if not base.exists():
            continue
        for child in [
            base / stem,
            base / "data_seq" / stem,
            base / "seq" / stem,
            base / "sequences" / stem,
        ]:
            if child.is_dir():
                candidates.append(child)

    if (annot_path.parent / "img").is_dir():
        candidates.append(annot_path.parent)

    if annot_path.parent.is_dir():
        candidates.append(annot_path.parent)

    for p in dataset_root.rglob(stem):
        if p.is_dir():
            candidates.append(p)

    seen: set = set()
    uniq: List[Path] = []
    for c in candidates:
        rc = c.resolve()
        if rc not in seen:
            seen.add(rc)
            uniq.append(c)
    return uniq


def _resolve_frame_dir(
    seq_dir: Path,
    dataset_root: Path,
    imgs_root: Path,
    dataset_name: str,
) -> Path:
    """
    Return the directory that actually contains the frame images for seq_dir.

    Priority:
      1. Extracted frames in imgs_root (data/<ds>/<seq>/img/ by default)
      2. Legacy img/ subfolder inside seq_dir
      3. seq_dir itself (frames at the sequence root)
    """
    imgs_dir = _seq_img_dir(seq_dir, dataset_root, imgs_root, dataset_name)
    if imgs_dir.is_dir() and len(_image_files(imgs_dir)) > 0:
        return imgs_dir

    legacy_img = seq_dir / "img"
    if legacy_img.is_dir() and len(_image_files(legacy_img)) > 0:
        return legacy_img

    return seq_dir


def _pick_best_seq_dir(
    candidates: List[Path],
    dataset_root: Path,
    imgs_root: Path,
    dataset_name: str,
) -> Optional[Path]:
    best: Optional[Path] = None
    best_count = 0
    for c in candidates:
        frame_dir = _resolve_frame_dir(c, dataset_root, imgs_root, dataset_name)
        count, _ = _validate_frame_dir(frame_dir, delete_bad=False)
        if count > best_count:
            best_count = count
            best = c
    return best if best_count > 0 else None


def _build_record(
    dataset_name: str,
    seq_dir: Path,
    annot_path: Path,
    dataset_root: Path,
    imgs_root: Path,
) -> Optional[Dict[str, object]]:
    """
    Build one CSV row for a sequence.

    ``seq_path`` is set to ``frame_dir`` — the directory that contains the
    frame images — so that ``TrackingSequence.frame_path()`` resolves to a
    real file via ``os.path.join(seq_path, frame_pattern.format(idx))``.
    ``frame_pattern`` is inferred relative to that same directory.
    """
    frame_dir = _resolve_frame_dir(seq_dir, dataset_root, imgs_root, dataset_name)
    frames_all = _image_files(frame_dir)
    if not frames_all:
        return None
    frames, bad = _partition_decodable_images(frames_all)
    if bad:
        removed = _cleanup_bad_images(bad)
        print(
            f"  [warn] removed {removed} corrupted frame files while indexing: {frame_dir}"
        )
    if not frames:
        return None

    frame_idxs = _frame_indices(frames)
    if not frame_idxs:
        return None

    ann_len = _annotation_len(annot_path)
    if ann_len <= 0:
        return None

    start_idx = frame_idxs[0]
    end_idx_from_ann = start_idx + ann_len - 1
    end_idx = _contiguous_end_index(
        frame_idxs=frame_idxs,
        start_idx=start_idx,
        max_end_idx=end_idx_from_ann,
    )
    # Keep index range aligned to both annotations and actually present frames.
    n_frames = end_idx - start_idx + 1
    if n_frames <= 0:
        return None

    size = _first_valid_image_size(frames)
    if size is None:
        print(f"  [warn] no decodable frames for sequence: {frame_dir}")
        return None
    w, h = size
    class_name = _clean_class_name(seq_dir.name)

    return {
        # seq_path == frame_dir so TrackingSequence.frame_path() works directly.
        "seq_path":      str(frame_dir.resolve()),
        "img_path":      str(frame_dir.resolve()),
        "annot_path":    str(annot_path.resolve()),
        "start_idx":     int(start_idx),
        "end_idx":       int(end_idx),
        "n_frames":      int(n_frames),
        "ann_len":       int(ann_len),
        "H":             int(h),
        "W":             int(w),
        "class":         class_name,
        "dataset":       dataset_name,
        # Pattern is relative to seq_path (== frame_dir); no subdir prefix needed.
        "frame_pattern": _infer_frame_pattern(frame_dir),
    }


# ---------------------------------------------------------------------------
# Compatibility filter
# ---------------------------------------------------------------------------

def _compat_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    before = len(df)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=REQUIRED_COLUMNS)

    for col in ["start_idx", "end_idx", "n_frames"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start_idx", "end_idx", "n_frames"])

    df["start_idx"] = df["start_idx"].astype(int)
    df["end_idx"]   = df["end_idx"].astype(int)
    df = df[df["end_idx"] >= df["start_idx"]]
    df["n_frames"]  = (df["end_idx"] - df["start_idx"] + 1).astype(int)
    df = df[df["n_frames"] > 1]

    for col in ["seq_path", "annot_path", "class", "dataset"]:
        df[col] = df[col].astype(str)

    dropped = before - len(df)
    return df.reset_index(drop=True), dropped


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_dataset_rows(
    dataset_root: Path,
    dataset_name: str,
    imgs_root: Path,
) -> pd.DataFrame:
    annot_files = _find_annotation_files(dataset_root)
    rows: List[Dict[str, object]] = []
    used_seq_paths: set = set()

    for annot in tqdm(annot_files, desc=f"Indexing {dataset_name}"):
        candidates = _candidate_seq_dirs_from_annot(annot, dataset_root)
        seq_dir = _pick_best_seq_dir(candidates, dataset_root, imgs_root, dataset_name)
        if seq_dir is None:
            continue

        seq_key = str(seq_dir.resolve())
        if seq_key in used_seq_paths:
            continue

        rec = _build_record(dataset_name, seq_dir, annot, dataset_root, imgs_root)
        if rec is None:
            continue

        rows.append(rec)
        used_seq_paths.add(seq_key)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a generic training dataset index CSV. "
            "Extracts video frames alongside each sequence (under --data by default) "
            "and skips sequences whose frames are already present."
        )
    )
    p.add_argument("--data", type=str, default="data",
                   help="Data root containing dataset subfolders (auto-resolves data/dataset).")
    p.add_argument("--imgs-root", type=str, default=None,
                   help="Root where extracted frames are stored. "
                        "Default: same as --data, producing data/<ds>/<seq>/img/. "
                        "Set this explicitly to keep frames in a separate tree.")
    p.add_argument("--output", type=Path, default=None,
                   help="Directory for CSV outputs (default: same as --data).")
    p.add_argument("--combined-name", type=str, default="train_dataframe.csv",
                   help="Combined CSV filename.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Specific dataset folder names to include. "
                        "Default: all subfolders except metadata.")
    p.add_argument("--skip-extraction", action="store_true",
                   help="Skip video-to-frames extraction entirely.")
    p.add_argument("--jpg-quality", type=int, default=95,
                   help="JPEG quality for extracted frames (1-95, default 95).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for frame extraction (default 4).")
    p.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-extract frames even when img/ already contains frames.",
    )
    p.add_argument(
        "--no-validate-frames",
        action="store_true",
        help="Disable post-extraction frame decode validation (faster, less safe).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root  = _resolve_data_dir(args.data)
    imgs_root  = _resolve_imgs_root(args.imgs_root, data_root)
    output_dir = args.output.resolve() if args.output else data_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if args.datasets:
        dataset_dirs = [d for d in args.datasets if (data_root / d).is_dir()]
    else:
        dataset_dirs = [
            p.name for p in data_root.iterdir()
            if p.is_dir() and p.name != "metadata"
        ]

    dataset_dirs = sorted(dataset_dirs)

    print(f"Data root  : {data_root}")
    print(f"Imgs root  : {imgs_root}  ← extracted frames go here")
    print(f"Output dir : {output_dir}")
    print(f"Datasets   : {dataset_dirs}")
    if not args.skip_extraction:
        backend = "ffmpeg" if _ffmpeg_available() else "cv2 (ffmpeg not found)"
        print(f"Extraction : enabled [{backend}]  (skipped per-sequence if frames exist)")
        if args.force_reextract:
            print("             force re-extract: ON")
        if not args.no_validate_frames:
            print("             frame validation: ON (corrupted outputs removed/retried)")
        else:
            print("             frame validation: OFF")
    else:
        print("Extraction : skipped (--skip-extraction)")

    all_frames: List[pd.DataFrame] = []
    partial_combined_path = output_dir / f"{args.combined_name}.partial"

    interrupted = False
    try:
        for ds_name in dataset_dirs:
            ds_root = data_root / ds_name
            print(f"\n── {ds_name} ──────────────────────────")

            if not args.skip_extraction:
                extract_all_videos(
                    ds_root,
                    imgs_root=imgs_root,
                    dataset_name=ds_name,
                    jpg_quality=args.jpg_quality,
                    workers=args.workers,
                    force_reextract=args.force_reextract,
                    validate_frames=not args.no_validate_frames,
                )

            df = discover_dataset_rows(ds_root, ds_name, imgs_root=imgs_root)
            df, dropped = _compat_filter(df)

            out_path = output_dir / f"{ds_name}.csv"
            df.to_csv(out_path, index=False)

            print(f"  sequences      : {len(df)}")
            print(f"  unique classes : {df['class'].nunique() if not df.empty else 0}")
            print(f"  total frames   : {int(df['n_frames'].sum()) if not df.empty else 0:,}")
            if dropped:
                print(f"  dropped rows   : {dropped}")
            print(f"  saved to       : {out_path}")

            if not df.empty:
                all_frames.append(df)
                partial = pd.concat(all_frames, ignore_index=True, sort=False)
                partial, _ = _compat_filter(partial)
                partial.to_csv(partial_combined_path, index=False)
                print(f"  partial combined: {partial_combined_path} ({len(partial)} rows)")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[warn] Interrupted. Keeping the latest partial CSV.")

    if all_frames and not interrupted:
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        combined, dropped = _compat_filter(combined)
        combined_path = output_dir / args.combined_name
        combined.to_csv(combined_path, index=False)
        if partial_combined_path.exists():
            partial_combined_path.unlink()

        print("\n── Combined ───────────────────────────")
        print(f"  rows         : {len(combined)}")
        if dropped:
            print(f"  dropped rows : {dropped}")
        print(f"  saved to     : {combined_path}")
    elif all_frames:
        print(f"[warn] Partial combined CSV preserved at: {partial_combined_path}")
    else:
        print("\n[warn] No sequences discovered; combined CSV was not written.")


if __name__ == "__main__":
    main()
