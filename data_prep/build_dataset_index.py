"""
Generic dataset index builder for SiamRAM training.

This script discovers tracking sequences automatically from a data root,
then writes CSV indexes compatible with utils/dataset.py.

Compatibility-required columns are guaranteed:
    seq_path, annot_path, start_idx, end_idx, n_frames, class, dataset

Typical usage:
    python data_prep/build_dataset_index.py
    python data_prep/build_dataset_index.py --data /path/to/data --output /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
REQUIRED_COLUMNS = [
    "seq_path",
    "annot_path",
    "start_idx",
    "end_idx",
    "n_frames",
    "class",
    "dataset",
]


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


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _annotation_len(annot_path: Path) -> int:
    try:
        lines = [ln.strip() for ln in annot_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
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


def _image_size(first_image: Path) -> Tuple[int, int]:
    with Image.open(first_image) as img:
        return img.size


def _infer_frame_pattern(seq_path: Path) -> Optional[str]:
    for sub in (Path(""), Path("img")):
        base = seq_path / sub if str(sub) else seq_path
        frames = _image_files(base)
        if not frames:
            continue
        stem = frames[0].stem
        if not stem.isdigit():
            continue
        width = len(stem)
        ext = frames[0].suffix.lower()
        fmt = f"{{:0{width}d}}{ext}"
        return f"{sub.as_posix()}/{fmt}" if str(sub) else fmt
    return None


def _clean_class_name(name: str) -> str:
    cls = re.sub(r"[-_]?\d+$", "", name)
    return cls if cls else name


def _find_annotation_files(dataset_root: Path) -> List[Path]:
    annots = []
    for p in dataset_root.rglob("*.txt"):
        low = p.name.lower()
        # exclude likely metadata/docs text files
        if low in {"readme.txt", "license.txt"}:
            continue
        annots.append(p)
    return sorted(annots)


def _candidate_seq_dirs_from_annot(annot_path: Path, dataset_root: Path) -> List[Path]:
    stem = annot_path.stem
    candidates: List[Path] = []

    # common direct sibling patterns
    for base in [annot_path.parent, annot_path.parent.parent, dataset_root]:
        if not base.exists():
            continue
        for child in [base / stem, base / "data_seq" / stem, base / "seq" / stem, base / "sequences" / stem]:
            if child.is_dir():
                candidates.append(child)

    # parent sequence with groundtruth file inside sequence folder
    if (annot_path.parent / "img").is_dir():
        candidates.append(annot_path.parent)

    # fallback scan: exact directory name match anywhere under dataset root
    for p in dataset_root.rglob(stem):
        if p.is_dir():
            candidates.append(p)

    # unique preserve order
    seen = set()
    uniq = []
    for c in candidates:
        rc = c.resolve()
        if rc not in seen:
            seen.add(rc)
            uniq.append(rc)
    return uniq


def _pick_best_seq_dir(candidates: List[Path]) -> Optional[Path]:
    best: Optional[Path] = None
    best_count = 0
    for c in candidates:
        for d in [c, c / "img"]:
            frames = _image_files(d)
            if len(frames) > best_count:
                best_count = len(frames)
                best = c
    return best if best_count > 0 else None


def _build_record(dataset_name: str, seq_dir: Path, annot_path: Path) -> Optional[Dict[str, object]]:
    frame_dir = seq_dir / "img" if (seq_dir / "img").is_dir() else seq_dir
    frames = _image_files(frame_dir)
    if not frames:
        return None

    frame_idxs = _frame_indices(frames)
    if not frame_idxs:
        return None

    ann_len = _annotation_len(annot_path)
    if ann_len <= 0:
        return None

    start_idx = frame_idxs[0]
    end_idx = min(frame_idxs[-1], start_idx + ann_len - 1)
    if end_idx < start_idx:
        return None

    w, h = _image_size(frames[0])
    class_name = _clean_class_name(seq_dir.name)

    return {
        "seq_path": str(seq_dir.resolve()),
        "annot_path": str(annot_path.resolve()),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "n_frames": int(end_idx - start_idx + 1),
        "H": int(h),
        "W": int(w),
        "class": class_name,
        "dataset": dataset_name,
        "frame_pattern": _infer_frame_pattern(seq_dir),
    }


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
    df["end_idx"] = df["end_idx"].astype(int)
    df = df[df["end_idx"] >= df["start_idx"]]
    df["n_frames"] = (df["end_idx"] - df["start_idx"] + 1).astype(int)
    df = df[df["n_frames"] > 0]

    for col in ["seq_path", "annot_path", "class", "dataset"]:
        df[col] = df[col].astype(str)

    dropped = before - len(df)
    return df.reset_index(drop=True), dropped


def discover_dataset_rows(dataset_root: Path, dataset_name: str) -> pd.DataFrame:
    annot_files = _find_annotation_files(dataset_root)
    rows: List[Dict[str, object]] = []
    used_seq_paths = set()

    for annot in tqdm(annot_files, desc=f"Discovering {dataset_name}"):
        candidates = _candidate_seq_dirs_from_annot(annot, dataset_root)
        seq_dir = _pick_best_seq_dir(candidates)
        if seq_dir is None:
            continue

        # Avoid duplicate rows when multiple annotation files map to same sequence dir
        seq_key = str(seq_dir.resolve())
        if seq_key in used_seq_paths:
            continue

        rec = _build_record(dataset_name, seq_dir, annot)
        if rec is None:
            continue

        rows.append(rec)
        used_seq_paths.add(seq_key)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a generic training dataset index CSV.")
    p.add_argument("--data", type=str, default="data", help="Data root (auto-resolves data/dataset).")
    p.add_argument("--output", type=Path, default=None, help="Directory for CSV outputs.")
    p.add_argument("--combined-name", type=str, default="train_dataframe.csv", help="Combined CSV filename.")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder names to include. Default: all subfolders except metadata.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = _resolve_data_dir(args.data)
    output_dir = args.output.resolve() if args.output else (BASE_DIR / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if args.datasets is not None and len(args.datasets) > 0:
        dataset_dirs = [d for d in args.datasets if (data_root / d).is_dir()]
    else:
        dataset_dirs = [p.name for p in data_root.iterdir() if p.is_dir() and p.name != "metadata"]

    dataset_dirs = sorted(dataset_dirs)
    print(f"Data root  : {data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Datasets   : {dataset_dirs}")

    all_frames: List[pd.DataFrame] = []

    for ds_name in dataset_dirs:
        ds_root = data_root / ds_name
        df = discover_dataset_rows(ds_root, ds_name)
        df, dropped = _compat_filter(df)

        out_path = output_dir / f"{ds_name}.csv"
        df.to_csv(out_path, index=False)

        print(f"\n{ds_name} index built")
        print(f"  sequences      : {len(df)}")
        print(f"  unique classes : {df['class'].nunique() if not df.empty else 0}")
        print(f"  total frames   : {int(df['n_frames'].sum()) if not df.empty else 0:,}")
        print(f"  dropped rows   : {dropped}")
        print(f"  saved to       : {out_path}")

        if not df.empty:
            all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        combined, dropped = _compat_filter(combined)
        combined_path = output_dir / args.combined_name
        combined.to_csv(combined_path, index=False)

        print("\nCombined training dataframe built")
        print(f"  rows         : {len(combined)}")
        print(f"  dropped rows : {dropped}")
        print(f"  saved to     : {combined_path}")
    else:
        print("\n[warn] No sequences discovered; combined CSV was not written.")


if __name__ == "__main__":
    main()
