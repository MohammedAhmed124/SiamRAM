"""Adapters that read official tracking-benchmark layouts off disk into a common Sequence form.

Frames are always the original JPEG files. Never a video re-encode.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sequence:
    """One benchmark sequence: ordered frame paths, init box and per-frame ground truth."""

    name: str
    frames: list[Path]
    init_box: tuple[float, float, float, float]
    gt: np.ndarray | None  # Nx4 x,y,w,h; NaN rows where the target is absent/unlabelled


# --------------------------------------------------------------------------- io


def _frames(directory: Path) -> list[Path]:
    """Ordered image paths inside a frame directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"frame directory not found: {directory}")
    return sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )


def read_boxes(path: Path) -> np.ndarray:
    """Nx4 x,y,w,h array from a comma-, tab-, semicolon- or space-delimited box file."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\t ;]+", line) if p]
        if len(parts) < 4:
            raise ValueError(f"{path}: expected 4 numbers, got {line!r}")
        rows.append([float(p) for p in parts[:4]])  # float() parses NaN/nan directly
    if not rows:
        raise ValueError(f"{path}: empty annotation file")
    return np.asarray(rows, dtype=float)


def _read_flags(path: Path) -> np.ndarray:
    """Per-frame 0/1 flags from a one-line or one-per-line file."""
    text = path.read_text(encoding="utf-8-sig")
    return np.asarray([int(float(v)) for v in re.split(r"[,\t \n]+", text.strip()) if v], dtype=int)


def _dig(root: Path, *names: str) -> Path:
    """Descend into a named subdirectory when the caller pointed one level too high."""
    for name in names:
        if (root / name).is_dir():
            return root / name
    return root


def _pick_dir(root: Path, *rel: str) -> Path:
    """First existing directory among the given relative candidates."""
    for r in rel:
        if (root / r).is_dir():
            return root / r
    raise FileNotFoundError(f"none of {rel} found under {root}")


def _pick_file(*candidates: Path) -> Path:
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"none of these annotation files exist: {list(map(str, candidates))}")


def _subdirs(root: Path) -> list[Path]:
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


# ------------------------------------------------------------------ normalisation


def _blank_absent(gt: np.ndarray) -> np.ndarray:
    """NaN out rows that encode absence as NaN or as a zero/negative-sized box."""
    gt = gt.astype(float, copy=True)
    bad = ~np.isfinite(gt).all(axis=1) | (gt[:, 2] <= 0) | (gt[:, 3] <= 0)
    gt[bad] = np.nan
    return gt


def _sequence(name: str, frames: list[Path], gt: np.ndarray | None) -> Sequence:
    """Build a Sequence, aligning frame and ground-truth lengths."""
    if gt is None:
        return Sequence(name, frames, (0.0, 0.0, 0.0, 0.0), None)
    if len(gt) == 1 and len(frames) > 1:
        # Initialisation-only annotation (GT withheld): keep the box, drop the GT.
        return Sequence(name, frames, tuple(float(v) for v in gt[0]), None)
    gt = _blank_absent(gt)
    if len(frames) != len(gt):
        print(f"[warn] {name}: {len(frames)} frames vs {len(gt)} gt rows; truncating to the shorter")
        n = min(len(frames), len(gt))
        frames, gt = frames[:n], gt[:n]
    if not np.isfinite(gt[0]).all():
        raise ValueError(f"{name}: first-frame ground truth is absent, cannot initialise")
    return Sequence(name, frames, tuple(float(v) for v in gt[0]), gt)


# ----------------------------------------------------------------------- loaders
#
# TODO(layout): bench/DATASETS.md did not exist when these were written. Every layout
# below is an assumption to be checked against the real archives; each loader states the
# one it assumes. Alternative directory/file names are tried where the releases are known
# to disagree, and a wrong guess fails loudly with the paths it looked for.


def _dtb70(root: Path) -> list[Sequence]:
    """DTB70/<Seq>/img/00001.jpg + DTB70/<Seq>/groundtruth_rect.txt, comma-delimited, no absence."""
    root = _dig(root, "DTB70")
    out = []
    for d in _subdirs(root):
        gt = read_boxes(_pick_file(d / "groundtruth_rect.txt", d / "groundtruth.txt"))
        out.append(_sequence(d.name, _frames(d / "img"), gt))
    return out


def _uavdt(root: Path) -> list[Sequence]:
    """UAV-benchmark-S/<S0101>/img000001.jpg + anno/<S0101>_gt.txt, comma-delimited, no absence."""
    root = _dig(root, "UAVDT", "UAV-benchmark-SOT_v1.0")
    seq_root = _pick_dir(root, "UAV-benchmark-S", "data_seq", "sequences", ".")
    anno_dir = root / "anno" if (root / "anno").is_dir() else root
    out = []
    for d in _subdirs(seq_root):
        gt_path = _pick_file(
            anno_dir / f"{d.name}_gt.txt", anno_dir / f"{d.name}.txt",
            d / "groundtruth_rect.txt", d / "groundtruth.txt",
        )
        frame_dir = d / "img" if (d / "img").is_dir() else d
        out.append(_sequence(d.name, _frames(frame_dir), read_boxes(gt_path)))
    return out


def _visdrone2018(root: Path) -> list[Sequence]:
    """sequences/<seq>/img0000001.jpg + annotations/<seq>.txt, comma-delimited.

    Some test-dev releases ship only the first-frame box; those sequences load with gt=None.
    """
    root = _dig(root, "VisDrone2018-SOT-test-dev", "VisDrone2018-SOT-val", "VisDrone2018-SOT-train")
    seq_root = _pick_dir(root, "sequences", "data_seq")
    anno_root = _pick_dir(root, "annotations", "anno")
    out = []
    for d in _subdirs(seq_root):
        gt = read_boxes(_pick_file(anno_root / f"{d.name}.txt", anno_root / f"{d.name}_gt.txt"))
        seq = _sequence(d.name, _frames(d), gt)
        if seq.gt is None:
            print(f"[warn] {d.name}: annotation holds only the init box; sequence is not evaluable")
        out.append(seq)
    return out


def _uav123_groups(anno_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Map each frame-folder name to its ordered (suffix, annotation) subsequences."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(anno_dir.glob("*.txt")):
        base, _, suffix = p.stem.rpartition("_")
        if base and suffix.isdigit():
            groups[base].append((int(suffix), p))
        else:
            groups[p.stem].append((0, p))
    for v in groups.values():
        v.sort()
    return groups


def _uav123_family(root: Path, anno_sub: str, seq_sub: str, stride: int = 1) -> list[Sequence]:
    """data_seq/<set>/<folder>/000001.jpg + anno/<set>/<seq>.txt, comma-delimited, NaN absence.

    TODO(layout): sequences named <folder>_<k> are assumed to partition <folder> into
    contiguous blocks in k order, matching configSeqs.m. The frame-coverage warning below
    fires if that is wrong for any folder. UAV123_10fps falls back to every third frame of
    data_seq/UAV123 when data_seq/UAV123_10fps is absent; that alignment is a guess.
    """
    root = _dig(root, "UAV123", "Dataset_UAV123")
    anno_dir = _pick_dir(root, f"anno/{anno_sub}", f"anno/{anno_sub.lower()}", "anno")
    seq_root = _pick_dir(root, f"data_seq/{seq_sub}", "data_seq", "sequences")
    if stride > 1 and seq_root.name != seq_sub:
        print(f"[warn] data_seq/{seq_sub} missing; sampling every {stride}rd frame of {seq_root}")
    else:
        stride = 1

    out = []
    for base, members in sorted(_uav123_groups(anno_dir).items()):
        all_frames = _frames(seq_root / base)[::stride]
        offset = 0
        for _suffix, anno_path in members:
            gt = read_boxes(anno_path)
            frames = all_frames[offset:offset + len(gt)]
            offset += len(gt)
            out.append(_sequence(anno_path.stem, frames, gt))
        if offset != len(all_frames):
            print(f"[warn] {base}: subsequences cover {offset} of {len(all_frames)} frames "
                  f"- check the official start/end frame table")
    return out


def _uavtrack112(root: Path, long_only: bool) -> list[Sequence]:
    """data_seq/<seq>/*.jpg + anno/<seq>.txt, comma-delimited, no absence.

    TODO(layout): the _L subset membership is not derivable from the tree, so it must come
    from anno/UAVTrack112_L/ or a UAVTrack112_L.txt name list.
    """
    root = _dig(root, "UAVTrack112")
    seq_root = _pick_dir(root, "data_seq", "sequences")
    anno_root = _pick_dir(root, "anno", "annotations")

    names = [d.name for d in _subdirs(seq_root)]
    if long_only:
        if (anno_root / "UAVTrack112_L").is_dir():
            anno_root = anno_root / "UAVTrack112_L"
            names = [p.stem for p in sorted(anno_root.glob("*.txt"))]
        elif (root / "UAVTrack112_L.txt").is_file():
            names = [n for n in (root / "UAVTrack112_L.txt").read_text().split() if n]
        else:
            raise FileNotFoundError(
                f"UAVTrack112_L subset list not found. Provide {root / 'UAVTrack112_L.txt'} "
                f"(one sequence name per line) or an anno/UAVTrack112_L directory."
            )

    out = []
    for name in names:
        gt = read_boxes(_pick_file(anno_root / f"{name}.txt", anno_root / f"{name}_gt.txt"))
        frame_dir = seq_root / name
        if (frame_dir / "img").is_dir():
            frame_dir = frame_dir / "img"
        out.append(_sequence(name, _frames(frame_dir), gt))
    return out


def _avist(root: Path) -> list[Sequence]:
    """sequences/<seq>/00000001.jpg + anno/<seq>.txt, with absence in the separate
    full_occlusion/<seq>_full_occlusion.txt and out_of_view/<seq>_out_of_view.txt flag files."""
    root = _dig(root, "AVisT", "avist")
    seq_root = _pick_dir(root, "sequences", "data_seq")
    anno_root = _pick_dir(root, "anno", "annotations")
    out = []
    for d in _subdirs(seq_root):
        gt = _blank_absent(read_boxes(_pick_file(anno_root / f"{d.name}.txt")))
        for sub, suffix in (("full_occlusion", "_full_occlusion"), ("out_of_view", "_out_of_view")):
            flag_path = root / sub / f"{d.name}{suffix}.txt"
            if not flag_path.is_file():
                flag_path = root / sub / f"{d.name}.txt"
            if flag_path.is_file():
                flags = _read_flags(flag_path)
                gt[: len(flags)][flags[: len(gt)] == 1] = np.nan
        out.append(_sequence(d.name, _frames(d), gt))
    return out


def _lasot(root: Path) -> list[Sequence]:
    """<class>/<class>-<n>/img/00000001.jpg + groundtruth.txt, absence in the sibling
    full_occlusion.txt and out_of_view.txt; restricted to the 280 names in testing_set.txt."""
    root = _dig(root, "LaSOT", "LaSOTBenchmark", "lasot")
    listing = _pick_file(root / "testing_set.txt", root.parent / "testing_set.txt")
    out = []
    for name in listing.read_text(encoding="utf-8-sig").split():
        d = root / name.rsplit("-", 1)[0] / name
        if not d.is_dir():
            d = root / name  # releases that do not nest sequences under their category folder
        if not d.is_dir():
            raise FileNotFoundError(
                f"testing_set.txt lists {name} but neither "
                f"{root / name.rsplit('-', 1)[0] / name} nor {root / name} exists")
        gt = _blank_absent(read_boxes(d / "groundtruth.txt"))
        for flag_name in ("full_occlusion.txt", "out_of_view.txt"):
            flags = _read_flags(_pick_file(d / flag_name))
            gt[: len(flags)][flags[: len(gt)] == 1] = np.nan
        out.append(_sequence(name, _frames(d / "img"), gt))
    return out


def _trackingnet(root: Path) -> list[Sequence]:
    """TEST/frames/<seq>/0.jpg (0-based, unpadded) + TEST/anno/<seq>.txt.

    The test ground truth is withheld: anno holds the first-frame box only, so every
    sequence loads with gt=None and is scored by the evaluation server instead.
    """
    root = _dig(_dig(root, "TrackingNet"), "TEST", "Test", "test")
    frames_root = _pick_dir(root, "frames")
    anno_root = _pick_dir(root, "anno")
    out = []
    for d in _subdirs(frames_root):
        frames = sorted(_frames(d), key=lambda p: int(p.stem))
        gt = read_boxes(_pick_file(anno_root / f"{d.name}.txt"))
        out.append(_sequence(d.name, frames, gt))
    return out


def _vot_lt2021(root: Path) -> list[Sequence]:
    """<seq>/color/00000001.jpg + <seq>/groundtruth.txt, comma-delimited, `nan,nan,nan,nan` absence."""
    root = _dig(root, "VOT-LT2021", "vot_lt2021", "sequences")
    out = []
    for d in _subdirs(root):
        gt_path = d / "groundtruth.txt"
        if not gt_path.is_file():
            continue
        frame_dir = d / "color" if (d / "color").is_dir() else d
        out.append(_sequence(d.name, _frames(frame_dir), read_boxes(gt_path)))
    if not out:
        raise FileNotFoundError(f"no <seq>/groundtruth.txt found under {root}")
    return out


LOADERS = {
    "dtb70": _dtb70,
    "uavdt": _uavdt,
    "visdrone2018": _visdrone2018,
    "visdrone_sot": _visdrone2018,  # bench/download.py's name for the same set
    "vot_lt2021": _vot_lt2021,
    "uav123": lambda r: _uav123_family(r, "UAV123", "UAV123"),
    "uav123_10fps": lambda r: _uav123_family(r, "UAV123_10fps", "UAV123_10fps", stride=3),
    "uav20l": lambda r: _uav123_family(r, "UAV20L", "UAV123"),
    "uavtrack112": lambda r: _uavtrack112(r, long_only=False),
    "uavtrack112_l": lambda r: _uavtrack112(r, long_only=True),
    "avist": _avist,
    "lasot": _lasot,
    "trackingnet": _trackingnet,
}


def load_sequences(dataset_name: str, data_root) -> list[Sequence]:
    """Read one benchmark off disk. dataset_name is a key of LOADERS."""
    key = dataset_name.strip().lower()
    if key not in LOADERS:
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of {sorted(LOADERS)}")
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    return LOADERS[key](root)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="List sequences found for one benchmark.")
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args()
    seqs = load_sequences(a.dataset, a.data_root)
    for s in seqs:
        n_absent = 0 if s.gt is None else int(np.isnan(s.gt[:, 0]).sum())
        print(f"{s.name:32s} frames={len(s.frames):6d} absent={n_absent:5d} init={s.init_box}")
    print(f"{len(seqs)} sequences, {sum(len(s.frames) for s in seqs)} frames")
