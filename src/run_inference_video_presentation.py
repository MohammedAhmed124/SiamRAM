"""
Run SiamRAM on a manifest split and write presentation-friendly debug videos.
============================================================================

Reads the manifest (default ``data/metadata/contestant_manifest.json``), takes
the ``public_lb`` split, and for each sequence calls the presentation visualiser
(``vis.test_model_presentation.run_inference``) which renders the full annotated output — main
frame with tracking overlays, an initialization-template inset, and the exact
tracker search region. The appended side column and bottom statistics strip are
omitted so the output keeps the source dimensions.

Manifest entries look like:

    "dataset1_basketball_4": {
        "video_path":      "dataset1/basketball_4/basketball_4.mp4",
        "annotation_path": "dataset1/basketball_4/annotation_first_box.txt"
    }

paths are relative to the data dir (default ``data/``). The first non-empty line
of the annotation is the init box.

Outputs mirror the metwally on-disk layout — one folder per dataset, one folder
per video:

    <outputs-dir>/<dataset>/<video_name>/<video_file>.mp4    ← annotated video
    <outputs-dir>/<dataset>/<video_name>/<video_file>.txt    ← per-frame boxes

The model is chosen with --config (a bare filename resolves under src/config).

This is a copy of ``run_inference_video.py`` that imports the presentation
renderer. The tracker/output semantics stay the same, but annotated videos use
heavier visual overlays for EKF proposals, YOLO candidates, and motion arrows.

Run from the project root
-------------------------
    # default (Small) model, public_lb -> outputs_presentation/
    python src/run_inference_video_presentation.py

    # render only dataset4/group2 for presentation
    python src/run_inference_video_presentation.py --dataset dataset4 --seq-name group2

    # tiny model into a custom output dir
    python src/run_inference_video_presentation.py --config inference_config_tiny.yaml --outputs-dir outputs_presentation/tiny

    # FAST-INFERENCE mode (metwally-cleaned): no rendering, just track + write
    # per-frame bbox .txt and a latency report — much faster.
    python src/run_inference_video_presentation.py --config inference_config_tiny.yaml --fast
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make the repo root + src importable so this runs as `python src/run_inference_video.py`
# from the project root without any PYTHONPATH juggling.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import predictor  # noqa: E402
from predictor import load_model, read_init_box  # noqa: E402
from vis.test_model_presentation import run_inference  # noqa: E402


def _resolve(rel_path: str, data_dir: Path, split: str) -> Path:
    """
    Resolve a manifest-relative path. Manifest paths look like
    "dataset1/basketball_4/basketball_4.mp4"; the actual files may sit either
    directly under the data dir, or under a per-split subfolder inside it
    (e.g. data/public_lb/dataset1/basketball_4/...). We try both, then the repo.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    candidates = (data_dir / p, data_dir / split / p, _REPO_ROOT / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return data_dir / p  # best guess (used only for the error message)


def _dataset_video_names(entry: dict):
    """
    Derive (dataset, video_name, video_file) for the metwally output layout.

    Prefers the manifest's explicit ``dataset`` / ``seq_name`` fields (as in
    data/metadata/contestant_manifest.json); otherwise parses them out of the
    video path, e.g. "dataset1/Car_video/Car_video.mp4" ->
    ("dataset1", "Car_video", "Car_video.mp4").
    """
    p = Path(entry["video_path"])
    video_file = p.name
    dataset = entry.get("dataset") or p.parent.parent.name or "misc"
    video_name = entry.get("seq_name") or p.parent.name or p.stem
    return dataset, video_name, video_file


def main():
    parser = argparse.ArgumentParser(
        description="Render source-sized SiamRAM presentation videos for a manifest split."
    )
    parser.add_argument("--config", default=None,
                        help="Inference config: a bare filename resolved under src/config "
                             "(e.g. inference_config_tiny.yaml) or a path. Default: inference_config.yaml")
    parser.add_argument("--manifest", default="data/metadata/contestant_manifest.json",
                        help="Manifest JSON (default: data/metadata/contestant_manifest.json)")
    parser.add_argument("--split", default="public_lb", help="Split to run (default: public_lb)")
    parser.add_argument("--data", default="data", help="Base dir manifest paths resolve against (default: data)")
    parser.add_argument("--outputs-dir", default="outputs_presentation",
                        help="Where to write videos (default: outputs_presentation)")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N sequences (0 = all)")
    parser.add_argument("--dataset", default=None,
                        help="Only run entries from this dataset, e.g. dataset4")
    parser.add_argument("--seq-name", "--sequence", dest="seq_name", default=None,
                        help="Only run entries with this sequence name, e.g. group2")
    parser.add_argument("--key", default=None,
                        help="Only run the exact manifest key, e.g. dataset4/group2")
    parser.add_argument("--fast", "--no-video", dest="fast", action="store_true",
                        help="Fast-inference mode (metwally-cleaned): track only via the prefetch "
                             "loop, write per-frame bbox .txt + a latency report, and skip all "
                             "rendering. Much faster than producing annotated videos.")
    args = parser.parse_args()

    # Pick the model config. --config wins; otherwise fall back to whatever
    # predictor resolved (SIAMRAM_INFERENCE_CONFIG env var, else the default).
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = _SRC / "config" / cfg_path
        if not cfg_path.exists():
            raise SystemExit(f"config not found: {cfg_path}")
        predictor._CONFIG_PATH = cfg_path

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _REPO_ROOT / manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if args.split not in manifest:
        raise SystemExit(f"split {args.split!r} not found in {manifest_path} "
                         f"(available: {', '.join(manifest)})")

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir

    sequences = list(manifest[args.split].items())
    if args.key:
        sequences = [(k, e) for k, e in sequences if k == args.key]
    if args.dataset:
        sequences = [
            (k, e)
            for k, e in sequences
            if _dataset_video_names(e)[0] == args.dataset
        ]
    if args.seq_name:
        sequences = [
            (k, e)
            for k, e in sequences
            if _dataset_video_names(e)[1] == args.seq_name
        ]
    if not sequences:
        filters = ", ".join(
            part
            for part in (
                f"key={args.key!r}" if args.key else "",
                f"dataset={args.dataset!r}" if args.dataset else "",
                f"seq_name={args.seq_name!r}" if args.seq_name else "",
            )
            if part
        )
        raise SystemExit(f"no sequences matched {filters or 'the requested split'}")
    if args.limit:
        sequences = sequences[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Path(predictor._CONFIG_PATH).name
    output_video = not args.fast
    mode = "fast-inference (bbox only)" if args.fast else "video render"
    print(f"Device: {device} | config: {cfg} | split: {args.split} | "
          f"sequences: {len(sequences)} | mode: {mode}")

    model = load_model(device=device)

    ran, skipped = 0, []
    total_start = time.perf_counter()
    for key, entry in sequences:
        video_rel = entry["video_path"]
        annot_rel = entry["annotation_path"]
        dataset, video_name, video_file = _dataset_video_names(entry)
        # metwally layout: <outputs>/<dataset>/<video_name>/<video_file>
        output_path = os.path.join(args.outputs_dir, dataset, video_name, video_file)
        print(f"\nRunning: {dataset}/{video_name}")
        try:
            video_path = _resolve(video_rel, data_dir, args.split)
            annot_path = _resolve(annot_rel, data_dir, args.split)
            if not video_path.exists():
                raise FileNotFoundError(
                    f"video not found: {video_rel} (looked in {data_dir} and {data_dir / args.split})")
            if not annot_path.exists():
                raise FileNotFoundError(
                    f"annotation not found: {annot_rel} (looked in {data_dir} and {data_dir / args.split})")
            init_box = read_init_box(str(annot_path))
            run_inference(
                initial_bbox=init_box,
                video_path=str(video_path),
                tracker=model,
                output_path=output_path,
                output_video=output_video,
                main_frame_only=True,
            )
            ran += 1
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"  [skip] {dataset}/{video_name}: {exc}")
            skipped.append(f"{dataset}/{video_name}")

    print("\n" + "=" * 50)
    print(f"FINISHED · {ran}/{len(sequences)} sequences · {time.perf_counter() - total_start:.1f} sec")
    if output_video:
        print(f"Videos: {os.path.abspath(args.outputs_dir)}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped[:8])}" + (" …" if len(skipped) > 8 else ""))


if __name__ == "__main__":
    main()
