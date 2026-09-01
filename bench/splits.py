"""Record which benchmark sequences the head was trained on, and detect test-set overlap.

    python bench/splits.py --write            # regenerate splits/manifest.json
    python bench/splits.py --dataset dtb70 --data-root /data/dtb70
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MANIFEST = _REPO_ROOT / "splits" / "manifest.json"
TRAIN_INDEX = _REPO_ROOT / "data" / "train_dataframe.csv"

# data/ folders that hold benchmarks we evaluate on. dataset5 is UAV123, whose sequence
# names are shared with UAV123@10fps. dataset1/3/4 are MTC-AIC4, UAVTrack112 and UAV20L,
# none of which are in the evaluated set.
FOLDER_BENCHMARKS = {
    "dataset2": ["dtb70"],
    "dataset5": ["uav123", "uav123_10fps"],
}


def train_sequences() -> dict[str, list[str]]:
    """Benchmark sequence names present in data/train_dataframe.csv, keyed by dataset."""
    import pandas as pd

    df = pd.read_csv(TRAIN_INDEX)
    df["seq"] = (df["seq_path"].str.replace("\\", "/", regex=False)
                 .str.rstrip("/").str.split("/").str[-2])
    out: dict[str, set[str]] = {}
    for folder, benchmarks in FOLDER_BENCHMARKS.items():
        names = set(df.loc[df["dataset"] == folder, "seq"])
        for b in benchmarks:
            out.setdefault(b, set()).update(names)
    return {k: sorted(v) for k, v in sorted(out.items())}


def write_manifest() -> dict:
    """Regenerate splits/manifest.json from the training index."""
    trained = train_sequences()
    manifest = {
        "source": "data/train_dataframe.csv",
        "note": "Sequence names of evaluated benchmarks that appear in the training index. "
                "Any of these showing up in a test set is leakage.",
        "trained_on": trained,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest() -> dict[str, list[str]]:
    """trained_on mapping from the committed manifest, or {} if it is absent."""
    if not MANIFEST.is_file():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8")).get("trained_on", {})


def overlap(dataset: str, sequence_names) -> list[str]:
    """Test sequences of this dataset that also appear in the training index."""
    trained = set(load_manifest().get(dataset, ()))
    return sorted(set(sequence_names) & trained)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Regenerate splits/manifest.json")
    ap.add_argument("--dataset")
    ap.add_argument("--data-root")
    args = ap.parse_args()

    if args.write:
        m = write_manifest()
        for k, v in m["trained_on"].items():
            print(f"{k}: {len(v)} sequences in the training index")
        print(f"wrote {MANIFEST}")
        return 0

    if not (args.dataset and args.data_root):
        ap.error("give --write, or both --dataset and --data-root")

    from bench.datasets import load_sequences

    names = [s.name for s in load_sequences(args.dataset, args.data_root)]
    bad = overlap(args.dataset, names)
    print(f"{args.dataset}: {len(names)} test sequences, {len(bad)} also in the training index")
    if bad:
        print("LEAKAGE:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
