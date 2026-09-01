"""Pack per-sequence predictions into the zip the TrackingNet evaluation server expects.

    python bench/pack_trackingnet.py --results results/siamram --data-root /vol/bench \
        --out submission.zip

The server (https://eval.ai/web/challenges/challenge-page/1805) reads every *.txt in the
zip and matches it to a test sequence by basename, so the entries are written flat.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.datasets import load_sequences, read_boxes  # noqa: E402

N_TEST_SEQUENCES = 511


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="Directory holding <sequence>.txt")
    ap.add_argument("--data-root", required=True, help="Directory holding TrackingNet/TEST")
    ap.add_argument("--out", required=True, help="Zip to write")
    a = ap.parse_args()

    results = Path(a.results)
    sequences = load_sequences("trackingnet", a.data_root)
    problems = []
    if len(sequences) != N_TEST_SEQUENCES:
        problems.append(f"{a.data_root} holds {len(sequences)} test sequences, expected "
                        f"{N_TEST_SEQUENCES} - the TEST split is incomplete")

    entries = {}
    for s in sequences:
        f = results / f"{s.name}.txt"
        if not f.is_file():
            problems.append(f"missing prediction file {f}")
            continue
        boxes = read_boxes(f)
        if len(boxes) != len(s.frames):
            problems.append(f"{f}: {len(boxes)} rows vs {len(s.frames)} frames")
            continue
        entries[f"{s.name}.txt"] = "\n".join(",".join(f"{v:g}" for v in row) for row in boxes) + "\n"

    if problems:
        print(f"{len(problems)} problem(s); no zip written:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in entries.items():
            z.writestr(name, text)
    print(f"wrote {out} with {len(entries)} sequences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
