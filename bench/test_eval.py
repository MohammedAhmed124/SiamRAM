"""Self-check for the OPE metrics: synthetic cases whose values are known analytically.

    python bench/test_eval.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.datasets import _blank_absent, _lasot, _trackingnet, read_boxes  # noqa: E402
from bench.eval import (center_error, evaluate_sequence, iou, norm_center_error,  # noqa: E402
                        norm_precision_auc, precision, success_auc)

N = 50
GT = np.tile([10.0, 20.0, 10.0, 10.0], (N, 1))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_iou():
    assert close(iou(GT, GT).mean(), 1.0)
    disjoint = GT + np.array([1000.0, 0, 0, 0])
    assert close(iou(disjoint, GT).mean(), 0.0)
    # half-height box fully inside: 50 / (50 + 100 - 50) = 0.5
    half = np.tile([10.0, 20.0, 10.0, 5.0], (N, 1))
    assert close(iou(half, GT).mean(), 0.5)
    # degenerate prediction contributes zero overlap, not a divide-by-zero
    zero = np.tile([10.0, 20.0, 0.0, 0.0], (N, 1))
    assert close(iou(zero, GT).mean(), 0.0)


def test_center_errors():
    shifted = GT + np.array([3.0, 4.0, 0.0, 0.0])
    assert close(center_error(shifted, GT).mean(), 5.0)
    # normalised by the 10x10 GT box
    assert close(norm_center_error(shifted, GT).mean(), 0.5)


def test_perfect_predictions():
    # OTB/pysot use a strict `>` over 21 thresholds including 1.0, so IoU 1.0 clears
    # 20 of them. Matching that convention is what makes the numbers comparable.
    assert close(success_auc(GT, GT), 20.0 / 21.0)
    assert close(precision(GT, GT), 1.0)
    assert close(norm_precision_auc(GT, GT), 1.0)


def test_precision_threshold():
    # Precision@20px counts centre error <= 20, so 20.0 passes and 21.0 does not.
    assert close(precision(GT + np.array([20.0, 0, 0, 0]), GT), 1.0)
    assert close(precision(GT + np.array([21.0, 0, 0, 0]), GT), 0.0)
    half = np.concatenate([GT[: N // 2], GT[N // 2:] + np.array([21.0, 0, 0, 0])])
    assert close(precision(half, GT), 0.5)


def test_success_auc_at_half_iou():
    # IoU is exactly 0.5 on every frame; the plot uses a strict `>` at each of the
    # 21 thresholds 0:0.05:1, so 10 of them (0 .. 0.45) count the frame.
    half = np.tile([10.0, 20.0, 10.0, 5.0], (N, 1))
    assert close(success_auc(half, GT), 10.0 / 21.0)


def test_norm_precision_auc():
    # Centre offset 2.5px on a 10px-wide box -> normalised error 0.25, which clears
    # 6 of the 11 thresholds 0:0.05:0.5 (0.25 .. 0.50).
    shifted = GT + np.array([2.5, 0.0, 0.0, 0.0])
    assert close(norm_precision_auc(shifted, GT), 6.0 / 11.0)


def test_absent_gt_is_skipped():
    gt = GT.copy()
    gt[::2] = np.nan
    pred = GT.copy()
    pred[::2] = 9999.0  # nonsense exactly where the GT is absent
    m = evaluate_sequence(pred, gt)
    assert m["frames"] == N // 2
    assert close(m["auc"], 20.0 / 21.0)
    assert close(m["prec20"], 1.0) and close(m["norm_prec"], 1.0)


def test_zero_sized_gt_is_absent():
    gt = GT.copy()
    gt[:10] = 0.0
    assert np.isnan(_blank_absent(gt)[:10]).all()
    assert evaluate_sequence(GT, gt)["frames"] == N - 10


def test_length_mismatch_truncates():
    pred = np.concatenate([GT, GT[:5]])
    assert evaluate_sequence(pred, GT)["frames"] == N
    assert evaluate_sequence(GT[:10], GT)["frames"] == 10


def test_read_boxes_delimiters():
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "comma.txt": "1,2,3,4\n5,6,7,8\n",
            "tab.txt": "1\t2\t3\t4\n5\t6\t7\t8\n",
            "space.txt": "1 2 3 4\n5  6 7 8\n\n",
        }
        for name, text in cases.items():
            p = Path(tmp) / name
            p.write_text(text)
            assert np.array_equal(read_boxes(p), np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)), name
        nan_file = Path(tmp) / "nan.txt"
        nan_file.write_text("1,2,3,4\nNaN,NaN,NaN,NaN\n")
        assert np.isnan(read_boxes(nan_file)[1]).all()


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_lasot_absence_merge():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "testing_set.txt", "airplane-1\n")          # bear-2 is a training seq
        seq = root / "airplane" / "airplane-1"
        _write(seq / "groundtruth.txt", "1,2,3,4\n5,6,7,8\n9,10,11,12\n13,14,15,16\n")
        _write(seq / "full_occlusion.txt", "0,1,0,0\n")            # one line, comma-separated
        _write(seq / "out_of_view.txt", "0\n0\n0\n1\n")            # one value per line
        for i in range(1, 5):
            (seq / "img").mkdir(exist_ok=True)
            (seq / "img" / f"{i:08d}.jpg").touch()
        (s,) = _lasot(root)
        assert s.name == "airplane-1" and len(s.frames) == 4
        assert s.init_box == (1.0, 2.0, 3.0, 4.0)
        assert np.isnan(s.gt[[1, 3]]).all(), "occluded and out-of-view frames must be NaN"
        assert np.isfinite(s.gt[[0, 2]]).all()


def test_trackingnet_numeric_frame_order():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "TEST"
        frames = root / "frames" / "vid_0"
        frames.mkdir(parents=True)
        for i in range(12):
            (frames / f"{i}.jpg").touch()          # 0,1,10,100,2 lexically
        _write(root / "anno" / "vid_0.txt", "10,20,30,40\n")
        (s,) = _trackingnet(Path(tmp))
        assert [p.stem for p in s.frames] == [str(i) for i in range(12)]
        assert s.gt is None, "TrackingNet test ground truth is withheld"
        assert s.init_box == (10.0, 20.0, 30.0, 40.0)


def test_uav123_family_finds_both_archive_roots():
    # Dataset_UAV123.zip unpacks to UAV123/, Dataset_UAV123_10fps.zip to UAV123_10fps/.
    # Both must resolve from the dataset's own --data-root.
    import tempfile
    from pathlib import Path

    from PIL import Image

    from bench.datasets import load_sequences

    for key, top in (("uav123", "UAV123"), ("uav123_10fps", "UAV123_10fps")):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / top
            (root / "anno" / top).mkdir(parents=True)
            frames = root / "data_seq" / top / "bike1"
            frames.mkdir(parents=True)
            for i in range(1, 4):
                Image.new("RGB", (8, 8)).save(frames / f"{i:06d}.jpg")
            (root / "anno" / top / "bike1.txt").write_text(("1,2,3,4" + chr(10)) * 3)
            seqs = load_sequences(key, td)
            assert [s.name for s in seqs] == ["bike1"], key
            assert len(seqs[0].frames) == 3, key


def test_leakage_overlap_matches_manifest():
    # The committed manifest records the DTB70 and UAV123 sequences the head was trained on.
    from bench.splits import load_manifest, overlap
    trained = load_manifest()
    assert trained, "splits/manifest.json missing - run: python bench/splits.py --write"
    known = trained["dtb70"][:3]
    assert overlap("dtb70", known) == sorted(known)
    assert overlap("dtb70", ["definitely-not-a-sequence"]) == []
    # LaSOT and TrackingNet were never in the training index.
    assert overlap("lasot", ["airplane-1"]) == []
    assert overlap("trackingnet", ["anything"]) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
