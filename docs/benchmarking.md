# Benchmarking SiamRAM for publication

How we produce tracking numbers that can sit in a paper next to published trackers.

Companion docs: [`bench/DATASETS.md`](../bench/DATASETS.md) (where each dataset comes from and
what it looks like on disk) and [`MODAL.md`](../MODAL.md) (running the sweep on cloud GPUs).

---

## 0. Start here

Read in this order. About 20 minutes.

1. **This document, sections 1–5.** What we are producing and why the current numbers cannot
   be used. Section 5 is the one that saves you a day.
2. **[`bench/DATASETS.md`](../bench/DATASETS.md) — only your own dataset's section.** It has the
   download route and the exact on-disk layout.
3. **[`MODAL.md`](../MODAL.md)** when you are ready to run something.

Then, before touching anything:

```bash
python bench/test_eval.py     # 12 checks, ~1 second. If this fails, stop and say so.
```

### Two shared gates — these happen once, for everybody

Nobody's individual results mean anything until both are done. Whoever owns them, say so in
the channel before starting.

- **Gate A — toolkit validation.** Reproduce SiamABC's *published LaSOT* AUC to within ±0.5
  with our evaluator. LaSOT is the sensitive check: 280 long sequences expose frame-ordering,
  absence-handling and protocol bugs that short UAV clips hide entirely. **If this does not
  reproduce, every number any of us produces is unverifiable.** Do not start dataset work
  that depends on it.
- **Gate B — TensorRT warm-up.** One person runs one dataset to completion first, which
  builds and commits the engines. If several people start from cold at once they all build
  into the same cache directory. See "Running several datasets in parallel" in `MODAL.md`.

### Splitting by dataset

Two people per dataset. Different datasets write to different paths
(`/results/<dataset>/<tracker>`), so you will not collide once Gate B is done.

| Dataset | Start it | Why |
|---|---|---|
| **LaSOT** | **immediately** | 248 GB download and 685k frames — 62% of all inference. The long pole, and Gate A depends on it. |
| **DTB70** | **immediately** | Blocked on a manual Baidu download that needs a human with an account. Unblock it early or it becomes the last thing standing. |
| UAV123 | after Gate B | |
| UAV123@10fps | after Gate B | Same archive family as UAV123 — worth pairing with that team. |
| VisDrone | after Gate B | Carries the partial-occlusion attribute split (column 9 of `<seq>_attr.txt`). |
| TrackingNet | last | Produces a submission zip, not a score. Rate-limited uploads, so do it once when everything else is settled. |

### What "done" means for a dataset

1. `python bench/download.py --dataset <name> --dest <dir>` completes, or the manual upload is in.
2. `python bench/datasets.py --dataset <name> --data-root <dir>` lists the expected sequence
   count (70, 123, 123, 35, 280, 511) and the expected frame counts.
3. `bench/eval.py --protocol-check` reports **no** missing files and **no** length mismatches.
4. SiamABC S-Tiny, SiamABC S-Small and SiamRAM full have all run.
5. The per-sequence `.txt` predictions are pulled down and archived. They cannot be
   regenerated later once configs move on, and reviewers ask for them.
6. Anything you assumed about the layout is written down — see section 8.

Report the sequence count and the protocol-check output when you claim a dataset is done.
A silently truncated result file is the failure mode we are most likely to ship.

---

## 1. Why this exists

The numbers in `ablation/results/` cannot be compared to any published tracker. Five reasons,
all verifiable in this repo:

| # | Problem | Evidence |
|---|---|---|
| 1 | **Ground truth is not the official ground truth.** These are MTC-AIC4 re-annotations of benchmark sequences, not the benchmarks. | `data/dataset2/Animal1/annotation.txt` has fractional boxes (`1005.4,515.04,63,66`); official DTB70 GT is integer. Every row of `ablation/results/full/sequence_metrics.csv` is tagged `annotation_source: aic` with a separate `local_label` column mapping `dataset2/Animal1` → `DTB70/Animal1`. |
| 2 | **Input is an MP4 re-encode.** Published trackers run on the original JPEG frames; H.264 re-encoding changes pixels and therefore changes tracker output. | `ablation/manifest.json` points every sequence at `<seq>/<seq>_30.mp4`. |
| 3 | **The metric is a challenge composite.** `0.6·AUC + 0.4·NormPrec`, averaged per sequence then averaged again. No tracking paper reports this. | `ablation/run_ablation.py:61` |
| 4 | **It does not run from a clean checkout.** | `run_ablation.py:187` shells out to `submission_to_sequence_metrics.py`, which does not exist. `video-links.csv` does not exist. `data/dataset4/bike1/` has an empty `img/` and a `.corrupt.bak`. |
| 5 | **Config drift.** The saved ablation and the current system are different systems and cannot share a table. | `ablation/configs/full.yaml` → `model_size: M`, `conf_threshold: 0.5`. `src/config/inference_config.yaml:94,311` → `model_size: S`, `conf_threshold: 0.65`. |

There is also a **leakage** problem. `data/train_dataframe.csv` indexes 253 sequences drawn from
all five data folders (53 DTB70, 84 UAVTrack112, 13 UAV20L, 94 UAV123, 9 AIC-4), and
`src/training/train_head.py:265` randomly splits that combined index — so 244 of the 325
evaluated sequences were also training sequences.

**We are not repairing the AIC-4 scorer.** `bench/` replaces it with the standard OPE protocol
over official data. That is less code and it is the only thing reviewers accept.

---

## 2. What was built

```
bench/
  datasets.py          adapters: official benchmark layout on disk -> Sequence objects
  download.py          acquisition (gdown / HuggingFace / plain HTTPS / manual instructions)
  run_tracker.py       one-pass OPE runner -> per-sequence prediction .txt
  eval.py              Success AUC, Precision@20px, Norm. Precision, --protocol-check
  pack_trackingnet.py  builds the TrackingNet eval-server submission zip
  test_eval.py         12 analytic self-checks on the metric implementations
  DATASETS.md          verified download routes and on-disk layouts
modal_app.py           Modal app: image, volumes, download/run/evaluate fan-out
MODAL.md               operator guide
```

Nothing under `src/`, `ablation/` or `data/` was modified. This is purely additive.

### The interface

```bash
python bench/download.py   --dataset uav123 --dest /data/uav123
python bench/run_tracker.py --dataset uav123 --data-root /data/uav123 \
                            --config src/config/inference_config.yaml --out results/uav123/siamram
python bench/eval.py       --dataset uav123 --data-root /data/uav123 \
                            --results results/uav123 --trackers siamram,siamabc --protocol-check
```

`run_tracker.py` writes `<seq>.txt` (one `x,y,w,h` per frame) and `<seq>_time.txt` per sequence —
the raw prediction files that reviewers ask for and that cannot be reconstructed later. Archive them.

---

## 3. The datasets

Final set: **UAV123, UAV123@10fps, VisDrone2018-SOT, DTB70, LaSOT, TrackingNet.**

| Dataset | Seqs | ~Frames | Acquisition | Table |
|---|---:|---:|---|---|
| DTB70 | 70 | 16k | **manual** — images are Baidu Pan only; GT and attributes come from the official GitHub repo | 1 |
| UAV123 | 123 | 113k | gdown | 1 |
| UAV123@10fps | 123 | 38k | gdown | 1 |
| VisDrone2018-SOT test-dev | 35 | 26k | gdown | 1 |
| LaSOT (test) | 280 | 685k | HuggingFace `l-lt/LaSOT` | 2, 3 |
| TrackingNet (test) | 511 | 226k | HuggingFace `SilvioGiancola/TrackingNet`, `TEST/*` only (35 GB) | 2 |

~1.10M frames per configuration ≈ 1.75 GPU-hours at our measured 175 FPS desktop median.
Ten configurations (7 ablation rows + full + SiamABC S-Tiny + S-Small) ≈ 18 GPU-hours.
LaSOT is 62% of that, so **run the four UAV sets first** and Table 1 lands early.

- **Table 1 — UAV comparison.** DTB70 + UAV123 + UAV123@10fps + VisDrone. This is the
  Aba-ViTrack / ORTrack / TCTrack reporting convention, so most of the table can be *copied*
  from published papers rather than run.
- **Table 2 — Generic tracking.** LaSOT + TrackingNet. SiamABC's own benchmarks, giving direct
  comparability against the base tracker.
- **Table 3 — Long-term evidence, derived from LaSOT.** LaSOT ships per-frame `out_of_view.txt`
  and `full_occlusion.txt`, so recovery success, time-to-recovery and false-reacquisition rate
  are computed over its absence episodes. This is our own metric definition, not a standardised
  one — state that plainly in the paper.

UAVDT is deliberately absent despite being a common column in this literature: its images
download fine, but the project page's link to the SOT ground truth returns 404, so there is
nothing to score against.

---

## 4. Metrics — and one thing that will trip you up

`bench/eval.py` implements the OTB/UAV123 OPE metrics directly:

- **Success AUC** — mean over IoU thresholds `0:0.05:1` of the fraction of frames above each.
- **Precision@20px** — fraction of frames with centre error ≤ 20 px.
- **Normalized Precision** — centre error normalised by GT box size, AUC over `0:0.05:0.5`.

Frames where the ground truth is absent (NaN, or a zero/negative-sized box) are skipped.

> **A perfect tracker scores 0.952, not 1.0.** OTB, pysot and pytracking all use a strict `>`
> across the 21 thresholds *including 1.0*, so IoU 1.0 clears only 20 of them: 20/21 = 0.952.
> Using `>=` instead inflates every AUC by ~4.8 points — ten times the tolerance of our
> reproduction gate, and it would look like a real improvement. `bench/test_eval.py` pins this.

Run `python bench/test_eval.py` after touching anything in `eval.py`. 12 checks, all analytic.

---

## 5. Traps already found (do not re-discover these)

- **TrackingNet's scorer silently zero-fills.** Its `metrics.py` matches prediction files to GT
  by basename and substitutes **all-zero boxes** for any sequence with no matching file, rather
  than rejecting the submission. An incomplete upload does not error — it just scores badly.
  `pack_trackingnet.py` therefore refuses to write unless all 511 sequences are present with
  matching row counts.
- **TrackingNet frame filenames are 0-based and unpadded** (`0.jpg, 1.jpg, 10.jpg`), which sort
  lexicographically as 0, 1, 10, 100, 2… The loader sorts numerically by integer stem.
- **LaSOT test membership comes from `testing_set.txt`, never from a directory scan.** Each
  category zip contains 16 training and 4 testing sequences, and SiamABC trains on LaSOT-train —
  scanning the tree would silently pull training data into the test set.
- **Always pass `--protocol-check`.** A prediction file whose row count does not match the
  sequence frame count is the most common silent evaluation bug in this literature.
- **TensorRT engines are GPU-architecture specific.** An A10G engine will not load on an H100.
  `modal_app.py` keys the cache directory by GPU model and builds it once per config.

---

## 6. The split (leakage fix)

```
TRAIN (confidence head):  GOT-10k + LaSOT-train + COCO2017 + TrackingNet-train  (SiamABC's recipe)
TUNE  (fixed thresholds): dataset1 — MTC-AIC4, 19 sequences, private
TEST  (untouched):        DTB70, UAV123, UAV123@10fps, VisDrone2018,
                          LaSOT-test, TrackingNet-test
```

Only the fixed thresholds need the tuning set — `conf_threshold`, `drm_tau_sim`,
`reacq_threshold`, `app_match_threshold`, `yolo_conf`. The `*_auto_*` knobs adapt at runtime and
need no tuning data; that is worth stating in the paper.

The claim is **"official train/test splits throughout, no sequence overlap"** — not "zero
training data from evaluated datasets", since SiamABC trains on LaSOT-train and
TrackingNet-train.

---

## 7. Order of work

1. Acquire the four UAV datasets (DTB70 needs the manual Baidu step).
2. Write `splits/manifest.json` and the train/test overlap assertion. **Does not exist yet.**
3. **Validation gate — do this before anything else runs.** Reproduce SiamABC's *published
   LaSOT* AUC to ±0.5 with our toolkit. LaSOT is the sensitive check: 280 long sequences will
   expose a frame-ordering, absence-handling or protocol bug that short UAV clips hide. If this
   does not reproduce, every number downstream is unverifiable regardless of how good SiamRAM is.
4. Run SiamABC S-Tiny/S-Small + SiamRAM full on all six datasets.
5. Run the 7 ablation rows.
6. Jetson: latency, memory, power, thermal.
7. Fill copied baseline numbers from the source papers — verify each against the paper itself,
   never against a survey's table. Mark copied vs measured in the table caption.

---

## 8. Known-unverified

- `run_tracker.py`'s real model path has only been exercised with a stub tracker; the first
  Modal run is its first real test.
- The UAV123 `<folder>_<k>` subsequence partitioning is an assumption (it warns on frame-count
  mismatch).
- LaSOT `testing_set.txt` should list 280 names (70 categories × 4). One HTTP read of it
  reported 400. The loader reads whatever the file lists rather than hardcoding, but confirm
  the count after the first download.
- `modal_app.py` has never been executed — no Modal token was available when it was written.
