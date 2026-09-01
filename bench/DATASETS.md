# Benchmark datasets

Ten visual-object-tracking benchmarks, how to get each one onto a headless
Linux box (Modal), and what the files look like once unpacked.

Everything below was verified against live hosts on 2026-09-01. Archive sizes
are exact byte counts read from the servers' `Content-Range` headers; internal
layouts were verified by range-reading the archives' zip central directories
(no full downloads). Where no non-interactive route exists, that is said
plainly rather than guessed at.

`bench/download.py` implements the automatable half:

```
python bench/download.py --dataset <name> --dest /vol/bench
```

## (a) Summary

| dataset | seqs | download | fetch method | automatable? |
|---|---|---|---|---|
| DTB70 | 70 | not published (Baidu-only) | Baidu Pan (images), GitHub (GT) | **manual** (GT: yes) |
| VisDrone-SOT test-dev | 35 | 11.27 GiB | Google Drive | **yes** (direct) |
| UAV123 | 123 | 13.08 GiB | Google Drive | **yes** (direct) |
| UAV123@10fps | 123 | 4.37 GiB | Google Drive | **yes** (direct) |
| LaSOT test | 280 of 1400 | 248 GB (whole set — no test-only archive) | Hugging Face | **yes** (`huggingface_hub`) |
| TrackingNet TEST | 511 | 35 GB | Hugging Face | **yes** (`huggingface_hub`) |

Google Drive files here are all >100 MB, so `wget`/`curl` hit the virus-scan
interstitial. **Do not use `gdown` for these.** `gdown==6.0.0` rebuilds every link
as `uc?id=<id>` and discards the `resourcekey`, and the two UAV123 archives are
pre-2021 links that need it: without it Drive serves a sign-in page, not the file.
`bench/download.py` talks to Drive directly instead - it normalises the link with
`drive_uc_url()`, keeps the resourcekey, and submits the virus-scan confirm form.
Verified against the live host: both UAV123 archives resolve to their exact pinned
byte counts. A `Quota exceeded` page means the file has been fetched too often
today; that is transient, so retry in a few hours.

---

## (b) Per dataset

### 1. DTB70 — 70 sequences

- Project page / official repo: <https://github.com/flyers/drone-tracking>
  (Li & Yeung, *Visual Object Tracking for Unmanned Aerial Vehicles*, AAAI 2017)
- **Fetch: Baidu Pan only.** The README's single download link is
  <https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ>. Searched Google Drive
  mirrors, Hugging Face, Kaggle, OpenDataLab and hyper.ai/OpenBayes (which
  lists DTB70 at <https://hyper.ai/en/datasets/5159> but exposes no torrent or
  direct URL) — **no non-interactive mirror exists.** Baidu Pan requires an
  account and its client for >files, so it is effectively unusable from Modal.
- **Ground truth and attributes are automatable**, even though the images are
  not: all 70 annotation files plus all 70 attribute files are committed to the
  official repo.

```bash
# annotations only (~2 MB); images still need the manual Baidu step
git clone --depth 1 https://github.com/flyers/drone-tracking /tmp/dtb70-anno
# /tmp/dtb70-anno/experiments/anno/<Seq>.txt      -> == <Seq>/groundtruth_rect.txt
# /tmp/dtb70-anno/experiments/anno/att/<Seq>.txt  -> 11 attribute flags
```

- Size: not published anywhere; 15,777 frames at 1280×720 JPEG.
- Layout (OTB-style, per `experiments/util/configDTBSeqs.m`, `nz=5`):

```
DTB70/
  <SeqName>/                     # Animal1, BMX2, ChasingDrones, RaceCar1, ...
    img/00001.jpg ... %05d.jpg
    groundtruth_rect.txt
```

- GT format: one line per frame, `x,y,w,h` (top-left + size), comma-delimited,
  values may be fractional (`1005.4,515.04,63,66`). **No absence/occlusion
  flags** — every frame has a box.
- Attributes: `experiments/anno/att/<Seq>.txt`, a single line of 11
  comma-separated 0/1 flags for the whole sequence. `experiments/perfPlot.m`
  names the first ten: scale variation, aspect-ratio variation, **occlusion**,
  deformation, fast camera motion, in-plane rotation, out-of-plane rotation,
  out of view, background clutter, similar objects around (the file carries an
  11th column, motion blur, that the toolkit's name list omits). DTB70 does
  **not** separate partial from full occlusion.

### 2. VisDrone-SOT test-dev — 35 sequences

- Project page: <http://www.aiskyeye.com/>; mirror with direct links:
  <https://github.com/VisDrone/VisDrone-Dataset> (Task 3: Single-Object Tracking)
- Fetch: **Google Drive, automatable** — the file is named
  `VisDrone2019-SOT-test-dev.zip` (the SOT test-dev split is shared by the 2018
  and 2019 editions), id `1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC`,
  12,097,003,333 B (11.27 GiB). Ground truth is included for test-dev.

```bash
python bench/download.py --dataset <name> --dest /vol/bench   # -> VisDrone2019-SOT-test-dev.zip
unzip -q VisDrone2019-SOT-test-dev.zip -d /vol/bench
```

- Layout (verified from the archive):

```
VisDrone2019-SOT-test-dev/
  sequences/<seq>/img0000001.jpg ... img%07d.jpg   # 35 dirs, e.g. uav0000011_00000_s
  annotations/<seq>.txt                            # 35 files
  attributes/<seq>_attr.txt                        # 35 files
```

  (the `img` prefix + 7 digits is `nz=7` in the toolkit's `util/configSeqs.m`;
  test-**challenge** ships `initialization/` instead of `annotations/`.)
- GT format (verified sample): one line per frame, `x,y,w,h` as integers,
  comma-delimited, no header. **No absence flags** — matches the toolkit's
  submission spec `<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>`
  (<https://github.com/VisDrone/VisDrone2018-SOT-toolkit>).
- Attributes (what we care about for partial occlusion): `attributes/<seq>_attr.txt`
  is a single line of **12** comma-separated 0/1 flags for the whole sequence.
  Order, from `perfPlot.m` in the SOT toolkit:

  | col | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | attr | ARC | BC | CM | FM | FOC | IV | LR | OV | **POC** | SOB | SV | VC |

  Partial Occlusion is **column 9**. Example: `0,0,1,1,1,0,0,0,1,1,1,1`.
- Counts (SOT toolkit README): train 86 clips / 69,941 frames, val 11 / 7,046,
  test 35 / 29,367.

### 3. UAV123 (123 seqs)

- Project page: <https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset>
  (Mueller, Smith & Ghanem, ECCV 2016)
- Fetch: **Google Drive, automatable, but old-style links with a `resourcekey`.**
  `Dataset_UAV123.zip`, 14,049,397,769 B (13.08 GiB), covers **both** UAV123 and
  UAV20L. The page's "FTP" buttons all point at a Drive folder now, so Drive is
  the only route.

```bash
python bench/download.py --dataset <name> --dest /vol/bench   # -> Dataset_UAV123.zip
unzip -q Dataset_UAV123.zip -d /vol/bench
```

- UAV123@10fps is a separate archive, `Dataset_UAV123_10fps.zip`,
  4,688,528,930 B (4.37 GiB) — trivially available, same recipe:

```bash
python bench/download.py --dataset <name> --dest /vol/bench   # -> Dataset_UAV123_10fps.zip
```

  (Also on the page: annotation-details PDF `0B6sQMCU1i4NbWUI5Nk5wempDQUU`,
  modified tracker benchmark `0B6sQMCU1i4NbeWxnalFqSWE3WTQ`, results
  `0B6sQMCU1i4NbdEhzdDJJekJuQWM`.)
- Layout (verified from both archives):

```
UAV123/
  data_seq/UAV123/<seq>/000001.jpg ... %06d.jpg
  anno/UAV123/<seq>.txt          # 123 tracks
  anno/UAV123/att/<seq>.txt      # 12 attribute flags
  anno/UAV20L/<seq>.txt          # 20 long tracks, images reused from data_seq/UAV123
  anno/UAV20L/att/<seq>.txt
UAV123_10fps/
  data_seq/UAV123_10fps/<seq>/%06d.jpg
  anno/UAV123_10fps/<seq>.txt
```

  Note UAV123 has 123 *tracks* over 91 image folders: long clips are split
  (`bird1_1`, `bird1_2`, `bird1_3` all read frames from `data_seq/UAV123/bird1`
  with different start/end frames). Use the frame ranges in
  `pytracking/evaluation/uavdataset.py` or `got10k/datasets/uav123.json`.
- GT format: one line per frame, `x,y,w,h`, comma-delimited. Frames where the
  target is fully occluded or out of view are written as literal
  `NaN,NaN,NaN,NaN` — that is the absence flag.
- Attributes: `anno/<version>/att/<seq>.txt`, one line of **12** comma-separated
  0/1 flags. Order verified by cross-checking the archive's files against
  `pytracking/evaluation/dataset_attribute_specs/UAV123_attributes.json`:

  | col | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | attr | SV | ARC | LR | FM | FOC | **POC** | OV | BC | IV | VC | CM | SOB |

  Partial Occlusion is **column 6**. Example: `bike1` = `1,1,0,1,0,0,0,0,1,1,1,1`
  → SV, ARC, FM, IV, VC, CM, SOB.
- Optional unofficial mirror (no gdown, plain HTTPS, community upload — sizes
  match the official zip at 14,037,198,671 B but it is **not** author-hosted, so
  treat as a fallback):
  `https://huggingface.co/datasets/xche32/UAV123/resolve/main/UAV123.tar.gz`

### 4. LaSOT — 280 testing sequences (of 1,400)

- Project page: <https://hengfan2010.github.io/projects/LaSOT/> (backup of
  <http://vision.cs.stonybrook.edu/~lasot/>, whose TLS chain does not validate
  from a clean box; Fan et al., CVPR 2019 / IJCV 2021)
- **There is no testing-set-only archive.** The project page offers the whole
  conference set (~227 GB) as one OneDrive file, three Google Drive parts or a
  Baidu link, plus a *per-category* set of 70 zips — and each category zip holds
  all 20 of that category's sequences, 16 training and 4 testing. All 70
  categories appear in the testing split, so the 280 test sequences cannot be
  fetched without the training ones.
- Fetch: **Hugging Face, automatable, no token** — `l-lt/LaSOT` (maintainer Heng
  Fan, the dataset's author) mirrors the 70 category zips (248 GB total,
  e.g. `airplane.zip` 2.46 GB, `basketball.zip` 3.93 GB) plus `training_set.txt`
  and `testing_set.txt` at the repo root. `huggingface-hub==1.13.0` is already
  pinned in `requirements.txt`.

```bash
python bench/download.py --dataset lasot --dest /vol/bench     # 248 GB
# equivalent to:
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('l-lt/LaSOT', repo_type='dataset', local_dir='/vol/bench/LaSOT', \
                    allow_patterns=['*.zip','*.txt'])"
```

- Layout (per the `l-lt/LaSOT` README and `pytracking/evaluation/lasotdataset.py`):

```
LaSOT/
  testing_set.txt                  # 280 names, one per line: "airplane-1", "airplane-9", ...
  training_set.txt                 # 1,120 names
  <class>/<class>-<n>/             # 70 classes x 20 sequences
    img/00000001.jpg ... %08d.jpg
    groundtruth.txt
    full_occlusion.txt
    out_of_view.txt
    nlp.txt                        # natural-language description, unused here
```

- GT format: `groundtruth.txt`, one line per frame, `x,y,w,h`, comma-delimited,
  integers. Every frame carries a box — including frames where the target is not
  visible, which is why the two flag files matter.
- Absence flags: `full_occlusion.txt` and `out_of_view.txt`, one 0/1 value per
  frame. pytracking reads them with `delimiter=','` and treats a frame as visible
  only when both are 0; `bench/datasets.py` NaNs out those rows, exactly as it
  does for AVisT. `_read_flags` accepts both a single comma-separated line and
  one value per line, so the file's line layout does not need to be pinned down.
- The loader reads `testing_set.txt` and touches nothing else, so the 1,120
  training sequences may be deleted after extraction.
- **Assumption not verified** (no download performed): that `<class>.zip`
  expands to `<class>/<class>-<n>/…` rather than to a bare `<class>-<n>/`. If it
  does the latter, move the directories one level down; the loader raises
  `testing_set.txt lists <name> but <path> does not exist` naming the path it
  expected.

### 5. TrackingNet — TEST split, 511 sequences

- Official repo: <https://github.com/SilvioGiancola/TrackingNet-devkit>
  (Müller et al., ECCV 2018). 30 chunks in total; the test chunk is one of them.
- Fetch: **Hugging Face, automatable, no token.** The devkit's
  `download_TrackingNet.py --chunk TEST` pulls from per-file Google Drive ids
  listed in CSVs and is quota-fragile; the same authors now publish the dataset
  at `SilvioGiancola/TrackingNet` (1.14 TB total), where `TEST/` is 35 GB and can
  be fetched on its own.

```bash
python bench/download.py --dataset trackingnet --dest /vol/bench    # 35 GB
# equivalent to:
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('SilvioGiancola/TrackingNet', repo_type='dataset', \
                    local_dir='/vol/bench/TrackingNet', allow_patterns='TEST/*')"
# then unzip TEST/zips/<seq>.zip -> TEST/frames/<seq>/   (what extract_frame.py does)
```

- Layout (verified against the HF file listing and
  `pytracking/evaluation/trackingnetdataset.py`):

```
TrackingNet/
  TEST/
    zips/<seq>.zip          # 511 zips, 4 MB - 226 MB, frames stored flat
    frames/<seq>/0.jpg 1.jpg ... N.jpg
    anno/<seq>.txt          # 511 files, 13-16 bytes each
```

  Sequence names are YouTube-style ids with a clip suffix, e.g. `-Dz7OIf54b0_0`,
  `0-6LB4FqxoE_0`.
- **Frame numbering is 0-based and unpadded**, unlike every other benchmark here.
  Lexical sorting gives `0, 1, 10, 100, 2, …`, so `_trackingnet` sorts by
  `int(path.stem)`; `_frames` is left alone because the other loaders rely on its
  lexical order.
- GT format: **withheld.** `anno/<seq>.txt` holds the first-frame box only
  (`x,y,w,h`, comma-delimited), so `_sequence` returns `gt=None` and `bench/eval.py`
  reports these sequences as not evaluable. Scores come from the server.
- Submission: <https://eval.ai/web/challenges/challenge-page/1805>. The devkit's
  `metrics.py` — which is the server-side scorer, run locally as
  `python metrics.py --GT_zip GT.zip --subm_zip subm.zip` — collects every entry
  in the zip that ends in `.txt` and is not under `__MACOSX`, matches it to a
  ground-truth annotation by **basename** (`<seq>.txt`), and parses it with
  `pd.read_csv(sep=",", names=["subm_x1","subm_y1","subm_w","subm_h"])`. So:
  comma-delimited `x,y,w,h`, one line per frame, one file per sequence, and the
  in-zip directory structure is free — flat at the zip root is what
  `bench/pack_trackingnet.py` writes. A sequence whose file is missing or
  ambiguously named is silently scored as all-zero boxes rather than rejected,
  which is exactly why the packer refuses to write an incomplete zip:

```bash
python bench/pack_trackingnet.py --results results/siamram \
    --data-root /vol/bench --out submission.zip
```

  It fails, listing every offender, unless all 511 sequences are present and each
  prediction file has exactly as many rows as its sequence has frames.

---

## (c) Manual steps required

Three things cannot be automated from a cloud box. In each case the human does
the fetch on a normal desktop and uploads the archive to the Modal volume.

1. **DTB70 images.** Log in to Baidu Pan, download
   <https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ>, upload the zip to the
   volume and unpack to `<dest>/DTB70/<Seq>/img/%05d.jpg`. The ground truth and
   attribute files do **not** need this step — clone
   `https://github.com/flyers/drone-tracking` and copy `experiments/anno/<Seq>.txt`
   to `<dest>/DTB70/<Seq>/groundtruth_rect.txt`.

2. **UAVTrack112 / UAVTrack112_L (whole dataset).** Log in to Baidu Pan,
   download <https://pan.baidu.com/s/1HK7zCKaa_olToGVzLrOpqA> with code `xb41`,
   upload and unpack. There is no alternative host.

3. **UAVDT SOT ground truth.** `UAV-benchmark-S.zip` downloads fine and gives
   the 50 image sequences, but the project page's SOT-toolkit Drive link that
   held `anno/<seq>_gt.txt` is dead (404). Options, in order of preference:
   - email the UAVDT authors via <https://sites.google.com/view/grli-uavdt> and
     ask for the SOT toolkit / annotation archive, then upload it; or
   - take the annotations out of a third-party restructured bundle — SGLATrack
     publishes `uavdt/{anno,sequences}` ready to use at
     <https://pan.baidu.com/s/1p0H_hHGUAc3fWkD3wlfNcw?pwd=e22r> (code `e22r`),
     which is Baidu again and covers DTB70 + UAVTrack112 + VisDrone in the same
     bundle, so it may be worth doing this one download instead of three.

Note also that `bench/download.py --dataset uavdt` exits non-zero after
downloading and extracting the images, precisely because the annotations are
still missing.
