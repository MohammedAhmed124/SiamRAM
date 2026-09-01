# Gate A — validating `bench/eval.py` against a published number

`bench/eval.py` has never been checked against a result somebody else published. Until it is,
every AUC it prints — including SiamRAM's 0.4814 on LaSOT — is unverifiable.

The check: run the **authors' own SiamABC**, with the **authors' own weights**, through **our**
`bench/eval.py`, on the **same** LaSOT test split, and see whether we land on their published
number. Nothing from `src/` or `checkpoints/` is involved: `src/models/SiamABC/` is a modified
fork and `checkpoints/SiamABC_epoch_zero_checkpoint.pth` has unknown provenance.

---

## 1. The target numbers

Paper: Ram Zaveri, Shivang Patel, Yu Gu, Gianfranco Doretto, *Improving Accuracy and
Generalization for Efficient Visual Tracking*, WACV 2025, pp. 9450–9460.
arXiv: <https://arxiv.org/abs/2411.18855> · Code: <https://github.com/wvuvl/SiamABC>

**Table 3** ("Comparative Study with other SOTA approaches on various benchmarks including
AVisT, NFS30, UAV123, TrackingNet, GOT-10k, and LaSOT"):

| Variant | LaSOT AUC | LaSOT Prec. | UAV123 AUC | UAV123 Prec. |
|---|---:|---:|---:|---:|
| S-Tiny (FBNetV2 backbone, `model_size='S'`) | **0.590** | 0.607 | **0.662** | 0.856 |
| S-Small (ResNet-50 backbone, `model_size='M'`) | **0.607** | 0.622 | **0.681** | 0.858 |

Table 3 reports no normalized precision for LaSOT or UAV123, so `norm_prec` from our evaluator
has no published counterpart — ignore that column for this gate.

**DTTA is ON in Table 3.** This is not a guess:

- Table 5 ("Comparative study on test-time adaptation (TTA) approaches") gives S-Tiny AVisT AUC
  0.458 with *No TTA* and **0.472** with *DTTA (ours)*. Table 3's S-Tiny AVisT AUC is **0.472**.
- Table 6 ("Case study on parameter-free dynamic updates"), whose header explicitly says
  **(No DTTA)**, gives the chosen *Ours* update strategy LaSOT AUC 0.572 / Prec. 0.592 —
  below Table 3's 0.590 / 0.607.

So the 0.590 target requires the DTTA batch-norm correction **and** the dynamic-update rule.
`--no-dtta` should land near 0.572, and that is a useful second data point if the main run misses.

Protocol facts from the paper, section 4 *Training*/3.2 *Dynamic Update*/3.3 *DTTA*:

- Template input 128x128, search input 256x256; template crop offset 0.2, search offset 2.0.
- Dynamic update: running average of classification scores,
  `rho_bar_t = (1 - lambda_D) rho_bar_{t-1} + lambda_D rho_t` with `lambda_D = 0.25`;
  a counter fires every `N = 60` frames and the dynamic template/search are refreshed when
  `rho_t > rho_bar_{t-1}`.
- DTTA: BN statistics blended with instance statistics at `lambda_BN = 0.1`, applied to the
  classification and box-regression heads only.

Toolkit: the upstream README credits
[pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) for evaluation, i.e. the standard
OTB/pysot success-plot convention that `bench/eval.py` already implements (21 thresholds
0:0.05:1, strict `>`). The upstream `eval_SiamABC.py` imports `eval_data` and `eval_toolkit`
modules that are **not** in the repo, so their harness cannot be run as published.

### Known gaps between the published protocol and what upstream ships

These are the reasons a miss may not be our evaluator's fault. Read them before concluding anything.

- `core/config/tracker/siam_tracker.yaml` ships `N: 150`, `dynamic_update: false`,
  `smooth: false`. Our runner overrides these to `N: 60`, `dynamic_update: true`,
  `smooth: true` (`PAPER_TRACKER_CONFIG`) to match the paper. With `smooth: false` the cosine
  window and scale/ratio penalty are bypassed entirely.
- `SiamABCTracker.initialize` hardcodes `self.update_lambda = 0.1`; the paper says
  `lambda_D = 0.25`. The README's own checklist lists "Commit dynamic update module" as
  **unticked**, so the released update rule is not necessarily the one that produced Table 3.
- The README checklist also lists "Commit hyperparameter for tracking window for each dataset"
  as **unticked**, and points at [Ocean's `tune_tpe.py`](https://github.com/researchmm/TracKit/blob/master/tracking/tune_tpe.py)
  for tuning `penalty_k` / `window_influence` / `lr`. The shipped values (0.062 / 0.38 / 0.765)
  are therefore not guaranteed to be the LaSOT ones. Use `--tracker-opt` to sweep them.
- Five checkpoints ship per variant (`model_S_Tiny_v1..v5.pt`); the README says to "choose the
  best for your sequence" and does not say which produced Table 3. We default to `v1`.

## 2. Running it

Weights are not a separate download: they ship in-tree at
`assets/S_Tiny/model_S_Tiny_v{1..5}.pt` and `assets/S_Small/model_S_Small_v{1..5}.pt`, so the
pinned clone *is* the weight fetch. Pinned commit: `b1c94e06fdf2dd3cb14ed07b05e38aa4601ece03`.

Locally:

```bash
python bench/baselines/siamabc_official.py --fetch-only        # clone repo + weights
python bench/baselines/siamabc_official.py \
    --dataset lasot --data-root /data/lasot \
    --out results/lasot/siamabc_official_tiny --variant tiny
python bench/eval.py --results results/lasot --dataset lasot \
    --data-root /data/lasot --trackers siamabc_official_tiny,inference_config --protocol-check
```

On Modal (the clone lands on the data volume, not in the image):

```bash
modal run modal_app.py::run_baseline --dataset lasot --variant tiny
modal run modal_app.py::evaluate --dataset lasot --trackers siamabc_official_tiny,inference_config
```

UAV123 is the cheap secondary check — a few hours instead of a day — but LaSOT is the one that
matters: 280 long sequences with absence labels expose ordering and protocol bugs that short
UAV clips hide.

## 3. Pass / fail

**Pass:** our `bench/eval.py` Success AUC for `siamabc_official_tiny` on LaSOT is within
**±0.005 absolute (±0.5 AUC points)** of 0.590 — i.e. 0.585 to 0.595. Same tolerance against
0.607 for `small`, 0.662 / 0.681 for UAV123.

Precision@20px should also land near 0.607 (tiny) / 0.622 (small). Precision agreeing while AUC
misses points at the success-plot convention; both missing together points at the tracker run.

**Fail:** anything outside that band. Do not adjust `bench/eval.py` to close the gap — that is
exactly the mistake this gate exists to prevent. Diagnose first.

## 4. If it fails, check in this order

1. **Protocol check output.** `--protocol-check` reports missing files and frame-count
   mismatches before any metric. A length mismatch invalidates everything below it.
2. **Frame ordering.** `bench/datasets.py::_frames` sorts by filename. LaSOT's `img/` is
   zero-padded 8-digit, so lexicographic == numeric. Confirm with
   `python bench/datasets.py --dataset lasot --data-root ...` that frame counts match
   `groundtruth.txt` row counts per sequence.
3. **Absence handling.** `_lasot` NaNs out frames flagged in `full_occlusion.txt` and
   `out_of_view.txt`, and `evaluate_sequence` drops them. pysot-toolkit's LaSOT evaluation
   scores **all** frames. This alone can move AUC by more than a point. Try scoring without the
   absence mask to see whether that is the whole gap.
4. **The strict-`>` AUC convention.** `success_auc` averages `(iou > t)` over 21 thresholds
   including 1.0, so a perfect tracker scores 20/21 = 0.952. Using `>=`, or 101 thresholds, or
   dropping the 1.0 threshold, each shifts AUC by a fixed offset — a suspiciously constant
   offset across every sequence is the signature.
5. **DTTA on/off.** Re-run with `--no-dtta`; if that lands near 0.572 while the DTTA run does
   not land near 0.590, the DTTA path is misconfigured, not the evaluator.
6. **Wrong weight variant.** `--variant tiny` must be compared against 0.590, `small` against
   0.607. Mixing them up produces a ~1.7-point error that looks exactly like an evaluator bug.
   The log line prints the variant and the checkpoint filename — check it.
7. **Wrong input resolution.** The log prints the resolved tracker config; confirm
   `template_size: 128` and `instance_size: 256`.
8. **Colour order.** Upstream reads frames with PIL (RGB); `cv2.imread` returns BGR. The runner
   converts, but if you change the read path, a silent BGR feed costs several AUC points.
9. **Tracking hyperparameters.** Last resort, per the gaps in section 1: sweep
   `--tracker-opt penalty_k=... --tracker-opt window_influence=... --tracker-opt lr=...`.
   If the gate only passes after a sweep, say so — the evaluator is validated, but the
   reproduction is not clean.

Whatever the outcome, write the measured numbers and the diagnosis into this file. A gate that
was run and missed is information; a gate nobody recorded is not.
