# BackTrack — Experiment Report (backward-tracking template-update verification)

**Author:** Claude (Opus 4.8), code-driven
**Date:** 2026-06-05
**Status:** Concluded — net **negative** on the 89-video test set at every configuration tested (best −0.0065 vs base). **The implementation was subsequently removed from the codebase** (it offered no benefit here); this report is retained as the historical record. The file/line references in §2 point to code as it existed at removal time and are no longer live.
**Source paper:** Lee, Choi, Lee, Yoo, Yang, Hwang, *"BackTrack: Robust template update via Backward Tracking of candidate template"* (arXiv:2308.10604), Samsung SAIT / KAIST / AITRICS.

---

## 0. TL;DR

We ported BackTrack to gate SiamABC's online dynamic-template update: before committing a candidate template, track it **backward** through the recent past frames; commit only if the backward tracklet reproduces the forward tracklet (count of IoU>0.5 hits > `M_thres` **and** start-frame IoU > `sigma_thres`). The intent is to stop *model drift* from committing a distractor that merely looks like the target. It was implemented and then tuned through five progressively fairer configurations.

**Result: no configuration beat base. Every variant lands −0.006 to −0.014 below base. BackTrack is redundant with — and slightly worse than — SiamRAM's existing template-admission gates and recovery stack.**

| Variant | Final score | Δ vs base | Verdict |
|---|---|---|---|
| **Base** (all research features off) | **0.7719** | — | reference |
| BT σ=0.7, `min_area=4096` — pre-fix (reject-spiral) | 0.7583 | −0.0136 | broken: `acc=0` |
| BT σ=0.7, `min_area=4096` — post per-clip fix | 0.7592 | −0.0127 | accepts, but size-gate freezes small targets |
| BT σ=0.8, `min_area=0` (paper-optimal gating) | 0.7644 | −0.0075 | fair test; still redundant |
| **BT σ=0.8, `min_area=0`, replace-select-gate (floor 0)** | **0.7654** | **−0.0065** | BackTrack as sole arbiter; best result, still < base |

(Final score = `W1·AUC + W2·NormPrec` over datasets 2–5; base AUC 0.7351 / NormPrec 0.8271.)

**Why it didn't work** — one structural reason, grounded in the source paper:

> On a well-tracked benchmark (AUC 0.73) backed by SiamRAM's recovery stack, dynamic-template updates are **overwhelmingly beneficial**, so any gate that *rejects* updates costs more (staleness) than it saves (avoided drift). BackTrack wins on distractor-dense LaSOT/LaSOText where bad updates are common; here, bad updates are rare — the base tracker already prevents them — so BackTrack's vetoes are mostly **false rejections of good updates.**

The paper itself flags this: combining BackTrack with an existing confidence head gave "no feasible synergy" (−1.0% on LaSOText). SiamRAM has an *even stronger* front-end (`conf≥0.83` + `IoU≥0.6` admission **plus** YOLO/DRM occlusion+distractor recovery), so BackTrack has almost nothing left to usefully catch.

---

## 1. Background — what BackTrack is and the hypothesis

The danger BackTrack targets is **model drift**: an inaccurate or ill-timed dynamic-template update (e.g. a look-alike distractor) replaces the good template and degrades all subsequent tracking. Appearance-only confidence cannot tell a genuine target from a similar distractor because it ignores the *past trajectory*.

**The mechanism (paper §"Proposed Method"):** given a candidate template `z*_N` extracted at frame `N` and a buffer of past `(image, forward_bbox)` pairs, re-run the tracker on each past frame using **only the candidate** as the template (backward tracking), and compare each backward box to the stored forward box by IoU:
- **count metric** (Eq. 3): `M` = number of past frames with `IoU(fwd, bwd) > 0.5`;
- **start-IoU metric** (Eq. 4): `Σ0` = IoU at the oldest frame.

Commit the candidate iff `M > M_thres AND Σ0 > sigma_thres`, with `M_thres = floor(N · sigma_thres)`. Paper default `sigma_thres ≈ 0.9`, `N = 10–15`. Two efficiency tricks: **early rejection** (skip backward tracking if the candidate bbox is smaller than the template input resolution `64²` — "already blurred") and **early termination** (stop at the first IoU<0.5 miss). The backward tracker need not be accurate — a small/cheap tracker suffices.

**The SiamRAM hypothesis.** SiamABC already refreshes its dynamic template every N frames from the newest high-confidence frame in memory (`select_representatives`). Wrap that commit with BackTrack so trajectory-inconsistent candidates (distractors) are rejected. The host wires SiamABC's own forward function as the backward tracker.

---

## 2. Implementation map (files & entry points)

All code is **config-gated and off by default.** Disabled → the legacy immediate-commit dynamic-template path is byte-for-byte unchanged.

| Concern | Location |
|---|---|
| Verifier (count/start-IoU metrics, early reject/terminate, range/step machine) | `models/siamram/backtrack.py` → `BackTrackVerifier.verify/_backtrack` |
| Candidate selection + BackTrack gate call | `models/SiamABC/tracker/SiamABC_Tracker.py` → `select_representatives()` (~L605) |
| Backward-tracking wiring (template swap, past-frame window, `run_track_for_candidate`) | `models/SiamABC/tracker/SiamABC_Tracker.py` → `_backtrack_accepts()` (~L649) |
| Verifier construction | `models/siamram/tracker.py` → `BackTrackVerifier(...)` (~L914) |
| Per-clip re-attach of verifier onto live inner tracker | `models/siamram/tracker.py` → `_attach_inner_feature_state()` |
| Config dataclass / flatten | `models/siamram/config.py` → `BackTrackConfig` |
| YAML knobs | `config/inference_config_experimental.yaml` → `ram_tracker.backtrack:` |
| Flag-gated telemetry (`BT(cand … acc … rej …)` funnel) | `models/siamram/tracker.py` → `_tlm_*`; inner `_record()` in `_backtrack_accepts` |

---

## 3. The journey (what we tried, in order)

### 3.1 v0 — the reject-spiral (`acc=0`)
First run: **0.7583, −0.0136**. Telemetry: `BT(cand=1822 acc=0 rej=1822 [small=1168 miss=6 count=648 sigma=0])`. BackTrack **never accepted a single candidate** across 89 videos — it was pure update-suppression (its worst possible form).

**Root cause — a structural reject-spiral, two compounding bugs:**
1. **Fixed `M_thres` vs. a short sliding window.** `M_thres = floor(15·0.7) = 10` requires **>10 IoU hits**. The host builds the backward window from a *sliding* memory deque (≤30 frames, but **≤10 when the target moves fast** via `target_motion_adapt`). With ≤10 frames, the count gate is **mathematically impossible** to pass.
2. **Unbounded `k_step` growth.** On every reject, `k_step += 1`; the host subsamples `memory[:latest_idx][::k_step]`, so growing `k_step` only *removes* frames (unlike the paper, the host never *widens* the range — it ignores `tau_start`). Since it never accepts, `k_step` climbs forever, shrinking the window further → permanent spiral.

### 3.2 Fix #1 — dynamic `M_thres` + `k_step` cap
Two config-gated knobs (defaults make BackTrack able to accept; legacy restorable):
- `backtrack_relative_m_thres` (default true): `M_thres = floor(min(N, len(past_frames)) · sigma_thres)` — scales the threshold to the *actual* number of backward frames.
- `backtrack_max_k_step` (default 1): pins `k_step` so reject-driven growth can't collapse the sliding window.

Result: **0.7592**, `BT(cand=1841 acc=698 rej=1143 [small=1129 miss=9 count=0 sigma=5])`. The `count` gate dropped to **0** — the trap was gone. But still −0.0127, now for a different reason.

### 3.3 The size-gate diagnosis — small targets frozen
Of 1143 rejections, **1129 were `small`** (early-rejection, bbox area < `64²=4096`px²). On small-target clips (bus1-n, couple, jogging2, person15, **all of dataset5** — uav/person/boat) *every* candidate tripped the size gate → the dynamic template was **frozen for the whole clip** (`acc=0`, all `small`). That staleness was the dominant cost.

The paper ties `min_sz_Z` to the **template input resolution** and calls early rejection a *speed-only* optimization with "no performance degeneration" — *on the assumption such candidates get rejected by backward tracking anyway*. That assumption breaks here: SiamABC re-crops every candidate to `template_crop_size=128` regardless, so the "already blurred" rationale is weak, and the gate just freezes valid small-target updates.

### 3.4 Fix #2 — `sigma_thres` and the dead size gate
Two pure-config changes, both grounded in the paper:
- **`backtrack_sigma_thres: 0.7 → 0.8`.** Paper supplementary **Table 6** (OSTrack/LaSOT, N=15): σ=0.7 gives **+0.7%** (the *worst* row) while σ=0.8/0.9 give **+2.0/+2.2%**. At σ=0.9 the paper's backward test rejects ~35% of candidates (hit ratio 65%) — *that rejection is where the benefit comes from*. Our σ=0.7 run accepted ~98% (rejected 2%), too lenient to catch anything.
- **`backtrack_min_candidate_area: 4096 → 0`.** Removes the size short-circuit; let the backward test decide for all candidates.

Result: **0.7644, −0.0075**. `BT(cand=1897 acc=1758 rej=139 [small=0 miss=26 count=0 sigma=113])`. Size freezes gone; BackTrack now does its real job (rejects 7.3% on trajectory disagreement). This is BackTrack's **paper-optimal gating** — and it's still −0.0075.

### 3.5 The redundancy realization → replace the gate
The 139 rejections are the problem: **every candidate reaching BackTrack has already passed SiamABC's `conf≥0.83` + `IoU≥0.6` admission gates.** BackTrack's `sigma` vetoes are overruling updates the base tracker already validated → net staleness, not drift-prevention. This is exactly the paper's "no synergy with a confidence head" finding.

The only path to a *win* is the paper's intended usage: make BackTrack **replace** the confidence gate rather than stack on it. Two config-gated knobs:
- `backtrack_replace_select_gate` (default false): drop the `score ≥ dynamic_update_threshold` (0.83) candidate-selection scan; the candidate becomes the newest stored frame with `score ≥ backtrack_select_conf_floor`, and BackTrack is the **sole** arbiter (paper-faithful — the N-th frame becomes the candidate unconditionally).
- `backtrack_select_conf_floor` (default 0.0): 0.0 = full replace; raise toward 0.65 to "loosen but not fully replace."

Result (`replace_select_gate=true`, floor 0.0): **0.7654, −0.0065**. `BT(cand=1931 acc=1765 rej=166 [small=0 miss=60 count=0 sigma=106])`.

---

## 4. Results — the replace-gate run (best configuration)

```
flags(sprt=off bt=on:sigma=0.8 drm=off)
BT(cand=1931 acc=1765 rej=166 [small=0 miss=60 count=0 sigma=106])   occ_entries=54
Mean AUC 0.7298 · NormPrec 0.8189 · Final 0.7654
```

**Final 0.7654 vs base 0.7719 → −0.0065.** Best of five configs, still below base.

---

## 5. Root cause — redundancy, not a tuning miss

Full gate-replacement gained only **+0.0010** over the stacked version (0.7644 → 0.7654), and `cand` barely moved (1897 → 1931, +34). The reason is in the gate structure: the **admission** gate (stage 1, in `update()`) filters frames *before* memory (`score > running_confidence` EMA), so by the time the **selection** gate runs, nearly everything is already high-confidence. The selection threshold was never the binding constraint — so replacing it exposed almost no new candidates.

The trend across all five configs is **monotonic toward base from below**: every loosening helps a little, but only by approaching "do nothing." There is no configuration where BackTrack's rejections are net-positive on this set. The 166 trajectory vetoes (mostly `sigma`) are predominantly *false* rejections — the candidates were genuine target frames the appearance gate already blessed, and not updating cost accuracy.

---

## 6. Why BackTrack has no headroom here (the durable lessons)

1. **Update-rejection is net-harmful on a well-tracked set.** When template updates are overwhelmingly good (base AUC 0.735, strong admission gates), any verification gate that *rejects* updates trades a large staleness cost for a tiny drift-avoidance benefit. BackTrack's value scales with the *false-positive update rate* of the base tracker — which is low here.
2. **Redundant with the existing front-end.** SiamRAM already gates updates by confidence **and** IoU continuity, and handles the distractor/occlusion cases BackTrack targets via its YOLO+DRM recovery stack. The paper's own "no synergy with a confidence head" result predicts the stacking penalty; SiamRAM's front-end is stronger still.
3. **The binding gate is admission, not selection.** Candidate eligibility is set by what enters memory, not by the selection scan — so "replacing the selection gate" is nearly a no-op. Truly making BackTrack primary would require loosening the *admission* gate, which admits weaker templates BackTrack still can't usefully filter and changes base behavior globally (predicted worse — **not pursued**).

These points predict the observed outcome and would apply to any backward-tracking template-verification port on a saturated, low-distractor benchmark.

---

## 7. Config reference

Block: `ram_tracker.backtrack:` in `config/inference_config_experimental.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `backtrack_enabled` | `false` | Master switch. Off → verifier never built; legacy immediate-commit path unchanged. |
| `backtrack_sigma_thres` | `0.8` | Start-frame IoU threshold (Eq. 4) and basis for `M_thres`. Paper Table 6: 0.7 worst, 0.8/0.9 best at N=15. |
| `backtrack_n_window` | `15` | `N` — frames per backward cycle; sizes `M_thres`. |
| `backtrack_min_candidate_area` | `0` | Early-rejection size floor (px²). `0` disables the size short-circuit (it froze small-target updates here). Paper uses `64²`. |
| `backtrack_max_backward_steps` | `0` | Hard cap on backward passes per cycle. `0` = no cap. |
| `backtrack_relative_m_thres` | `true` | `M_thres = floor(min(N, len(past_frames))·sigma)` — scales threshold to the real window length. Fixes the `acc=0` reject-spiral. `false` = legacy fixed `floor(N·sigma)`. |
| `backtrack_max_k_step` | `1` | Cap on per-reject `k_step` growth. `1` pins the window (correct for this sliding-window host). `≤0` = legacy unbounded growth. |
| `backtrack_replace_select_gate` | `false` | Make BackTrack the **sole** update arbiter by replacing the `score≥0.83` candidate-selection gate with its backward-track verdict. The paper's intended usage. |
| `backtrack_select_conf_floor` | `0.0` | Min candidate score in replace mode. `0.0` = newest frame regardless (full replace). Raise toward `0.65` to loosen-not-replace. Inert unless `replace_select_gate` is true. |

**Coupled tracker knobs (separate block, `tracker:`):** `template_admit_conf_threshold` (0.83) and `template_admit_iou_threshold` (0.6) are the *admission* gates that actually bound the candidate pool. Loosening them is the only untested lever to make BackTrack truly primary — **predicted harmful, not pursued.**

---

## 8. How to reproduce

All runs use the 89-video test set; only the `ram_tracker.backtrack:` flags change. **No code edits needed.** Turn on `ram_tracker.telemetry.research_telemetry_enabled: true` for the per-video and cumulative `BT(cand … acc … rej [small|miss|count|sigma])` funnel.

- **Base (reference 0.7719):** `backtrack_enabled: false`.
- **Reject-spiral repro (0.7583):** `backtrack_enabled: true`, `backtrack_sigma_thres: 0.7`, `backtrack_min_candidate_area: 4096`, `backtrack_relative_m_thres: false`, `backtrack_max_k_step: 0`.
- **Post per-clip + dynamic-M_thres fix (0.7592):** as above but `backtrack_relative_m_thres: true`, `backtrack_max_k_step: 1`.
- **Paper-optimal gating (0.7644):** `backtrack_sigma_thres: 0.8`, `backtrack_min_candidate_area: 0`, `backtrack_replace_select_gate: false`.
- **Replace-gate / best (0.7654):** `backtrack_sigma_thres: 0.8`, `backtrack_min_candidate_area: 0`, `backtrack_replace_select_gate: true`, `backtrack_select_conf_floor: 0.0`.

---

## 9. Conclusion & recommendation

BackTrack is **correctly implemented** (the reject-spiral was a real bug, now fixed and unit-verified) but has **no headroom on this benchmark** in any of the five configurations — best case 0.7654 < base 0.7719. The cause is structural: on a saturated, low-distractor set with SiamRAM's strong update gates and recovery stack, verifying-and-rejecting template updates costs more staleness than it prevents drift. Recommended state — the default — is **off**:

```
backtrack_enabled: false
```

If BackTrack is ever revisited, do it on **distractor-dense evaluation data** (à la DAM4SAM's DiDi) where false-positive template updates are common, and consider letting it **replace** rather than stack on the admission gates (loosen `template_admit_conf_threshold`/`template_admit_iou_threshold` alongside `backtrack_replace_select_gate`). On this benchmark, expectations should stay at "back toward base," not a win.

**Reusable wins from this experiment (independent of BackTrack's verdict):**
- Fixed the **reject-spiral** (dynamic `M_thres` + `k_step` cap) — a candidate-verifier whose acceptance threshold exceeded the available window could *never* accept.
- Identified the **size-gate / small-target freeze** — a paper "speed-only" optimization that silently freezes updates on small-target clips.
- Confirmed the **admission-vs-selection** gate structure — the binding constraint on candidate eligibility is memory admission, not the selection scan.
- Reused the **flag-gated research telemetry** (`BT(...)` funnel) that made every run self-documenting.

---

## Appendix — related research features (same investigation)

| Feature | Paper | Verdict on this set |
|---|---|---|
| **SPRT** e-process occlusion entry | arXiv:2602.12983 | Dead at `entry_patience:1`; `and`/`replace` modes cost ~−0.04. Off. |
| **BackTrack** template-update verify | arXiv:2308.10604 | This report — net negative at all 5 configs (best −0.0065). Off. |
| **DRM introspection** | arXiv:2411.17576 | Neutral-to-negative; distractor-bank variant harmful (bank poisoning). Off. See `DRM_INTROSPECTION_EXPERIMENT.md`. |
