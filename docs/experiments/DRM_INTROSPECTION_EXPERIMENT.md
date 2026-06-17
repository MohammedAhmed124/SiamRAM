# DRM Introspection — Experiment Report (DAM4SAM-style distractor memory)

**Author:** Claude (Opus 4.8), code-driven
**Date:** 2026-06-05
**Status:** Concluded — net **neutral-to-negative** on the 89-video test set. All code retained, **config-gated, off by default.**
**Source paper:** Videnović, Lukežič, Kristan, *"A Distractor-Aware Memory for Visual Object Tracking with SAM2"* (arXiv:2411.17576) — the work that originated the RAM/DRM split SiamRAM already borrows.

---

## 0. TL;DR

We ported DAM4SAM's **introspection-based DRM (Distractor-Resolving Memory) update** to SiamRAM: when the tracker's own response map reveals a competing secondary peak (a distractor) during reliable tracking, record that frame so re-acquisition can tell the target from the look-alike. It was implemented in four progressively more aggressive forms, each fairly tested.

**Result: no form helped on this benchmark; the most complete form actively hurt.**

| Variant | Final score | Δ vs base | Verdict |
|---|---|---|---|
| **Base** (all research features off) | **0.7719** | — | reference |
| DRM positive anchors (after per-clip fix) | 0.7720 | **~0.0000** | neutral |
| DRM + scale-aware divergence | 0.7720 | ~0.0000 | neutral (fixes a dead knob, no score effect) |
| DRM + distractor bank + `gamma=0.3` | **0.7529** | **−0.0190** | **harmful — reverted** |

(Final score = `W1·AUC + W2·NormPrec` over datasets 2–5; base AUC 0.7351 / NormPrec 0.8271.)

**Why it didn't work** — two independent reasons, both grounded in the source paper:
1. **No every-frame readout.** In SAM2, the memory (RAM+DRM) is *cross-attended by every frame*, so DRM anchors continuously shape localization. SiamABC's "DRM" is a re-ID/recovery bank consulted **only during occlusion/distractor recovery** — and this set has only ~47–55 occlusion entries across 89 videos. The writes happen; almost nothing reads them.
2. **Saturated benchmark.** DAM4SAM's authors built the distractor-distilled *DiDi* dataset precisely because standard sets are "no longer challenging… high performance overwhelms the total score." This test set is exactly that kind: AUC 0.735, sparse occlusion. DRM's gain is structurally invisible on it.

The distractor-bank variant additionally **poisoned** the negative bank (see §5).

---

## 1. Background — what DRM is and the hypothesis

DAM4SAM splits the tracker memory by *function*:
- **RAM** (Recent Appearance Memory) — recent target looks, for segmentation/localization accuracy.
- **DRM** (Distractor-Resolving Memory) — *anchor* frames captured when a distractor is present, for robustness/re-detection.

DRM's **introspection update rule** (paper §3.2.2): write a DRM anchor only when **all** hold —
1. target present;
2. ≥ `δ=5` frames since the last DRM update;
3. **hypothesis divergence** — a box fit to the predicted mask vs. the union with the *alternative* mask; if the area ratio < `θ_anc=0.7`, a competing distractor is detected;
4. **reliable tracking** — predicted IoU > `θ_iou=0.8` **and** target area within `θ_area=20%` of the median area over the last `θ_M=10` frames.

**The SiamRAM hypothesis.** SiamABC has no masks, but it has a 16×16 FCOS-style response map. The analogue of SAM2's "alternative mask" is the **second-strongest local peak**. If we detect a strong, separated secondary peak during reliable tracking, we can (a) record DRM anchors and/or (b) record the distractor itself, so the recovery pipeline discriminates target from look-alike.

---

## 2. Implementation map (files & entry points)

All code is **config-gated and off by default**. Disabled → the legacy paths are byte-for-byte unchanged.

| Concern | Location |
|---|---|
| Response-map secondary-peak signal + divergence ratio | `models/siamram/drm_introspection.py` → `secondary_peak_divergence(...)` |
| Introspection gate (paper's 4 conditions) | `models/siamram/drm_introspection.py` → `DrmIntrospectionUpdater.evaluate/should_update` |
| Inner tracker computes & stashes the signal | `models/SiamABC/tracker/SiamABC_Tracker.py` → `track()` DRM block (~L1094) |
| Per-clip re-attach of inner feature state | `models/siamram/tracker.py` → `_attach_inner_feature_state()` (~L1422) |
| Positive DRM anchor write | `models/siamram/tracker.py` → `_maybe_update_drm_introspection()` (~L3004) → `memory.add_drm_anchor()` |
| Negative distractor-bank write | `models/siamram/tracker.py` → `_maybe_write_distractor_anchor()` → `_add_distractor_descriptor()` (~L1979) |
| DRM bank consumed (recovery scoring) | `models/siamram/memory.py` → `drm_match()` (`lam_app·cos_sim …`) |
| Negative bank consumed (suppressor) | `models/siamram/memory.py` → `drm_match()` `− gamma·dist_sim` (~L355) |
| Config dataclass / flatten | `models/siamram/config.py` → `DrmIntrospectionConfig` |
| YAML knobs | `config/inference_config_experimental.yaml` → `ram_tracker.drm_introspection:` |
| Flag-gated telemetry | `models/siamram/tracker.py` → `_TLM_KEYS`, `_tlm_format()` |

---

## 3. The journey (what we tried, in order)

### 3.1 v0 — positive anchors only, and the bug that hid everything
First version wrote positive DRM anchors via `memory.add_drm_anchor()`. Manual test: **byte-identical to base** (0.7719). Telemetry showed the divergence signal stuck at the inert default `1.0` and inner counters at 0.

**Root cause — per-clip inner-state loss.** The inner `SiamABCTracker` is reset/recreated per clip and its `__init__` restores inert defaults (`_drm_introspection_enabled=False`, etc.). The outer tracker attached the feature flags **once** in its own `__init__`, so from clip 2 onward the live inner tracker had lost them. DRM never actually ran.

**Fix:** `_attach_inner_feature_state()` — a single source of truth called from **both** `__init__` and `initialize()` (the per-clip entry), so every clip's inner tracker carries the flags. Telemetry confirmed the signal went live (`ratio≠1.0`, inner counters non-zero).

### 3.2 First fair test — positive anchors are neutral
With the signal real, DRM fired **492 anchors** across 89 videos. Score: **0.7720 vs 0.7719** — neutral. The DRM funnel:
```
DRM(gate_calls=34967 pass=492 [not_present=0 not_due=982 reliab=597 size=3161 diverg=0 sec=29735] anchors=492)
```

### 3.3 The dead-knob diagnosis — `theta_anc` does nothing
`diverg=0` in the funnel means the divergence gate **never** fails — the area ratio is *always* < 0.7. On a 16×16 grid with a fixed 3×3 peak box, any secondary peak ≥3 cells away yields ratio ≤ 0.5. The paper's ratio is meaningful because it fits boxes to **masks** (real spatial extent); our coarse-grid port collapsed it. So `θ_anc` was a tuning knob wired to nothing.

**Fix — scale-aware divergence** (`drm_introspection_scale_aware_divergence`, default true): scale the divergence box **and** the non-max suppression to the target's grid-cell extent (derived from the crop-space predicted box + `instance_size`). A secondary peak within the target's own footprint is now correctly read as a response shoulder (ratio→1.0, no divergence), and `θ_anc` becomes a real selectivity control. Verified by unit smoke (small target → diverges; large target → does not). **Score impact on this set: none** — targets occupy few grid cells, so the gate still rarely binds, and (per §0) DRM has no headroom here regardless.

### 3.4 The architectural realization
A user question — *"which parameter controls the distractor-bank score?"* — exposed the real issue. Tracing the consumption path: positive anchors feed `drm_match` via `lam_app·cos_sim` (a **positive** appearance term, weight = 1), but they carry the **target's own descriptor** — redundant with the existing positive refs, so adding them barely moves the max-appearance score. The genuine discrimination lever in this codebase is the **negative distractor bank**, weighted by `gamma`, which the introspection feature **never fed**.

### 3.5 The distractor-bank variant (the real mechanism)
New, config-gated path (`drm_introspection_distractor_bank_enabled`, default false):
1. `secondary_peak_divergence` now also returns the secondary-peak grid location.
2. The inner tracker maps it to a frame-space box (riding the same `_rescale_bbox` transform as the predicted box, shifted by the primary↔secondary grid offset).
3. On reliable frames with a detected distractor, the outer tracker extracts a descriptor at that box and writes it to the **negative bank** (`_add_distractor_descriptor` → local deque **and** `memory.add_distractor`).
4. `drm_match` penalizes recovery candidates matching the bank via `− gamma·dist_sim`.

This is the **only** introspection path that touches the discrimination lever. Tested with `drm_gamma: 0.3`.

---

## 4. Results — the distractor-bank run

```
flags(sprt=off bt=off drm=on)
DRM(gate_calls=33648 pass=485 [reliab=509 size=3044 diverg=0 sec=28642]
    anchors=485 dist_anchors=1022)   occ_entries=47
Mean AUC 0.7177 · NormPrec 0.8058 · Final 0.7529
```

**Final 0.7529 vs base 0.7719 → −0.0190.** Net harmful.

---

## 5. Root cause of the regression — bank poisoning

The smoking gun: `dist_anchors=1022` is **more than 2× the positive `anchors=485`**. The distractor write fires on nearly every reliable frame with a secondary peak. On a 16×16 map most "secondary peaks" are the **target's own response shoulder or background**, not a genuine look-alike. So the negative bank fills with descriptors **similar to the target itself**. During re-acquisition, `gamma·dist_sim` then penalizes candidates that look like those descriptors — i.e. **it penalizes the real target**, corrupting recovery. The shifted `occ_entries` (55→47) is a second-order symptom of changed recovery outcomes.

This is the classic negative-bank failure mode: a single impure distractor sample poisons the suppressor. A targeted guard exists (only bank a secondary descriptor when `cos_sim(sec, target) < τ`), but it was **not pursued** — on a low-distractor set even a clean bank has little to suppress, so the expected upside is "back toward neutral," not a win.

---

## 6. Why DRM has no headroom here (the durable lessons)

1. **Architecture mismatch.** SAM2's memory is *read every frame* via cross-attention; DRM anchors continuously shape localization. SiamABC's "DRM" is a re-ID/recovery bank read **only in occlusion/distractor recovery** (`drm_match`, distractor mode). With ~47–55 recovery episodes across 89 videos, writes rarely get read. **Porting a memory mechanism without porting its readout yields a no-op.**
2. **Dataset saturation.** DAM4SAM built the *DiDi* dataset (≥⅓ frames with strong distractors) specifically because standard sets hide distractor-handling gains. This test set (AUC 0.735, sparse occlusion) is exactly the saturated kind. **Distractor-robustness features cannot be measured here — let alone shown to help.**
3. **Coarse-grid signal is a weak distractor proxy.** A 16×16 response map's "second peak" is too often the target's own structure; using it as a distractor sample without a purity guard poisons downstream suppression.

These three points predict the observed outcome and would apply to any similar appearance/distractor-memory port on this benchmark.

---

## 7. Config reference

Block: `ram_tracker.drm_introspection:` in `config/inference_config_experimental.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `drm_introspection_enabled` | `false` | Master switch. Off → inner tracker never computes the secondary-peak signal; legacy DRM write path unchanged. |
| `drm_introspection_theta_anc` | `0.7` | Divergence trigger (ratio < this ⇒ distractor). Real knob **only** with scale-aware on. |
| `drm_introspection_theta_iou` | `0.8` | Reliability gate on per-frame score. |
| `drm_introspection_theta_area` | `0.2` | Size-stability gate (fraction of median area over `theta_M`). |
| `drm_introspection_theta_M` | `10` | Frames for the median-area stability test. |
| `drm_introspection_delta` | `5` | Min frame spacing between **positive** anchor writes. |
| `drm_introspection_secondary_min_ratio` | `0.5` | Min secondary/primary peak strength to count as a real competitor. |
| `drm_introspection_scale_aware_divergence` | `true` | Scale divergence box + suppression to target grid extent (fixes the dead `theta_anc`). `false` = legacy degenerate 3×3. |
| `drm_introspection_distractor_bank_enabled` | `false` | Write secondary-peak (distractor) descriptors into the **negative** bank. **Inert unless a recovery `gamma` > 0.** Harmful on this set (§5). |

**Coupled knob (separate block, `ram_tracker.occlusion.phase2_final_drm.drm`):** `drm_gamma` (default `0`) — weight of the negative-bank suppressor in `drm_match`. The distractor-bank feature does **nothing** unless this (and/or `distractor_occ_drm_gamma`) is raised above 0.

---

## 8. How to reproduce

All runs use the 89-video test set; only the flags below change. **No code edits needed** — everything is in `config/inference_config_experimental.yaml`.

- **Base (reference 0.7719):** all of `drm_introspection_enabled`, `drm_introspection_distractor_bank_enabled` = `false`; `drm_gamma: 0`.
- **DRM positive anchors (neutral 0.7720):** `drm_introspection_enabled: true`; distractor bank `false`; `drm_gamma: 0`.
- **DRM + distractor bank (harmful 0.7529):** `drm_introspection_enabled: true`; `drm_introspection_distractor_bank_enabled: true`; `drm_gamma: 0.3`.

Turn on `ram_tracker.telemetry.research_telemetry_enabled: true` to print the per-video and cumulative `DRM(... anchors=… dist_anchors=…)` funnel used throughout this report.

---

## 9. Conclusion & recommendation

DRM introspection is **correctly implemented** but has **no headroom on this benchmark** in any form, and the distractor-bank variant is **harmful** (bank poisoning). Recommended state — the current default — is **off**:

```
drm_introspection_enabled: false
drm_introspection_distractor_bank_enabled: false
drm_gamma: 0
```

If DRM is ever revisited, do it on **distractor-dense evaluation data** (à la DiDi), and only with: (a) the scale-aware divergence on, (b) a **purity guard** on distractor-bank writes (`cos_sim(sec, target) < τ`), and (c) `drm_gamma` tuned upward. Without an every-frame memory readout, expectations should stay modest regardless.

**Reusable wins from this experiment (independent of DRM's verdict):**
- Fixed the **per-clip inner-state loss** bug (`_attach_inner_feature_state`) — affected any inner-tracker feature flag, not just DRM.
- Fixed the **degenerate `theta_anc`** (scale-aware divergence) — `theta_anc` is now a real knob.
- Added **flag-gated research telemetry** (the `DRM(...)`/`BT(...)`/`SPRT(...)` funnel) that makes every run self-documenting.

---

## Appendix — related research features (same investigation)

| Feature | Paper | Verdict on this set |
|---|---|---|
| **SPRT** e-process occlusion entry | arXiv:2602.12983 | Dead at `entry_patience:1`; `and`/`replace` modes cost ~−0.04. Off. |
| **BackTrack** template-update verify | arXiv:2308.10604 | Reject-spiral bug fixed (dynamic `m_thres` + `k_step` cap); fair-tested across 5 configs — net negative at all (best −0.0065). Off. See `BACKTRACK_EXPERIMENT.md`. |
| **DRM introspection** | arXiv:2411.17576 | This report — neutral-to-negative. Off. |
