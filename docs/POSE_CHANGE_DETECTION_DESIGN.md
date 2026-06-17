# Appearance / Pose-Change Detection — Design Discussion

> **Status:** design discussion only. Nothing here is implemented yet. This
> document captures the full reasoning, the proposed mechanisms, and the exact
> existing signals each one reuses, so implementation can start from a single
> reference.
>
> **Goal:** detect when the *currently tracked* object is undergoing a heavy
> appearance change — rotation, pose change, walking into shadow/dark, sudden
> deformation — so the system can react (e.g. accelerate the dynamic-template
> update, lower the admit threshold, write a new DRM anchor) **and avoid falsely
> entering occlusion-recovery mode just because the object changed how it looks.**

---

## 1. The real problem (why the naive approach fails)

Pose change, "entering the dark", and genuine occlusion **all produce the same
primary symptom**: the appearance-match score drops. You therefore cannot use
"score dropped → react" on its own — that rule cannot tell the three apart.

The danger is **asymmetric**:

- Treating a **pose change as occlusion** → we needlessly enter recovery, freeze
  the template, and may lock onto a distractor. Annoying, but recoverable.
- Treating an **occlusion as a pose change** and reacting by *updating the
  template faster* → we learn the **occluder** into the template. That is
  permanent drift / **template poisoning**. Catastrophic and usually
  unrecoverable.

So the detector must be **discriminative** (separate "changed look" from "gone"),
and the reaction must be **safety-interlocked** (never fast-adapt appearance
unless we are confident the target is still genuinely *there*).

---

## 2. The unifying framework: decouple Localization (L) from Appearance (A)

Track two **orthogonal** per-frame signals:

- **A — appearance match:** how much the current crop looks like what we
  remember.
- **L — localization confidence:** is there a *single, sharp, motion-consistent*
  target where we predicted it?

The decision is a 2×2:

|              | **A high**                | **A low**                              |
| ------------ | ------------------------- | -------------------------------------- |
| **L high**   | normal tracking           | **POSE / APPEARANCE CHANGE → adapt**   |
| **L low**    | rare (distractor-in-wrong-place) → freeze | **OCCLUSION / LOSS → recovery** |

The entire mechanism lives in the **A-low / L-high** cell: the object is
unmistakably *there*, moving as predicted, sharply localized — it just *looks
different*. That is pose / lighting change, and it is exactly where we want to
**accelerate template adaptation and veto occlusion entry.** Occlusion sits one
cell away (A-low **and** L-low), so **L is the discriminator** that keeps the two
apart.

### 2.1 The critical rule: L must be appearance-*independent*

If L includes any appearance-similarity term, then "A low" drags "L low" with it,
the two axes collapse, and you can no longer distinguish pose-change from
occlusion. Therefore:

- **Peak *height* (`pred_score` / `latest_primary_peak_score`) belongs to A, not
  L.** It is literally the appearance-match score at the chosen cell.
- **L is built only from geometry and map-*shape* signals**: *is there a single,
  sharp blob where motion predicts it* — regardless of how well it matches the
  template.

---

## 3. The A axis — how appearance match is measured

Both ingredients already exist:

- **DRM self-score:** `AppearanceMemory.score_target_against_drm`
  (`models/siamram/memory.py:451`). This already scores the genuine target
  against its own DRM bank and is already EMA-smoothed to drive the adaptive
  `drm_margin="auto"` estimator. Reuse it directly. (With the current config the
  per-anchor score is **pure appearance cosine** — `drm_lam_app: 1`, all other
  weights `0` — so this is a clean appearance signal.)
- **Peak height:** `latest_primary_peak_score` / `pred_score` on the SiamABC
  tracker — the response-map confidence at the chosen location.

**Detecting "A dropped":** compare a fast read of A against its own recent
baseline (a slow EMA), or against the auto-margin's EMA. A sustained drop below
baseline = candidate appearance-change event (to be confirmed by L).

---

## 4. The L axis — how localization confidence is computed

**Most of L is already computed in the codebase**; it is simply not yet
assembled into one number. Four components, all appearance-independent:

### 4.1 KF motion innovation — the cleanest L signal
`SiamRAMExperimentTracker._mahalanobis_distance_sq`
(`models/siamram/tracker.py:2249`):

```
d² = (obs − kf_pred)ᵀ · Σ⁻¹ · (obs − kf_pred)
```

Pure geometry, already χ²-calibrated (the distractor gate uses
`distractor_mode_mahalanobis_threshold = 9.21` = 99% for 2 DOF). Low `d²` = the
target is exactly where constant-velocity motion predicts → strong evidence we
are still tracking the real object through a pose change. A spike = jump /
occlusion.

- Normalize: `L_motion = exp(−d² / τ)` (soft), or a gate at a **gentler**
  threshold than the distractor gate.

### 4.2 Peak dominance / unimodality — the PSR analogue
`secondary_peak_divergence` (`models/siamram/drm_introspection.py:65`) already
extracts the **second-strongest** response peak every frame and stores:
`latest_primary_peak_score`, `latest_secondary_peak_score`,
`latest_secondary_peak_ratio` (computed in `run_track`, where
`secondary_peak_divergence` is called around
`models/SiamABC/tracker/SiamABC_Tracker.py:1201`).

This is *shape*, not absolute level:

- a **dim but unimodal** map (weak secondary) = pose change / fading light;
- a **multi-modal** map (strong secondary, low divergence ratio) = competing
  hypothesis = occlusion / distractor.

- Normalize: `L_shape = 1 − (secondary_score / primary_score)`, clamped to
  [0, 1]; optionally fold in `latest_secondary_peak_ratio` for spatial
  separation.

### 4.3 Frame-to-frame spatial continuity
`latest_iou_score` / `latest_iou_score_ema` on the SiamABC tracker (already read
by SiamRAM at `models/siamram/tracker.py:3630`). Smooth motion keeps consecutive
boxes overlapping; a collapse or teleport drops it.

- Normalize: `L_iou = latest_iou_score_ema` (already in [0, 1]).

### 4.4 Size / aspect stability
The introspection module already tracks a size-history deque
(`theta_area = 20%` over `theta_M = 10` frames; see
`models/siamram/drm_introspection.py` header). Rotation changes aspect
*smoothly*; occlusion makes the box *collapse or jump*. Use size **jumpiness**
(deviation from the rolling median), not absolute change:

- Normalize: `L_size = 1 − min(1, |area − median| / (theta_area · median))`.

### 4.5 Combining the components — AND-semantics, not a weighted sum
These are near-*necessary* conditions for "cleanly localized": if *any one* says
"not a clean, predictable target", L should be low. So use a **product (or
soft-min)**:

```
L = L_motion · L_shape · L_iou        (· L_size)
```

A weighted sum would let a great IoU mask a motion-innovation spike — exactly the
case (a distractor sitting near the predicted box) where we must **not** declare
"pose change". The product won't.

**Safety floor:** additionally put a hard gate on the motion term — if
`d² > threshold`, force `L ≈ 0` regardless of the other terms. Never fast-adapt
the template when the target is not where motion says it should be.

---

## 5. Detector designs (ranked by fit with this system)

These are the candidate ways to drive the **A axis / change event**; all use L
(Section 4) as the localization gate.

### ① DRM-bank novelty + localization gate — *most compatible with the memory design*
The DRM bank is already a 50-deep multi-view appearance model
(`drm_capacity: 50`) — a sampled *manifold* of every look the target has had. So
define:

```
novelty = 1 − max cosine-sim(current_crop, any DRM anchor)
```

reusing the machinery of `score_target_against_drm`.

- novelty low → current look is "in-distribution" (a remembered pose) → safe.
- novelty high **and L high** → genuinely new appearance of the *real* target →
  fire the response **and write the new look as a DRM anchor**
  (`AppearanceMemory.add_drm_anchor`, `models/siamram/memory.py:222`, already
  exists for introspective writes). This *grows the manifold*, which also makes
  future re-acquisition robust to that pose.
- novelty high **and L low** → occlusion; do not touch the bank.

Elegant because one concept (DRM as appearance memory) powers both *detection*
and the *cure*, and it composes with the top-k aggregation knobs
(`drm_score_aggregation` / `drm_score_aggregation_topk`).

### ② Fast/slow appearance EMA divergence — *cheapest, most robust*
Keep two EMAs of the target descriptor: a **fast** one (adapts in a few frames)
and a **slow** one (stable over many). Their divergence `1 − cos(fast, slow)`
spikes precisely during *active* appearance change and settles once the new look
stabilizes — giving a clean **onset/offset** signal for hysteresis. Equivalent in
spirit: compare current-crop similarity to the **static** template vs the
**dynamic** template — when *both* drop together, the change is happening *now* (a
dynamic template that already tracked the drift would still match).

### ③ Response-map PSR / entropy — *the best occlusion discriminator (the L oracle)*
This is largely **already built** via `secondary_peak_divergence` (Section 4.2).
A sharp unimodal map = "target clearly here, just dimmer" (pose change); a flat or
multi-modal map = "target lost" (occlusion). Entropy of the soft-maxed response
map is the alternative formulation if the full map is available.

### Illumination special case ("entering the dark")
Worth a dedicated cheap check because it has a tell occlusion lacks: it is
**global**. Track mean luminance of the search crop **and a background ring**
around the box. If the target box and the surrounding ring dim **together** while
edge structure is preserved → illumination shift, not occlusion (an occluder dims
the box but **not** the background). Feed the same "appearance change, not loss"
conclusion.

> **Note:** do **not** re-normalize the crop before the frozen backbone — the
> system is deliberately careful about byte-for-byte backbone input. Use
> illumination only as a *gating signal*, never as a preprocessing change.

---

## 6. Reaction policy (once "A-low / L-high" fires)

Most levers already exist; this just sequences them, with **hysteresis** so the
system does not flutter in and out:

1. **Veto / delay occlusion entry — the single most important action.** Occlusion
   entry already requires the score to stay low for `entry_patience` consecutive
   frames, gated by `enter_occlusion_on_loss` (see
   `models/siamram/tracker.py:603`, grace period around
   `models/siamram/tracker.py:3540`). While pose-change is active, **extend
   `entry_patience`** (or inject a hold/veto) on that same low-score counter.
   This directly delivers "don't enter occlusion on pose change."
2. **Accelerate template adaptation transiently.** Drop the dynamic-template
   interval (reuse the `template_rate` auto infrastructure) and/or lower
   `admit_conf_threshold` for the duration, so the genuinely-changing target
   refreshes the template instead of being rejected for "low confidence".
3. **Expand the DRM manifold.** `add_drm_anchor` on the new look — **only while L
   is high.**
4. **Decay back** to steady parameters once the fast/slow EMAs reconverge (the
   change has finished).

---

## 7. The non-negotiable safety interlock

**L gates everything.** Acceleration of template update, lowering the admit
threshold, and DRM writes happen **only when L is high.** The instant L drops
(peak goes diffuse / KF innovation spikes), appearance adaptation must **freeze**
and the normal occlusion path takes over — even mid-event.

This single rule is what guarantees an occluder is never learned: pose-change
handling makes the system **more permissive about appearance** but **never more
permissive about localization.**

---

## 8. Integration / hook points

- **Where the signals live:**
  - Peak signals + frame-to-frame IoU live on the **SiamABC** tracker
    (`self.tracker.latest_*`).
  - Mahalanobis distance + KF state live on the **SiamRAM** tracker.
  - SiamRAM already reaches across for `latest_iou_score`
    (`models/siamram/tracker.py:3630`), so **L is naturally assembled at the
    SiamRAM level** — the same place the `entry_patience` occlusion-entry
    decision and the template-rate levers live. Detector, veto, and reaction all
    co-locate cleanly.
- **Decoupled-module style:** follow the existing pattern of `auto_margin.py`,
  `auto_template_rate.py`, and `drm_introspection.py` — a tracker-agnostic object
  that consumes scalars (A, the four L terms, frame index) and emits a single
  decision + state, with the tracker owning the actual reactions. When disabled,
  the tracker never constructs it and existing paths are byte-for-byte untouched.

---

## 9. Practical wrinkles

- **KF warmup.** The motion prior only engages after
  `kf_motion_stable_frames: 15`. Before that, `L_motion` is not meaningful — fall
  back to `L_shape · L_iou` during warmup, and only let appearance-adaptation
  accelerate once the KF is live.
- **Threshold reuse.** The Mahalanobis machinery is currently tuned for the
  distractor jump-gate (`meas_var = 25`, threshold `9.21`). For L, use the same
  *distance* but a separate, **gentler** threshold — a pose change should not have
  to clear as strict a bar as a distractor teleport.
- **Anchor write rate.** Confirm whether `add_drm_anchor` writes are rate-limited
  before wiring step 6.3 (the introspection path enforces a `delta = 5` frame gap;
  the manual path may not).

---

## 10. Recommended starting spine

If one path is to be built first:

- **A axis:** DRM-bank novelty (① — reuses `score_target_against_drm`, composes
  with the top-k work).
- **L axis:** the product `L_motion · L_shape · L_iou` from Section 4, with the
  Mahalanobis hard floor.
- **Hysteresis:** the fast/slow EMA (②) for clean onset/offset of the event.
- **Reaction:** veto via `entry_patience` extension (6.1) + transient
  template-rate / admit-threshold relaxation (6.2) + gated `add_drm_anchor`
  (6.3), all under the L interlock (Section 7).

**One-line summary:** `L = (KF-innovation goodness) × (peak unimodality) ×
(frame-to-frame IoU)`, motion as a hard gate; `A = DRM self-score + peak height`.
**Pose change = A drops while L holds** — that is the cell to act on.

---

## 11. Open questions to verify before implementation

1. Does SiamABC expose the **full raw response map** per frame (not just the two
   peaks), in case entropy is preferred over the secondary-peak ratio?
2. The exact "low-score" quantity that `entry_patience` counts, so the veto hooks
   the *same* signal it gates on.
3. Whether `add_drm_anchor` is rate-limited on the manual path.
4. Availability of the **KF covariance Σ** at the moment of the occlusion-entry
   check (needed to compute `L_motion` there).
