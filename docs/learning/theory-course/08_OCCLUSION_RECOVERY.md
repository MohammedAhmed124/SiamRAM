# 08 — Occlusion Recovery

> Prerequisites: doc 03 (matcher confidence), doc 05 (EKF: predict, coast, reseed, velocity), doc 06 (heavy camera motion), doc 07 (RAM/DRM and the composite re‑acquisition score). This document explains what happens when the target is **lost** — hidden, off‑frame, or otherwise unmatchable. We cover how loss is *detected* (with hysteresis and guards), how the tracker *coasts* on the EKF while searching, the multi‑**phase** re‑detection state machine (fast re‑try → detector candidates → memory scoring → verification), the *geometry* of the growing search region, and how a winner is *verified* before normal tracking resumes. Implemented across `occlusion_recovery.py` and the conductor in `tracker.py`.

---

## 1. Two worlds, one switch

Recall the structural fact from doc 00: **per frame, exactly one of `normal update` or `occlusion update` runs.** A single boolean `in_occlusion` selects the world. The normal world (docs 03/06) assumes the target is visible and trusts the matcher; the occlusion world assumes it is *not* visible and runs a structured search instead.

While `in_occlusion` is true, the public output is special: the tracker emits either zeros or the *held box* (the EKF's best guess), with score $0.0$ and the `in_occlusion` flag set. The "real" internal position is carried in `held_box`; only the **first frame after successful re‑acquisition** emits genuine coordinates again. So from the outside, occlusion looks like "no confident answer," while inside, the machinery is busy coasting and searching.

```
                        score < threshold, sustained
   NORMAL  ───────────────────────────────────────────────►  OCCLUSION
     ▲                                                            │
     │            verified re‑acquisition (confirmed streak)      │
     └────────────────────────────────────────────────────────────┘
```

---

## 2. Detecting loss

A tracker must not panic at a single bad frame, nor dawdle while the target is truly gone. SiamRAM uses **hysteresis** plus several guards.

### 2.1 The effective threshold

Each frame computes an **effective loss threshold** (`_compute_effective_threshold`). It is the configured confidence threshold, *switched* for far/tiny targets: if the object is "long‑distance" (its median area is a tiny fraction of the frame), a separate, lower `long_distance_conf_threshold` is used (such targets genuinely score lower even when tracked correctly — the same difficulty‑awareness that motivates the adaptive thresholds of doc 10), and the matcher's search context is *widened* (offset $+0.5$) to keep the small target in view. The threshold itself can be a fixed number or **adaptive** (doc 10).

### 2.2 Entry hysteresis

A loss is declared only after the score stays below threshold for **several consecutive frames** (`entry_patience`). A streak counter `_entry_streak` increments on each below‑threshold frame and resets to $0$ on any healthy frame; occlusion is entered when

$$
\text{\_entry\_streak} \ge \text{effective\_entry\_patience}.
$$

This **hysteresis** (requiring persistence before switching states) is the standard cure for chattering: a one‑frame dip from blur or a passing shadow won't flip the state, but a real disappearance will. Under heavy camera motion the patience can be *raised* (`entry_patience_high_motion`), giving the matcher extra frames to catch up to a fast pan before crying loss.

### 2.3 Guards against false entry

Two more guards before committing:

- **Grace window.** During the first $N$ frames of a sequence (`no_occlusion_first_n_frames`), occlusion entry is *forbidden* and the streak held at $0$. Early on, the dynamic template is still settling (doc 03 §9), so low scores are expected and must not be mistaken for loss.
- **Heavy‑camera‑motion block.** If heavy camera motion is detected (doc 06 §4) *and* the target is not exiting the frame, the streak is blocked from accumulating and entry is refused — a confidence dip during a fast pan is "search lag," not loss. (Exception: if the target is genuinely *exiting* an edge, §4.2, entry is allowed even under camera motion.)

### 2.4 Exit‑direction detection

At the moment of entry, the tracker checks whether the target was *leaving the frame* and, if so, which **edge** (`_detect_exit_direction`). This sets `_out_of_frame` and `_exit_edge`, which change the search geometry (§4.2): an off‑frame target should be searched for at the edge it left, not in the middle.

### 2.5 Loss‑cause classification (clean history for the EKF)

When entering occlusion, the EKF must coast on a *clean* velocity. But the last few frames before loss may be corrupted (the box was already drifting/shrinking). `_classify_loss_cause` decides how many recent history frames to discard, by casting **two votes**:

**Area‑trend vote.** Fit a straight line to the recent box *areas* $a_0, \dots, a_{n-1}$ over time indices $t_i = 0,\dots,n-1$, by least squares. The slope of the best‑fit line (doc 01 §4.1 idea; the closed form is)

$$
\text{slope} = \frac{\sum_i (t_i - \bar t)(a_i - \bar a)}{\sum_i (t_i - \bar t)^2}
$$

measures whether the box was shrinking (negative slope) or steady/growing. Normalize by the median area, $\text{norm\_slope} = \text{slope}/\operatorname{median}(a)$. If $\text{norm\_slope} \ge$ a small negative threshold (i.e. *not* meaningfully shrinking), vote **camera_motion** (the target was full‑size right up to loss → it didn't dissolve behind something, the camera lurched).

**Camera vote.** If the exponentially‑weighted camera displacement (doc 06 §4.1) exceeds a threshold, vote **camera_motion**.

Only if **both** votes say `camera_motion` is the cause classified as camera motion (and the recent history is kept *clean*, skip $=0$). Otherwise it is `occlusion`, and a number of recent frames are skipped so the EKF rebuilds its velocity from pre‑corruption history. Requiring *both* votes to agree makes the "trust the history" decision conservative — the cost of wrongly keeping corrupted history (a bad coast direction) is high.

---

## 3. Coasting: the EKF runs the search

Once in occlusion, there is no trustworthy measurement, so the EKF **coasts** (doc 04 §8, doc 05 §C.5): it keeps predicting (with camera compensation while in frame; pure covariance inflation $P\leftarrow P+Q$ while out of frame), its uncertainty growing each frame. Every occlusion frame (`occlusion_update`):

1. reads the EKF's predicted box and sets the **held box** to it (clamped to the frame);
2. sets the **search center** $(s_{cx}, s_{cy})$ to the EKF center;
3. reads the EKF **velocity** (used by the DRM motion terms, doc 07 §4.3/4.7);
4. increments the occlusion frame counter `_occ_frames`;
5. handles **out‑of‑frame** edge pinning and re‑entry (§4.2);
6. routes to the current **phase** (§4).

So the EKF is the *navigator* during occlusion: it says "the target is probably here and heading this way," and the search machinery looks there. The growing uncertainty is reflected in a growing search region (§4.1) — the longer the target is gone, the wider we look.

---

## 4. The phase state machine

Recovery proceeds through phases keyed off `_occ_frames` (and a separate `reacq_confirm` sub‑state). The dispatcher routes:

```
   occlusion frame
        │
        ├─ reacq_confirm active? ──► PHASE: reacq‑confirm   (tentative lock‑on streak)   §5
        │
        ├─ _occ_phase == 0      ──► PHASE: siam             (fast SiamABC re‑try)        §4.3
        │
        ├─ 1 ≤ _occ_phase ≤ N   ──► PHASE: collect          (gather YOLO candidates)     §4.4
        │
        └─ _occ_phase >  N      ──► PHASE: final‑DRM         (score + verify)             §4.5
```

The intuition is *cheap‑first*: try the fast appearance matcher at the predicted spot; if that fails, spend a few frames collecting object‑detector candidates; then score them against long‑term memory and verify the best.

### 4.1 The growing search ROI (`_get_yolo_search_roi`)

The detector (YOLO) is run not on the whole frame but on a **region of interest (ROI)** that *grows geometrically with time*, centered on the EKF search center. This is the spatial expression of the EKF's growing uncertainty. With the object's representative size $\text{obj\_size}$ (median of recent sizes, doc 01 §7.2) and time‑step count

$$
\text{steps} = \Big\lfloor \frac{\text{\_occ\_frames}}{\text{growth\_every}}\Big\rfloor,
$$

the **expansion factor** grows by a constant ratio per step, capped:

$$
\text{effective\_expand} = \min\big(\underbrace{\text{roi\_start\_expand}\cdot \text{growth\_factor}^{\text{steps}}}_{\text{geometric growth}},\ \ \text{yolo\_search\_expand}\big),
$$

and the (square) ROI side is $\text{side} = \text{obj\_size}\cdot\text{effective\_expand}$, centered on $(s_{cx}, s_{cy})$ and clamped to the frame. **Geometric** growth ($\times$ ratio per step) means the search widens slowly at first, then accelerates — so a quickly‑returning target is found in a tight, low‑false‑positive window, while a long absence eventually triggers a broad sweep. The cap stops it from growing without bound. Running YOLO on a crop (not the full frame) also means fewer distractors in view and faster inference.

```
   ROI side vs occlusion time (geometric, then capped):
   side
    │                         ┌──────── cap (yolo_search_expand)
    │                    __--‾‾
    │               _--‾‾
    │          _--‾‾
    │   __--‾‾                  side = obj_size · roi_start_expand · growth_factor^steps
    │‾‾                          (small & tight early → wide sweep later)
    +----+----+----+----+----+----► _occ_frames
```

A separate, looser parameter set is used for **tiny/far** targets (they need proportionally wider search), and during the warm‑up window or after a failed recovery the ROI is simply centered on the frame to give the tracker a stable area to re‑lock.

### 4.2 Out‑of‑frame handling

If the EKF center leaves the frame by more than a margin, the target is marked `_out_of_frame` with an `_exit_edge` (left/right/top/bottom). Then:

- the **search center is pinned to that edge** (e.g. exited right → $s_{cx} = W-1$), and the ROI becomes a **strip** along the edge rather than a centered square — because the target will *re‑enter* from the edge it left;
- the EKF **coasts by pure covariance inflation** (no camera warp), since there is no in‑frame content to compensate against;
- **re‑entry** is declared when the EKF center comes back inside the frame *and* its velocity points **inward** (e.g. exited right → $v_x < 0$). Requiring inward velocity (not just an inside position) prevents a noisy coast from prematurely declaring the target back.

### 4.3 Phase: fast SiamABC re‑try (`occ_phase_siam`)

The cheapest attempt first: run the matcher (doc 03) seeded at the EKF‑predicted location. The EKF has been coasting on the target's last good velocity, so for a *brief* occlusion (the target slips behind a thin pole for a few frames) the predicted spot is often right, and the matcher re‑locks immediately — no detector needed. If the matcher's score clears the re‑acquisition bar, recovery jumps to verification (§5). If not, advance to the collect phase.

### 4.4 Phase: candidate collection (`occ_phase_collect`)

For a handful of frames (`_cand_collection_frames`), run YOLO on the growing ROI (§4.1) and **accumulate** the detected boxes as candidates, also recording the camera velocities seen during these frames. Collecting over several frames (rather than one) gives more chances to catch the target the moment it reappears, and builds a richer candidate set for scoring. YOLO is **class‑agnostic** or class‑filtered depending on configuration — but it provides *generic object proposals*, not identity; identity is decided next, by memory.

### 4.5 Phase: memory scoring + verification (`occ_phase_final_drm`)

Now the decisive step. Score the collected candidates against the long‑term appearance memory using the **composite DRM score** of doc 07 §4 (`drm_match` / `_occlusion_memory_match`), passing the EKF velocity (for the motion‑direction terms) and the search center (for the distance penalty). This yields a ranked shortlist. Then **verify**: for each top candidate, re‑run the matcher *as if the target were there* (`run_track_for_candidate`, doc 03 §..) — it injects the candidate as a hypothesis, computes the matcher's score there, and restores state. (The KF motion prior is *suppressed* during this, doc 06 §2.4, because a genuinely re‑acquired target is far from the stale trajectory and would be unfairly demoted.) A candidate is accepted only if its verified matcher score clears the **re‑acquisition threshold** `reacq_threshold` (e.g. $\ge 0.70$; itself possibly adaptive, doc 10).

If no candidate is verified, the tracker keeps coasting (the ROI grows further next frame). If detections exist but none survive scoring, a gentle **nudge** (`_nudge_toward_nearest`) biases the held box a small step toward the most plausible nearby detection (closest, weighted by appearance similarity) — so the EKF search center is pulled in roughly the right direction without committing to an unverified box. This is a *soft* correction, in keeping with the "prefer soft over hard" theme.

Using DRM (long‑term, occlusion‑surviving anchors) here — rather than the short‑term RAM or the matcher's drifting dynamic template — is the whole point: after a long absence, only a *stable, curated identity memory* can tell the target from a look‑alike that the detector also proposed. (Early in a sequence, before DRM has filled, recovery falls back to RAM matching, doc 07 §2.2.)

---

## 5. Verification and the confirmation streak

A single high verification score could be a fluke, so re‑acquisition is **confirmed over multiple frames** before exiting occlusion. On a successful verification, the tracker enters a tentative lock‑on (`begin_reacq_confirmation`): it seeds the matcher at the candidate box and, for the next `_reacq_confirm_frames` frames, requires the matcher to *stay* above `reacq_threshold`. A consecutive‑frame streak counter tracks this; only when it is satisfied does the tracker **commit**:

- set `in_occlusion = False` (return to the normal world);
- **reseed** the EKF at the re‑acquired box with a fresh velocity (doc 05 §C.5) — *not* nudge, because this is a legitimate teleport and the old trajectory is stale;
- reset the occlusion counters and emit the real coordinates.

This confirmation streak is, again, hysteresis (§2.2) — but on the *exit* side: just as we required persistence before declaring loss, we require persistence before declaring recovery. It trades a frame or two of latency for a large drop in false re‑acquisitions onto clutter or impostors.

---

## 6. The whole recovery loop

```
   ENTER OCCLUSION  (entry_streak ≥ patience, guards passed)               §2
     │  classify loss cause → keep/skip recent history → rebuild EKF velocity §2.5
     ▼
   each occlusion frame:
     EKF coast → held_box, search center, velocity; grow ROI; edge logic    §3,§4.1,§4.2
     │
     ├─ phase siam:     matcher at EKF spot — re‑lock if confident           §4.3
     ├─ phase collect:  run YOLO on ROI, accumulate candidates               §4.4
     └─ phase final‑DRM: composite‑score candidates (doc 07) → verify top‑k   §4.5
                          (verified ≥ reacq_threshold?)
                              │ yes
                              ▼
                      reacq‑confirm: stay confident for K frames             §5
                              │ confirmed
                              ▼
                      reseed EKF, in_occlusion=False, emit real box → NORMAL  §5
```

The design philosophy in one line: **coast on physics, propose with a detector, decide with long‑term memory, and confirm before you trust.** Every stage is cheap‑before‑expensive and conservative‑before‑committal, so the tracker neither wastes compute nor snaps onto the wrong object.

---

## 7. Recap

- A single `in_occlusion` flag selects between the normal and occlusion worlds; during occlusion the public output is the held/EKF box (or zeros) with score $0$.
- **Loss detection** uses an **effective threshold** (lowered+widened for tiny targets), **entry hysteresis** (`entry_patience` consecutive low‑score frames), a **grace window**, and a **heavy‑camera‑motion block**; a **loss‑cause** vote (area‑slope + camera displacement) decides how much recent history to trust.
- The **EKF coasts** (camera‑compensated in frame, covariance‑inflating out of frame), providing the search center, held box, and velocity; out‑of‑frame targets are searched for at the edge they left and require inward velocity to re‑enter.
- The **phase machine** goes fast‑matcher → **detector candidate collection** → **DRM composite scoring + matcher verification**, with a **geometrically growing ROI** mirroring the EKF's growing uncertainty.
- A winner must clear `reacq_threshold` and survive a **confirmation streak** (exit hysteresis); commitment **reseeds** the EKF and returns to normal tracking.

Next: **`09_FRAME_DYNAMICS.md`** — a small, self‑contained input augmentation that surfaces *motion* in the search crop to help the matcher lock onto tiny moving targets.
