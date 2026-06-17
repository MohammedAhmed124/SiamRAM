# 11 — Putting It All Together: The Conductor

> Prerequisites: ideally all prior documents, but especially doc 00 (the pipeline skeleton), doc 03 (matcher), doc 05 (EKF), doc 06 (motion priors), doc 08 (occlusion). This final document shows the **conductor** — the master per‑frame routine `update()` in `tracker.py` — that wires every subsystem of this series into one coherent state machine. We trace a frame end‑to‑end, explain each branch and *why* it is ordered as it is, and walk through three concrete scenarios. *(The real code also contains distractor‑mode and spike‑watcher branches; per the scope of these notes those are deprecated/out of scope and are omitted here — wherever they would sit, we say so.)*

---

## 1. The two‑state machine

At the top level SiamRAM is a state machine with two macro‑states, selected each frame by the boolean `in_occlusion` (doc 08 §1):

```
                     score < threshold for `entry_patience` frames
                          (guards passed: grace over, not heavy‑cam)
   ┌──────────────┐  ───────────────────────────────────────────────►  ┌──────────────┐
   │    NORMAL     │                                                     │   OCCLUSION   │
   │  (matcher +   │                                                     │ (EKF coast +  │
   │  motion +     │  ◄───────────────────────────────────────────────  │  detect +     │
   │  memory)      │      verified re‑acquisition + confirmation streak  │  DRM + verify)│
   └──────────────┘                                                     └──────────────┘
```

**The invariant:** per frame, *exactly one* of `_normal_update` or `_occlusion_update` runs. They are disjoint worlds with separate logic; `in_occlusion` is the single switch between them. Everything below is the choreography around that switch.

---

## 2. The per‑frame entry (`update` / `_update_impl`)

Every frame, before either world runs, the conductor does a fixed preamble — the steps shared by both states:

```
 1. proc_frame = prescale(frame)                       full‑res → proc space (doc 02 §6)
 2. frame_idx += 1 ; reset per‑frame scratch (yolo cache, …)
 3. H, H_reliable, gray = estimate_homography(proc)     camera motion (doc 05 Part B)
    store last_H, last_H_reliable                       (consumed by EKF, GMC, heavy‑motion)
 4. is_heavy_camera_motion(proc)                         refresh the heavy‑motion verdict (doc 06 §4)
 5. EKF step:
       if in_occlusion and out_of_frame:  P ← P + Q      coast, no warp (doc 05 §C.5)
       else:                              EKF.predict(H, H_reliable)   camera‑aware predict (doc 05 §C)
 6. branch:
       if in_occlusion:  (bbox, score) = occlusion_update(proc)        doc 08
       else:             (bbox, score) = normal_update(proc)           §3 below
 7. prev_gray = gray                                     set up next frame's optical flow
 8. scale bbox back to full‑res (× 1/s); emit (bbox, score, in_occlusion, yolo)
```

Three things to notice in the ordering:

- **Camera motion is estimated first**, because *both* worlds need it: the EKF predict uses $H$, GMC uses $H$, the heavy‑motion gate uses $H$, and the loss‑cause classifier uses the camera‑displacement history derived from $H$. Estimate once, share everywhere.
- **The EKF predicts before the branch.** Whether we are tracking or occluded, the motion belief advances every frame — so when occlusion begins, the EKF is already coasting on a fresh, camera‑compensated prediction; and when tracking, the predicted center is available to place the search.
- **Coordinate discipline.** The frame is prescaled to *proc* space at step 1 and the answer scaled back at step 8 (doc 02 §6). Everything in between — every subsystem in this series — is proc‑space. This is what lets the matcher, EKF, YOLO, and memory all speak the same pixel units.

---

## 3. The normal update, ordered (`_normal_update`)

When `in_occlusion` is false, this is the hot path that runs every visible frame. Its order is *load‑bearing* — each step depends on the previous ones — so we annotate why each is where it is.

```
 (a) template‑freeze / memory‑freeze bookkeeping            (distractor‑mode lifecycle — out of scope)
 (b) GMC search prior: recenter search on H·(prev box)      doc 06 §3   ── before the matcher looks
 (c) template‑rate adaptation: set N / window from motion    doc 10 §4   (or legacy binary adapt)
 (d) frame‑dynamics: blend motion saliency into the crop     doc 09      ── tiny targets only
 (e) pred_bbox, score = SiamABC.update(frame)                doc 03      ── the visual match
       (inside: optional KF‑centered search + KF‑IoU fusion  doc 06 §2)
 (f) class warm‑up / detectability probe (early frames)      (YOLO sanity checks; see §3.1)
 (g) effective_threshold = compute_effective_threshold()     doc 08 §2.1 (tiny‑switch; may be adaptive)
       [ here the real code branches into distractor‑mode / spike handling — OUT OF SCOPE ]
 (h) heavy_cam_motion, disp = is_heavy_camera_motion(pred)    doc 06 §4
 (i) sample adaptive controllers on confident frames:         doc 10
       reacq_threshold  ← AutoDrmMargin / reacq EMA
       conf_threshold   ← AutoConfThreshold EMA
 (j) occlusion‑entry decision:                                doc 08 §2
       grace window? heavy‑cam block? → hold streak at 0
       else if score < effective_threshold: entry_streak += 1
       else: entry_streak = 0
       if entry_streak ≥ effective_entry_patience (and not in grace):
            classify loss cause, rebuild EKF, set in_occlusion = True; hand off to occlusion next frame
 (k) HEALTHY frame (no entry): commit
       EKF.update(pred_bbox)                                  doc 05 §C.4
       record camera‑motion history; update stable references
       admit appearance descriptor to RAM (→ maybe promote to DRM)  doc 07
       (SiamABC also admits to its memory window + refreshes z̃)     doc 03 §9
 → return (pred_bbox, score)
```

### 3.1 The early‑frame YOLO probes (brief)

Steps (f) run only in the first frames. **Class warm‑up** runs the detector to learn which object *class* the target most likely is, so later occlusion‑phase YOLO can optionally filter to that class (doc 08 §4.4). The **detectability probe** checks whether the detector can even see the target at all; if not, the system leans more on the appearance matcher and less on detector‑based recovery. These are *configuration* of the recovery machinery, not part of the per‑frame match, which is why they sit off to the side.

### 3.2 Why entry is gated so heavily

Step (j) is where NORMAL can flip to OCCLUSION, and it is deliberately reluctant (doc 08 §2): a **grace window** early in the sequence, a **heavy‑camera‑motion block**, **hysteresis** (`entry_patience` consecutive sub‑threshold frames), and patience that *increases* under camera motion. The reason: entering occlusion needlessly is expensive (the matcher stops trusting itself, the detector spins up, the output goes to "no answer"), and most low‑score frames are *not* true losses (blur, a fast pan, a hard but visible target). The gates cost a frame or two of latency on a true loss in exchange for far fewer false losses — the same latency‑for‑reliability trade as the exit‑side confirmation streak (doc 08 §5).

### 3.3 Why the EKF updates *only* on healthy frames

Step (k) calls `EKF.update` only when no occlusion entry happened — i.e., when we have a *trustworthy* measurement. On a frame that triggers entry, the matcher's box is suspect, so we do **not** feed it to the EKF; instead the loss‑cause classifier (doc 08 §2.5) decides how much recent history to discard so the EKF coasts on a clean velocity. Feeding a bad box to the filter would corrupt exactly the velocity the recovery search depends on.

---

## 4. The full wiring: where every subsystem plugs in

A single table mapping the series to the conductor:

| Subsystem (doc) | Runs in | When in the frame | Role |
|---|---|---|---|
| Homography / camera motion (05) | both | preamble step 3 | shared camera estimate $H$ + reliability |
| `BBoxEKF` predict (04, 05) | both | preamble step 5 | advance motion belief (camera‑aware) |
| Heavy‑camera‑motion gate (06) | both | preamble 4 / normal (h) | block false entry, drive adaptation |
| GMC search prior (06) | normal | (b) before matcher | recenter search on camera‑comp. location |
| Template‑rate controller (10) | normal | (c) | set dynamic‑template cadence from motion |
| Frame dynamics (09) | normal | (d) | motion‑saliency blend (tiny targets) |
| SiamABC matcher (03) | normal | (e) | the visual match → box + score |
| KF motion prior + fusion (06) | normal | inside (e) | reweight response toward motion |
| Effective threshold (08, 10) | normal | (g) | the loss bar (tiny‑switch / adaptive) |
| Adaptive controllers (10) | normal | (i) | retune thresholds to this video |
| Occlusion entry (08) | normal | (j) | NORMAL → OCCLUSION switch |
| `BBoxEKF` update (05) | normal | (k) | correct motion from a good box |
| RAM/DRM admission (07) | normal | (k) | grow appearance memory |
| EKF coast + search nav (05, 08) | occlusion | step 5 / occlusion | extrapolate, steer the search |
| Phase machine: siam/collect/DRM (08) | occlusion | occlusion | re‑detect the target |
| DRM composite scoring (07) | occlusion | occlusion phase | rank re‑acquisition candidates |
| Re‑acquisition verify + confirm (08) | occlusion | occlusion phase | OCCLUSION → NORMAL switch |

Read top to bottom, it is the whole tracker: *estimate camera motion → advance the motion belief → (if tracking) steer & run the matcher, fuse motion, decide if lost, else update motion & memory → (if lost) coast, detect, score against memory, verify, recover.*

---

## 5. Three worked frames

To make the choreography concrete, here are three representative frames narrated through the conductor.

### 5.1 A healthy frame (target visible, calm scene)

```
 preamble: H estimated (reliable, tiny motion); EKF.predict warps center ~0 px, adds velocity.
 normal:   (b) GMC prior ~no shift (camera still).
           (e) SiamABC matches; sharp peak; score 0.88.
           (g) effective_threshold 0.55; (h) not heavy‑cam.
           (i) controllers fold 0.88 into their EMAs (confident frame).
           (j) 0.88 > 0.55 → entry_streak = 0 (no loss).
           (k) EKF.update toward the box (velocity refined); descriptor admitted to RAM
               (and, if it agrees with neighbors, promoted to DRM); z̃ refreshed on cadence.
 emit:     box scaled back to full‑res, score 0.88, in_occlusion = False.
```

Everything agrees; the safety nets are quietly maintaining state for the day things go wrong.

### 5.2 Entering occlusion (target slips behind a wall)

```
 frames t..t+2: score falls to 0.30, 0.28, 0.31 (< 0.55). Scene calm (not heavy‑cam, grace over).
           (j) entry_streak climbs 1 → 2 → 3.   At streak ≥ entry_patience (say 3):
               loss‑cause vote: box was full‑size up to loss BUT camera barely moved
                  → not both camera votes → cause = OCCLUSION → skip a few corrupted history frames,
                  rebuild EKF velocity from clean history.
               in_occlusion = True.
 next frame: OCCLUSION world. EKF coasts (predict with whatever camera motion); held_box = EKF box;
             phase 0 (siam) re‑tries at the predicted spot — still hidden, score low; advance to collect.
 emit during occlusion: held/EKF box (or zeros), score 0.0, in_occlusion = True.
```

The hysteresis ensured a few low frames didn't trigger this prematurely; the clean‑history step ensured the coast direction is right.

### 5.3 Re‑acquiring (target reappears across the frame)

```
 occlusion, _occ_frames large: ROI has grown (geometric); YOLO proposes 3 boxes in the wide ROI.
 phase final‑DRM: composite‑score each candidate against DRM anchors
                  (appearance + IoU + motion‑direction along EKF velocity + recency − distance penalty).
                  Best candidate scores 0.83 ≥ skip_threshold → shortlist = [best].
 verify: run_track_for_candidate seeds the matcher there (KF prior suppressed); matcher score 0.78
         ≥ reacq_threshold (0.70) → begin reacq‑confirm.
 confirm: matcher stays ≥ 0.70 for K consecutive frames → COMMIT:
          reseed EKF at the new box with fresh velocity; in_occlusion = False; emit real coords.
 back to NORMAL: next frame is an ordinary healthy frame (§5.1), now locked onto the true target.
```

Long‑term memory (DRM) told the true target from the other detected objects; verification + confirmation made sure before trusting it; the EKF reseed (not nudge) acknowledged the legitimate teleport.

---

## 6. The design philosophy, in one view

Step back from the code and the whole system is one idea applied repeatedly: **a fast, myopic appearance matcher, surrounded by layered, mostly‑soft safety nets, each handling one failure mode, each conservative about taking control and conservative about giving it up.**

- *Appearance* (doc 03) is fast but myopic and identity‑blind.
- *Motion* (docs 04–06) supplies physics: where the target should be (EKF), where the camera moved it (homography/GMC), and which response peak is trajectory‑consistent (KF fusion).
- *Memory* (doc 07) supplies long‑term identity, so the target can be told from look‑alikes after a gap.
- *Occlusion recovery* (doc 08) supplies the structured fallback when appearance fails: coast on physics, propose with a detector, decide with memory, confirm before trusting.
- *Perception aids* (doc 09) make hard (tiny) targets more visible.
- *Adaptive controllers* (doc 10) make every threshold fit the difficulty of the current video instead of a global guess.

And throughout, recurring principles you can now name on sight:

1. **Hysteresis** before switching states (enter *and* exit occlusion only after persistence).
2. **Soft penalties over hard kills** (IoU‑gate scales the score by $0.25$, doesn't zero it; nudges over commits).
3. **Scale normalization** (displacements in box diagonals; noise scaled by target size) so one rule fits all target sizes.
4. **Measure the genuine target, threshold relative to it** (the adaptive controllers).
5. **Estimate once, share everywhere** (one homography feeds five consumers).
6. **Graceful fallback when a signal is missing** (no homography → no warp; empty DRM → RAM match; feature off → exact no‑op).

That is SiamRAM: not one clever trick, but a disciplined stack of small, individually understandable, mathematically grounded mechanisms — every one of which you have now derived from first principles.

---

## 7. Where to go next

- Re‑read **doc 05** with the conductor in mind: you will see exactly when `predict`, `update`, `reseed`, and the coast path fire.
- Trace a real video with debug logging and watch `entry_streak`, the heavy‑motion verdict, and the phase counter move through the transitions of §5.
- The two subsystems we deliberately skipped — **distractor mode** and the **spike watcher** — build on the same primitives (camera‑compensated displacement, appearance similarity, hysteresis); the foundations in docs 01–07 are exactly what you'd need to read them, should you choose to.

You now understand the whole tracker, end to end, mathematically. That was the goal.
