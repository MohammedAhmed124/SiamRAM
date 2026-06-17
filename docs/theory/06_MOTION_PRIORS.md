# 06 — Motion Priors: Steering the Matcher with Physics

> Prerequisites: doc 03 (the response map, the search crop, why the target must stay inside it), doc 04 (Kalman filter), doc 05 (homography, the EKF, camera displacement). A **prior** is a belief held *before* looking at the current image. This document covers the three ways SiamRAM injects motion knowledge into the otherwise appearance‑only matcher: a **target‑motion Kalman prior** (where the object is going), the **camera‑motion GMC prior** (where the background went), and the **heavy‑camera‑motion** gate that tells the rest of the system when image motion is dominated by the camera. The recurring theme: *appearance says where the target looks like it is; motion says where it should plausibly be; the strongest decision is where they agree.*

---

## 1. Two questions, two priors

A moving image can change for two independent reasons (doc 05 Part C): the **target** moved, or the **camera** moved. SiamRAM has a separate prior for each:

| Prior | Question it answers | Mechanism | Source |
|------|--------------------|-----------|--------|
| Target‑motion Kalman | "Where will the object go, given its own recent velocity?" | a constant‑velocity KF over the box; reweights the response map / recenters the crop | `KalmanMotionPrior` |
| Camera‑motion **GMC** | "Where did the whole image move because the camera moved?" | warp the previous box by the homography $H$; use it as the search center | `apply_gmc_search_prior` |

They are complementary. If the camera is fixed and the object walks right, the target prior helps and the camera prior has nothing to do. If the camera pans and the object is stationary, the camera prior helps and the target prior alone would *mistake the pan for object motion*. If both move, the camera prior handles the global shift and the target prior handles the residual.

A note to prevent confusion: there are now **two** Kalman filters in SiamRAM. The **`BBoxEKF`** of doc 05 (center + velocity, camera‑aware) is the *outer* motion model — it drives the occlusion search and the EKF predict at the top of every frame. The **`KalmanMotionPrior`** of this document is a *second, inner* filter (full box, no camera term) used *only* to reweight the matcher's response map. They serve different roles and do not share state.

---

## 2. The target‑motion Kalman prior (`KalmanMotionPrior`)

This is a small constant‑velocity Kalman filter (doc 04 §6) over the **full box**, in the SAMURAI style. Its job is not to *be* the answer but to *score* the matcher's candidate locations by motion consistency.

### 2.1 State and model

The state is the box and its velocity — eight numbers:

$$
\mathbf{x} = (c_x,\ c_y,\ w,\ h,\ v_x,\ v_y,\ v_w,\ v_h).
$$

So it tracks not only how the center moves $(v_x, v_y)$ but also how the box *grows/shrinks* $(v_w, v_h)$. The constant‑velocity transition $F$ (an $8\times8$ matrix) adds each velocity to its position‑like coordinate:

$$
c_x \leftarrow c_x + v_x,\quad c_y \leftarrow c_y + v_y,\quad w \leftarrow w + v_w,\quad h \leftarrow h + v_h,\quad (\text{velocities unchanged}),
$$

i.e. $F = \begin{bmatrix} I_4 & I_4 \\ 0 & I_4\end{bmatrix}$ (the block form of doc 04 §6). The measurement is the full box, so $H = [\,I_4 \mid 0\,]$ reads off $(c_x,c_y,w,h)$ from the state.

### 2.2 Scale‑adaptive noise (the SAMURAI detail)

A clever, tuning‑free choice: the process and measurement noise standard deviations are made **proportional to the target's height** $h$. Using constants $\sigma_{\text{pos}} = 1/20$ and $\sigma_{\text{vel}} = 1/160$ (the standard MOT/ByteTrack convention), the per‑frame process noise standard deviations are

$$
\boldsymbol\sigma_Q = \big(\underbrace{\sigma_{\text{pos}}h,\ \sigma_{\text{pos}}h,\ \sigma_{\text{pos}}h,\ \sigma_{\text{pos}}h}_{\text{position-like}},\ \underbrace{\sigma_{\text{vel}}h,\ \sigma_{\text{vel}}h,\ \sigma_{\text{vel}}h,\ \sigma_{\text{vel}}h}_{\text{velocity-like}}\big),
$$

and $Q = \operatorname{diag}(\boldsymbol\sigma_Q^2)$ (variance is standard deviation squared, doc 01 §5.2); the measurement noise $R$ uses the position‑like block. **Why scale by $h$?** Because a "big" motion for a tiny far‑away target (a few pixels) is a "small" motion for a large near target (tens of pixels). Tying the noise to the target's size makes the filter behave the *same way relative to the target* regardless of how big it appears — no per‑video retuning. (At seeding, the initial covariance uses inflated versions of these, e.g. $2\sigma_{\text{pos}}h$ for position and $10\sigma_{\text{vel}}h$ for velocity, encoding "unsure at start," echoing doc 05 §C.1.)

### 2.3 The three operations

- **`seed(box)`** — initialize the state at a box with **zero velocity** and the inflated covariance. Called at init and whenever the filter must restart.
- **`predict()`** — advance one frame by $F$ and grow the covariance by $Q$ (doc 04 §3 predict).
- **`update(box)`** — the standard linear Kalman update (doc 04 §5) toward the matcher's chosen box.

It also exposes **`predicted_bbox()`** (the current state as an `xywh` box, used for the fusion below) and **`center_distance(box)`** (Euclidean distance between the state center and a box center, used for the reseed test).

### 2.4 Stability gating: never fight a fresh start

The prior is only allowed to *influence* selection after the tracker has been **stable** for several consecutive confident frames (`kf_motion_stable_frames`). The reason is fundamental: right after (re)initialization the filter's velocity is meaningless (seeded at zero), so trusting it would *hurt*. The gate is bookkept by a streak counter:

- On a **confident** frame (score $\ge$ a threshold) that is *not* a big jump, the filter updates and the streak increments.
- If the new box is a big jump from the prediction — center distance $>$ `reseed_dist` $\times$ box diagonal — the filter **reseeds** at the new box and the streak **resets to 0**. This covers legitimate teleports (a reacquisition writing the box directly): the prior should restart there, not slowly drag toward it.
- On a **low‑confidence** frame the filter neither updates nor counts; the streak resets. (It "coasts," doc 04 §8.)

The prior is "active" only when the streak $\ge$ `kf_motion_stable_frames` (and not explicitly suppressed). During recovery candidate‑verification the prior is *suppressed* outright (`_kf_suppress`): a re‑acquired target is, by definition, far from the stale pre‑occlusion trajectory, so the motion prior would wrongly demote exactly the right answer.

### 2.5 KF‑IoU response fusion (the core idea)

When the prior is active, SiamRAM blends a **motion‑consistency score** into the matcher's response map *before* the argmax (doc 03 §10, step 6). For every grid cell, the matcher already has its decoded candidate box (doc 03 §5.3). Map the KF's predicted box into the same crop coordinates (using the crop→frame mapping of doc 02 §4.4, inverted), then compute, per cell, the **IoU** (doc 02 §3) between that cell's candidate box and the KF box:

$$
\text{KF\_IoU}(i,j) = \operatorname{IoU}\big(\text{candidate box}(i,j),\ \text{KF predicted box}\big)\in[0,1].
$$

High where the candidate sits where the trajectory expects the target; low where it contradicts the motion. The fused selection map is the convex blend

$$
\boxed{\ \text{fused}(i,j) = (1 - w)\cdot\text{appearance}(i,j) + w\cdot\text{KF\_IoU}(i,j)\ },\qquad w = \text{kf\_motion\_weight}\in[0,1].
$$

This is a **weighted average** (doc 01 §7.3) of "looks like the target" and "is where the target should be." With $w=0$ it is pure appearance (legacy behavior); as $w$ grows, motion gets more say. The classic payoff: when a crossing look‑alike briefly produces a slightly higher *appearance* peak, its *low* KF‑IoU (it is off the trajectory) demotes it, and the true target — consistent with motion — wins. The danger, symmetrically, is that too large a $w$ makes the tracker follow a *predicted* trajectory through a genuine sudden turn; hence $w$ is kept modest (e.g. $0.15$–$0.2$).

```
   appearance map        KF‑IoU map (from trajectory)      fused = (1−w)·app + w·iou
   ┌──────────────┐      ┌──────────────┐                  ┌──────────────┐
   │ . ▃ . . █ . .│      │ . ▆ . . ▁ . .│                  │ . ▆ . . ▅ . .│  ← true target (▆)
   │ . . . . . . .│  ⊕   │ . . . . . . .│        =         │ . . . . . . .│    kept; look‑alike
   │ . . . ▂ . . .│      │ . . . . . . .│                  │ . . . ▁ . . .│    (█→▅) demoted
   └──────────────┘      └──────────────┘                  └──────────────┘
   (█ look‑alike beats     (trajectory favors              (motion breaks the tie
    target slightly)        the true target ▆)               toward the target)
```

### 2.6 KF‑centered search (keeping fast movers in the crop)

The fusion above reweights *within* a crop, but it can't help if the target already left the crop. A second, optional use of the same prior addresses that: center the **search crop** at the KF‑predicted center instead of the previous box's center (`kf_motion_center_search`). Only the *center* moves; the crop *size* still comes from the tracker's own box estimate, so the crop geometry and the size‑penalty's `prev_size` are unaffected. For a fast mover, placing the crop *ahead* of the target (where motion predicts it is going) keeps it inside the search region — recall (doc 03 §3) that a target outside the crop is simply unfindable. This is "predict‑then‑associate," the MOT pattern: use the motion model to place the search, then let appearance find the exact location.

---

## 3. The camera‑motion GMC prior (`apply_gmc_search_prior`)

GMC = **Global Motion Compensation**. Where the target prior asks "where is the object going," GMC asks "where did the image move because the camera moved," and uses the answer to *recenter the search* before the matcher runs.

### 3.1 What it does

Given the reliable homography $H$ from doc 05, GMC warps the **previous target box** into the current frame (doc 05 A.3): take the four corners of the previous box, push each through $H$ (perspective divide), and build the axis‑aligned box around the warped corners (`_warp_bbox_with_homography`):

$$
b_{\text{prior}} = \text{AABB}\big(\{\,H\!\cdot\!\tilde{\mathbf c}_k\,\}_{k=1}^4\big),
$$

clipped to the frame. Warping *all four corners* (rather than just shifting by $(dx,dy)$) means the prior also reflects mild zoom and rotation, not only translation. This warped box is written into SiamABC's internal search state (`_set_tracker_search_prior`), so the matcher's next search crop is centered on the **camera‑compensated** location. As the GMC note in the repo puts it: *"Before visual matching, start looking from the camera‑compensated location."* It does **not** force the answer — the matcher still does the visual match and can land anywhere in the (now better‑placed) crop.

```
   without GMC: crop centered on last box      with GMC: crop centered on H·(last box)
   ┌───────────────┐                            ┌───────────────┐
   │█ target drifted│  ← camera panned, target  │       █ target │  ← crop follows the
   │  to crop edge  │     near to leaving crop   │     centered   │     camera; target
   └───────────────┘                            └───────────────┘     stays central
```

### 3.2 When GMC is skipped

GMC is intentionally conservative — a *wrong* prior is worse than none (doc 05 B.5). It is skipped when:

- there is no current box, or no homography $H$ (e.g. first frame);
- $H$ is **unreliable** and reliability is required (`gmc_prior_require_reliable_h`);
- $H$ fails the **plausibility gates** of doc 05 §D.1 (translation/scale/rotation/corner caps);
- the warp produces an invalid box.

(There is also a skip during distractor mode, which is out of scope here.) In any skip case the matcher just uses its ordinary last‑box‑centered crop. The robustness payoff of GMC is exactly in the cases real video throws constantly — fast pans, shake, small zoom — where the target would otherwise drift to the crop edge.

### 3.3 GMC vs. KF‑centered search — do they fight?

Both want to set the search center. They are not contradictory, but they interact: GMC writes a camera‑compensated box into the tracker state, and the KF prior may then *see that as an external move* and reseed (resetting its stability streak, §2.4). This is by design — after another subsystem moves the search, the stale object‑velocity prediction shouldn't be trusted until motion is re‑established. The practical upshot: with GMC on, the KF prior tends to be less continuously active. Neither is "wrong"; they encode different, occasionally overlapping, knowledge, and the conservative reseed keeps them from compounding errors.

---

## 4. Heavy camera motion: a gate for the whole system

The third use of camera motion is not a search prior at all — it is a **boolean signal**, "is the image motion right now dominated by the camera?", consumed by several other subsystems (occlusion entry in doc 08; template‑rate adaptation in doc 10). Getting this right prevents a large class of false alarms: *during a fast pan, the matcher's confidence dips and the box lags — but the target is not lost, the camera is just moving faster than the search can follow.* Treating that dip as "target gone" would trigger needless occlusion recovery.

### 4.1 The displacement signals

Two scalars are combined (`is_heavy_camera_motion`):

- **Instantaneous displacement** — this frame's camera‑center displacement from $H$ (doc 05 §D.2): warp the frame center through $H$, measure how far it moved.
- **Weighted‑history displacement** — an *exponentially weighted average* (doc 01 §7.3) of recent per‑frame camera displacements, with weights growing toward the present:

$$
\text{weighted\_disp} = \frac{\sum_i w_i\, d_i}{\sum_i w_i},\qquad w_i = e^{(i - n)/\tau}\ \text{(larger for recent } i),
$$

where $d_i$ are the stored displacements. Using a *history* (not just this frame) smooths out single‑frame estimate noise so a sustained pan is detected even if one frame's $H$ was momentarily weak. The code takes the **max** of the instantaneous and weighted values, $\;\text{max\_disp} = \max(\text{inst},\ \text{weighted})$, so either a sudden lurch or a sustained pan can trip the gate.

### 4.2 Scale‑normalizing the displacement

A raw pixel displacement means different things for a tiny vs. a large target — the same recurring scale issue (doc 02 §2, doc 06 §2.2). So the displacement is also normalized by a **reference box diagonal** $d_{\text{ref}}$ (the current box's diagonal, falling back to the frame diagonal if no box):

$$
\text{max\_norm} = \frac{\text{max\_disp}}{d_{\text{ref}}}.
$$

This bbox‑relative number ("the camera moved by $X$ box‑diagonals this frame") is the scale‑independent measure, and it is also published for the adaptive template‑rate controller (doc 10).

### 4.3 The verdict

Heavy camera motion is declared if *either* an absolute pixel gate or a normalized gate is exceeded (each can be disabled by setting its threshold to $0$):

$$
\text{heavy} = \big[\text{max\_disp} \ge \tau_{\text{px}}\big]\ \lor\ \big[\text{max\_norm} \ge \tau_{\text{norm}}\big].
$$

This boolean then *blocks* premature occlusion entry (doc 08), *extends* the patience before declaring loss, and can *adapt* the template‑update rate (doc 10). It is one of the most leveraged signals in the system, which is why it draws on both an absolute and a scale‑relative criterion and on both instantaneous and smoothed history — robustness layered on robustness.

---

## 5. How a frame uses all three (ordering)

In the normal (non‑occlusion) update, the priors are applied in this order, *before and around* the SiamABC forward of doc 03:

```
   1. heavy‑camera‑motion verdict refreshed (used later for gating)            [§4]
   2. GMC: if H reliable & plausible, recenter the search crop on H·(prev box) [§3]
   3. (frame‑dynamics motion saliency may be blended into the crop — doc 09)
   4. SiamABC forward → appearance response map                                [doc 03]
   5. if the KF prior is active: fuse KF‑IoU into the response map, then argmax [§2.5]
      (and/or the crop was KF‑centered in step 2's place — §2.6)
   6. ... downstream: thresholding, EKF update, memory, occlusion entry ...
```

So GMC acts *before* the matcher (it changes where we look); the KF fusion acts *inside* the matcher's decoding (it changes which peak wins); the heavy‑motion verdict acts *after* (it changes how we interpret a low score). Three different insertion points for three different kinds of motion knowledge.

---

## 6. Recap

- SiamRAM holds **two** motion priors: a **target‑motion** Kalman prior (object's own velocity) and a **camera‑motion** GMC prior (global image shift). They answer different questions and combine cleanly.
- The **`KalmanMotionPrior`** is an 8‑state constant‑velocity KF over the full box with **height‑scaled noise** (tuning‑free across target sizes). It is gated to act only after a **stability streak**, reseeds on big jumps, and is suppressed during recovery verification.
- **KF‑IoU fusion** blends a motion‑consistency IoU into the response map, $\text{fused} = (1-w)\,\text{appearance} + w\,\text{KF\_IoU}$, demoting motion‑inconsistent look‑alikes; **KF‑centered search** places the crop ahead of fast movers.
- The **GMC prior** warps the previous box by the homography to recenter the search on the camera‑compensated location, guarded by reliability and physical‑plausibility gates; it *steers*, never *forces*.
- **Heavy‑camera‑motion** is a leveraged boolean built from instantaneous + exponentially‑weighted, absolute + scale‑normalized displacement; it prevents the system from mistaking pan‑induced confidence dips for target loss.

Next: **`07_APPEARANCE_MEMORY_RAM_DRM.md`** — the long‑term identity memory that lets the tracker *re‑recognize* the target after it has been lost, and the composite score that ranks re‑acquisition candidates.
