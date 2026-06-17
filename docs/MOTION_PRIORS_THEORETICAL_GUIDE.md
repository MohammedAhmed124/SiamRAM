# Motion Priors In SiamRAM: A Student-Friendly Theoretical Guide

This note explains how the motion-related tracking options interact in SiamRAM.
It is written for a student who is still building intuition, so it starts from
the tracking ideas before discussing the exact configuration.

The two main ideas are:

1. **Target-motion prior**: predict where the object itself is likely to move.
2. **Camera-motion prior**: predict how the whole image moved because the camera
   moved.

These are different. A car driving to the right is target motion. A camera
panning to the right is camera motion. In video tracking, the image can change
because of either one, or both at the same time.

---

## 1. The Basic Tracking Problem

A tracker receives a video one frame at a time.

In the first frame, the target is given:

```text
frame 0 + target box = initial knowledge
```

For every later frame, the tracker must answer:

```text
where is the same object now?
```

SiamRAM uses a Siamese visual tracker called SiamABC for the short-term visual
tracking part. The idea is:

```text
template crop: "what the target looks like"
search crop:   "where to look in the new frame"
```

The tracker compares the template crop against the search crop. It then produces
a small grid of scores called a **response map**.

The response map can be imagined as:

```text
each grid cell says:
"how likely is the target around this position?"
```

The highest-scoring cell usually becomes the selected target position.

---

## 2. What Is A Search Crop?

SiamABC does not search the whole image every frame. That would be expensive and
would create many false matches.

Instead, it searches around the last known target position:

```text
previous target box -> larger surrounding crop -> neural network input
```

The search crop is larger than the object. For example, if the object is a small
box, SiamABC crops extra context around it. This lets the object move a little
between frames while still staying inside the crop.

The important theoretical point:

```text
if the target is outside the search crop, SiamABC cannot directly match it
```

So crop placement matters a lot. A good search crop keeps the target inside the
area that the visual tracker sees.

---

## 3. What Is A Response Map?

After SiamABC sees the search crop, it outputs a response map.

If the response map is 16 by 16, it has 256 possible coarse positions:

```text
16 * 16 = 256 candidate locations
```

Each candidate has an appearance score:

```text
high score = looks like the target
low score  = does not look like the target
```

Without any motion prior, the tracker simply trusts the visual appearance score.
That can fail when there are similar-looking objects.

Example:

```text
target car and another similar car both appear in the crop
```

The wrong car may look slightly more similar for one frame. A pure appearance
tracker can jump to it. This is called an **identity switch**.

---

## 4. Why Motion Priors Exist

A motion prior adds a simple expectation:

```text
the target will probably continue moving in a physically reasonable way
```

This does not replace visual matching. It only helps the tracker choose between
visual possibilities.

A useful mental model is:

```text
appearance says: "this candidate looks like the object"
motion says:     "this candidate is where the object should plausibly be"
```

The final tracking decision is strongest when both agree.

---

## 5. Methodology 1: Target-Motion Kalman Prior

The target-motion prior tries to model the target object's own movement.

In this repo, it is the `kf_motion` feature under `tracker:`.

KF means **Kalman filter**. You do not need advanced math to understand the idea.
A Kalman filter is a predictor-corrector:

```text
predict where the object should be
observe where the tracker found it
correct the motion estimate
```

### 5.1 What The Kalman Filter Stores

The Kalman filter stores a simple state:

```text
center x
center y
width
height
velocity x
velocity y
width change velocity
height change velocity
```

In compact form:

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

This means it remembers not only where the object is, but also how it has been
moving.

### 5.2 The Predict Step

Before tracking the next frame, the filter predicts:

```text
next center = current center + velocity
```

If the object has been moving right by 8 pixels per frame, the prediction moves
the expected center about 8 pixels right.

This is a constant-velocity assumption. It is simple, causal, and fast.

### 5.3 The Observe Step

After SiamABC selects a box, the Kalman filter receives that box as a
measurement.

If the tracker is confident, the filter updates its state:

```text
prediction + observed box = improved estimate
```

If the tracker is not confident, the filter does not trust that frame. It does
not update from a possibly wrong visual match.

### 5.4 Stability Gating

The filter is only allowed to influence tracking after enough confident frames.

The purpose is to avoid this failure:

```text
bad early prediction -> bad motion prior -> worse tracking
```

So the feature uses:

```yaml
kf_motion_stable_frames: 1
kf_motion_score_threshold: 0.55
```

The meaning:

```text
only after enough frames with score >= threshold can the motion prior participate
```

If confidence drops, the stable streak resets.

### 5.5 Reseeding

Sometimes the tracker is intentionally moved by another subsystem, such as
reacquisition after occlusion.

If the current box is suddenly far away from the Kalman prediction, the filter
should not slowly drag itself there. It should restart at the new location.

This is reseeding.

The config:

```yaml
kf_motion_reseed_dist: 2.0
```

The meaning:

```text
if the new box center is more than about 2 box diagonals away,
restart the filter at that box
```

---

## 6. Target-Motion Prior Feature A: Response-Map Fusion

This is the "weight the response map toward motion" part.

For each response-map cell, SiamABC already has:

```text
appearance score
```

The Kalman filter adds:

```text
motion consistency score
```

The motion score is based on how well the candidate box overlaps the
Kalman-predicted box. This is measured with IoU:

```text
IoU = overlap area / union area
```

High IoU means:

```text
this candidate is close to where the motion model expected the target
```

Low IoU means:

```text
this candidate contradicts the expected motion
```

The two scores are blended:

```text
fused_score = (1 - weight) * appearance_score + weight * motion_score
```

Config:

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
```

If `kf_motion_weight` is low, appearance dominates. If it is high, motion has
more influence.

Student intuition:

```text
weight = 0.0 -> ignore motion
weight = 0.2 -> mostly visual, lightly motion-aware
weight = 0.8 -> very strongly motion-biased
```

A high value can help during ambiguity, but it can also over-trust a bad motion
prediction.

---

## 7. Target-Motion Prior Feature B: Centering The Search Crop

This is the "nudge the search region in the direction of motion" part.

The normal crop center is based on the previous tracker box:

```text
previous selected box -> search crop center
```

With KF-centered search, the crop center is based on the Kalman prediction:

```text
Kalman-predicted center -> search crop center
```

Config:

```yaml
tracker:
  kf_motion_center_search: True
```

Important detail:

```text
the crop center moves, but the crop size still uses the tracker box size
```

This keeps the crop geometry stable while shifting it toward where the object is
expected to be.

This feature is especially useful when the target is moving fast. If the crop
always stays centered on the old box, the target may drift toward the crop edge
or leave the crop. Motion-centered search tries to place the crop ahead of the
target.

---

## 8. Methodology 2: Camera-Motion GMC Prior

GMC means **Global Motion Compensation**.

It models the movement of the camera or the background, not the movement of the
target itself.

Example:

```text
the camera pans right
the entire scene shifts left in the image
```

If the target is actually standing still in the world, it may still move in the
image because the camera moved. GMC tries to estimate that global image motion.

### 8.1 What GMC Estimates

GMC estimates a transformation between two consecutive frames:

```text
previous frame -> current frame
```

This transformation is represented as a homography:

```text
H
```

You can think of `H` as a matrix that moves points from the old frame into the
new frame:

```text
point_current ~= H * point_previous
```

It can represent translation, mild rotation, scale, and perspective-like image
motion.

### 8.2 Why The Target Region Is Avoided

GMC wants background motion.

The target can move differently from the background. If the target dominates the
motion estimate, GMC might accidentally learn target motion instead of camera
motion.

So the system tries to estimate motion from the scene around the target, not from
the target itself.

### 8.3 Reliability Checks

A homography is not always trustworthy.

It can be bad if:

```text
the frame is blurry
there are too few background features
the scene is low texture
many moving objects confuse the estimate
the camera motion is too extreme
```

So GMC has reliability and plausibility checks.

The config includes:

```yaml
ram_tracker:
  gmc_prior:
    gmc_prior_require_reliable_h: true
    gmc_prior_max_translation_frac: 0.25
    gmc_prior_min_scale: 0.7
    gmc_prior_max_scale: 1.4
    gmc_prior_max_rotation_deg: 25.0
    gmc_prior_max_corner_displacement_frac: 0.25
```

The theoretical meaning:

```text
only use GMC if the estimated camera motion looks physically plausible
```

### 8.4 Warping The Previous Target Box

If GMC is valid, SiamRAM warps the previous target box into the current frame.

The theoretical steps:

1. Take the four corners of the previous target box.
2. Move each corner through the homography `H`.
3. Build a new axis-aligned box around the moved corners.
4. Use that warped box as a better search prior.

This gives:

```text
previous target box + camera motion = camera-compensated target box
```

If the object itself did not move, this warped box should be close to the new
visual location.

---

## 9. How The Two Priors Are Different

Target-motion KF asks:

```text
where should the object move based on its own recent velocity?
```

Camera-motion GMC asks:

```text
where did the whole image move because the camera moved?
```

They answer different questions.

If the camera is fixed and the target walks right:

```text
KF helps
GMC has little to do
```

If the camera pans and the target is stationary:

```text
GMC helps
KF alone may mistake camera motion for target motion
```

If the camera pans and the target also moves:

```text
GMC handles background/camera shift
KF handles target's own remaining motion
```

That is the theoretical reason they can be useful together.

---

## 10. How They Interact In One Frame

This is the bit-by-bit frame flow.

### Step 1: A New Frame Arrives

SiamRAM receives the next video frame.

The tracker has a previous target box from the last frame.

### Step 2: Camera Motion Is Estimated

The camera-motion subsystem compares the previous frame with the current frame.

It estimates:

```text
H = global background motion
```

It also decides:

```text
is H reliable?
```

### Step 3: GMC May Adjust The Inner Search Prior

If:

```text
gmc_prior_enabled = true
H exists
H is reliable enough
H passes plausibility gates
```

then SiamRAM warps the current target box by `H` and writes that box into the
inner SiamABC tracker as the search prior.

In plain language:

```text
"Before visual matching, start looking from the camera-compensated location."
```

### Step 4: The Kalman Filter Begins The Frame

The target-motion Kalman filter checks the current inner tracker box.

If the box was externally changed, for example by GMC, the Kalman filter may
reseed itself.

This is important:

```text
GMC can move the starting box
KF notices the starting box moved
KF may restart its stability streak
```

So if GMC rewrites the inner tracker state every frame, the KF target-motion
prior may become less active unless the motion remains consistent enough.

This is not necessarily a bug. It is a safety behavior. The tracker avoids using
a stale object-motion prediction after another subsystem has changed the search
state.

### Step 5: KF-Centered Search May Move The Crop Center

If the KF prior is active and `kf_motion_center_search` is true, SiamABC centers
the search crop at the KF-predicted target center.

If the KF prior is not active, SiamABC uses the normal current tracker box as
the crop center.

### Step 6: SiamABC Extracts The Search Crop

The selected search center becomes the actual crop fed to the neural network.

This is the crop that matters for model behavior.

Note for visualization:

```text
the right-side "SEARCH CTX" panel in the debug video may not show this live crop
```

That panel can be based on the dynamic-template frame/bbox, so it may look
unchanged even when the actual model crop was shifted.

### Step 7: SiamABC Produces A Response Map

The neural network compares template features with the search crop and produces
a response map.

Each response-map cell has an appearance score.

### Step 8: KF Response-Map Fusion May Reweight The Candidates

If the KF prior is active and `kf_motion_enabled` is true, every candidate gets a
motion consistency score.

Then the final selection map becomes:

```text
fused_score = (1 - weight) * appearance_score + weight * motion_score
```

The highest fused score is decoded into the predicted box.

### Step 9: The Final Box Is Chosen

SiamABC decodes the winning response-map cell into a box in the search crop.

That box is mapped back into full-frame coordinates.

### Step 10: The Motion States Are Updated

If the visual tracker is confident, the Kalman filter updates from the selected
box.

SiamRAM also updates its outer history, EKF, memory, confidence logic, and
occlusion/distractor logic.

---

## 11. Why Results Can Look Identical

Motion priors do not always visibly change results.

Common reasons:

### Reason 1: The Appearance Peak Already Wins

If the visual response map has one clear correct peak, the motion prior has
nothing important to change.

In that case:

```text
appearance-only result = motion-aware result
```

That is a good outcome, not a failure.

### Reason 2: The Prior Is Gated Off

KF motion only acts after stable confident frames.

If confidence drops or the filter reseeds, the feature temporarily turns off.

### Reason 3: GMC Can Reset KF Stability

When GMC writes a new search prior into the inner tracker, the KF can see that as
an external move and reseed.

If this happens often, the two priors may not both be active in the way you
expect.

### Reason 4: The Debug Panel May Not Show The Live Crop

The visual debug panel named `SEARCH CTX` may show a dynamic-template search
context, not the actual crop used on the current frame.

So:

```text
side panel unchanged does not prove live crop unchanged
```

### Reason 5: The Reported Score Is Still Appearance-Based

The KF-fused map can choose the winning location, but the reported confidence
score may still be read from the original appearance score at that location.

So the visible score may look similar even when the selection map changed.

---

## 12. The Current Important Config Knobs

Target-motion KF lives under:

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_stable_frames: 1
  kf_motion_score_threshold: 0.55
  kf_motion_reseed_dist: 2.0
  kf_motion_center_search: True
```

Camera-motion GMC search prior lives under:

```yaml
ram_tracker:
  gmc_prior:
    gmc_prior_enabled: true
    gmc_prior_require_reliable_h: true
    gmc_prior_skip_in_distractor_mode: true
    gmc_prior_max_translation_frac: 0.25
    gmc_prior_min_scale: 0.7
    gmc_prior_max_scale: 1.4
    gmc_prior_max_rotation_deg: 25.0
    gmc_prior_max_corner_displacement_frac: 0.25
```

The main switches are:

```text
kf_motion_enabled          -> allow response-map motion fusion
kf_motion_weight           -> how strongly motion affects the response map
kf_motion_center_search    -> center crop using target-motion prediction
gmc_prior_enabled          -> center/search from camera-compensated bbox first
```

---

## 13. How To Run Target-Motion Fusion And KF-Centered Search Together

If by "both" you mean the two target-motion features:

```text
1. response-map weighting toward target motion
2. search-crop centering toward target motion
```

use:

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_stable_frames: 1
  kf_motion_score_threshold: 0.55
  kf_motion_reseed_dist: 2.0
  kf_motion_center_search: True

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: false
```

This isolates target-motion behavior. It is the cleanest setup for studying how
KF response-map fusion and KF crop centering behave without camera-motion prior
interference.

Run one video with annotated output:

```bash
uv run run_single_video.py 42
```

Or run one exact manifest key:

```bash
uv run run_inference.py --run_split all --video_key dataset1/volleyball --output_video
```

Run a quick one-sequence smoke test:

```bash
uv run run_inference.py --run_split public_lb --max_sequences 1 --output_video
```

---

## 14. How To Run Target-Motion KF And Camera-Motion GMC Together

If by "both" you mean:

```text
1. target-motion Kalman prior
2. camera-motion GMC prior
```

then enable both blocks:

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_stable_frames: 1
  kf_motion_score_threshold: 0.55
  kf_motion_reseed_dist: 2.0
  kf_motion_center_search: True

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: true
    gmc_prior_require_reliable_h: true
    gmc_prior_skip_in_distractor_mode: true
    gmc_prior_max_translation_frac: 0.25
    gmc_prior_min_scale: 0.7
    gmc_prior_max_scale: 1.4
    gmc_prior_max_rotation_deg: 25.0
    gmc_prior_max_corner_displacement_frac: 0.25
```

Then run:

```bash
uv run run_single_video.py 42
```

Or:

```bash
uv run run_inference.py --run_split all --video_key dataset1/volleyball --output_video
```

For a leaderboard-style run:

```bash
uv run run_inference.py --run_split public_lb --max_sequences 0
```

For a small test:

```bash
uv run run_inference.py --run_split public_lb --max_sequences 1
```

---

## 15. Recommended Student Experiments

Use the same video and change only one idea at a time.

### Experiment A: No Motion Prior

```yaml
tracker:
  kf_motion_enabled: false
  kf_motion_center_search: false

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: false
```

This is the visual-only reference.

### Experiment B: KF Response Fusion Only

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_center_search: false

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: false
```

This tests whether response-map reweighting changes peak selection.

### Experiment C: KF Fusion Plus KF-Centered Search

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_center_search: True

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: false
```

This tests pure target-motion steering.

### Experiment D: GMC Only

```yaml
tracker:
  kf_motion_enabled: false
  kf_motion_center_search: false

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: true
```

This tests camera-motion compensation without target-motion KF.

### Experiment E: KF Plus GMC

```yaml
tracker:
  kf_motion_enabled: True
  kf_motion_weight: 0.2
  kf_motion_center_search: True

ram_tracker:
  gmc_prior:
    gmc_prior_enabled: true
```

This tests the full combined motion-aware setup.

---

## 16. How To Interpret What You See

When comparing runs, do not only look at the visible side panel.

Look for:

```text
does the predicted bbox avoid jumping to similar objects?
does the target stay inside the crop during fast movement?
does the tracker recover faster after camera movement?
does the tracker over-trust motion and miss a real sudden turn?
```

Motion priors are most visible when:

```text
there are look-alike distractors
the target moves near the crop boundary
the camera pans or shakes
the response map has multiple plausible peaks
```

They are least visible when:

```text
the target is easy
the response map has one strong correct peak
the object barely moves
the prior is gated off by low confidence
```

---

## 17. Safe Starting Values

For student experiments, start conservative:

```yaml
kf_motion_weight: 0.2
kf_motion_stable_frames: 1
kf_motion_score_threshold: 0.55
kf_motion_center_search: True
```

If the tracker over-trusts motion, try:

```yaml
kf_motion_weight: 0.15
kf_motion_stable_frames: 3
```

If the tracker needs stronger motion help, try:

```yaml
kf_motion_weight: 0.3
```

Be careful with very high values such as:

```yaml
kf_motion_weight: 0.8
```

That makes the motion prior very strong. It can help in some ambiguous cases,
but it can also force the tracker to follow a wrong predicted trajectory.

---

## 18. Final Mental Model

The cleanest way to remember the interaction is:

```text
GMC answers: "how did the camera move?"
KF answers:  "how did the target move?"
SiamABC answers: "where does the target visually match?"
```

When they work together:

```text
GMC places the search near the camera-compensated location
KF nudges the search and response map toward target-motion consistency
SiamABC still makes the visual match
SiamRAM still applies confidence, memory, occlusion, and distractor logic
```

So these priors do not replace the tracker. They guide it.

