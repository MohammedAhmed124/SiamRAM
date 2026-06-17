# GMC In SiamRAM: Theoretical Step-By-Step Flow

This note explains what happens in SiamRAM when the **GMC prior** is enabled.
It is written as a theoretical runtime description, not as a code walkthrough.

GMC means **Global Motion Compensation**. In SiamRAM, it estimates how the
camera moved between two consecutive frames and uses that estimate to shift the
expected target location before the SiamABC visual tracker searches.

The short version:

```text
previous target box
    + estimated background/camera motion
    = better search prior for the next frame
```

GMC does **not** replace SiamABC. It only changes where SiamABC starts looking.
The final box still comes from the normal visual tracking, confidence, motion,
occlusion, and distractor logic.

---

## 1. A New Frame Arrives

For every video frame, SiamRAM first moves into its internal processing
coordinate space. If the input frame is too large, it is downscaled to the
configured processing resolution.

The important theoretical point is:

```text
all GMC, EKF, SiamABC, YOLO, and memory logic work in the same processed frame
coordinate system
```

Only at the end is the final bbox scaled back to the original video resolution.

---

## 2. SiamRAM Estimates Camera Motion Before Tracking

Before SiamABC predicts the target box for the current frame, SiamRAM estimates
the motion between:

```text
previous frame -> current frame
```

This motion is represented as a matrix called a **homography**, usually written:

```text
H
```

Conceptually, `H` maps a point from the previous frame into the current frame:

```text
p_current ~= H * p_previous
```

If the camera panned, tilted, shook, zoomed, or rotated slightly, `H` describes
that global image motion.

---

## 3. The Frame Is Converted For Motion Estimation

SiamRAM does not estimate GMC on the full RGB/BGR image directly.

It first:

1. Downscales the current frame by the optical-flow scale.
2. Converts the downscaled frame to grayscale.
3. Compares this grayscale frame with the previous grayscale frame.

If there is no previous grayscale frame, for example on the first frame, no GMC
motion can be estimated yet.

In that case:

```text
H = None
reliable = False
```

The tracker then continues without a GMC search prior.

---

## 4. SiamRAM Chooses A Homography Estimator

The current implementation supports two theoretical camera-motion modes:

```text
classic
accurate
```

### Classic Mode

Classic mode is the default fast path.

It treats camera motion as a mostly global affine-like transformation:

```text
translation + rotation + scale + mild shear
```

The theoretical process is:

1. Place a regular grid of points over the previous frame.
2. Remove grid points that fall inside or near the target box.
3. Track the remaining background points into the current frame using optical flow.
4. Use RANSAC to fit a robust background transform.
5. Convert that transform into a 3x3 homography-style matrix `H`.

The target region is excluded because the target may move independently from
the camera. GMC wants background motion, not target motion.

### Accurate Mode

Accurate mode is slower but more expressive.

The theoretical process is:

1. Detect good visual feature points in the previous grayscale frame.
2. Mask out the target area so the object itself does not dominate the estimate.
3. Track those features forward into the current frame.
4. Track them backward to check consistency.
5. Keep only points that survive the forward/backward test.
6. Fit a full homography with RANSAC or MAGSAC.

If accurate mode fails to produce a usable transform, SiamRAM falls back to the
classic estimator.

---

## 5. The Homography Gets A Reliability Flag

The estimator returns two things:

```text
H
reliable
```

`H` is the estimated camera/background motion.

`reliable` says whether the estimate looked trustworthy enough. The main idea is
that enough tracked background points must agree with the same global motion.

If too few points agree, the transform may be random, caused by blur, caused by
the moving target, or caused by a low-texture scene. Then it is marked
unreliable.

---

## 6. SiamRAM Stores The Latest Camera Motion

After estimating camera motion, SiamRAM stores:

```text
last_H
last_H_reliable
```

These values are used by several subsystems during the same frame:

```text
GMC search prior
EKF prediction
heavy-camera-motion detection
loss-cause classification
camera-compensated velocity logic
```

So GMC is not just a visual-tracker trick. The same camera-motion estimate also
helps SiamRAM reason about whether apparent motion came from the target or from
the camera.

---

## 7. The EKF Predicts With Camera Motion

SiamRAM's EKF tracks the target center and velocity:

```text
[cx, cy, vx, vy]
```

When a reliable homography exists, the EKF first warps the previous target
center by camera motion:

```text
camera-shifted center = H * previous center
```

Then it adds the target's own velocity:

```text
predicted center = camera-shifted center + target velocity
```

This separates two effects:

```text
apparent image motion = camera motion + real target motion
```

That is one of the main theoretical reasons GMC helps. Without it, camera pan
and target motion are mixed together.

---

## 8. SiamRAM Enters The Normal Tracking Step

If the tracker is not already in occlusion recovery, it enters the normal update
path.

This is where the GMC search prior may be applied.

The prior is attempted only if:

```text
gmc_prior_enabled = true
```

If this switch is off, SiamRAM still may estimate camera motion for other logic,
but it will not inject the warped bbox into SiamABC as a search prior.

---

## 9. SiamRAM Checks Whether The GMC Prior Should Be Skipped

Even when enabled, the GMC prior is skipped in several cases.

It is skipped if:

```text
there is no current target bbox
there is no homography H
the homography is unreliable and reliability is required
the tracker is in distractor mode and skip-in-distractor-mode is enabled
```

The distractor-mode skip is intentional. Distractor mode is trying to resolve
identity ambiguity between similar-looking objects. SiamRAM keeps that logic less
dependent on camera motion so a bad global transform does not push the tracker
toward the wrong identity.

---

## 10. SiamRAM Runs Plausibility Gates On The Motion

If a homography exists, SiamRAM still does not trust it blindly.

It extracts theoretical motion statistics from `H`:

```text
dx                  horizontal translation
dy                  vertical translation
scale               approximate zoom factor
rotation_deg        approximate rotation angle
max_corner_disp     largest displacement of any frame corner
```

Then it rejects the GMC prior if the motion is too extreme.

The default theoretical gates are:

```text
translation must be <= 25% of the larger frame dimension
scale must stay between 0.7 and 1.4
rotation must be <= 25 degrees
corner warp must be <= 25% of the frame diagonal
```

These gates protect the tracker from using a mathematically valid but physically
unlikely transform.

Example:

```text
If RANSAC accidentally fits a huge rotation because the scene is blurred,
the plausibility gate rejects it and SiamABC tracks normally.
```

---

## 11. SiamRAM Warps The Previous Target Box

If the homography passes all checks, SiamRAM applies it to the previous target
bbox.

The bbox is represented as:

```text
[x, y, w, h]
```

Theoretical warp process:

1. Convert the bbox into four corners.
2. Project each corner through the homography `H`.
3. Find the min/max x and y of the warped corners.
4. Build a new axis-aligned bbox around the warped corners.
5. Clip the bbox so it stays inside the frame.

So the prior is not just:

```text
x += dx
y += dy
```

It can also reflect mild zoom and rotation because all four corners are warped.

---

## 12. The Warped Box Becomes SiamABC's Search Prior

After warping, SiamRAM injects the warped bbox into SiamABC's internal tracking
state.

The meaning is:

```text
"For this frame, start the visual search from this camera-compensated location."
```

This does not force the final answer. It changes the search center and state
that SiamABC uses before running its normal matching.

The next operation is still:

```text
SiamABC.update(current_frame)
```

SiamABC then compares the target template against the search crop and outputs:

```text
predicted bbox
confidence score
```

---

## 13. SiamABC Produces The Actual Visual Prediction

Once SiamABC runs, the GMC prior has already done its job.

The final predicted bbox for the frame comes from SiamABC's visual matching, not
directly from `H`.

The theoretical relationship is:

```text
GMC says: "look around here"
SiamABC says: "the target visually matches best here"
```

This distinction matters. GMC can help the tracker stay centered during camera
motion, but a bad visual match can still be rejected later by SiamRAM's other
logic.

---

## 14. SiamRAM Then Applies The Rest Of Its Normal Logic

After SiamABC predicts a bbox and score, SiamRAM continues with the rest of the
normal tracking pipeline.

This can include:

```text
class warmup
YOLO detectability probing
confidence thresholding
distractor-mode checks
hard-jump rejection
heavy-camera-motion checks
occlusion-entry logic
EKF update
appearance-memory admission
history updates
```

GMC helps place the search, but it does not bypass these safeguards.

---

## 15. Heavy Camera Motion Is Also Derived From GMC

SiamRAM uses the homography to estimate how much the camera moved.

A common scalar summary is:

```text
how far the frame center moves after applying H
```

That value feeds the heavy-camera-motion logic.

If camera motion is heavy, SiamRAM may:

```text
avoid treating a temporary low score as true occlusion
avoid entering distractor mode because of a camera-induced jump
adapt dynamic-template update behavior if configured
classify a loss as camera-motion-related
```

The theory is:

```text
during fast camera movement, low tracker confidence may mean "search lag"
instead of "target disappeared"
```

---

## 16. Camera-Compensated Velocity Is Recorded

During healthy tracking, SiamRAM records camera motion history.

Later, when estimating target velocity, it can subtract the camera component:

```text
observed center displacement - camera displacement = target's own motion
```

This is especially important before occlusion recovery.

If the target is lost after a camera pan, SiamRAM wants the recovery search to be
guided by the target's real motion, not by the camera pan that moved the whole
scene.

---

## 17. The Previous Gray Frame Is Updated

After the frame update finishes, the current grayscale frame becomes the
previous grayscale frame for the next iteration.

So the next frame will estimate:

```text
current frame -> next frame
```

This makes GMC a continuous frame-to-frame motion estimate.

---

## 18. What Happens When GMC Fails

If GMC cannot estimate a reliable transform, SiamRAM does not crash and does not
force a bad prior.

It simply skips the GMC search-prior injection for that frame.

The tracker then behaves like:

```text
SiamABC visual tracking
+ EKF prediction without reliable camera warp
+ normal SiamRAM confidence / memory / occlusion logic
```

This fallback behavior is important because real video often contains:

```text
motion blur
low texture
large foreground objects
scene cuts
fast zooms
rolling shutter artifacts
partial occlusion
```

In those cases, "no GMC prior" is safer than "wrong GMC prior".

---

## 19. Why GMC Helps

Without GMC, SiamABC's search crop is centered around the last target location.
That works when the target moves smoothly and the camera is stable.

But if the camera moves quickly, the whole image shifts. The target may appear
far away from its previous image coordinates even if it did not move much in the
real world.

Without compensation:

```text
camera pan -> target appears displaced -> SiamABC search crop may be stale
```

With GMC:

```text
camera pan -> estimate global image motion -> warp previous bbox -> search near expected new location
```

This makes the tracker more robust to:

```text
camera shake
fast pan
small zoom
small rotation
temporary visual lag
```

---

## 20. What GMC Is Not

GMC is not an object detector.

GMC is not an identity matcher.

GMC is not the final tracking decision.

GMC is not guaranteed to help if the background itself is moving, if the scene
has strong parallax, or if the homography is unreliable.

In SiamRAM, GMC is best understood as:

```text
a camera-motion-aware search prior
```

It helps the visual tracker start from a more physically plausible location.

---

## Full Theoretical Timeline

```text
Frame t arrives
  |
  v
Prescale frame into processing coordinates
  |
  v
Convert to grayscale for motion estimation
  |
  v
Compare previous gray frame with current gray frame
  |
  v
Estimate global camera/background motion H
  |
  v
Mark H as reliable or unreliable
  |
  v
Store last_H and last_H_reliable
  |
  v
EKF predicts target center using H if reliable
  |
  v
If normal tracking and gmc_prior_enabled:
  |
  v
Check skip conditions
  |
  v
Check translation / scale / rotation / corner-warp plausibility
  |
  v
Warp previous bbox by H
  |
  v
Inject warped bbox into SiamABC as search prior
  |
  v
Run SiamABC visual update
  |
  v
Get predicted bbox and confidence score
  |
  v
Apply SiamRAM confidence, distractor, jump, occlusion, memory, and EKF logic
  |
  v
Store current gray frame as previous gray frame
  |
  v
Return final bbox in original video coordinates
```

---

## One-Sentence Summary

When enabled, GMC in SiamRAM estimates frame-to-frame camera motion, warps the
previous target box into the current frame, gives that warped box to SiamABC as
a better search prior, and then lets the normal SiamRAM tracking pipeline decide
whether the visual result is valid.
