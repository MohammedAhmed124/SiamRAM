# SiamRAM, From the Ground Up — Reading Guide & System Overview

> **Who this is for.** A student who likes mathematics and wants to *understand*, not just use, a modern single‑object visual tracker. We assume you know high‑school algebra and a little calculus (what a derivative is). Everything else — vectors, matrices, probability, Gaussians, Kalman filters, homographies, convolution — is built up from scratch in these notes. Nothing is hand‑waved.

> **Scope.** These notes describe the **SiamRAM** tracker in this repository, component by component, *theoretically and mathematically*. They deliberately **ignore "distractor mode"** (the identity‑arbitration subsystem) **and the spike watcher** that feeds it — that line of work is a deprecated/failed experiment and is out of scope here. Everything else — the visual matcher, the motion filter, the camera‑motion estimator, the appearance memory, occlusion recovery, frame‑dynamics motion injection, the adaptive controllers, and how they connect — is covered in full.

---

## 0. How to read this series

The documents are numbered. **Read them in order the first time.** The early ones build tools and core subsystems; the later ones assemble those subsystems into the full tracker.

| # | File | What it teaches | Depends on |
|---|------|-----------------|-----------|
| 00 | `00_READING_GUIDE.md` | *(this file)* the problem, the map, notation | — |
| 01 | `01_MATHEMATICAL_FOUNDATIONS.md` | vectors, matrices, norms, dot/cosine, sigmoid, argmax, expectation, Gaussians, EMA, interpolation | — |
| 02 | `02_BOUNDING_BOXES_AND_IMAGE_GEOMETRY.md` | boxes, IoU, crops, context, coordinate spaces, the score grid | 01 |
| 03 | `03_SIAMESE_TRACKING_SIAMABC.md` | the visual matcher: correlation, attention, FCOS decoding, penalties, dual templates | 01, 02 |
| 04 | `04_PROBABILITY_AND_BAYESIAN_FILTERING.md` | Bayes' rule, recursive estimation, the Kalman filter derived from scratch, the EKF | 01 |
| 05 | `05_HOMOGRAPHY_AND_EKF.md` | projective geometry, homographies, optical flow, RANSAC, and SiamRAM's camera‑aware EKF | 01, 04 |
| 06 | `06_MOTION_PRIORS.md` | the constant‑velocity Kalman prior, KF‑IoU response fusion, the GMC search prior, heavy‑motion gating | 03, 04, 05 |
| 07 | `07_APPEARANCE_MEMORY_RAM_DRM.md` | appearance descriptors, short‑term RAM, long‑term DRM, the composite re‑acquisition score | 01, 02 |
| 08 | `08_OCCLUSION_RECOVERY.md` | loss detection, EKF‑guided search, the phase machine, YOLO re‑detection, verification | 05, 06, 07 |
| 09 | `09_FRAME_DYNAMICS.md` | injecting short‑term motion saliency into the search crop for tiny targets | 02, 03 |
| 10 | `10_ADAPTIVE_CONTROLLERS.md` | exponential moving averages as controllers; the three auto‑tuners | 01 |
| 11 | `11_PUTTING_IT_ALL_TOGETHER.md` | the per‑frame conductor and the full state machine | all |

If you only have time for one "deep" file, the user‑requested centerpiece is **05 (homography + EKF)**, but it leans on **04 (Kalman filtering)**, so read those two together.

---

## 1. The problem: single‑object tracking

You are given a video — a sequence of images (frames) $I_0, I_1, I_2, \dots$ — and, **on the first frame only**, a rectangle telling you where a target object is:

$$
\text{given: } I_0 \text{ and a box } b_0 = (x_0, y_0, w_0, h_0).
$$

Your job: for every later frame $I_t$, output a box $b_t$ that bounds *the same object*. You are never told the answer again. You must follow that one object — and only that object — through everything the world throws at it.

This is **single‑object tracking (SOT)**, also called *model‑free* tracking because you are not told what *class* of thing you are tracking (it might be a fish, a drone, a player, a car). You only know its appearance from one frame.

### 1.1 Why this is hard

If the target always stayed sharp, fully visible, alone, and still relative to the camera, a simple template‑matcher would suffice. Real video breaks each of those assumptions:

- **Appearance change.** The object rotates, deforms, changes lighting, changes scale. The frame‑0 crop stops looking like the object.
- **Camera motion.** The camera pans, zooms, shakes. The object can move dozens of pixels in the *image* without moving at all in the *world*.
- **Occlusion / leaving frame.** The object disappears behind something, or exits the frame entirely, then comes back — possibly far from where it vanished.
- **Look‑alikes (distractors).** Another similar object passes by, and a naive matcher silently jumps onto the impostor. *(SiamRAM has a "distractor mode" for this; we set it — and the spike watcher that triggers it — aside by design.)*
- **Tiny / low‑texture targets.** A small, faint, or distant object barely registers in appearance; its *motion* is often the strongest cue that it is there at all.

SiamRAM is, essentially, **a fast visual matcher wrapped in a stack of mathematical safety nets**, each one designed to survive one of these failure modes.

### 1.2 The one mental model

Hold this picture in your head for the whole series:

```
            ┌──────────────────────────────────────────────────────┐
            │                   SiamRAM (the wrapper)                │
            │                                                        │
            │   motion model (EKF)      camera‑motion (homography)   │
            │   appearance memory       occlusion recovery           │
            │   spike rejection         adaptive thresholds          │
            │                                                        │
            │        ┌───────────────────────────────────┐          │
            │        │  SiamABC  — the fast visual matcher │          │
            │        │  "given last position, where now?" │          │
            │        └───────────────────────────────────┘          │
            └──────────────────────────────────────────────────────┘
```

- The **inner matcher** (SiamABC, doc 03) is *fast and precise but myopic*. It only looks in a small window near where the target was last, and it has no memory and no notion of physics — only "what looks most like the template, right here, right now."
- The **outer layer** (everything else) *watches* the matcher. It predicts where the target should be (motion + camera models), remembers what it looks like (appearance memory), notices when the matcher's confidence collapses (loss detection), runs a structured search to find it again (occlusion recovery), and rejects physically implausible jumps (spike rejection). It also *tunes its own thresholds* to the difficulty of the current video (adaptive controllers).

Every later document is one of those safety nets, explained in full.

---

## 2. Notation used throughout

We fix conventions now so later math is unambiguous.

**Scalars, vectors, matrices.** Scalars are italic lowercase: $x, w, s$. Vectors are bold lowercase: $\mathbf{x} = (x_1, \dots, x_n)$, treated as **column** vectors unless stated. Matrices are uppercase: $A, H, P$. The transpose is $A^{\!\top}$; the inverse is $A^{-1}$; the $n\times n$ identity is $I_n$ (or just $I$).

**Frames and time.** $I_t$ is the image at discrete time (frame index) $t = 0, 1, 2, \dots$. A quantity "at time $t$" gets subscript $t$; "predicted before seeing the measurement" gets a minus superscript, e.g. $\hat{\mathbf{x}}_t^-$; "after correction" gets a plus, $\hat{\mathbf{x}}_t^+$.

**Bounding boxes.** Two formats appear constantly (doc 02 details them):

$$
b = (x, y, w, h) \quad\text{(top‑left corner + size)},\qquad
b = (x_1, y_1, x_2, y_2)\quad\text{(two corners)}.
$$

The **center** of a box is $(c_x, c_y) = \big(x + \tfrac{w}{2},\, y + \tfrac{h}{2}\big)$.

**Pixels and coordinates.** Image coordinates are $(u, v)$ or $(x, y)$ with $x$ to the right and $y$ **downward** (the standard image convention — the origin is the top‑left corner). A point in the image is $\mathbf{p} = (x, y)$.

**Probability.** $p(\mathbf{x})$ is a probability density; $\mathbb{E}[\cdot]$ is expectation; $\mathrm{Cov}[\cdot]$ is covariance; $\mathcal{N}(\boldsymbol\mu, \Sigma)$ is the Gaussian with mean $\boldsymbol\mu$ and covariance $\Sigma$.

**Indicator / clamp.** $\mathbb{1}[\text{condition}]$ is $1$ when the condition holds, else $0$. $\operatorname{clip}(z, a, b) = \min(\max(z, a), b)$ clamps $z$ into $[a,b]$.

---

## 3. The processing pipeline in one breath

Before diving into subsystems, here is the journey of a single frame through SiamRAM. Every box and arrow below is expanded — with its mathematics — somewhere in docs 03–11. Treat this as the skeleton.

```
 full‑resolution frame  I_t   (H × W × 3, color)
        │
        ▼  (1) prescale to a bounded working resolution; all internal math is in "proc" coordinates
 proc frame
        │
        ▼  (2) estimate camera motion: homography H between previous and current proc frame   [doc 05]
        │
        ▼  (3) EKF predict: warp the target's predicted center by H, then add its own velocity [doc 05]
        │
   in occlusion?
    ├── yes ──►  OCCLUSION RECOVERY  (EKF‑guided search → YOLO candidates → memory match → verify) [doc 08]
    │
    └── no  ──►  NORMAL UPDATE                                                                      [docs 03,06,09,10]
                 │
                 ▼ optionally inject a camera‑compensated search prior (GMC)                       [doc 06]
                 ▼ optionally blend short‑term motion saliency into the search crop                [doc 09]
                 ▼ SiamABC forward: response map → box + confidence score                          [doc 03]
                 ▼ (optional) Kalman motion fusion reweights the response map                      [doc 06]
                 ▼ compute the effective loss threshold (possibly adaptive)                        [doc 10]
                 ▼ score < threshold for enough consecutive frames? → ENTER OCCLUSION              [doc 08,11]
                 ▼ else: EKF update, admit appearance to memory, refresh dynamic template          [docs 05,07,03]
        │
        ▼  (4) scale the chosen box back to full resolution, emit (box, score, in_occlusion)
 output b_t
```

Two structural facts to anchor on now, because they recur:

1. **Exactly one of `normal update` or `occlusion recovery` runs per frame.** They are two disjoint "worlds." The tracker is a state machine with these two macro‑states (plus the distractor world we are ignoring).

2. **All internal computation happens in a single, downscaled "proc" coordinate space**, and only the final box is scaled back to the original video resolution. So when later docs talk about pixels, displacements, or box diagonals, they mean *proc pixels* unless noted. This keeps every subsystem (matcher, motion, camera, YOLO, memory) speaking the same units.

---

## 4. The cast of subsystems

A quick glossary so the names in later docs are familiar. Each links to its full treatment.

- **SiamABC** *(doc 03)* — the inner Siamese neural matcher. Compares a *template* of the target to a *search region* and outputs a dense map of "is the target here?" scores plus box geometry. Fast, local, appearance‑only.

- **EKF** — Extended Kalman Filter *(docs 04, 05)* — a recursive estimator of the target's **center and velocity** $(c_x, c_y, v_x, v_y)$. It predicts where the target will be next frame and corrects itself from the matcher's output. It can fold in **camera motion** through a homography.

- **GMC** — Global Motion Compensation *(doc 06)* — uses the estimated camera homography to *shift the search region* to where the target should appear after the camera moved.

- **KF motion prior** *(doc 06)* — a second, constant‑velocity Kalman filter over the full box $(c_x, c_y, w, h, \dots)$ whose predicted box reweights the matcher's response map toward motion‑consistent locations (the "SAMURAI" idea).

- **Appearance memory: RAM + DRM** *(doc 07)* — RAM is a short‑term buffer of recent confident target descriptors; DRM ("Dynamic Reference Memory") is a curated long‑term bank used to *re‑identify* the target after it is lost. Matching uses a composite of appearance, overlap, motion direction, and recency.

- **Occlusion recovery** *(doc 08)* — the structured re‑detection pipeline that runs once the matcher's confidence has collapsed: it coasts on the EKF, grows a search region, runs an object detector (YOLO), scores candidates against DRM, then verifies a winner by re‑running SiamABC.

- **Frame dynamics** *(doc 09)* — injects short‑term motion saliency (frame differences) into the search crop to help lock onto tiny moving targets.

- **Adaptive controllers** *(doc 10)* — small exponential‑moving‑average estimators that retune the confidence threshold, the re‑acquisition margin, and the template‑update rate to each video's difficulty.

- **The conductor** *(doc 11)* — the master per‑frame routine (`update`) that wires all the above into a coherent state machine.

---

## 5. A note on honesty and on neural networks

Two caveats that matter for a careful reader.

**We describe the code that runs.** Where a research paper's *intent* and the *implementation* differ, these notes follow the implementation, because that is what produces the tracker's behavior. Citations to the source look like `tracker.py` or name a function.

**The neural network is not derived here.** SiamABC's *weights* come from training, which is its own large topic. These notes explain the **architecture and the math of how the network is used at inference** — what it consumes, what it outputs, how those outputs are turned into a box, and why each operation is shaped the way it is. We treat the trained convolutional backbone as a given function $f_\theta$ that maps an image crop to a feature map; doc 03 explains everything built around it. That is the right altitude for understanding the *tracker* (as opposed to re‑deriving deep learning).

---

## 6. What you should be able to do after each milestone

- **After doc 03:** explain how a Siamese tracker turns two image crops into a bounding box, and why a cosine window and a scale penalty stop it from teleporting.
- **After doc 05:** derive the Kalman prediction/update equations, explain what a homography is and how it is estimated from optical flow + RANSAC, and read the EKF's Jacobian line by line.
- **After doc 07:** explain the multi‑term re‑acquisition score and why each term is there.
- **After doc 11:** trace a single frame through the entire state machine and say, at each branch, *why* the tracker does what it does.

Turn to **`01_MATHEMATICAL_FOUNDATIONS.md`** to assemble the toolkit.
