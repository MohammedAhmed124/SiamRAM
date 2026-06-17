# 05 — Homography Estimation and the Camera‑Aware EKF

> Prerequisites: doc 01 (matrices, Jacobian, quotient rule, atan2), doc 02 (boxes, coordinate spaces), doc 04 (the Kalman filter and the EKF). **This is the centerpiece.** We build, from scratch: homogeneous coordinates and what a homography *is*; how SiamRAM estimates the frame‑to‑frame camera motion from optical flow + RANSAC; how it decomposes and sanity‑checks that motion; and finally how its `BBoxEKF` folds the camera motion into the prediction step — including a line‑by‑line derivation of the Jacobian the code computes. By the end you should be able to read `models/siamram/motion.py` and the homography functions in `tracker.py` and understand every line.

---

## Part A — Projective geometry and the homography

### A.1 The motivating picture

When the camera pans, tilts, rotates, or zooms between two frames, the *entire image* shifts in a structured way. A static background point at pixel $(x, y)$ in frame $t-1$ appears at some new pixel $(x', y')$ in frame $t$. If we can model that map $(x,y)\mapsto(x',y')$ for the background, we can *predict where a stationary target would appear* after the camera moved — and, more importantly, *separate* camera‑induced image motion from the target's own motion. That map, for a moving camera viewing a roughly planar or distant scene, is a **homography**.

```
   frame t−1                         frame t (camera panned right + slight zoom)
   ┌───────────────┐                 ┌───────────────┐
   │   ★ (x,y)     │     H            │ ★ (x',y')     │      a background point ★
   │      • target │  ────────►       │   • target?   │      moved in the image only
   │               │                  │               │      because the camera moved
   └───────────────┘                 └───────────────┘
```

### A.2 Homogeneous coordinates

To write perspective maps as matrix multiplications, we use **homogeneous coordinates**. A 2‑D point $(x, y)$ is represented by a 3‑vector

$$
\tilde{\mathbf{p}} = \begin{bmatrix} x \\ y \\ 1\end{bmatrix},
$$

and — this is the key rule — **any nonzero scalar multiple represents the same 2‑D point**:

$$
\begin{bmatrix} x \\ y \\ 1\end{bmatrix}
\ \equiv\
\begin{bmatrix} \lambda x \\ \lambda y \\ \lambda\end{bmatrix}\quad(\lambda\neq0).
$$

To go back to ordinary ("inhomogeneous") coordinates from a general homogeneous vector $(a, b, c)$, you **divide by the third coordinate**:

$$
(a, b, c) \ \longrightarrow\ \Big(\tfrac{a}{c},\ \tfrac{b}{c}\Big).
$$

That division is the **perspective division** — and it is the nonlinearity at the heart of this whole document. Why bother? Because in this representation, *every* projective transformation (translation, rotation, scale, shear, and true perspective foreshortening) becomes a single $3\times3$ matrix multiply. Without homogeneous coordinates, translation can't be written as a matrix multiply at all; with them, everything can.

### A.3 The homography matrix

A **homography** is a $3\times3$ matrix $H$ acting on homogeneous points:

$$
\tilde{\mathbf{p}}' = H\,\tilde{\mathbf{p}},
\qquad
H = \begin{bmatrix} h_{00} & h_{01} & h_{02} \\ h_{10} & h_{11} & h_{12} \\ h_{20} & h_{21} & h_{22}\end{bmatrix}.
$$

Writing it out and applying the perspective division (A.2), a point $(x,y)$ maps to

$$
\boxed{\
x' = \frac{h_{00}x + h_{01}y + h_{02}}{h_{20}x + h_{21}y + h_{22}},
\qquad
y' = \frac{h_{10}x + h_{11}y + h_{12}}{h_{20}x + h_{21}y + h_{22}}.
\ }
$$

Stare at this — these two formulas are *literally* the lines in `BBoxEKF.predict` and `_h_translation_magnitude`. Reading off the structure:

- $h_{02}, h_{12}$ are **translation** (they shift $x', y'$).
- The top‑left $2\times2$ block $\begin{bmatrix} h_{00} & h_{01} \\ h_{10} & h_{11}\end{bmatrix}$ does **rotation, scale, and shear**.
- The bottom row $h_{20}, h_{21}$ produces **perspective**: when nonzero, the denominator depends on position, so the map is *not* affine — far parts of the image transform differently from near parts. This is exactly what a tilting camera does.
- $H$ is defined **only up to scale** (multiplying all nine entries by the same constant gives the same map, by A.2), so it has $8$ degrees of freedom, not $9$. Conventionally $h_{22}\approx1$.

**Special case — affine.** If the bottom row is $(0,0,1)$, the denominator is constant $=1$ and the map becomes linear‑plus‑translation:

$$
\begin{bmatrix} x' \\ y'\end{bmatrix} = \begin{bmatrix} h_{00} & h_{01} \\ h_{10} & h_{11}\end{bmatrix}\begin{bmatrix} x \\ y\end{bmatrix} + \begin{bmatrix} h_{02} \\ h_{12}\end{bmatrix}.
$$

This is an **affine** transform (6 DOF). SiamRAM's fast "classic" estimator (Part B) produces an even more restricted form — a *similarity* transform: rotation + uniform scale + translation (4 DOF) — and stores it in the top two rows of a $3\times3$ $H$ with bottom row $(0,0,1)$. The "accurate" estimator produces a full $8$‑DOF homography.

### A.4 Decomposing $H$ into interpretable numbers

SiamRAM never uses the raw $H$ for thresholds; it first extracts human‑meaningful motion statistics (`_gmc_motion_stats`). Given $H$ (treating the relevant block as a scaled rotation $\begin{bmatrix} a & * \\ b & *\end{bmatrix}$ with $a = h_{00}$, $b = h_{10}$):

$$
\underbrace{dx = h_{02},\quad dy = h_{12}}_{\text{translation (px)}},
\qquad
\underbrace{\text{scale} = \sqrt{a^2 + b^2}}_{\text{zoom factor}},
\qquad
\underbrace{\theta = \operatorname{atan2}(b, a)}_{\text{rotation (rad)}}.
$$

The scale and rotation come from recognizing that a rotation‑by‑$\theta$‑and‑scale‑by‑$\sigma$ matrix is $\sigma\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix} = \begin{bmatrix} a & -b \\ b & a\end{bmatrix}$, so $a = \sigma\cos\theta,\ b = \sigma\sin\theta$, giving $\sigma = \sqrt{a^2+b^2}$ and $\theta = \operatorname{atan2}(b,a)$ (doc 01 §7.1). The code reports $\theta$ in degrees.

It also computes the **maximum corner displacement**: warp the four frame corners through $H$ and take the largest distance moved,

$$
\text{max\_corner\_disp} = \max_{k\in\text{corners}} \big\| H\!\cdot\!\tilde{\mathbf{c}}_k \big|_{\text{÷ persp}} - \mathbf{c}_k \big\|.
$$

This single scalar captures the *worst‑case* image motion implied by $H$ (including perspective), which is a robust summary for the plausibility gates (Part D).

---

## Part B — Estimating the homography from two frames

We have the model; now, how do we *find* $H$ from two consecutive grayscale frames? The pipeline is: pick points → track them with optical flow → robustly fit $H$ to the agreeing points → flag reliability. Crucially, we estimate from **background** points, masking out the target.

### B.1 Why mask the target

We want the *camera/background* motion, not the *target's* motion. The target generally moves differently from the background; if it dominated the point set, the fit would learn target motion and the whole compensation would be backwards. So both estimators **exclude points on or near the current target box** (padding the box by ~15–20% of its size). The fast path simply drops grid points inside the padded box; the accurate path builds a mask image with the box region zeroed so the feature detector won't place points there.

### B.2 Working at reduced resolution

Estimation runs on a **downscaled** grayscale image (scale $s=$ `flow_scale`, e.g. $0.5$) for speed. Everything tracked is in downscaled pixels; the resulting transform must be converted back to proc‑resolution. For the classic (affine) path the conversion is: keep the linear $2\times2$ block (a ratio, scale‑invariant) and **divide the translation by $s$**. Why? If a proc point $\mathbf p$ becomes a small‑image point $s\mathbf p$, and the affine map on the small image is $A_{\text{lin}}(s\mathbf p) + \mathbf t_{\text{small}}$, then converting the result back to proc by dividing by $s$ gives $A_{\text{lin}}\mathbf p + \mathbf t_{\text{small}}/s$ — the linear part is unchanged and the translation scales by $1/s$. That is exactly `H[0,2] /= scale; H[1,2] /= scale`. (The accurate path applies the equivalent full conjugation $H_{\text{proc}} = D^{-1} H_{\text{small}} D$ with $D = \operatorname{diag}(s,s,1)$.)

### B.3 Optical flow: tracking a point from one frame to the next

**Optical flow** estimates, for a point in frame $t-1$, where it went in frame $t$. SiamRAM uses the **Lucas–Kanade** method. Here is the full idea — it is beautiful and only needs a derivative and a least‑squares solve.

**Brightness constancy.** Assume a small image patch keeps the same brightness as it moves by a small displacement $(\Delta x, \Delta y)$ over one frame:

$$
I_t(x + \Delta x,\ y + \Delta y) = I_{t-1}(x, y).
$$

Taylor‑expand the left side to first order (doc 01 §4.1) about $(x,y,t)$:

$$
I_{t-1}(x,y) + I_x\,\Delta x + I_y\,\Delta y + I_t' \approx I_{t-1}(x,y),
$$

where $I_x = \partial I/\partial x$, $I_y = \partial I/\partial y$ are the **spatial image gradients** (how brightness changes across pixels) and $I_t' = I_t - I_{t-1}$ is the **temporal change** (how the pixel's brightness changed between frames). Cancelling $I_{t-1}(x,y)$:

$$
\boxed{\ I_x\,\Delta x + I_y\,\Delta y + I_t' = 0.\ }
$$

This is the **optical‑flow constraint equation** — one equation, two unknowns $(\Delta x, \Delta y)$. A single pixel is not enough to solve it (this is the famous **aperture problem**: looking through a tiny hole, you can only sense motion *across* an edge, not *along* it).

**Lucas–Kanade's fix.** Assume the displacement is *the same* for all pixels in a small window $\mathcal W$ around the point. Then every pixel $i\in\mathcal W$ gives one equation, and we have an overdetermined linear system:

$$
\underbrace{\begin{bmatrix} I_x^{(1)} & I_y^{(1)} \\ \vdots & \vdots \\ I_x^{(n)} & I_y^{(n)}\end{bmatrix}}_{A}\begin{bmatrix}\Delta x \\ \Delta y\end{bmatrix} = \underbrace{\begin{bmatrix} -I_t'^{(1)} \\ \vdots \\ -I_t'^{(n)}\end{bmatrix}}_{\mathbf b}.
$$

Solve it in the **least‑squares** sense (the displacement that best satisfies all the equations) via the *normal equations*:

$$
\begin{bmatrix}\Delta x \\ \Delta y\end{bmatrix} = (A^{\!\top}A)^{-1}A^{\!\top}\mathbf b.
$$

Here $A^\top A$ is the $2\times2$ "structure tensor" $\begin{bmatrix}\sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2\end{bmatrix}$; it is invertible exactly when the window has gradients in *two* directions (a corner, not a flat patch or a single edge) — which is why good points to track are **corners** (next paragraph).

**Pyramids and the window.** A first‑order Taylor expansion is only valid for *small* displacements, but cameras can move fast. The fix is a **pyramid**: build downscaled copies of both frames, estimate the (small, in those coarse pixels) flow at the coarsest level, upscale that estimate, refine at the next level, and so on. This is `maxLevel` in the code; the window size is `winSize`. Iterations stop on a small change or a max count (`TERM_CRITERIA`).

**Which points?**
- The **classic** path tracks a fixed regular **grid** of points (e.g. every $50$ px), dropping those inside the padded target box. Simple and fast.
- The **accurate** path first runs `goodFeaturesToTrack` (a corner detector, Shi–Tomasi: it scores candidate windows by the smaller eigenvalue of $A^\top A$ — large means "trackable corner") to find up to `homo_max_corners` strong, well‑conditioned points, masked away from the target. Tracking corners (where the structure tensor is well‑conditioned) gives far more reliable flow than arbitrary points.

**Forward–backward check (accurate path).** A tracked point can drift or jump. The accurate path tracks each point forward ($t-1\to t$) *and then back* ($t\to t-1$); if the round trip doesn't return near the start ($\|\mathbf p_{\text{back}} - \mathbf p\| > 1.5$ px), the point is discarded. This cheap consistency test removes most bad correspondences before fitting.

```
   forward–backward consistency:
       p ──flow t−1→t──►  p'
       p ◄──flow t→t−1──  p'        keep iff ‖p_back − p‖ ≤ 1.5 px
       (a reliable track returns to where it started)
```

### B.4 RANSAC: fitting a transform when some points lie

Even after the FB check, some correspondences are wrong (a background point that happens to sit on a *moving* object, repetitive texture, etc.). A plain least‑squares fit would be dragged off by these **outliers**. **RANSAC** (RANdom SAmple Consensus) is the standard robust remedy:

1. Randomly pick the *minimal* number of correspondences needed to define the transform (e.g. $4$ point‑pairs for a homography).
2. Fit a candidate transform to just those.
3. Count how many *all* correspondences agree with it — the **inliers** — where "agree" means the point maps to within a reprojection threshold (e.g. $2.5$–$3$ px).
4. Repeat for many random samples; keep the candidate with the most inliers, then refit using all its inliers.

RANSAC's power: as long as *some* random sample happens to contain only true inliers, the correct transform will win the vote, no matter how many outliers are present. The classic path uses `cv2.estimateAffinePartial2D` with `RANSAC` (fits the 4‑DOF similarity transform). The accurate path uses `cv2.findHomography` with `USAC_MAGSAC` (MAGSAC++ is a modern RANSAC variant that avoids a hard inlier threshold by marginalizing over it — more robust, slightly slower) to fit the full 8‑DOF homography.

```
   data with outliers (×):              RANSAC picks the line/transform
        •  •                            most points agree with, ignoring ×:
      •      • ×                              •  •
    •    ×      •                           •──────•───•───•      ← consensus
   •        •     •   ×                    •     ×       •
        (× = outliers, • = inliers)            (× = outliers, rejected)
```

### B.5 The reliability flag

The estimate is not trusted blindly. RANSAC reports the inlier set; SiamRAM computes the **inlier ratio**

$$
\text{inlier\_ratio} = \frac{\#\,\text{inliers}}{\#\,\text{tracked points}},
\qquad
\text{reliable} = \big[\text{inlier\_ratio} \ge \text{homo\_inlier\_threshold}\big].
$$

A high ratio means *many* background points agree on a single global motion — strong evidence the estimate is real. A low ratio means the points disagree (blur, low texture, too many independent movers, a scene cut), so the transform may be garbage and is flagged **unreliable**. Several downstream consumers (the EKF warp, the GMC prior, the spike test we are skipping) *require* `reliable` before they will use $H$. On the first frame there is no previous frame, so the estimate is simply `(None, reliable=False)` and everyone falls back to "no camera motion." Handling that "no estimate" case gracefully — rather than forcing a bad warp — is a recurring safety principle.

---

## Part C — The camera‑aware EKF (`BBoxEKF`)

We now have the homography $H$ and its reliability. Time to fold it into a Kalman filter that tracks the target's center and velocity. This is `models/siamram/motion.py`, and it is a textbook EKF (doc 04 §7) with one nonlinear twist: the prediction warps the center through $H$.

### C.1 State, measurement, and the noise matrices

The state is the 2‑D constant‑velocity state of doc 04 §6:

$$
\mathbf{x} = \begin{bmatrix} c_x \\ c_y \\ v_x \\ v_y\end{bmatrix}\in\mathbb{R}^4.
$$

**Why center‑only, not the box corners or size?** Because velocity should be *immune to box reshaping*. At occlusion onset the matcher's box often shrinks; if velocity were derived from corners, that shrink would inject a fake velocity. Tracking only the center decouples motion from size. The width/height are carried separately as a slowly smoothed pair `(_bw, _bh)` (an EMA, §C.5), used only to turn the center back into a box on output.

The measurement is the center, so the (linear!) measurement matrix is

$$
H_{\text{meas}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\end{bmatrix},
\qquad H_{\text{meas}}\,\mathbf{x} = (c_x, c_y).
$$

(We write $H_{\text{meas}}$ to avoid clashing with the *homography* $H$; in code it is `_H_mat`.) Initial covariances (diagonal, doc 01 §2.4):

$$
P_0 = \operatorname{diag}(10, 10, 100, 100),\quad
Q = \operatorname{diag}(q, q, 4q, 4q),\quad
R = m\,I_2,
$$

with $q=$ `process_noise` and $m=$ `meas_noise`. Three design choices to read here:
- **Velocity starts far more uncertain than position** ($100$ vs $10$ in $P_0$): we know roughly where the target is at init, but nothing about its velocity.
- **Velocity gets $4\times$ the process noise of position** ($4q$ vs $q$ in $Q$): the constant‑velocity assumption is violated mainly through *velocity* changing (the target accelerates/turns), so we let the filter distrust its own velocity more — exactly the $Q$‑tuning logic of doc 04 §5.1.
- **$R = mI_2$** treats the two measured coordinates as equally, independently noisy.

### C.2 The nonlinear prediction function

One frame's prediction does two things, in order: (1) warp the *current* center by the camera homography (where would a point sitting at the target go, if it only moved with the camera?), then (2) add the target's own velocity. With $(c_x, c_y)$ the current center, the camera‑warped center is the perspective map of Part A.3:

$$
g_x(c_x, c_y) = \frac{h_{00}c_x + h_{01}c_y + h_{02}}{D},\qquad
g_y(c_x, c_y) = \frac{h_{10}c_x + h_{11}c_y + h_{12}}{D},\qquad
D = h_{20}c_x + h_{21}c_y + h_{22},
$$

(when $H$ is reliable; otherwise $g_x = c_x,\ g_y = c_y$, i.e. no camera warp). Then the full predicted state is

$$
f(\mathbf x) = \begin{bmatrix} g_x(c_x,c_y) + v_x \\ g_y(c_x,c_y) + v_y \\ v_x \\ v_y\end{bmatrix}.
$$

This is the exact $f$ the EKF uses to move the **mean**: `x[0] = g_x + x[2]`, `x[1] = g_y + x[3]`, velocities unchanged. It is nonlinear precisely because $g_x, g_y$ contain the division by $D$. The decomposition it encodes is the whole reason camera compensation works:

$$
\text{apparent image motion} \;=\; \underbrace{\text{camera motion}}_{\text{the }g\text{ warp}} \;+\; \underbrace{\text{the target's own motion}}_{\text{the }+v\text{ term}}.
$$

Without the warp, a camera pan and a target walk are indistinguishable to the filter; with it, they are separated.

### C.3 The Jacobian — derived line by line

To propagate the **covariance** ($P^- = F P^+ F^\top + Q$, doc 04 §7.2) the EKF needs $F = \partial f/\partial\mathbf x$, the $4\times4$ Jacobian of $f$ at the current state (doc 01 §4.2). Let's compute all sixteen entries; only four are nontrivial.

The velocity rows are trivial: $v_x, v_y$ don't change, so rows 3 and 4 of $F$ are $(0,0,1,0)$ and $(0,0,0,1)$. The center's dependence on velocity is also trivial: $\partial(g_x + v_x)/\partial v_x = 1$, $\partial(g_y+v_y)/\partial v_y = 1$, giving $F_{0,2}=1$ and $F_{1,3}=1$. (These are the constant‑velocity couplings — the same `F[0,2]=1; F[1,3]=1` you saw in doc 04 §6.)

The interesting part is how the *warped center* depends on the *current center* — the four partial derivatives

$$
\frac{\partial g_x}{\partial c_x},\qquad \frac{\partial g_x}{\partial c_y},\qquad \frac{\partial g_y}{\partial c_x},\qquad \frac{\partial g_y}{\partial c_y}.
$$

Take $g_x = N_x/D$ with numerator $N_x = h_{00}c_x + h_{01}c_y + h_{02}$ and denominator $D = h_{20}c_x + h_{21}c_y + h_{22}$. Apply the **quotient rule** (doc 01 §4.3), $\frac{d}{dc}\frac{N}{D} = \frac{N'D - ND'}{D^2}$:

$$
\frac{\partial g_x}{\partial c_x} = \frac{\dfrac{\partial N_x}{\partial c_x}\,D - N_x\,\dfrac{\partial D}{\partial c_x}}{D^2}
= \frac{h_{00}\,D - N_x\,h_{20}}{D^2}.
$$

Now the simplifying trick the code uses: since $g_x = N_x/D$, we have $N_x = g_x D$, so

$$
\frac{\partial g_x}{\partial c_x} = \frac{h_{00}D - g_x D\, h_{20}}{D^2} = \frac{h_{00} - g_x\,h_{20}}{D}.
$$

That is the code's `A = (H[0,0] - new_cx_h * H[2,0]) / denom`, with `new_cx_h` $= g_x$ and `denom` $=D$ (plus the $10^{-8}$ guard, doc 01 §1.3). The other three are identical in form:

$$
\boxed{
\begin{aligned}
A = \frac{\partial g_x}{\partial c_x} = \frac{h_{00} - g_x h_{20}}{D}, &\qquad
B = \frac{\partial g_x}{\partial c_y} = \frac{h_{01} - g_x h_{21}}{D},\\[4pt]
C = \frac{\partial g_y}{\partial c_x} = \frac{h_{10} - g_y h_{20}}{D}, &\qquad
D_{\!J} = \frac{\partial g_y}{\partial c_y} = \frac{h_{11} - g_y h_{21}}{D}.
\end{aligned}}
$$

(We write $D_J$ for the fourth entry to avoid clashing with the denominator $D$; the code calls it `D`.) Assembling all sixteen entries:

$$
F = \begin{bmatrix}
A & B & 1 & 0 \\
C & D_{\!J} & 0 & 1 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

**Sanity check.** If there is no camera motion (or $H$ unreliable), $g_x = c_x$ and $g_y = c_y$, the denominator is $1$, $h_{20}=h_{21}=0$, $h_{00}=h_{11}=1$, $h_{01}=h_{10}=0$, so $A=1, B=0, C=0, D_J=1$ and

$$
F = \begin{bmatrix}1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1\end{bmatrix},
$$

which is exactly the plain constant‑velocity transition of doc 04 §6 ($F$ adds velocity to position). So the camera term is a clean *generalization*: turn the camera off and you recover the ordinary CV Kalman filter. With this $F$, the covariance prediction is the standard $P^- = F P^+ F^\top + Q$.

### C.4 The update step

The measurement is the matcher's box center (or, during recovery, a re‑detection). Since the measurement model is **linear** ($h(\mathbf x) = H_{\text{meas}}\mathbf x$), the update is the *ordinary* linear Kalman update of doc 04 §5, no Jacobian needed there. With measured center $\mathbf z = (c_x^{\text{meas}}, c_y^{\text{meas}})$:

$$
\begin{aligned}
\mathbf y &= \mathbf z - H_{\text{meas}}\,\hat{\mathbf x}^- && \text{(innovation)}\\
S &= H_{\text{meas}}\,P^-\,H_{\text{meas}}^{\!\top} + R\\
K &= P^-\,H_{\text{meas}}^{\!\top}\,S^{-1}\\
\hat{\mathbf x}^+ &= \hat{\mathbf x}^- + K\,\mathbf y\\
P^+ &= (I - K H_{\text{meas}})\,P^-
\end{aligned}
$$

Because of the off‑diagonal coupling in $P^-$ (built up by the predict step, doc 04 §5), this position‑only measurement also corrects the **velocity** estimate — that is how the EKF infers how fast the target is moving from a stream of position measurements. This is exactly the code: `innov = z - H_mat @ x; S = H_mat @ P @ H_mat.T + R; K = P @ H_mat.T @ inv(S); x = x + K@innov; P = (I - K@H_mat)@P`.

### C.5 Box size, output, reseed, nudge, and coasting

A few practical methods complete the filter:

- **Size EMA.** On each update the tracked width/height are smoothed toward the measured box (doc 01 §6): $\;\_bw \leftarrow 0.85\,\_bw + 0.15\,w_{\text{meas}}$ and likewise for height. So size changes gradually and a single bad box can't resize the output.
- **`get_bbox`** turns the state back into a box: from center $(c_x, c_y)$ and the smoothed $(\_bw, \_bh)$, the top‑left is $(c_x - \_bw/2,\ c_y - \_bh/2)$.
- **`get_velocity`** returns $(v_x, v_y)$; **`get_uncertainty`** returns $\sqrt{\overline{\operatorname{diag}(P_{[:2]})}}$, a scalar standard deviation of the position estimate (how unsure the filter is about where the target is).
- **`reseed(box, velocity)`** hard‑resets the state to a new position and velocity and *re‑inflates* the covariance ($P=\operatorname{diag}(25,25,40,40)$). This is used when an *external* event teleports the target legitimately — e.g. a successful re‑acquisition after occlusion (doc 08). Without reseeding, the filter would slowly drag itself toward the new location and report wrong velocities meanwhile; reseeding says "forget the old trajectory, start fresh here."
- **`nudge_position(box)`** moves the center *without* touching velocity — a gentle correction used during recovery when we want to bias the search center toward a plausible detection but not commit to it (doc 08).
- **Coasting during occlusion.** When the target is out of frame, there is no warp to apply and no measurement to update from, so the code simply inflates uncertainty: $P \leftarrow P + Q$ each frame (this is the "no measurement → coast, covariance grows" case of doc 04 §8). The filter keeps extrapolating with growing uncertainty until the target is re‑found. When in occlusion but *in* frame, it still does the normal `predict(H, reliable)` so it keeps following camera motion while it searches.

---

## Part D — Using and trusting the camera motion

The EKF *consumes* $H$ in its predict step, but $H$ also drives several outer decisions. Two are worth stating here (the search‑prior use is doc 06; loss‑cause classification is doc 08).

### D.1 Plausibility gates (`gmc_motion_is_valid`)

Even a "reliable" $H$ can be physically absurd (RANSAC can occasionally fit a large bogus rotation in a blurred, low‑texture scene). Before any *search‑prior* use, SiamRAM checks the decomposed statistics (A.4) against caps:

$$
\begin{aligned}
|dx|,\,|dy| &\le \text{max\_translation\_frac}\cdot \max(W, H) && (\text{e.g. } 25\%\ \text{of the long edge})\\
\text{min\_scale} \le \text{scale} &\le \text{max\_scale} && (\text{e.g. } [0.7,\ 1.4])\\
|\theta| &\le \text{max\_rotation\_deg} && (\text{e.g. } 25^\circ)\\
\text{max\_corner\_disp} &\le \text{max\_corner\_frac}\cdot\sqrt{W^2 + H^2} && (\text{fraction of the frame diagonal})
\end{aligned}
$$

If any is exceeded, the transform is rejected for that purpose ("a mathematically valid but physically unlikely transform"). This is a *belt‑and‑suspenders* design: `reliable` filters statistically bad fits; the plausibility gates filter statistically fine but physically implausible ones.

### D.2 Camera displacement magnitude

A scalar "how far did the camera move this frame" is obtained by warping the **frame center** $(W/2, H/2)$ through $H$ and measuring how far it moved (`_h_translation_magnitude`):

$$
\text{cam\_disp} = \big\|\,(g_x, g_y)\big|_{(W/2,\,H/2)} - (W/2,\,H/2)\,\big\|.
$$

This single number, accumulated into a short history, feeds the "heavy camera motion" detector (doc 06) and the loss‑cause vote at occlusion entry (doc 08). It is the same perspective formula as everywhere else, evaluated at the frame center.

---

## Recap

- **Homogeneous coordinates** let every camera transform be a $3\times3$ matrix $H$; the **perspective division** $(x',y') = \big(\tfrac{h_{00}x+h_{01}y+h_{02}}{D}, \tfrac{h_{10}x+h_{11}y+h_{12}}{D}\big)$ is the lone nonlinearity, and it is literally the EKF's predict math.
- $H$ decomposes into translation $(dx,dy)$, scale $\sqrt{a^2+b^2}$, rotation $\operatorname{atan2}(b,a)$, and a worst‑case corner displacement.
- SiamRAM estimates $H$ from **background** points (target masked) via **Lucas–Kanade optical flow** (brightness constancy + least squares over a window + pyramids, with a forward–backward consistency check on the accurate path) and a **RANSAC** robust fit (4‑DOF similarity for "classic", 8‑DOF MAGSAC homography for "accurate"), flagged **reliable** by the inlier ratio.
- The **`BBoxEKF`** is a constant‑velocity EKF over $(c_x,c_y,v_x,v_y)$ whose **prediction warps the center through $H$ then adds velocity**, separating camera motion from target motion. Its Jacobian's four nontrivial entries are $\tfrac{h_{0\cdot}-g_x h_{2\cdot}}{D}$ and $\tfrac{h_{1\cdot}-g_y h_{2\cdot}}{D}$, derived by the quotient rule; turning the camera off recovers the plain CV filter.
- The **update** is the ordinary linear Kalman update (measuring the center), which also corrects velocity through the covariance coupling; size is carried as a separate EMA; `reseed`/`nudge`/coasting handle teleports and occlusion.
- Outer code **trusts** $H$ only after a reliability flag *and* physical‑plausibility gates, and reads a scalar camera‑displacement from it.

Next: **`06_MOTION_PRIORS.md`**, where the camera motion and a second Kalman filter become *priors* that steer the visual matcher's search and reweight its response — the bridge from "we know where the target should be" to "so look there."
