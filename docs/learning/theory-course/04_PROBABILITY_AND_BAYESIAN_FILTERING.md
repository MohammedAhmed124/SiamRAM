# 04 — Probability, Bayesian Filtering, and the Kalman Filter

> Prerequisites: doc 01 (vectors, matrices, inverse, Gaussian, covariance, Bayes' rule). This document derives the **Kalman filter** from first principles and then the **Extended Kalman Filter (EKF)** that SiamRAM actually uses. We build it slowly: the estimation problem, the state‑space model, the recursive Bayesian idea, the 1‑D Kalman filter (with full algebra), the vector Kalman filter, and finally linearization for nonlinear models. Doc 05 then plugs SiamRAM's specific motion + camera model into these equations. Nothing here is specific to SiamRAM yet — it is the theory you must own first.

---

## 1. The problem: estimating a hidden state from noisy data

Imagine you want to know a target's true position and velocity at each frame. You cannot observe them directly. What you *can* observe is the matcher's reported box — a *noisy, indirect* measurement of position (and not of velocity at all). You also know something about *physics*: between two frames, a moving object roughly continues at its current velocity.

This is the universal **filtering** problem:

- There is a **hidden state** $\mathbf{x}_t$ (here: center position and velocity) that evolves over time.
- You receive **measurements** $\mathbf{z}_t$ that depend on the state but are corrupted by noise.
- You want the **best estimate** of $\mathbf{x}_t$ given *all* measurements so far, $\mathbf{z}_1, \dots, \mathbf{z}_t$ — computed **recursively** (update the estimate as each new measurement arrives, without re‑processing the whole history).

"Recursively" is essential for a real‑time tracker: at $30+$ fps you cannot reprocess thousands of past frames each step. You want to carry a compact summary of the past and fold in one new measurement per frame in constant time. The Kalman filter is the optimal way to do exactly that — *when* the model is linear and the noise is Gaussian.

---

## 2. The state‑space model

We formalize "physics" and "measurement" as two equations.

### 2.1 The process (motion) model

The state evolves by a linear map plus random disturbance:

$$
\mathbf{x}_t = F\,\mathbf{x}_{t-1} + \mathbf{w}_t,\qquad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q).
$$

- $\mathbf{x}_t\in\mathbb{R}^n$ is the state at time $t$.
- $F$ (the **state‑transition matrix**, $n\times n$) encodes the deterministic dynamics — e.g. "position next = position + velocity."
- $\mathbf{w}_t$ is **process noise**: a zero‑mean Gaussian capturing everything the linear model misses (the target accelerates, turns, etc.). Its covariance $Q$ says how much we *distrust* the motion model. Big $Q$ = "the target can deviate a lot from constant velocity."

### 2.2 The measurement model

The measurement is a linear function of the state plus noise:

$$
\mathbf{z}_t = H\,\mathbf{x}_t + \mathbf{v}_t,\qquad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R).
$$

- $\mathbf{z}_t\in\mathbb{R}^m$ is the measurement at time $t$.
- $H$ (the **measurement matrix**, $m\times n$) maps state to measurement — e.g. "we measure position but not velocity," so $H$ picks out the position coordinates.
- $\mathbf{v}_t$ is **measurement noise** with covariance $R$. Big $R$ = "the measurement is unreliable."

The two noises are assumed independent and white (uncorrelated across time). This pair of linear‑Gaussian equations is the **linear‑Gaussian state‑space model**, and it is the exact setting in which the Kalman filter is optimal.

```
  hidden:   x_0 ──F──► x_1 ──F──► x_2 ──F──► x_3   (+ process noise w each step)
                       │           │           │
                       H           H           H   (+ measurement noise v each step)
                       ▼           ▼           ▼
  observed:           z_1         z_2         z_3
```

### 2.3 The belief: a Gaussian over the state

Our knowledge of the state at any time is itself a **Gaussian** (doc 01 §5.4):

$$
\mathbf{x}_t \mid \mathbf{z}_{1:t} \sim \mathcal{N}(\hat{\mathbf{x}}_t,\ P_t),
$$

with mean $\hat{\mathbf{x}}_t$ (our point estimate) and covariance $P_t$ (our uncertainty). The Kalman filter is just the rule for updating this $(\hat{\mathbf{x}}, P)$ pair as time advances and measurements arrive. Because the model is linear‑Gaussian, the belief *stays* Gaussian forever (doc 01 §5.3, fact 2), so $(\hat{\mathbf{x}}, P)$ is a complete summary — that's why the filter is recursive and cheap.

---

## 3. The recursive Bayesian recipe (two half‑steps)

Each time step splits into **predict** (advance the belief through the motion model) and **update** (fold in the new measurement via Bayes' rule). This predict–update rhythm is the skeleton of every filter.

**Predict (a.k.a. time update).** Push the previous belief through the motion model. Intuitively, applying $F$ shifts the mean and the random kick $\mathbf{w}$ *spreads* the uncertainty:

$$
\hat{\mathbf{x}}_t^- = F\,\hat{\mathbf{x}}_{t-1}^+,
\qquad
P_t^- = F\,P_{t-1}^+\,F^{\!\top} + Q.
$$

The superscript $-$ means "before this frame's measurement" (the *prior*), $+$ means "after" (the *posterior*). Notice prediction *increases* uncertainty (it adds $Q$ and inflates $P$ by $F\cdots F^\top$): with no new data, we become less sure.

**Update (a.k.a. measurement update).** Use Bayes' rule (doc 01 §5.6) to combine the prior $\mathcal{N}(\hat{\mathbf{x}}_t^-, P_t^-)$ with the measurement likelihood $\mathcal{N}(\mathbf z_t;\, H\mathbf x, R)$. Because both are Gaussian, the posterior is Gaussian, and prediction *decreases* uncertainty (data makes us surer). The exact formulas are the Kalman update, derived next.

Let's first do the whole thing in **one dimension** so the algebra is fully visible, then state the vector version.

---

## 4. The Kalman filter in 1‑D (full derivation)

Strip everything to scalars to see the mechanism. State $x$ (say, position), motion $x_t = x_{t-1} + w$ with $w\sim\mathcal N(0, q)$, measurement $z = x + v$ with $v\sim\mathcal N(0, r)$.

### 4.1 Predict

If before the step we believe $x\sim\mathcal N(\hat x^+, p^+)$, then after applying $x_t = x_{t-1}+w$ (here $F=1$):

$$
\hat x^- = \hat x^+, \qquad p^- = p^+ + q.
$$

Mean unchanged (constant model), variance grows by the process noise $q$.

### 4.2 Update — combining two Gaussians

Now a measurement $z$ arrives. We have two pieces of information about $x$:

- the **prior** $\mathcal N(\hat x^-,\, p^-)$ from prediction;
- the **likelihood** from the measurement: given $x$, $z\sim\mathcal N(x,\, r)$, i.e. as a function of $x$ it is $\mathcal N(z,\, r)$.

By Bayes' rule the posterior is proportional to the **product** of the two Gaussians. Multiply the two exponentials and complete the square (this is the one derivation worth seeing once). The product of $\mathcal N(\mu_1, \sigma_1^2)$ and $\mathcal N(\mu_2, \sigma_2^2)$ (as functions of $x$) is proportional to a Gaussian with

$$
\frac{1}{\sigma^2} = \frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2},
\qquad
\mu = \sigma^2\!\left(\frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}\right).
$$

This is the famous rule: **precisions (inverse variances) add**, and the combined mean is the precision‑weighted average of the two means. Plug in $\mu_1=\hat x^-,\ \sigma_1^2=p^-$ and $\mu_2=z,\ \sigma_2^2=r$:

$$
\frac{1}{p^+} = \frac{1}{p^-} + \frac{1}{r},
\qquad
\hat x^+ = p^+\!\left(\frac{\hat x^-}{p^-} + \frac{z}{r}\right).
$$

### 4.3 Rewriting it as "predict + gain × surprise"

The form above is correct but opaque. Algebra turns it into the canonical Kalman form. Define the **Kalman gain**

$$
K = \frac{p^-}{p^- + r}.
$$

Then (you can verify by substitution) the update becomes

$$
\boxed{\ \hat x^+ = \hat x^- + K\,(z - \hat x^-),\qquad p^+ = (1 - K)\,p^-.\ }
$$

Read this carefully — it is the soul of the Kalman filter:

- $z - \hat x^-$ is the **innovation** (a.k.a. *residual* or *surprise*): how much the measurement disagrees with the prediction.
- $K$ decides **how much of the surprise to believe**. Look at the two extremes:
  - If the measurement is very reliable ($r\to0$), then $K\to1$: trust the measurement fully, $\hat x^+ \to z$.
  - If the measurement is useless ($r\to\infty$), then $K\to0$: ignore it, $\hat x^+ \to \hat x^-$.
  - $K$ also grows when the prior is uncertain ($p^-$ large): if we don't trust our own prediction, we lean on the data.
- The variance always **shrinks** on update ($p^+ = (1-K)p^- \le p^-$): a measurement, however noisy, can only reduce uncertainty.

So the Kalman filter is a *self‑tuning weighted average* between prediction and measurement, where the weight is set, optimally, by their relative uncertainties. That sentence is the entire intuition; everything else is bookkeeping to make it work for vectors.

```
   prior N(x̂⁻, p⁻)        measurement N(z, r)        posterior N(x̂⁺, p⁺)
        _-•-_                  _--•--_                     _•_
      _/  ↑  \_              _/   ↑   \_         ×   ──►   /│\     narrower & between
     /   x̂⁻   \             /     z    \                 / │ \    (precision‑weighted)
    ────────────           ─────────────                ───┼───
                                                          x̂⁺ = x̂⁻ + K(z − x̂⁻)
```

---

## 5. The vector Kalman filter

The 1‑D logic lifts verbatim to vectors; scalars become matrices, division becomes matrix inverse. With state $\mathbf x\in\mathbb R^n$, measurement $\mathbf z\in\mathbb R^m$, and the model of §2:

**Predict:**

$$
\hat{\mathbf{x}}_t^- = F\,\hat{\mathbf{x}}_{t-1}^+,
\qquad
P_t^- = F\,P_{t-1}^+\,F^{\!\top} + Q.
$$

**Update:**

$$
\begin{aligned}
\mathbf{y}_t &= \mathbf{z}_t - H\,\hat{\mathbf{x}}_t^- && \text{(innovation: measurement − predicted measurement)}\\
S_t &= H\,P_t^-\,H^{\!\top} + R && \text{(innovation covariance: how surprising can the surprise be?)}\\
K_t &= P_t^-\,H^{\!\top}\,S_t^{-1} && \text{(Kalman gain: optimal trust in the surprise)}\\
\hat{\mathbf{x}}_t^+ &= \hat{\mathbf{x}}_t^- + K_t\,\mathbf{y}_t && \text{(corrected state estimate)}\\
P_t^+ &= (I - K_t H)\,P_t^- && \text{(corrected, shrunken covariance)}
\end{aligned}
$$

Match each line to the 1‑D version: $\mathbf y_t$ is $z - \hat x^-$; $S_t$ generalizes $p^- + r$ (the total uncertainty in the innovation, mapped through $H$ plus measurement noise $R$); $K_t = P^- H^\top S^{-1}$ generalizes $\tfrac{p^-}{p^-+r}$ (note the $S^{-1}$ — that matrix inverse is the vector analogue of dividing by $p^-+r$); $\hat{\mathbf x}^+ = \hat{\mathbf x}^- + K\mathbf y$ is identical in form; $P^+ = (I-KH)P^-$ generalizes $(1-K)p^-$.

Why $P^- H^\top$ in the gain? $P^- H^\top$ is the **cross‑covariance between state error and innovation** — it says *which* state coordinates the innovation is informative about. This is how a measurement of *position only* still corrects the *velocity* estimate: if position and velocity are correlated in $P^-$ (and after a predict step they are, because velocity drove position), then a position surprise updates velocity too, through the off‑diagonal of $P^- H^\top$. That coupling is the magic that lets a Kalman filter infer an unmeasured quantity (velocity) from a measured one (position).

### 5.1 Tuning $Q$ and $R$

The filter's behavior is governed entirely by the noise covariances:

- **$R$ large** (untrusted measurements) → small gain → the filter smooths heavily, leaning on its motion model. Good against noisy detections; bad if the model is wrong.
- **$Q$ large** (untrusted model) → $P^-$ inflates → large gain → the filter follows measurements closely. Good for erratic motion; bad against noisy detections.

The ratio $Q/R$ is the single most important tuning quantity: it sets where the filter sits on the smoothing‑vs‑responsiveness spectrum. SiamRAM picks these by hand for its tracker (doc 05) — e.g. velocity gets *more* process noise than position, encoding "the object's speed can change more abruptly than the model assumes."

---

## 6. The constant‑velocity model (what SiamRAM uses)

The motion model SiamRAM's filters use is **constant velocity (CV)**: "between frames, position advances by velocity, and velocity stays the same (up to noise)." For a 1‑D position $c$ with velocity $v$, the state is $\mathbf x = (c, v)$ and one frame ($\Delta t = 1$) gives

$$
c_t = c_{t-1} + v_{t-1},\qquad v_t = v_{t-1},
$$

which in matrix form is $\mathbf x_t = F\mathbf x_{t-1}$ with

$$
F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}.
$$

Check: $F\begin{bmatrix} c \\ v\end{bmatrix} = \begin{bmatrix} c + v \\ v\end{bmatrix}$. ✓ The "$1$" in the top‑right couples velocity into next position; that off‑diagonal is what makes velocity *observable* from position measurements (§5).

For 2‑D tracking the state is $(c_x, c_y, v_x, v_y)$ and $F$ is the $4\times4$ block version: each position gets its velocity added, each velocity persists. We measure position only, so

$$
H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0\end{bmatrix},\qquad
H\,(c_x, c_y, v_x, v_y)^{\!\top} = (c_x, c_y)^{\!\top}.
$$

This is *exactly* the structure of SiamRAM's `BBoxEKF` (doc 05): a 4‑D state, a $2\times4$ measurement matrix that reads off the center, and constant‑velocity $F$ — except that the prediction also folds in camera motion, which makes it *nonlinear* and forces the "Extended" upgrade we cover now.

---

## 7. When the model is nonlinear: the Extended Kalman Filter

The Kalman filter assumed *linear* $F$ and $H$. But SiamRAM's prediction warps the target's position through a **camera homography** (doc 05), and a homography involves a *division* (perspective), which is **not linear**. The same happens in countless applications (radar with range/bearing, GPS, robotics). The fix is the **Extended Kalman Filter (EKF)**: keep the exact nonlinear functions for propagating the *mean*, but **linearize** them (doc 01 §4.2, the Jacobian) for propagating the *covariance*.

### 7.1 Nonlinear state‑space model

$$
\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{w}_t,
\qquad
\mathbf{z}_t = h(\mathbf{x}_t) + \mathbf{v}_t,
$$

with $f$ and $h$ now general (possibly nonlinear) functions.

### 7.2 The trick: linearize about the current estimate

Near our current estimate $\hat{\mathbf x}$, replace $f$ by its tangent (first‑order Taylor expansion, doc 01 §4.2):

$$
f(\mathbf x) \approx f(\hat{\mathbf x}) + F\,(\mathbf x - \hat{\mathbf x}),\qquad
F = \left.\frac{\partial f}{\partial \mathbf x}\right|_{\hat{\mathbf x}} \quad(\text{the Jacobian of } f).
$$

Likewise $H = \partial h/\partial\mathbf x$ at the predicted state. The Jacobian $F$ is the local linear stand‑in for the curved map: it tells the covariance how the uncertainty ellipse gets rotated/stretched by the nonlinearity. The EKF then runs the linear Kalman equations with these *locally computed* $F$ and $H$:

**Predict (EKF):**

$$
\hat{\mathbf{x}}_t^- = f(\hat{\mathbf{x}}_{t-1}^+),
\qquad
P_t^- = F\,P_{t-1}^+\,F^{\!\top} + Q.
$$

Note the **mean uses the exact $f$** (no approximation), while the **covariance uses the Jacobian $F$** (the linear approximation). That split is the entire idea of the EKF.

**Update (EKF):** identical to §5 but with the innovation using the exact $h$,

$$
\mathbf{y}_t = \mathbf{z}_t - h(\hat{\mathbf{x}}_t^-),\quad
S_t = H P_t^- H^{\!\top} + R,\quad
K_t = P_t^- H^{\!\top} S_t^{-1},\quad
\hat{\mathbf{x}}_t^+ = \hat{\mathbf{x}}_t^- + K_t\mathbf{y}_t,\quad
P_t^+ = (I - K_t H)P_t^-.
$$

In SiamRAM the measurement model is *already linear* (we measure the center, $h(\mathbf x)=H\mathbf x$ with the constant $2\times4$ matrix above), so only the **prediction** is nonlinear (because of the camera warp). That means SiamRAM only needs the Jacobian of the *prediction* function — and doc 05 derives exactly that Jacobian, term by term, including the quotient‑rule step (doc 01 §4.3) for the perspective division.

### 7.3 Caveats of the EKF

The linearization is an approximation, exact only to first order. It can be inaccurate if the function is strongly curved over the uncertainty region, and the filter can diverge if initialized badly or if $Q,R$ are mis‑set. For SiamRAM's mild per‑frame camera motions the linearization is excellent (the warp is nearly affine over one box), and the engineering guards (covariance resets, "reseed," plausibility gates in later docs) keep it stable. These are normal, expected parts of using an EKF in practice.

---

## 8. The filtering algorithm in one box

For reference, the full recursive loop you now understand:

```
 initialize  x̂₀⁺ , P₀⁺
 for each frame t = 1, 2, 3, ...:
     ── PREDICT ──────────────────────────────────────────
     x̂⁻ = f(x̂⁺)                 (exact nonlinear move; = F x̂⁺ if linear)
     F  = ∂f/∂x |_{x̂⁺}          (Jacobian; = F itself if linear)
     P⁻ = F P⁺ Fᵀ + Q           (uncertainty grows)
     ── UPDATE (only if a measurement z_t arrived) ───────
     y  = z_t − h(x̂⁻)           (innovation / surprise)
     S  = H P⁻ Hᵀ + R           (innovation covariance)
     K  = P⁻ Hᵀ S⁻¹             (Kalman gain)
     x̂⁺ = x̂⁻ + K y              (corrected estimate)
     P⁺ = (I − K H) P⁻          (uncertainty shrinks)
     ── (if no measurement: x̂⁺ = x̂⁻ , P⁺ = P⁻ — the filter "coasts") ──
```

The "coast" case in the last line is exactly what an occluded tracker does (doc 08): when the target is hidden there is no trustworthy measurement, so the filter keeps predicting (and its uncertainty keeps growing), extrapolating where the target probably went until it is re‑found.

---

## 9. Recap

- Filtering = recursively estimating a hidden, evolving state from noisy measurements, carrying a Gaussian belief $(\hat{\mathbf x}, P)$.
- A **state‑space model** has a linear **process** ($\mathbf x_t = F\mathbf x_{t-1} + \mathbf w$) and a linear **measurement** ($\mathbf z_t = H\mathbf x_t + \mathbf v$), with Gaussian noises $Q, R$.
- The **Kalman filter** is a predict (mean shifts, covariance grows by $Q$) / update (Bayes' product of Gaussians) loop. Its heart is $\hat{\mathbf x}^+ = \hat{\mathbf x}^- + K(\mathbf z - H\hat{\mathbf x}^-)$, a precision‑weighted blend of prediction and measurement.
- The **Kalman gain** $K = P^- H^\top S^{-1}$ optimally weights measurement against model; off‑diagonals of $P$ let a position measurement correct an unmeasured velocity.
- $Q/R$ tunes smoothing vs. responsiveness.
- The **constant‑velocity** model ($F$ adds velocity to position, $H$ reads position) is what SiamRAM tracks.
- The **EKF** handles nonlinear $f$ (SiamRAM's camera warp) by using the exact $f$ for the mean and its **Jacobian** for the covariance.

You now have everything to read the centerpiece. Next: **`05_HOMOGRAPHY_AND_EKF.md`** — what a homography is, how it's estimated from optical flow and RANSAC, and how SiamRAM's `BBoxEKF` folds it into the prediction step, Jacobian and all.
