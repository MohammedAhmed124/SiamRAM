# 01 — Mathematical Foundations

> This document builds, from scratch, every piece of mathematics the rest of the series uses. If you already know linear algebra and probability, skim it and move on; if you don't, read it carefully — later documents lean on every section here and assume no further background. Each idea is introduced, defined, given intuition, and connected to where SiamRAM uses it.

---

## 1. Vectors

A **vector** is an ordered list of numbers. We write a vector of $n$ numbers as a column:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n .
$$

$\mathbb{R}^n$ means "the set of all lists of $n$ real numbers." In SiamRAM, vectors are everywhere: a target's position is a vector $(c_x, c_y)$; the EKF's state is a vector $(c_x, c_y, v_x, v_y)$; an appearance "descriptor" is a vector of 512 numbers.

**Geometric picture.** A vector in $\mathbb{R}^2$ is an arrow from the origin to the point $(x_1, x_2)$. Addition is tip‑to‑tail; scaling by $\alpha$ stretches the arrow.

```
   x2
    ^
    |        • (3,2) = x
  2 +       ↗
    |     ↗
    |   ↗
    | ↗
    +--+--+--+---> x1
    0  1  2  3
```

### 1.1 Length (norm)

The **Euclidean norm** (length) of a vector is

$$
\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{\textstyle\sum_{i=1}^n x_i^2}.
$$

This is just the Pythagorean theorem in $n$ dimensions. The distance between two points $\mathbf{a}$ and $\mathbf{b}$ is the length of their difference, $\|\mathbf{a} - \mathbf{b}\|$. In code this is the `hypot` function for 2‑D: $\sqrt{\Delta x^2 + \Delta y^2}$. SiamRAM measures, e.g., how far a box center moved between frames as exactly this.

A **unit vector** has length $1$. Any nonzero vector can be **normalized** to unit length by dividing by its norm: $\hat{\mathbf{x}} = \mathbf{x}/\|\mathbf{x}\|$.

### 1.2 The dot product

The **dot product** (inner product) of two vectors of the same size is

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^{\!\top}\mathbf{b} = \sum_{i=1}^n a_i b_i .
$$

It produces a single number. Two facts make it indispensable:

1. **It relates to length:** $\mathbf{a}\cdot\mathbf{a} = \|\mathbf{a}\|^2$.
2. **It relates to angle:** if $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$, then

$$
\boxed{\;\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta\;}
$$

So the dot product is large and positive when two vectors point the same way, zero when they are perpendicular, and negative when they point oppositely. This single identity powers two SiamRAM ideas: **cosine similarity** of appearance descriptors (§3) and **direction consistency** of motion (doc 07).

### 1.3 Cosine similarity

Rearranging the boxed identity, the cosine of the angle between two vectors is

$$
\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|} \in [-1, 1].
$$

This is **cosine similarity**. It ignores the *magnitudes* of the vectors and compares only their *directions*. Why is that the right tool for appearance?

A neural network turns an image crop into a feature vector (a "descriptor"). Two crops of the *same* object produce descriptors pointing in nearly the same direction ($\cos\theta \approx 1$), even if overall brightness scales them differently; two crops of *different* objects point in different directions ($\cos\theta$ closer to $0$). Cosine similarity is brightness‑/scale‑robust, which is exactly what you want for "do these two crops show the same thing?" SiamRAM's `_cos_sim` computes precisely

$$
\operatorname{cossim}(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\| + \varepsilon},
$$

with a tiny $\varepsilon$ (e.g. $10^{-8}$) added to the denominator to avoid division by zero when a vector is all zeros. *(That $\varepsilon$ trick appears everywhere in numerical code; whenever you see "$+\,1\mathrm{e}{-8}$" in a denominator, it is guarding against divide‑by‑zero.)*

---

## 2. Matrices and linear maps

A **matrix** is a rectangular grid of numbers. An $m\times n$ matrix has $m$ rows and $n$ columns:

$$
A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn}\end{bmatrix}.
$$

### 2.1 Matrix–vector multiplication

A matrix multiplies a vector to produce another vector. If $A$ is $m\times n$ and $\mathbf{x}\in\mathbb{R}^n$, then $\mathbf{y} = A\mathbf{x}\in\mathbb{R}^m$ has entries

$$
y_i = \sum_{j=1}^n a_{ij}\,x_j .
$$

The key interpretation: **a matrix is a linear transformation.** $A\mathbf{x}$ takes the vector $\mathbf{x}$ and rotates/scales/shears/projects it. "Linear" means it respects addition and scaling: $A(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha A\mathbf{x} + \beta A\mathbf{y}$. Lines through the origin stay lines through the origin; the grid stays evenly spaced. This is why so much of tracking is matrix algebra: camera rotation, the constant‑velocity motion model, and the Kalman filter's state transition are all linear maps.

### 2.2 Matrix–matrix multiplication

If $A$ is $m\times k$ and $B$ is $k\times n$, their product $C = AB$ is $m\times n$ with

$$
c_{ij} = \sum_{\ell=1}^k a_{i\ell}\,b_{\ell j}.
$$

Composing transformations = multiplying matrices: applying $B$ then $A$ to a vector is $(AB)\mathbf{x}$. **Order matters:** in general $AB \neq BA$.

### 2.3 Identity, transpose, inverse

- The **identity matrix** $I_n$ has $1$s on the diagonal, $0$s elsewhere; $I\mathbf{x} = \mathbf{x}$ (it does nothing).
- The **transpose** $A^{\!\top}$ flips rows and columns: $(A^{\!\top})_{ij} = a_{ji}$.
- The **inverse** $A^{-1}$ (only for square matrices, and only when it exists) "undoes" $A$: $A^{-1}A = AA^{-1} = I$. So if $\mathbf y=A\mathbf x$ then $\mathbf x=A^{-1}\mathbf y$. Inverting is how the Kalman filter "solves" for the optimal correction (doc 04). For a $2\times2$ matrix there is a closed form; for larger ones the computer uses standard numerical routines (`np.linalg.inv`).

### 2.4 The diagonal and trace

A **diagonal matrix** has nonzero entries only on the diagonal; `np.diag([a,b,c])` builds one. Multiplying by a diagonal matrix just scales each coordinate independently. SiamRAM initializes covariance matrices as diagonals (doc 04/05), meaning "each state variable has its own uncertainty and they start uncorrelated."

### 2.5 Symmetric and positive‑(semi)definite matrices

A matrix is **symmetric** if $A = A^{\!\top}$. A symmetric matrix is **positive semidefinite (PSD)** if $\mathbf{x}^{\!\top}A\mathbf{x}\ge 0$ for all $\mathbf{x}$, and **positive definite** if the inequality is strict for $\mathbf{x}\neq 0$. Covariance matrices (§5.5) are always symmetric PSD. Intuition: $\mathbf{x}^{\!\top}A\mathbf{x}$ is a "generalized squared length," so a covariance can never assign negative variance to any direction.

---

## 3. Functions you will meet constantly

### 3.1 The sigmoid (logistic) function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

It squashes any real number into the open interval $(0,1)$. It is increasing, with $\sigma(0)=\tfrac12$, $\sigma(-\infty)=0$, $\sigma(+\infty)=1$. SiamABC's classification head outputs raw scores ("logits") in $(-\infty,\infty)$; applying $\sigma$ turns each into a probability‑like confidence in $(0,1)$ that "the target is at this location."

```
 σ(z)
 1.0 |                       _____------
     |                 __--‾‾
 0.5 |............__--‾.............  (σ(0)=0.5)
     |       _--‾
     |  __--‾
 0.0 |‾‾____________________________
     +----+----+----+----+----+----+ z
      -6   -4   -2    0    2    4   6
```

### 3.2 argmax

Given a list (or grid) of numbers, $\operatorname{argmax}$ returns the **index of the largest one** (as opposed to $\max$, which returns the value). If $s = (0.1, 0.9, 0.3)$, then $\max s = 0.9$ but $\operatorname{argmax} s = 2$ (the second index, $1$‑based). SiamABC decodes a box by taking the $\operatorname{argmax}$ over a 2‑D grid of confidence scores: "the target is at the grid cell with the highest score."

For a 2‑D grid flattened to 1‑D index $k$, the row and column are recovered by integer division and remainder: $r = \lfloor k / W\rfloor$, $c = k \bmod W$, where $W$ is the grid width. (You will see exactly this in the decoder.)

### 3.3 Exponential and its uses

$e^{z}$ (with $e \approx 2.718$) grows fast for positive $z$ and decays toward $0$ for negative $z$, always staying positive. Two recurring uses:

- **Decay weights.** $e^{-\alpha t}$ for $\alpha>0$, $t\ge 0$ starts at $1$ ($t=0$) and decays to $0$. SiamRAM uses $e^{-\alpha\cdot\text{age}}$ to make old memory entries count less (doc 07), and $e^{-(\cdots)}$ to penalize sudden scale changes (doc 03).
- **Gaussian bumps.** $e^{-d^2/(2\sigma^2)}$ is a smooth bump that is $1$ at $d=0$ and falls off with distance $d$; it appears in the spatial distance penalty (doc 07) and in soft training labels (doc 03).

### 3.4 Linear interpolation (lerp)

To blend smoothly between a "slow" value and a "fast" value as a knob $t$ moves from $0$ to $1$:

$$
\operatorname{lerp}(a, b, t) = a + (b - a)\,t = (1-t)\,a + t\,b,\qquad t\in[0,1].
$$

At $t=0$ you get $a$; at $t=1$ you get $b$; in between, a straight‑line blend. SiamRAM's adaptive template‑rate controller (doc 10) lerps the update period between a slow setting and a fast setting based on how much motion there is.

### 3.5 Clamping (clip)

$$
\operatorname{clip}(z, a, b) = \min\!\big(\max(z, a),\, b\big)
$$

forces $z$ into $[a,b]$: anything below $a$ becomes $a$, anything above $b$ becomes $b$. Used to keep boxes inside the frame, probabilities inside $[0,1]$, and adaptive thresholds inside their legal range.

---

## 4. A little calculus: derivatives, gradients, Jacobians

You need only the *idea* of a derivative plus how it generalizes to vectors.

### 4.1 Derivative

For a function $f(x)$ of one variable, the **derivative** $f'(x) = \frac{df}{dx}$ is the slope of $f$ at $x$ — how fast the output changes per unit change in input. Near a point $x_0$, $f$ is well approximated by its tangent line:

$$
f(x) \approx f(x_0) + f'(x_0)\,(x - x_0).
$$

This is the **first‑order (linear) approximation**, and it is the entire idea behind the *Extended* Kalman filter: replace a curved function by its tangent line near the current estimate so the linear Kalman machinery applies (doc 05).

### 4.2 Partial derivatives and the Jacobian

If a function has several inputs, $f(x_1,\dots,x_n)$, the **partial derivative** $\partial f/\partial x_j$ is its slope with respect to $x_j$ while holding the others fixed.

Now consider a function that maps a vector to a vector, $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$, written componentwise as $\mathbf f(\mathbf x)=\big(f_1(\mathbf x),\dots,f_m(\mathbf x)\big)$. Its **Jacobian** is the $m\times n$ matrix of all partial derivatives:

$$
J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[1.2ex]
\vdots & \ddots & \vdots \\[0.6ex]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}.
$$

The Jacobian is the multidimensional generalization of "slope": near $\mathbf{x}_0$,

$$
\mathbf{f}(\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J\,(\mathbf{x} - \mathbf{x}_0).
$$

When SiamRAM's EKF warps a point through a camera homography (a *nonlinear* operation, because of the perspective division), it computes exactly this Jacobian to linearize that warp; doc 05 derives every entry of it.

### 4.3 The quotient rule (the one calculus rule we actually re‑derive)

Because the homography warp involves a division, we will need the derivative of a ratio. If $f(x) = \dfrac{g(x)}{h(x)}$ then

$$
f'(x) = \frac{g'(x)\,h(x) - g(x)\,h'(x)}{h(x)^2}.
$$

Keep this handy; it is the only nontrivial differentiation in the whole series, and doc 05 walks through applying it.

---

## 5. Probability you actually need

Tracking is estimation under uncertainty, so we need a working vocabulary of probability. Doc 04 turns this into the Kalman filter.

### 5.1 Random variables and densities

A **continuous random variable** $X$ doesn't have a single value but a **probability density function (pdf)** $p(x)\ge 0$ describing how likely each value is. Probabilities are areas under the density: $P(a\le X\le b) = \int_a^b p(x)\,dx$, and the total area is $1$, $\int_{-\infty}^{\infty}p(x)\,dx = 1$.

### 5.2 Expectation and variance

The **expectation** (mean) is the average value, weighting each $x$ by its density:

$$
\mu = \mathbb{E}[X] = \int x\,p(x)\,dx .
$$

The **variance** measures spread — the average squared distance from the mean:

$$
\sigma^2 = \operatorname{Var}[X] = \mathbb{E}\big[(X-\mu)^2\big].
$$

Its square root $\sigma$ is the **standard deviation**, in the same units as $X$. Small $\sigma$ = confident/peaked; large $\sigma$ = uncertain/spread out.

### 5.3 The Gaussian (normal) distribution

The single most important distribution for tracking is the **Gaussian**:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

It is the classic symmetric "bell curve," centered at $\mu$, with width set by $\sigma$.

```
 p(x)            μ
    |           _-•-_
    |         _/     \_          width set by σ
    |        /         \
    |      _/           \_
    |   __-               -__
    |_-‾                     ‾-_
    +----+----+----+----+----+----  x
      μ-2σ  μ-σ   μ    μ+σ  μ+2σ
```

Why Gaussians dominate tracking:

1. **Two numbers describe everything.** A Gaussian is fully specified by its mean $\mu$ and variance $\sigma^2$. So "what we believe about the target's position" can be stored as just a center and an uncertainty.
2. **They stay Gaussian under linear maps.** If $X$ is Gaussian and you apply a linear transformation, the result is still Gaussian. This is *why* the Kalman filter is exact for linear systems: belief stays Gaussian forever, so you only ever propagate $(\mu, \sigma^2)$.
3. **The product of two Gaussians is (proportional to) a Gaussian.** Combining a prediction and a measurement — both Gaussian — yields another Gaussian. That product *is* the Kalman update.

### 5.4 The multivariate Gaussian

For a vector $\mathbf{x}\in\mathbb{R}^n$, the Gaussian generalizes to

$$
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det\Sigma}}\,\exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol\mu)^{\!\top}\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right),
$$

written $\mathbf{x}\sim\mathcal{N}(\boldsymbol\mu,\Sigma)$. Here $\boldsymbol\mu\in\mathbb{R}^n$ is the mean vector and $\Sigma$ (an $n\times n$ symmetric PSD matrix) is the **covariance matrix**.

### 5.5 Covariance

The covariance matrix generalizes variance to vectors:

$$
\Sigma = \operatorname{Cov}[\mathbf{x}] = \mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^{\!\top}\big].
$$

- The diagonal entry $\Sigma_{ii}$ is the variance of coordinate $i$ (its own uncertainty).
- The off‑diagonal $\Sigma_{ij}$ is the **covariance** between coordinates $i$ and $j$: positive if they tend to be above/below their means together, negative if oppositely, zero if unrelated.

Geometrically, a multivariate Gaussian's "$1\sigma$" contour is an ellipse (in 2‑D) or ellipsoid (higher‑D); $\Sigma$ sets the ellipse's size and orientation. In the EKF, $\Sigma$ (called $P$) is literally the tracker's belief about how uncertain it is in position and velocity, and how those uncertainties are coupled.

### 5.6 Bayes' rule (the engine of estimation)

For events/hypotheses $A$ and data $B$:

$$
\boxed{\;p(A\mid B) = \frac{p(B\mid A)\,p(A)}{p(B)}\;}
$$

Read it as: **posterior $\propto$ likelihood $\times$ prior.** Your belief about $A$ *after* seeing data $B$ ("posterior") equals how well $A$ explains $B$ ("likelihood") times what you believed *before* ("prior"), renormalized. Tracking is one long application of Bayes' rule: the *prior* is where the motion model predicts the target is; the *likelihood* is what the measurement (matcher / detector) says; the *posterior* is the corrected estimate. Doc 04 turns this exact sentence into the Kalman filter.

---

## 6. The exponential moving average (EMA)

This deserves its own section because SiamRAM uses it in at least a dozen places (running confidence, adaptive thresholds, blur reference, camera‑motion magnitude, the EKF's tracked box size, …).

You receive a stream of measurements $z_1, z_2, z_3, \dots$ and want a running estimate of their "recent average" that updates cheaply and forgets the distant past. The EMA does this with one line:

$$
\boxed{\;\hat{s}_t = (1-\alpha)\,\hat{s}_{t-1} + \alpha\, z_t\;},\qquad \alpha\in(0,1].
$$

The new estimate is a blend of the old estimate and the new measurement. The single parameter $\alpha$ (the "learning rate" or "smoothing factor") controls memory:

- $\alpha$ near $1$: trust the latest measurement, forget quickly (responsive, noisy).
- $\alpha$ near $0$: change slowly, heavy smoothing (stable, laggy).

**Why it's "exponential."** Unrolling the recursion shows each past measurement's weight decays geometrically:

$$
\hat{s}_t = \alpha\, z_t + \alpha(1-\alpha) z_{t-1} + \alpha(1-\alpha)^2 z_{t-2} + \cdots
$$

A measurement $k$ steps ago carries weight $\alpha(1-\alpha)^k$, which shrinks exponentially with $k$. The weights sum to $1$, so the EMA is a genuine weighted average. Its effective memory length is about $1/\alpha$ samples.

```
 weight on z_{t-k}
   |•                      α
   | •
   |   •                   α(1-α)
   |      •
   |          •            α(1-α)^2
   |               •  •
   +----+----+----+----+----+----> k (steps into the past)
     0    1    2    3    4    5
```

A close cousin is the **running confidence with a floor** used by the matcher (doc 03): an EMA of the per‑frame score that is never allowed below a floor value, so a hard sequence cannot collapse the tracker's internal confidence to zero.

---

## 7. Geometry helpers: angles, projections, robust averages

### 7.1 atan2 — recovering an angle

Given a 2‑D vector $(a, b)$, its angle from the positive $x$‑axis is $\theta = \operatorname{atan2}(b, a)$. Unlike plain $\arctan(b/a)$, `atan2` gets the **quadrant right** (it knows the signs of $a$ and $b$ separately) and handles $a=0$. SiamRAM uses it to read the rotation angle out of a camera transform (doc 05): if the top‑left $2\times2$ block of the transform is a scaled rotation $\begin{bmatrix} a & -b \\ b & a\end{bmatrix}$, then the rotation is $\operatorname{atan2}(b, a)$ and the scale is $\sqrt{a^2+b^2}$.

### 7.2 Median — a robust average

The **median** of a list is its middle value when sorted. Unlike the mean, a few wild values ("outliers") barely move it. SiamRAM uses medians where robustness matters: a baseline "typical step size" for spike detection (doc 09), a typical box area for memory admission (doc 07), a representative object size during occlusion (doc 08). Whenever the data may contain a few corrupt entries, the median is the safer summary.

### 7.3 Weighted average

Given values $v_1,\dots,v_n$ and nonnegative weights $w_1,\dots,w_n$, the **weighted average** is

$$
\bar v = \frac{\sum_i w_i v_i}{\sum_i w_i}.
$$

The EMA (§6) is a weighted average with exponentially decaying weights. SiamRAM also forms an *explicit* exponentially‑weighted average of recent camera displacements, with weights $w_i = e^{(\,i - n\,)/\text{(something)}}$ that grow toward the present, to estimate "how much is the camera moving lately" (doc 06).

---

## 8. Putting the toolkit to work — a preview

Every later document is built from the pieces above. As a sanity check that you have what you need, here is where each tool first becomes essential:

| Tool (this doc) | First essential use |
|---|---|
| Norm / distance (§1.1) | box‑center displacement, spike speed (docs 02, 09) |
| Cosine similarity (§1.3) | appearance matching (docs 03, 07) |
| Matrix–vector product (§2.1) | constant‑velocity model, homography warp (docs 04, 05) |
| Matrix inverse (§2.3) | Kalman gain (doc 04) |
| Sigmoid + argmax (§3.1–3.2) | turning the network's maps into a box (doc 03) |
| Exponential decay / Gaussian bump (§3.3) | scale penalty, recency & distance penalties (docs 03, 07) |
| Jacobian + linear approx (§4.2) | the EKF's camera‑warp linearization (doc 05) |
| Quotient rule (§4.3) | differentiating the perspective division (doc 05) |
| Gaussian + covariance (§5.3–5.5) | the belief state of the EKF (docs 04, 05) |
| Bayes' rule (§5.6) | the predict/update derivation (doc 04) |
| EMA (§6) | every adaptive threshold and running statistic (doc 10) |
| Median / weighted avg (§7) | robust baselines, heavy‑motion gating (docs 06, 09) |

With the toolkit in hand, turn to **`02_BOUNDING_BOXES_AND_IMAGE_GEOMETRY.md`**, where we make "a box on an image" precise and set up the coordinate spaces the whole tracker lives in.
