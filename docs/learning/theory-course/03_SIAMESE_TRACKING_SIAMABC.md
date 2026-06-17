# 03 — Siamese Tracking and the SiamABC Matcher

> Prerequisites: doc 01 (dot product, cosine similarity, sigmoid, argmax, exponential) and doc 02 (boxes, crops, the score grid, coordinate map‑back). This is the longest core document because the visual matcher is the engine the whole tracker wraps. We go from the idea of "compare two pictures" all the way to "here is a box and a confidence number," deriving every transformation in between. We will treat the trained convolutional backbone as a known function and explain everything built around it.

---

## 1. The Siamese idea: tracking as matching

A **Siamese network** is two (or more) copies of the *same* network — same architecture, **same weights** — applied to different inputs, whose outputs are then compared. The name is by analogy with Siamese twins: identical, joined.

In tracking, the two inputs are:

- a **template** $z$ — a crop of the target (doc 02 §4.3), "what we are looking for";
- a **search region** $x$ — a larger crop of the current frame near the last known position, "where we are looking."

Both are passed through the **same** feature extractor $f_\theta$ (the shared backbone), producing feature maps $f_\theta(z)$ and $f_\theta(x)$. We then ask, at every location of the search feature map: *how well does the template match here?* The answer is a 2‑D **response map**; its peak is the predicted location.

```
   template z ─► f_θ ─► φ_z  ┐
                              ├──►  compare (correlate)  ──►  response map  ──► peak = location
   search   x ─► f_θ ─► φ_x  ┘
            (same weights — that's what "Siamese" means)
```

Why share weights? Because we want the *same* notion of "appearance" applied to both crops, so that "looks like" is a meaningful comparison in a common feature space. Why use a learned $f_\theta$ instead of raw pixels? Because raw‑pixel matching breaks under lighting, rotation, and small deformation; a network trained on millions of pairs learns features where "the same object" stays close even as it changes.

---

## 2. Cross‑correlation: the comparison operation

The "compare" step is **cross‑correlation**. Take the template feature map $\varphi_z$ (small, e.g. $C\times 8\times8$ channels$\times$height$\times$width) and the search feature map $\varphi_x$ (larger, e.g. $C\times16\times16$). Slide the template over the search map; at each offset, compute the **dot product** of the overlapping features (summed over all $C$ channels):

$$
R(u, v) = \sum_{c}\sum_{i,j} \varphi_z[c, i, j]\;\varphi_x[c,\, i+u,\, j+v].
$$

This $R(u,v)$ is the response at offset $(u,v)$. Recall from doc 01 §1.2 that a dot product is large when two vectors point the same way — so $R$ is large where the search features locally resemble the template features. Cross‑correlation is just "the matching score, evaluated at every shift, in one operation." It is implemented as a convolution where the template plays the role of the filter kernel; on a GPU this is a single fast layer.

```
   search feature map φ_x                response map R
   ┌───────────────┐    template φ_z     ┌───────────────┐
   │ . . . . . . . │     ┌───┐  slide    │ . . ▁ ▃ . . . │
   │ . . . ███ . . │ ◄── │███│ ──►        │ . ▁ ▆ █ ▃ . . │  ← peak where template aligns
   │ . . . ███ . . │     └───┘            │ . . ▃ ▆ ▁ . . │     with the object in φ_x
   └───────────────┘                      └───────────────┘
```

SiamABC's head (the `BoxTower`/`connect_model`) is a learned, richer version of this: it cross‑correlates the (attention‑refined) template against the search features and then runs a few convolutional layers to emit two maps — **classification** and **regression** — described in §5. But the heart is still "correlate the template over the search region."

---

## 3. The dual‑template trick (why SiamABC survives change)

A plain Siamese tracker uses one template, cropped from frame 0, forever. That template is a perfect *identity anchor* (it is, by definition, the true target), but it goes stale: after the object rotates or the lighting shifts, the frame‑0 crop no longer matches well.

SiamABC keeps **two** templates:

- **Static template** $z$ — the frame‑0 crop. **Never changes.** It is the ground‑truth identity anchor, the thing that stops the tracker from slowly drifting onto a similar‑looking neighbor.
- **Dynamic template** $\tilde z$ — refreshed during tracking from a recent high‑confidence frame. It absorbs gradual appearance change (pose, lighting, scale).

Holding both is a deliberate tension: the static template says "this is the *real* target," the dynamic one says "this is what it looks like *now*." Fusing them gives robustness to change without losing the identity anchor. The "how do we decide when and from which frame to refresh $\tilde z$" question is the online‑update mechanism of §9.

Symmetrically, SiamABC also keeps a **dynamic search** context $\tilde x$ from the same recent confident frame, so the network can compare current context against recent context. So the network actually consumes **four** crops per frame: static template $z$, dynamic template $\tilde z$, current search $x$, dynamic search $\tilde x$.

---

## 4. Inside the network: backbone, neck, attention, head

We now name the pieces of `SiamABCNet` and what each does. You do not need to know how the weights were learned; you need to know the *shape* of the computation.

### 4.1 Shared backbone $f_\theta$ and the neck

Each of the four crops is passed through the **shared backbone** `encoder` (a lightweight MobileNet‑style network for the "S" model, a ResNet for "M"). The backbone is a stack of convolution + normalization + nonlinearity layers that produces a spatial **feature map**: a tensor $C_{\text{enc}}\times H'\times W'$ where each of the $H'\times W'$ spatial positions holds a $C_{\text{enc}}$‑dimensional descriptor of the local image content. Spatial resolution is reduced by the backbone's **stride** (doc 02 §5).

A **convolution** layer, for intuition, slides a small learned filter over the input and computes, at each position, a dot product between the filter and the local patch — the same "local matching" motif as §2, but with *learned* filters and many of them stacked. Early layers detect edges/textures; deep layers detect object‑level patterns. We take that as given.

The **neck** (`AdjustLayer`) is a $1\times1$ convolution that projects every position's $C_{\text{enc}}$‑dimensional descriptor down to a common width $C$ (default $256$), so all four feature maps share a channel count and can be combined. (A $1\times1$ convolution is just a learned linear map applied identically at every spatial position — a per‑pixel matrix multiply.)

### 4.2 Polarized self‑attention: fusing static + dynamic

The two template feature maps ($\varphi_z$ static, $\varphi_{\tilde z}$ dynamic) are **concatenated along channels** into a $2C$‑channel map and passed through a `FastParallelPolarizedSelfAttention` block; an `attention_neck` projects the result back to $C$ channels. The same is done for the two search maps.

You do not need the internal algebra of polarized attention to understand the tracker, but the *idea* is worth one paragraph because it is the "ABC" (Attention‑Based Correlation) in SiamABC. **Attention** is a learned, content‑dependent reweighting: the block computes, from the features themselves, a set of weights and uses them to emphasize informative channels and spatial locations and suppress the rest. "Self"‑attention means the weights are computed from the same map they reweight. "Polarized" refers to keeping a high‑resolution branch in each of the channel and spatial directions (so fine detail is not lost) and combining them. The practical effect: the fused template representation $\varphi_{\text{tmpl}}$ blends the *stable* identity from $z$ with the *current* appearance from $\tilde z$, weighting whichever is more informative per channel/location. Output shape is unchanged ($C\times H'\times W'$), so the downstream correlation is identical to §2.

```
  φ_z (static)  ┐ concat        ┌─ polarized ─┐   attention_neck
                ├──► [2C maps] ──┤ self‑attn   ├──► φ_tmpl  (C maps)
  φ_z̃ (dynamic) ┘                └─────────────┘
  (same structure fuses current search φ_x with dynamic search φ_x̃ → φ_srch)
```

### 4.3 The BoxTower head: two output maps

The fused template $\varphi_{\text{tmpl}}$ is cross‑correlated against the fused search $\varphi_{\text{srch}}$ (the §2 operation), and a small stack of convolutions (`towernum` blocks) turns the result into **two maps** over the $G\times G$ score grid (doc 02 §5, e.g. $16\times16$):

- **Classification map** $\;\text{cls}\in\mathbb{R}^{G\times G}$: one raw score (logit) per cell, "is the target centered here?"
- **Regression map** $\;\text{reg}\in\mathbb{R}^{4\times G\times G}$: four numbers per cell, describing the box edges (next section).

An optional small **IoU head** predicts, per location, how good the localization is (used as a confidence gate in §8).

---

## 5. Anchor‑free box decoding (FCOS style)

Now the crucial bridge from "maps" to "a box." SiamABC uses the **FCOS** parameterization: *anchor‑free, distance‑to‑edges* regression. This is clean and tuning‑free, and the encode/decode are exact inverses, so we present both.

### 5.1 What each cell predicts

Recall (doc 02 §5) that grid cell $(i,j)$ corresponds to a known crop pixel $(\text{grid\_x}, \text{grid\_y})$. The regression map's four channels at that cell are the **distances from that pixel to the four edges** of the target box:

$$
\text{reg}(i,j) = (\ell, t, r, b) = \big(\underbrace{\text{grid\_x}-x_1}_{\text{left}},\ \underbrace{\text{grid\_y}-y_1}_{\text{top}},\ \underbrace{x_2-\text{grid\_x}}_{\text{right}},\ \underbrace{y_2-\text{grid\_y}}_{\text{bottom}}\big).
$$

In words: "from me, the left edge is $\ell$ pixels to the left, the top is $t$ up, the right is $r$ right, the bottom is $b$ down." A cell is **inside** the box exactly when all four distances are positive. That positivity test is how training labels which cells are "positive" (target) versus "negative" (background).

```
        x1            grid_x          x2
   y1 ──┼───────────────┼──────────────┼──
        │      ◄── ℓ ──►•◄────── r ─────►│      • = grid pixel (grid_x, grid_y)
        │               │ ▲             │       ℓ = grid_x − x1   (left distance)
        │               │ │ t           │       t = grid_y − y1   (top distance)
        │               │ ▼             │       r = x2 − grid_x   (right distance)
   y2 ──┼───────────────┼──────────────┼──      b = y2 − grid_y   (bottom distance)
```

### 5.2 Encoding (training targets) — for completeness

Given a ground‑truth box $(x,y,w,h)$, the encoder broadcasts it over the whole grid and computes, per cell,

$$
\ell = \text{grid\_x} - x,\quad t = \text{grid\_y} - y,\quad r = (x+w) - \text{grid\_x},\quad b = (y+h) - \text{grid\_y},
$$

stacks $(\ell,t,r,b)$ as the regression target, and sets the classification label to $1$ where $\min(\ell,t,r,b) > 0$ (strictly inside) and $0$ otherwise. (This is `SiamABCBoxCoder.encode`.) Training nudges the network's two maps toward these targets; we don't derive the loss here, but note that some recipes use a *soft* Gaussian label centered on the box — a 2‑D bump $\exp(-d^2/2\sigma^2)$ (doc 01 §3.3) — instead of a hard $0/1$ mask, which produces smoother, more peaked score maps and hence a more reliable argmax at decode time.

### 5.3 Decoding (inference) — turning maps into a box

At inference, `SiamABCBoxCoder.decode` inverts the encoding:

1. **Activate** the classification logits into confidences: $\text{cls\_score} = \sigma(\text{cls})\in(0,1)^{G\times G}$ (doc 01 §3.1). (After the penalties of §6 it becomes the *penalized* map used for selection.)
2. **Reconstruct each cell's box edges** from the grid and the predicted distances:

$$
\begin{aligned}
x_1(i,j) &= \text{grid\_x}(i,j) - \ell(i,j), & x_2(i,j) &= \text{grid\_x}(i,j) + r(i,j),\\
y_1(i,j) &= \text{grid\_y}(i,j) - t(i,j), & y_2(i,j) &= \text{grid\_y}(i,j) + b(i,j).
\end{aligned}
$$

3. **Pick the winning cell** by argmax over the (penalized) classification map. Flattening the $G\times G$ grid to a 1‑D index $k$, the winner is $k^{*} = \operatorname{argmax}_k \text{score}_k$, and its row/column are $r_{\max}=\lfloor k^{*}/G\rfloor,\ c_{\max}=k^{*}\bmod G$ (doc 01 §3.2).
4. **Read off the box** at the winning cell and convert to `xywh`:

$$
b_{\text{crop}} = \big(x_1,\ y_1,\ x_2 - x_1,\ y_2 - y_1\big)\Big|_{(r_{\max}, c_{\max})}.
$$

This box is in **crop coordinates**; doc 02 §4.4 maps it back to proc coordinates. The confidence reported for the frame is the score at $(r_{\max}, c_{\max})$ — possibly transformed (§7).

That's the whole appearance pipeline: four crops → features → attention fusion → correlation → two maps → penalties → argmax → one box. Everything else in this document refines *which cell wins* and *how confident we should be in it*.

---

## 6. Penalties: stopping the tracker from teleporting

A pure argmax over the raw classification map is brittle. Two failure modes:

- A spurious high response near the **edge** of the search crop (clutter, the context ring, a passing object) can beat the true peak.
- The box can suddenly **change size/shape** frame‑to‑frame in a physically implausible way.

SiamABC multiplies the score map by penalties that encode the priors "the target is probably near the center of the search crop" and "the box probably has a similar size/shape to last frame." These come from `_confidence_postprocess`.

### 6.1 The cosine (Hanning) window

The **Hann window** in 1‑D of length $G$ is

$$
\text{hann}(n) = \tfrac12\Big(1 - \cos\tfrac{2\pi n}{G-1}\Big),\qquad n = 0,\dots,G-1,
$$

a smooth bump that is $0$ at the ends and $1$ in the middle. The 2‑D window is the **outer product** of two 1‑D Hann windows, $\text{window}(i,j) = \text{hann}(i)\,\text{hann}(j)$ — a smooth dome peaking at the grid center and decaying to $0$ at the borders.

```
 1‑D Hann window (length G):              2‑D window = outer product (a dome):
   1 |        __--‾‾--__                       low ░░░░░░░░░░░ low
     |     _-‾          ‾-_                         ░░▒▒▒▓▓▒▒░░
     |   _-                -_                       ░▒▓▓███▓▓▒░
   0 |_-‾                    ‾-_                     ░░▒▒▒▓▓▒▒░░
     +--------------------------- n            low ░░░░░░░░░░░ low
       0                      G-1                    (peak at center)
```

It is blended into the score map with an **influence** weight $\omega\in[0,1]$:

$$
\text{pscore} = (1-\omega)\,\text{pscore} + \omega\,\text{window}.
$$

With $\omega=0$ the window does nothing; as $\omega$ grows, off‑center peaks are increasingly suppressed. The prior it encodes is exactly "between consecutive frames the target moves only a little, so it should be near the center of a search crop that was centered on its last position." This single trick is what keeps Siamese trackers from jumping frame‑to‑frame.

### 6.2 The scale/aspect (size) penalty

This penalty demotes candidate boxes whose **size** or **aspect ratio** differ sharply from the previous box. Two ingredients, both built so that *any* change — growth or shrinkage — is penalized symmetrically.

First, the helper `limit`, which folds a ratio to be $\ge 1$:

$$
\operatorname{limit}(\rho) = \max\!\big(\rho,\ \tfrac{1}{\rho}\big) \ \ge 1.
$$

If $\rho=2$ (doubled) it returns $2$; if $\rho=0.5$ (halved) it also returns $2$. So "twice as big" and "half as big" are treated as equally suspicious — a symmetric penalty curve. (Recall doc 01 §3.5 for clamping intuition; `limit` is a related symmetry trick.)

Now define the candidate's "padded square size" with `squared_size` (doc 02 §4.2) for both the candidate box and the previous box, and form the **size‑change** and **aspect‑change** ratios:

$$
s_c = \operatorname{limit}\!\left(\frac{\text{squared\_size}(w_{\text{cand}}, h_{\text{cand}})}{\text{squared\_size}(w_{\text{prev}}, h_{\text{prev}})}\right),
\qquad
r_c = \operatorname{limit}\!\left(\frac{w_{\text{prev}}/h_{\text{prev}}}{w_{\text{cand}}/h_{\text{cand}}}\right).
$$

$s_c\ge1$ measures overall size change; $r_c\ge1$ measures aspect‑ratio change; both equal $1$ for no change. The penalty multiplies the score by

$$
\boxed{\ \text{penalty} = \exp\!\big(-(r_c\, s_c - 1)\,k\big)\ } \quad (0 < \text{penalty} \le 1),
$$

where $k = \text{penalty\_k} \ge 0$ is the strength knob. When the candidate matches the previous box, $r_c s_c = 1$, the exponent is $0$, and $\text{penalty}=1$ (no demotion). The more the size/shape changes, the larger $r_c s_c$, the more negative the exponent, the smaller the penalty (toward $0$). This is the exponential decay of doc 01 §3.3 used as a soft veto on implausible size jumps. With $k=0$ the penalty is identically $1$ (disabled).

```
 penalty
   1 |•__
     |   ‾‾--__               penalty = exp(−(r_c·s_c − 1)·k)
     |         ‾‾--__         = 1 when size/shape unchanged (r_c·s_c = 1)
     |               ‾‾--__   → 0 as the box change grows
   0 |                     ‾‾----______
     +----+----+----+----+----+----+----- r_c·s_c
       1   1.5   2   2.5   3   3.5  4
```

The final selection map is $\text{pscore} = \text{penalty}\odot\text{cls\_score}$ (elementwise), then the window of §6.1 is blended in, then argmax (§5.3) selects the winner on this *penalized* map.

### 6.3 Size smoothing of the chosen box

Even after selection, the *reported* box width/height are smoothed toward the previous size so they don't flicker. With a learning rate $\eta$ modulated by the winning cell's penalty and score,

$$
\text{lr} = \text{penalty}[r_{\max},c_{\max}]\cdot\text{score}[r_{\max},c_{\max}]\cdot\eta,
$$

the new size is an interpolation (doc 01 §3.4) between the previous size and the predicted size. A confident, plausible prediction (high penalty $\times$ score) moves the size more; an uncertain one barely nudges it. Position is taken from the decoder directly; only size is smoothed.

---

## 7. Turning the response into a confidence number

Downstream subsystems (occlusion entry, memory admission, re‑acquisition) all threshold a single per‑frame **confidence** in $[0,1]$. SiamABC offers two ways to compute it.

### 7.1 Peak score (default)

The simplest: the value of the (sigmoid) classification map at the winning cell, $\text{score} = \text{cls\_score}[r_{\max}, c_{\max}] \in (0,1)$. High peak = confident match.

### 7.2 Peak‑to‑sidelobe ratio (PSR), optionally

The peak value alone can be misleading: a *flat* map with a slightly elevated peak is far less trustworthy than a *sharp* lone spike of the same height. The **peak‑to‑sidelobe ratio** (from the MOSSE correlation‑filter tracker) measures sharpness. Exclude a small square window (half‑width $\rho$) around the peak; call the rest the *sidelobe*, with mean $\mu_s$ and standard deviation $\sigma_s$. Then

$$
\text{PSR} = \frac{g_{\text{peak}} - \mu_s}{\sigma_s},
$$

where $g_{\text{peak}}$ is the peak value. PSR is large when the peak stands far above a low, flat sidelobe and small when the map is flat or multi‑modal. Since PSR is unbounded (it can be $\sim3$ when confused, $40+$ when locked on) but the pipeline wants a $[0,1]$ number, it is squashed monotonically:

$$
\text{score} = \frac{\text{PSR}}{\text{PSR} + \kappa}\in[0,1),
$$

with $\kappa$ a constant (larger $\kappa$ = stricter). $\text{PSR}\le0$ or a degenerate sidelobe maps to $0$. Because the output is still in $[0,1]$, every existing threshold keeps its meaning whether the tracker is in "peak" or "psr" mode.

### 7.3 The optional IoU‑head gate

If the IoU head (§4.3) is active, it predicts a localization‑quality value $q\in(0,1)$ per cell. SiamABC can (a) multiply the confidence by $q^{\,p}$ for a power $p$ to fold quality into the score, and/or (b) use $q$ at the peak as a **gate**: if $q$ (smoothed by its own EMA, doc 01 §6) falls below a threshold, the frame's confidence is *penalized* (multiplied by $0.25$) rather than hard‑zeroed — a soft veto, because hard‑zeroing caused too many false occlusion entries on noisy IoU maps. This is a small but instructive design point: **soft, reversible penalties are preferred over hard kills** throughout SiamRAM, because a hard kill on a noisy signal is itself a failure mode.

---

## 8. Test‑Time Adaptation via Adaptive BatchNorm (brief)

Neural networks contain **BatchNorm** layers that standardize activations using running mean/variance statistics learned at training time. If the test video's lighting/sensor differs, those statistics are slightly off. SiamABC can **adapt** them at test time: an `AdaptiveBatchNorm` blends the stored statistics with the *current crop's* statistics, controlled by a scalar $\lambda$ (`_tta_lam`). With $\lambda=0$ it behaves as ordinary frozen BatchNorm (default); setting $\lambda$ to a small positive value turns adaptation on. This is a lightweight, training‑free way to absorb gradual illumination/white‑balance drift. It changes the *features*, not the geometry, so everything above is unaffected.

---

## 9. The online dual‑template update (RAM‑lite)

Section 3 promised a mechanism for refreshing the dynamic template $\tilde z$. Here it is. This is a *short‑term, geometry‑and‑confidence* memory living inside SiamABC — distinct from the appearance memory of doc 07, though related in spirit.

### 9.1 The memory window

SiamABC keeps a fixed‑size deque (`all_memory_imgs`) of the most recent **(frame, box)** pairs, of length $M$ (`memory_window_size`), with their confidence scores in a parallel deque. When a new confident frame arrives it is pushed; the oldest falls off. A running pointer tracks the highest‑scoring entry so the best frame can be found cheaply.

### 9.2 Two admission gates

A frame is admitted to the window only if it passes both:

1. **Score gate.** The score must beat the *running confidence* EMA, OR clear a hard update threshold. Formally, with running confidence $\bar c$ (an EMA, §9.4) and threshold $\tau_u$: admit if $\text{score} > \bar c$ or $\text{score} \ge \tau_u$. This keeps low‑confidence (possibly wrong) frames out of memory.
2. **IoU gate.** The new box must overlap the previous box: $\operatorname{IoU}(b_t, b_{t-1}) \ge \tau_{\text{iou}}$ (doc 02 §3, default $\approx0.3$). A sudden box jump (low IoU) is refused — it usually signals confusion, and storing it would corrupt the template.

### 9.3 The blur‑admit gate (optional)

A frame that is a transient **motion‑blur** smear should not refresh the template, or the smear gets baked in. SiamABC measures crop sharpness with the **variance of the Laplacian**: it resizes the target crop to a fixed $64\times64$ grayscale patch, applies the Laplacian operator (a discrete second derivative that responds to edges), and takes the variance of the result. A sharp crop has lots of high‑frequency edge energy → high variance; a blurred crop is low‑pass‑filtered → low variance. The gate is **relative**: the sharpness must clear a fraction (e.g. $0.6$) of a running EMA reference of recent confident‑frame sharpness. Crucially, the EMA reference is updated *even when a frame is vetoed*, so a *sustained* drop in sharpness (the target genuinely became lower‑texture) gradually lowers the bar and lets updates resume — only a one‑off spike is rejected. (Using a fixed absolute threshold would fail because sharpness depends on content and scale; a relative, self‑adjusting bar is the correct design.)

### 9.4 Selecting the dynamic template, on a cadence

Every $N$ frames (`select_representatives`), the tracker scans the window from newest to oldest for the first entry whose score clears the update threshold $\tau_u$, and refreshes $\tilde z$ (and $\tilde x$) from that frame. Choosing the *newest qualifying* frame keeps the dynamic template as current as possible while still requiring it to be confident. If no stored frame clears $\tau_u$, the previous $\tilde z$ is kept. (The cadence $N$ and window $M$ can themselves be adapted to motion — doc 10.)

### 9.5 Running confidence with a floor

The **running confidence** is an EMA of the per‑frame score (doc 01 §6),

$$
\bar c \leftarrow (1-\lambda)\,\bar c + \lambda\,\text{score},
$$

clamped below by a floor value:

$$
\bar c \leftarrow \max(\bar c,\ \bar c_{\min}).
$$

The EMA gives a sequence‑adaptive sense of "how confident am I usually," used as the dynamic admission bar in §9.2. The floor prevents a hard run from driving $\bar c$ to $0$, which would let *any* frame into memory and collapse the tracker — a guardrail, again of the "never let an adaptive quantity run away" kind that recurs in doc 10.

---

## 10. The per‑frame loop, assembled

Putting §§4–9 together, here is one SiamABC frame:

```
 1. extract search crop x around the previous (or motion‑predicted) box; preprocess     [doc 02 §4]
 2. backbone+neck → φ_x ;  φ_z, φ_z̃, φ_x̃ are cached (templates change only on cadence)   [§4.1]
 3. attention‑fuse (φ_z, φ_z̃)→φ_tmpl  and  (φ_x, φ_x̃)→φ_srch                             [§4.2]
 4. BoxTower: correlate → classification map (logits) + regression map (ℓ,t,r,b)        [§4.3,§5]
 5. cls_score = σ(cls);  apply size penalty × window → penalized selection map           [§6]
 6. (optional) Kalman‑motion fusion reweights the map                                    [doc 06]
 7. argmax → winning cell → reconstruct box edges → xywh in crop coords                   [§5.3]
 8. rescale box to proc coords; clamp to frame                                            [doc 02 §4.4]
 9. confidence = peak (or PSR), optionally IoU‑gated                                      [§7]
10. admit frame to memory if score+IoU(+blur) gates pass; every N frames refresh z̃       [§9]
11. update running confidence EMA (with floor)                                           [§9.5]
 → returns (box, confidence) to the SiamRAM conductor                                    [doc 11]
```

Steps 1–9 are "find the target this frame"; steps 10–11 are "keep the template fresh for next time." The output `(box, confidence)` is exactly what the outer SiamRAM layer consumes: the box is a candidate position, and the confidence is the single number the motion filter, memory, and occlusion logic all reason about.

---

## 11. Recap and limitations

- A Siamese tracker compares a **template** to a **search region** through a shared learned feature extractor; the comparison is **cross‑correlation**, whose peak is the location.
- SiamABC fuses a **static** (identity anchor) and a **dynamic** (current appearance) template via polarized self‑attention, then decodes a box with the **FCOS** distance‑to‑edges parameterization — anchor‑free, argmax‑selected.
- A **cosine window** and a **scale/aspect penalty** ($\exp(-(r_c s_c -1)k)$) encode the priors "near center" and "similar size," preventing frame‑to‑frame teleports.
- Confidence is the **peak** (or a squashed **PSR** for sharpness‑awareness), optionally gated by a learned IoU head — always with *soft* penalties over hard kills.
- An online **memory window** with score/IoU/blur gates refreshes the dynamic template on a cadence, and a floored **running‑confidence EMA** adapts the admission bar.

**What SiamABC cannot do alone:** it has no physics (it doesn't know the target's velocity), no long‑term identity memory (only a short window), no notion of camera motion, and no way to recover once the target leaves its search crop. Those are exactly the jobs of the outer SiamRAM layer. The very next thing it needs is a **motion model**, so the search crop can be centered where the target is *going*, not where it *was*. That is doc 04 (the theory of filtering) and doc 05 (the EKF + camera motion).
