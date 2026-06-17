# 09 — Frame Dynamics: Injecting Motion Saliency

> Prerequisites: doc 01 (absolute value, weighted blend, clamping), doc 02 (the search crop, coordinate windows), doc 03 (what the matcher consumes). This is a short, self‑contained document about an **input augmentation**: for tiny/faint targets, SiamRAM can blend a *short‑term motion saliency* signal into the search crop before the matcher sees it, so a moving speck pops out against static clutter. Implemented in `models/siamram/frame_dynamics.py`. It is independent of the motion *filters* (docs 05–06): those reason about *where* the target is; this changes *what the matcher sees*.

---

## 1. The problem: appearance collapses for tiny targets

A drone $20$ px wide in a $1080$p frame, or a distant low‑contrast object, barely registers as *appearance* — its descriptor is weak and easily confused with background texture. But such a target is usually **moving relative to the static background**, and motion is a powerful, almost orthogonal cue. The idea (from the Anti‑UAV literature: *"A Simple Detector with Frame Dynamics is a Strong Tracker"*) is to feed the network not just "what the scene looks like now" but also "what *changed*" — because what changed is, mostly, what moved.

---

## 2. Frame differencing: motion as a temporal gradient

The simplest motion signal is the **frame difference**: subtract consecutive frames. A static pixel barely changes between frames, so its difference is near $0$; a pixel a moving object just entered (or left) changes a lot, so its difference is large. Taking absolute values, the difference map is a **motion saliency** image — bright where motion happened, dark where the scene was still. It is, in effect, a discrete *temporal gradient* $|\partial I/\partial t|$ (compare the temporal term $I_t'$ in the optical‑flow derivation, doc 05 §B.3).

The source paper concatenates the current frame with two such differences into a $6$‑channel input,

$$
x_{\text{fd}} = \big[\,x_t,\ \ x_t - x_{t-1},\ \ x_t - x_{t-2}\,\big],
$$

and trains a detector from scratch on those $6$ channels. Two differences (one‑ and two‑frame gaps) capture motion at two short time scales, which helps with both fast and slow movers.

---

## 3. SiamRAM's constraint and its workaround

SiamRAM's matcher (doc 03) is a **frozen, pretrained, 3‑channel** network. We cannot change its input to $6$ channels, and we will not retrain it. So instead of *concatenating* the difference maps, SiamRAM **blends them additively** into the existing 3‑channel search crop — an "additive motion emphasis." Per frame:

1. Compute the two absolute short‑term differences over the registered search window (oldest→newest in the buffer give $x_{t-2}$ and $x_{t-1}$):

$$
\text{diff}_1 = |\,x_t - x_{t-1}\,|,\qquad \text{diff}_2 = |\,x_t - x_{t-2}\,|.
$$

2. Average the available differences into one saliency map (one map if only $x_{t-1}$ is buffered, two if both):

$$
\text{saliency} = \frac{1}{n}\sum_{k} \text{diff}_k,\qquad n\in\{1,2\}.
$$

Averaging the two mirrors the paper's two‑difference triplet while staying single‑(per‑channel) so it can be added to a 3‑channel crop.

3. **Register and resize** the saliency to the search crop's pixel grid (doc 02 §4.4): the differences are computed over the same window of the full frame the crop came from, then resized to the crop's $S\times S$ size so the saliency lines up pixel‑for‑pixel with what the backbone consumes.

4. **Scale, clip, blend, clamp.** Multiply the saliency by a gain `scale`, optionally **clip** it from above (to suppress outlier difference pixels from compression noise or a sudden global illumination change), then add it into the crop with a small weight $w$ (`blend_weight`), and re‑clamp to the valid 8‑bit range so the result is still a legal image:

$$
\boxed{\ \text{crop}' = \operatorname{clip}\!\Big(\text{crop} + w\cdot \min(\text{scale}\cdot\text{saliency},\ \text{clip}),\ \ 0,\ 255\Big).\ }
$$

The effect: pixels that *moved* are brightened relative to static clutter, nudging the matcher's response toward the moving target without overwriting appearance. With $w=0$ this is an exact no‑op (the crop is unchanged), so the augmentation can be disabled cleanly. The default $w$ is deliberately *small* (e.g. $0.06$) — the frozen backbone still relies mainly on appearance; the motion is a gentle emphasis, not a takeover.

```
   x_t        x_{t−1}        |x_t − x_{t−1}|         search crop'        what the matcher sees
   ┌─────┐    ┌─────┐        ┌─────┐                 ┌─────┐
   │  •  │ −  │ •   │   =    │  ▓▓ │  (moving speck   │  ✦  │  ← the mover is emphasized
   │     │    │     │        │     │   lit up; static │     │     relative to static texture
   └─────┘    └─────┘        └─────┘   background ≈0) └─────┘
```

---

## 4. The tiny‑only gate (why it's not always on)

Blending motion into *every* crop would corrupt appearance on the majority of frames where the target is large and clearly visible — exactly where the frozen backbone is already strong. So by default the blend is **restricted to tiny targets**: it is applied only when the box's area fraction of the frame is at or below a small threshold (`tiny_area_fraction`, e.g. $0.001$):

$$
\text{apply blend} \iff \frac{w_{\text{box}}\,h_{\text{box}}}{H\cdot W} \le \text{tiny\_area\_fraction}.
$$

This confines the augmentation to the regime where appearance collapses and motion genuinely helps, leaving normal/large‑target crops untouched. (Setting `tiny_only = False` blends on every frame — the unrestricted variant.) Note the gate uses the *target's* size, the same scale‑awareness that recurs throughout SiamRAM (docs 02, 06, 08).

---

## 5. State and edge cases

The processor **owns a tiny rolling buffer** of the previous two full frames (so the host carries no extra state). Two consequences:

- On the **first two frames** of a clip there are not yet two previous frames, so there is no motion signal — the crop is returned unchanged. (The buffer is still advanced so motion is available next frame.)
- The buffer stores **copies** of the frames, so a later in‑place edit of the caller's frame cannot corrupt the motion history.

When the feature is disabled the host never even constructs the processor, so the crop is byte‑for‑byte the legacy crop and no frame‑difference work runs — zero cost when off.

---

## 6. Where it sits in the frame

Frame dynamics acts in the normal update, *after* the camera/Kalman search‑placement priors (doc 06) and *before* the matcher extracts its features (doc 03):

```
   ... GMC recenters the search (doc 06) ...
        │
        ▼
   frame‑dynamics: blend motion saliency into the search crop (this doc)   ◄── tiny targets only
        │
        ▼
   SiamABC forward on the (possibly emphasized) crop (doc 03)
```

So it is purely a *perceptual* aid: it does not move the search or change any threshold; it makes the moving target slightly more salient in the pixels the matcher reads.

---

## 7. Recap

- Tiny/faint targets have weak appearance but usually **move** against a static background; **frame differencing** $|x_t - x_{t-k}|$ is a cheap motion‑saliency signal (a discrete temporal gradient).
- SiamRAM cannot extend its frozen 3‑channel backbone, so it **additively blends** the averaged absolute differences into the search crop: $\text{crop}' = \operatorname{clip}(\text{crop} + w\cdot\min(\text{scale}\cdot\text{saliency}, \text{clip}),\ 0, 255)$, with a small weight $w$.
- A **tiny‑only gate** restricts the blend to small targets so normal‑target appearance is never corrupted; $w=0$ or a non‑tiny target is an exact no‑op.
- It owns a 2‑frame buffer (no‑op for the first two frames), and is a *perceptual* aid only — it changes what the matcher sees, not where it looks.

Next: **`10_ADAPTIVE_CONTROLLERS.md`** — the small EMA‑based estimators that retune the tracker's key thresholds and rates to each video's difficulty, so no single fixed number has to fit every sequence.
