# 10 — Adaptive Controllers: Self‑Tuning Thresholds

> Prerequisites: doc 01 (EMA, linear interpolation, clamping), doc 03 (the confidence score, the dynamic‑template admission bar), doc 06 (heavy‑camera‑motion magnitude, target speed), doc 07 (the DRM composite score and `score_target_against_drm`), doc 08 (occlusion entry / re‑acquisition thresholds). This document explains the small **EMA‑based estimators** that retune SiamRAM's key thresholds and rates to *each video's difficulty*, so that no single fixed number has to work for an easy, distinctive target and a hard, tiny, low‑texture one at the same time. Implemented in `auto_conf_threshold.py`, `auto_margin.py`, and `auto_template_rate.py`.

---

## 1. Why fixed thresholds fail, and the general pattern

Consider the occlusion‑entry threshold (doc 08 §2). A distinctive target scores $\sim0.9$ when correctly tracked, so a $0.5$ entry bar means "a real, large confidence drop is needed before we declare loss." But a genuinely hard target may score $\sim0.55$–$0.65$ *even when perfectly tracked* — for it, that same $0.5$ bar would fire a false loss on nearly every frame. One number cannot serve both. The same tension afflicts the dynamic‑template admission bar, the re‑acquisition margin, and the template update rate.

The fix is a **controller** that observes how the genuine target behaves during healthy tracking and sets the threshold *relative to that*. All three controllers share one design pattern, worth stating once because it recurs:

> **Measure a signal from the real target during confident tracking → smooth it with an EMA → map that smoothed value to a threshold → guard with warm‑up, minimum‑samples, and confident‑only sampling.**

Two guards make this safe:

- **Warm‑up + minimum samples.** Until enough samples are collected, the controller reports a fixed warm‑up value, so an unwarmed controller behaves exactly like the legacy fixed threshold. (Construction also keeps the warm‑up value inside the legal output range, so `value` is always valid.)
- **Confident‑only sampling (avoid feedback loops).** The controllers fold in a sample only on *healthy* frames (score above the loss threshold, not in distractor mode), on a cadence (`due`/`n_frames`). This keeps genuine *pre‑loss declines* out of the statistic — otherwise the threshold would drift *down* as the target starts to disappear and *delay the very loss it should detect*, a dangerous positive‑feedback loop. Sampling only confident frames breaks that loop: the bar reflects how the target scores *when actually locked on*.

The EMA itself is doc 01 §6: $\hat s \leftarrow (1-\alpha)\hat s + \alpha z$, a one‑line recency‑weighted average with memory length $\sim 1/\alpha$.

---

## 2. `AutoConfThreshold` — EMA‑to‑range mapping

Used for the **occlusion‑entry** confidence threshold and (with its own knobs) the **dynamic‑template admission** bar (doc 03 §9.2). It maps an EMA of the target's own tracking score onto an output threshold range by a **linear map**.

### 2.1 The mechanism

Maintain $\hat s=$ EMA of the per‑frame confident score. Map it from an input range $[\text{in\_min}, \text{in\_max}]$ onto an output range $[\text{out\_min}, \text{out\_max}]$ (doc 01 §3.4–3.5):

$$
\text{frac} = \operatorname{clip}\!\Big(\frac{\hat s - \text{in\_min}}{\text{in\_max} - \text{in\_min}},\ 0,\ 1\Big),
\qquad
\boxed{\ \text{value} = \text{out\_min} + \text{frac}\cdot(\text{out\_max} - \text{out\_min}).\ }
$$

Read it: locate $\hat s$ as a fraction of the way through the input band (clamped to $[0,1]$ so values outside the band saturate), then place the threshold the same fraction of the way through the output band. So:

- A target the tracker is consistently **very confident** on (high $\hat s$, near `in_max`) gets a **high** entry bar (near `out_max`): a real, large confidence drop is required before declaring loss.
- A target it is only **weakly confident** on (low $\hat s$, near `in_min`) gets a **low** entry bar (near `out_min`): its normally‑low scores are *not* mistaken for occlusion.

```
   threshold (value)
   out_max ┤                       ┌────────  (confident target → strict bar)
           │                   ___/
           │               ___/  ← value = out_min + frac·(out_max − out_min)
   out_min ┤──────────────/
           └────┬───────────────┬──────────► EMA of target score  ŝ
             in_min           in_max
```

It is a *monotonic, saturating* remap: outside $[\text{in\_min},\text{in\_max}]$ it clamps to the nearest output bound. Degenerate input band ($\text{in\_max}\approx\text{in\_min}$) is handled by snapping to $0$ or $1$.

### 2.2 The lifecycle

`due(frame_idx)` returns true once `n_frames` have elapsed since the last sample (a cadence). `observe(score, frame_idx)` folds the score into the EMA and, once `samples ≥ min_samples`, refreshes `value` via the map. `reset()` (on re‑init) forgets everything and returns to `warmup`. Because the threshold the *next* frames compare against is refreshed *after* this frame, there is a one‑frame lag — harmless, and the same convention across all the controllers.

---

## 3. `AutoDrmMargin` — sit just below the target's own score

Used for the occlusion **re‑acquisition margin** $\mu$ (doc 07 §4.8, doc 08 §4.5): the bar a candidate's composite DRM score must clear. The insight (doc 07 §4.9): we can *measure how high the genuine target scores against its own DRM anchors* via `score_target_against_drm`, using the **same composite formula** the matcher uses for candidates — so the number is on the same scale as $\mu$. Set the margin **just below** where the true target lives, so real re‑acquisitions clear it but impostors (which score lower) don't.

### 3.1 The mechanism

Maintain $\hat s=$ EMA of the target's self‑score against DRM (sampled on confident frames). The margin is that EMA minus a small offset, clamped to a legal range:

$$
\boxed{\ \mu = \operatorname{clip}\big(\hat s - \delta,\ \ \mu_{\min},\ \ \mu_{\max}\big).\ }
$$

The offset $\delta$ (`delta`) is the safety gap: the true target typically scores around $\hat s$, so a bar at $\hat s - \delta$ admits it with margin to spare while staying above the lower scores a look‑alike produces. The clamp keeps $\mu$ inside $[\mu_{\min}, \mu_{\max}]$ regardless. For a low‑texture/small/fast target whose self‑score is genuinely lower, the margin *follows it down* (so valid re‑acquisitions aren't rejected); for a distinctive target it rides higher (so distractors are excluded). Same `due`/`observe`/`reset`/warm‑up lifecycle as §2.2 (with its own cadence `n_frames` and `min_samples`).

```
        score scale
   1.0 ┤   ŝ (target self‑score, EMA)  ──●        impostors score below here
       │                                 │  δ
   μ → ┤·································○·│··  margin = ŝ − δ  (clamped)
       │                                          ← real re‑acquisitions clear μ,
   0.0 ┤                                            look‑alikes usually don't
```

---

## 4. `AutoTemplateRate` — motion drives the update cadence

Used to set the dynamic‑template refresh cadence $N$ and the memory‑window size (doc 03 §9). The trade‑off: when the scene is **steady**, appearance is stable, so the template should update *slowly* (large $N$ / large window = lots of averaging, very stable); when the **camera or target moves fast**, appearance changes quickly, so it should update *fast* (small $N$ / small window = snap to the latest look). So the cadence should track *motion*.

### 4.1 The mechanism

Each frame, form a single **"fastness"** scalar $m\in[0,1]$ by combining the camera‑motion magnitude (doc 06 §4: the bbox‑normalized displacement) and the target's own speed, taking the **max** (so either one being fast drives fast updating). Smooth it with an EMA, $\hat m \leftarrow (1-\alpha)\hat m + \alpha m$, then **linearly interpolate** (doc 01 §3.4) the cadence and window between their slow‑end and fast‑end values as $\hat m$ goes $0\to1$:

$$
N = \operatorname{round}\big(\operatorname{lerp}(N_{\text{slow}},\ N_{\text{fast}},\ \hat m)\big),
\qquad
\text{window} = \operatorname{round}\big(\operatorname{lerp}(\text{win}_{\text{slow}},\ \text{win}_{\text{fast}},\ \hat m)\big),
$$

with $N_{\text{fast}} < N_{\text{slow}}$ and $\text{win}_{\text{fast}} < \text{win}_{\text{slow}}$ (both clamped $\ge 1$). At $\hat m = 0$ (steady) you get the slow, stable rate; at $\hat m = 1$ (fast) the fast, snappy rate; in between, a straight‑line blend. Before any observation it sits at the slow end, so an unobserved controller reproduces a fixed slow rate.

```
   N (frames between template refreshes)
   N_slow ┤●__
          │   ‾‾──__              N = lerp(N_slow, N_fast, m̂)
          │         ‾‾──__        steady scene → slow, stable updates
   N_fast ┤               ‾‾──●   fast motion  → frequent, snappy updates
          └────┬──────────────┬──► smoothed motion  m̂ ∈ [0,1]
              0               1
```

This controller *takes over* from the older binary "camera‑motion template adapt" / "target‑motion template adapt" switches when enabled (they would otherwise fight over $N$/window); when it is off, the legacy fixed‑rate or binary paths are untouched.

---

## 5. Other self‑adjusting quantities (cross‑references)

The same "never let a fixed number fail; never let an adaptive number run away" philosophy appears in a few places already covered:

- **Running confidence with a floor** (doc 03 §9.5): an EMA of the per‑frame score is the dynamic‑template admission bar, but it is clamped from below by a floor so a hard sequence can't collapse it to $0$ and let any frame into memory.
- **Blur‑admit reference** (doc 03 §9.3): a *relative* sharpness gate whose reference EMA updates even on a veto, so a *sustained* texture drop lowers the bar and lets updates resume — only a one‑off blur spike is rejected.
- **Effective threshold switching** (doc 08 §2.1): a non‑EMA but related idea — swap to a lower threshold (and wider search) for tiny targets, because their honest scores are lower.

These are not separate "controllers" with the §1 lifecycle, but they embody the same principle: thresholds should reflect the *observed difficulty* of the current target, and adaptive quantities need guardrails.

---

## 6. Why this design is safe

It is worth making explicit why letting the tracker move its own thresholds doesn't spiral:

1. **Sampling is gated to confident, healthy frames.** The statistics describe the target *when locked on*, not when failing — so the bars don't drift to mask a real loss (§1).
2. **EMAs are slow.** With a modest $\alpha$, a single odd frame barely moves any threshold; it takes a *sustained* change to shift the bar, which is exactly when shifting is warranted.
3. **Everything is clamped.** Each controller's output lives in a fixed legal range ($[\text{out\_min},\text{out\_max}]$, $[\mu_{\min},\mu_{\max}]$, $\ge 1$ for $N$/window), so no value can run off to an absurd extreme.
4. **Unwarmed = legacy.** Before `min_samples` are collected, each controller returns its fixed warm‑up value, so behavior degrades gracefully to the hand‑tuned constant rather than to something undefined. A numeric (non‑`"auto"`) config never even constructs the controller.

---

## 7. Recap

- A single fixed threshold can't fit both easy and hard targets; SiamRAM uses **EMA‑based controllers** that set thresholds *relative to how the genuine target behaves during confident tracking*.
- **`AutoConfThreshold`**: EMA of the target score → **linear map** $[\text{in\_min},\text{in\_max}]\to[\text{out\_min},\text{out\_max}]$ → occlusion‑entry / dynamic‑admission bar; confident targets get a strict bar, weak ones a lenient bar.
- **`AutoDrmMargin`**: EMA of the target's self‑score against its own DRM, minus a safety offset $\delta$, clamped → re‑acquisition margin that sits *just below* where the true target lives.
- **`AutoTemplateRate`**: a $[0,1]$ motion "fastness" (max of camera and target motion), EMA‑smoothed, **lerped** into the template cadence $N$ and window — slow when steady, fast when moving.
- All share guards — **warm‑up, minimum samples, confident‑only sampling, clamping** — so adaptation is responsive but cannot feed back on itself or run away.

Next, the finale: **`11_PUTTING_IT_ALL_TOGETHER.md`** — the per‑frame conductor that wires every subsystem in this series into one coherent state machine.
