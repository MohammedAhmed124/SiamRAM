# 07 — Appearance Memory: RAM and DRM

> Prerequisites: doc 01 (cosine similarity, exponential decay, Gaussian bump, weighted average), doc 02 (IoU, box center). This document explains how SiamRAM *remembers what the target looks like* so it can **re‑recognize** the target after losing it. There are two memories — a short‑term **RAM** buffer and a long‑term **DRM** (Dynamic Reference Memory) bank — and a multi‑term **composite score** used to rank re‑acquisition candidates during occlusion recovery (doc 08). The matcher (doc 03) handles short‑horizon "where did it move"; this memory handles long‑horizon "is *this* the same object." Implemented in `models/siamram/memory.py`.

---

## 1. Why a separate appearance memory?

The matcher's dynamic template (doc 03 §9) is a *short‑window* appearance model, refreshed continuously. That is perfect for following gradual change, but it has two weaknesses for the *recovery* problem:

1. It drifts. If the matcher briefly latched onto the wrong thing, the dynamic template absorbs that error.
2. It is a single fused template, not a *bank* of verified exemplars you can match a fresh detection against.

When the target has been **lost** (occlusion, doc 08) and an object detector proposes several candidate boxes, we need to ask "which of these, if any, is the *same object* we were tracking?" Answering that robustly needs (a) a stable set of *known‑good* appearance exemplars, and (b) a scoring rule that combines appearance with overlap, motion, and recency. That is exactly RAM + DRM + the composite score.

### 1.1 Appearance descriptors

A **descriptor** is a fixed‑length feature vector summarizing the look of a box's contents — produced by a small re‑identification network (OSNet) applied to the cropped region (`_extract_descriptor`). Two crops of the same object give descriptors pointing in nearly the same direction; different objects point differently. We compare descriptors with **cosine similarity** (doc 01 §1.3):

$$
\operatorname{cossim}(\mathbf a, \mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\,\|\mathbf b\| + \varepsilon}\in[-1, 1],
$$

brightness‑/scale‑robust, $1$ for "same look," near $0$ for "unrelated." This single number is the appearance signal throughout this document.

---

## 2. RAM — the short‑term buffer

**RAM** is a fixed‑capacity deque of recent **(box, descriptor)** pairs from confidently tracked frames. It is the raw material from which the long‑term DRM is curated, and it provides the "what did the target look like most recently" reference for simple matching.

### 2.1 Admission gates (`try_admit`)

A new (box, descriptor) is admitted only if it passes **two geometric coherence gates** — appearance is *not* checked at admission (a confidently tracked frame is assumed to be the target; the gates guard against the box having jumped or resized suspiciously):

1. **IoU gate.** The new box must overlap the previous box: $\operatorname{IoU}(b_t, b_{t-1}) \ge \tau_{\text{iou}}$ (doc 02 §3, e.g. $\tau_{\text{iou}}=0.4$). A sudden positional jump (low IoU) suggests confusion, not smooth tracking, so it is refused.

2. **Area gate.** The new box's area must be close to the **median** area of the buffer (doc 01 §7.2 — median for robustness to a few odd entries):

$$
\frac{\big|\,w_t h_t - \operatorname{median}(\text{areas})\,\big|}{\operatorname{median}(\text{areas}) + \varepsilon} \le \tau_{\text{area}}\quad(\text{e.g. } 0.25).
$$

This rejects frames where the box suddenly ballooned or collapsed (a partial occlusion or a bad match) — those would pollute the memory of "the target's size."

If both pass, the pair is appended (oldest evicted when full), and the candidate is immediately tested for **promotion** to DRM (§3). Tying admission to *geometry* rather than appearance is deliberate: we trust that confident, smoothly‑moving frames are the target, and we keep out the geometrically suspicious ones.

### 2.2 The simple match (`match`)

The most basic query: given candidate boxes in the current frame, extract their descriptors and return the one most similar (cosine) to the **most recent** RAM descriptor, if it clears a threshold. This is the fallback used when the long‑term DRM is still empty (early in a sequence). It answers "which candidate looks most like the target did a moment ago."

---

## 3. DRM — the long‑term reference bank

The matcher's dynamic template and RAM both track *recent* appearance, which is good for drift‑following but bad for *re‑identification after a long gap*: by the time the target reappears, "recent" may be corrupted or stale. **DRM** is a small, curated bank of **anchors** — verified, mutually‑consistent exemplars — meant to remain valid across an occlusion. Each anchor is a triple $(\,\text{box},\ \text{descriptor},\ \rho\,)$ where $\rho$ is the *time index* it was stored (its age stamp).

### 3.1 Promotion by agreement (`_try_promote_to_drm`)

A RAM entry is promoted to DRM only if it is **corroborated** by its neighbors. When a new descriptor $\mathbf d$ is admitted to RAM, look at the last $W$ RAM entries (a window, `window_W`) and count how many *agree* with $\mathbf d$ in appearance:

$$
\text{agreements} = \#\big\{\,\text{recent entry } \mathbf d_i :\ \operatorname{cossim}(\mathbf d, \mathbf d_i) \ge \tau_{\text{sim}}\,\big\}.
$$

If $\text{agreements} \ge m_{\min}$ (e.g. $\tau_{\text{sim}}=0.85$, $m_{\min}=3$), the entry is appended to DRM. The logic: an exemplar worth keeping long‑term should be *appearance‑consistent with several of its temporal neighbors* — that filters out one‑off lucky matches and keeps only stable, repeatedly‑confirmed looks of the target. (There is a second write path, `add_drm_anchor`, used by the introspection feature; the agreement path is the primary one.) DRM has a small capacity (e.g. $10$), so it holds a compact, high‑quality set rather than a long history.

```
   RAM (recent, by time →)                          DRM (curated anchors)
   [d₁][d₂][d₃][d₄][d₅][d₆ new]                       ┌─────────────────────┐
                  └──window W──┘                       │ (box, desc, ρ) × ≤10│
        new d₆ agrees (cossim ≥ τ_sim) with            │  verified, stable   │
        ≥ m_min of the window  ──promote──►            └─────────────────────┘
```

---

## 4. The composite re‑acquisition score (`drm_match`)

This is the heart of recovery scoring: given a set of candidate boxes (object‑detector proposals during occlusion, doc 08), score each by **how much it looks and behaves like the lost target**, and return the best (or a short list). Appearance alone is not enough — a look‑alike can score high on appearance — so the score *combines* appearance with overlap, motion direction, recency, and spatial plausibility. We build it term by term.

Setup: let the **reference box** $b_{\text{ref}}$ (center $(c_x^{\text{ref}}, c_y^{\text{ref}})$) be the last known target box, and $\mathbf v$ the target's velocity (from the EKF, doc 05). For each **DRM anchor** $k = (b_k, \mathbf d_k, \rho_k)$ and each **candidate** $c = (b_c, \mathbf d_c)$, the score combines the following.

### 4.1 Overlap term

$$
s_{\text{iou}} = \lambda_{\text{iou}}\cdot\operatorname{IoU}(b_k,\ b_c).
$$

Rewards a candidate that geometrically overlaps the anchor's box. (Most relevant when anchors are spatially near the candidate.)

### 4.2 Appearance term

$$
s_{\text{app}} = \lambda_{\text{app}}\cdot\operatorname{cossim}(\mathbf d_k,\ \mathbf d_c).
$$

Rewards a candidate whose descriptor matches the anchor's — the core "same object" signal.

### 4.3 Motion‑direction term

Does the *direction* from the reference to the anchor agree with where the target was heading? Form the displacement vector from reference center to anchor center, $\mathbf m_k = (c_x^k - c_x^{\text{ref}},\ c_y^k - c_y^{\text{ref}})$, and take the cosine of the angle between the velocity and that displacement (doc 01 §1.2–1.3), clamped to be non‑negative:

$$
\pi_k = \max\!\Big(0,\ \frac{\mathbf v\cdot\mathbf m_k}{\|\mathbf v\|\,\|\mathbf m_k\| + \varepsilon}\Big)\in[0,1].
$$

$\pi_k = 1$ when the anchor lies exactly in the direction of motion, $0$ when it is sideways or behind. The $\max(0,\cdot)$ means "no bonus for being behind the motion," but no penalty either (other terms handle that).

### 4.4 Recency term

Older anchors should count a little less. With $\text{age} = t - \rho_k$ (how many time steps ago the anchor was stored) and a decay rate $\alpha$ (doc 01 §3.3):

$$
s_{\text{time}} = \lambda_{\text{time}}\cdot e^{-\alpha\,\text{age}}.
$$

$e^{-\alpha\,\text{age}}$ is $1$ for a fresh anchor and decays toward $0$ for old ones — a soft preference for recent evidence without discarding the old.

### 4.5 (Optional) negative‑bank penalty

There is a subtractive term $-\gamma\cdot\max_j \operatorname{cossim}(\text{distractor}_j,\ \mathbf d_k)$ that penalizes anchors resembling a bank of *known non‑target* appearances. How that "distractor bank" is populated is tied to the distractor/spike machinery that is **out of scope** for these notes; for our purposes it is simply a small negative term that can demote an anchor that looks like a known impostor. With an empty bank (or $\gamma=0$) it vanishes.

### 4.6 Per‑anchor raw score and aggregation

The motion + recency (− penalty) part is computed per anchor once; the per‑anchor raw score for a candidate is

$$
\text{raw}_k = s_{\text{iou}} + s_{\text{app}} + \underbrace{\big(\lambda_{\text{mot}}\,\pi_k + \lambda_{\text{time}}\,e^{-\alpha\,\text{age}} - \gamma\,(\text{penalty})\big)}_{s_{\text{mot+time}}}.
$$

A candidate is scored against *every* anchor, then those per‑anchor scores are **aggregated** into one candidate score (`_aggregate_drm_anchor_scores`):

- **`max`** — the best matching anchor wins ("does this candidate match *any* good exemplar?"). Most permissive.
- **`mean`** (optionally **top‑$k$ mean**) — average over all anchors, or just the $k$ best ("does this candidate match the target's exemplars *on the whole*?"). More conservative; top‑$k$ is a middle ground that ignores the worst anchors.

### 4.7 Two candidate‑level adjustments

After aggregation, two terms that depend on the *candidate* (not per‑anchor) are added:

**Spatial distance penalty.** Candidates far from the current *search center* $(s_{cx}, s_{cy})$ are penalized with a Gaussian‑shaped falloff (doc 01 §3.3):

$$
\text{cand\_score} \mathrel{-}= \lambda_{\text{dist}}\Big(1 - e^{-\frac{1}{2}(d/\sigma)^2}\Big),\qquad d = \big\|(c_x^c, c_y^c) - (s_{cx}, s_{cy})\big\|.
$$

The bracket is $0$ at the search center (no penalty) and rises toward $\lambda_{\text{dist}}$ far away — a soft "the target should be near where we're looking" prior, with $\sigma$ setting how forgiving it is.

**Candidate‑direction term.** Like §4.3 but for the candidate itself: reward candidates lying in the direction of the target's velocity from the reference,

$$
\text{cand\_score} \mathrel{+}= \lambda_{\text{cand\_dir}}\cdot\frac{\mathbf v\cdot\mathbf m_c}{\|\mathbf v\|\,\|\mathbf m_c\| + \varepsilon},\qquad \mathbf m_c = (c_x^c - c_x^{\text{ref}},\ c_y^c - c_y^{\text{ref}}).
$$

(Here the cosine is *not* clamped to $\ge 0$: a candidate *behind* the motion is actively penalized, since the target was moving forward.)

### 4.8 Acceptance, ranking, and the skip shortcut

A candidate is kept only if its final composite score exceeds the **margin** $\mu$ (a threshold; can be adapted per‑video, doc 10). Survivors are sorted descending. Two outputs:

- If the **best** candidate's score is at least a high **skip threshold** (e.g. $0.80$), return *only* it — it is so good there is no need to verify alternatives.
- Otherwise, return the **top‑$k$** candidates for the verification stage (doc 08 will re‑run the matcher on each).

So `drm_match` is a *ranked shortlist generator*: appearance + overlap + motion + recency + spatial plausibility, summarized into one number per candidate, thresholded by a margin, with a fast path when one candidate is obviously the target.

```
   candidates ──► for each: combine over DRM anchors
                     s_iou + s_app + (λ_mot·π + λ_time·e^{−α·age} − γ·pen)
                     ──aggregate(max / top‑k mean)──► base score
                     − λ_dist·(1 − e^{−½(d/σ)²})        (far from search center?)
                     + λ_cand_dir·cos(v, m_c)            (along the velocity?)
                  ──► keep if > margin ──► sort ──► (best ≥ skip? return it) else top‑k
```

### 4.9 Scoring the *live* target (for auto‑tuning)

There is a sibling method, `score_target_against_drm`, that runs the *same composite formula* on the **current, genuine target** as if it were a re‑acquisition candidate. Why score something you already have? Because the *value* of that score tells you how high the true target typically scores against its own DRM — which is exactly the information needed to set the acceptance **margin** automatically (doc 10's `AutoDrmMargin`). It skips the spatial‑distance and candidate‑direction terms (degenerate for the live target, whose search center coincides with itself) and returns the composite, so the number is on the same scale as `margin`. This is a nice design motif: *measure how the real target scores under your own rule, and set the threshold just below that.*

---

## 5. How the pieces relate

```
   confident tracking frame
        │ (box, descriptor)
        ▼
   RAM.try_admit ── IoU gate + area gate ──► RAM buffer (short‑term, recent looks)
        │  on admit
        ▼
   promote? ── agreement ≥ m_min over window W ──► DRM bank (long‑term anchors)
                                                       │
   ... target lost (occlusion) ...                     │
        │ detector proposes candidates                 ▼
        └──────────────► drm_match: composite score over DRM anchors
                          (appearance + IoU + motion + recency + spatial)
                          ──► ranked shortlist ──► verification (doc 08)
```

- **RAM** = short‑term, geometry‑gated, raw recent exemplars.
- **DRM** = long‑term, agreement‑curated, stable anchors that survive an occlusion.
- **`drm_match`** = the composite ranker that turns "here are some candidate boxes" into "here is the one that is most plausibly the same target."

---

## 6. Recap

- Appearance is summarized by an OSNet **descriptor**; comparisons use **cosine similarity**.
- **RAM** admits confident frames through an **IoU gate** and a **median‑area gate** (geometry, not appearance), keeping a short window of recent (box, descriptor) pairs.
- **DRM** promotes a RAM entry only when it **agrees** (cosine $\ge\tau_{\text{sim}}$) with at least $m_{\min}$ of its recent neighbors — a curated, stable, long‑term anchor bank.
- The **composite re‑acquisition score** sums an **overlap** term, an **appearance** term, a **motion‑direction** term ($\max(0,\cos)$ of velocity vs. displacement), and an exponential **recency** term per anchor; aggregates (max / top‑$k$ mean); then applies a Gaussian **distance** penalty and a **candidate‑direction** term; accepts above a **margin**; and returns the best (skip shortcut) or a top‑$k$ shortlist.
- `score_target_against_drm` runs the same formula on the live target to drive the **adaptive margin** (doc 10).

Next: **`08_OCCLUSION_RECOVERY.md`** — how loss is *detected*, how the EKF *coasts* and steers a growing search, how the detector + DRM + matcher cooperate across phases to find the target again, and how a winner is *verified* before tracking resumes.
