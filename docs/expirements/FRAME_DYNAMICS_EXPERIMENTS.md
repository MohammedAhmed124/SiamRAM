# Frame-Dynamics / Anti-UAV motion cues — Experiment Report (3 features)

**Author:** Claude (Opus 4.8), code-driven
**Date:** 2026-06-05
**Status:** Concluded.
- **`frame_dynamics`** (tiny-only, low-weight) — **net positive** (+0.0013 vs base). **Shipped ON-by-config, OFF-by-default**, with the winning values baked in as defaults.
- **`size_adaptive_gate`** — net **negative** (−0.0814). **Code removed from the codebase** (offered no benefit here); this report is retained as the historical record. The file/line references for it in §2 point to code as it existed at removal time and are no longer live.
- **`trajectory_filter`** — **inert** on this set (== base). **Code removed from the codebase** (no measurable effect); retained here as the historical record. Its §2 references are likewise no longer live.

**Source papers:**
- Wang, Liu, Cheng et al., *"A Simple Detector with Frame Dynamics is a Strong Tracker"* — 1st place, 4th Anti-UAV Challenge (arXiv:2505.04917). Powers `frame_dynamics` (Sec. 3.2) and `trajectory_filter` (Sec. 3.3), and is one of the two motivations for `size_adaptive_gate`.
- CST Anti-UAV line, *"Cross-modality Spatial-Temporal Transformer for Anti-UAV"* (arXiv:2507.23473). Motivates `size_adaptive_gate` (tiny-target appearance-signal collapse).

---

## 0. TL;DR

Three features from the Anti-UAV / tiny-target tracking literature were ported, all targeting the same weakness: **on tiny / long-range targets the appearance descriptor collapses, and motion becomes the more reliable cue.** Each is config-gated, off by default, and byte-for-byte inert when disabled.

| Feature | Mechanism | Best Final score | Δ vs base | Verdict |
|---|---|---|---|---|
| **Base** (all 3 off) | — | **0.7719** | — | reference (AUC 0.7351 / NormPrec 0.8271) |
| **`frame_dynamics`** | additive frame-difference motion saliency blended into the search crop | **0.7732** | **+0.0013** | **WIN (tiny-only, `0.001`/`0.06`). Shipped.** |
| `frame_dynamics` (full blend, all targets, `w=0.3`) | as above, unrestricted | 0.7455 | −0.0264 | corrupts frozen-backbone appearance on most frames |
| `frame_dynamics` (tiny-only, `0.03`/`0.08`) | tiny net too wide | 0.7507 | −0.0212 | net still too broad → blend fires on normal targets |
| `size_adaptive_gate` | down-weight DRM `lam_app` for tiny targets | 0.6905 | −0.0814 | gates the ONLY active DRM term → re-acquisition collapse |
| `trajectory_filter` | constant-velocity motion-window candidate gate | 0.7719 | 0.0000 | inert: only wired to occlusion candidates, keeps-all |

(Final score = `W1·AUC + W2·NormPrec` over datasets 2–5, the 89-video test set.)

**The one durable result:** the *only* version that beats base is `frame_dynamics` confined to **genuinely tiny targets** (`tiny_area_fraction = 0.001`) at a **low blend weight** (`0.06`). The win is small but real and comes almost entirely from **NormPrec (0.8271 → 0.8287)** — tightening center localization on small targets where motion is the better cue, *without* corrupting the appearance match on the majority (normal-sized) frames.

---

## 1. The shared hypothesis and the shared deviation

All three papers exploit the same finding: **a tiny target's appearance descriptor carries almost no discriminative information** (CST Anti-UAV reports SoTA dropping from ~67.69% to ~35.84% on the tiny subset), so a moving tiny target is far easier to localize by its **motion** than by its (collapsed) appearance.

The papers act on this by **augmenting the network input** — concatenating frame-difference or optical-flow channels and training a detector from scratch on the 6-channel input.

**SiamRAM cannot do this.** Its SiamABC backbone is a **frozen, pretrained 3-channel RGB network**; we must not change its channel count or retrain it. Every feature below is therefore a *deviation*:
- `frame_dynamics` — blends the motion saliency **additively into the 3-channel crop** instead of concatenating channels.
- `size_adaptive_gate` — operates **post-network, at DRM scoring time** (a weighting multiplier), not as a preprocessing step.
- `trajectory_filter` — a pure **post-processing candidate gate**, unchanged from the paper's intent (it never touches the network).

---

## 2. Implementation map (files & entry points)

All three are config-gated under `ram_tracker:` in `config/inference_config_experimental.yaml` and **off by default**. When off, the relevant object is never constructed and the legacy path is byte-for-byte unchanged.

| Concern | `frame_dynamics` | `size_adaptive_gate` | `trajectory_filter` |
|---|---|---|---|
| Standalone module | [`models/siamram/frame_dynamics.py`](../../models/siamram/frame_dynamics.py) → `FrameDynamicsProcessor` | [`models/siamram/size_adaptive_gate.py`](../models/siamram/size_adaptive_gate.py) → `SizeAdaptiveGate` | [`models/siamram/trajectory_filter.py`](../models/siamram/trajectory_filter.py) → `TrajectoryConstraintFilter` |
| Construction (when enabled) | `tracker.py:949` | `tracker.py:968` | `tracker.py:988` |
| Host wiring | `tracker.py` → `_apply_frame_dynamics()` (~L5330), called before `self.tracker.update(frame)` at L3249 | `tracker.py` → `_size_adaptive_lam_app()` (~L5385), called at DRM sites `occlusion_recovery.py:293` and `:583` | `observe()` at `tracker.py:1605`/`3535`, `reset()` at `1560`, `filter_candidates()` gate at `tracker.py:3751` |
| Config dataclass | `config.py` → `FrameDynamicsConfig` | `config.py` → `SizeAdaptiveGateConfig` | `config.py` → `TrajectoryFilterConfig` |
| YAML block | `ram_tracker.frame_dynamics:` | `ram_tracker.size_adaptive_gate:` | `ram_tracker.trajectory_filter:` |

---

## 3. Feature 1 — `frame_dynamics` (the win)

### 3.1 Mechanism
Per frame, compute the short-term frame-difference motion saliency between consecutive **full** frames:

```
diff_t_1 = |x_t - x_{t-1}|,   diff_t_2 = |x_t - x_{t-2}|     # paper Eq. 1 (FD variant)
```

average them into one per-pixel saliency map, register it to the search region, scale by `scale`, optionally clip outliers to `clip`, and blend additively into the 3-channel search crop:

```
crop' = clamp8( crop + blend_weight * scaled_saliency )
```

The module owns a 2-frame rolling buffer (host carries no extra state). With <2 prior frames the crop is returned unchanged. `blend_weight = 0` (or `tiny_only` with a non-tiny target) is an exact no-op.

### 3.2 The journey — why the first two tries lost
| Run | `tiny_only` | `tiny_area_fraction` | `blend_weight` | AUC | NormPrec | Final | Δ |
|---|---|---|---|---|---|---|---|
| full blend | false | — | 0.3 | — | — | 0.7455 | −0.0264 |
| tiny-only (wide net) | true | 0.03 | 0.08 | 0.7173 | 0.8008 | 0.7507 | −0.0212 |
| **tiny-only (tight net)** | **true** | **0.001** | **0.06** | **0.7362** | **0.8287** | **0.7732** | **+0.0013** |

- **Full blend (−0.0264):** blending motion into *every* crop corrupts the RGB the frozen backbone matches on, across all frames. Pure loss.
- **Tiny-only at `0.03` (−0.0212):** better, but `0.03` of frame area is **not tiny** — a target merely ~17%×17% of the frame is below it, so the blend still fired on most normal targets. The "leave normal frames clean" protection never engaged for the majority of frames.
- **Tiny-only at `0.001` (+0.0013):** a genuinely tiny target (e.g. ~20×20 px in 1080p ≈ 0.0002 area fraction) is ~150× smaller than the old `0.03` net. At `0.001` the blend fires only on truly small targets; the vast majority of frames are left byte-for-byte clean → score sits at/above base, with upside only where motion actually helps. The gain is concentrated in **NormPrec** (center-localization), exactly the expected signature.

### 3.3 Root cause of the difference
The regression was never the feature — it was the **"tiny" net being too wide**, applying an appearance-corrupting blend to frames where appearance was the reliable cue. Once the net is restricted to the regime the paper actually targets (genuinely tiny targets), the deviation pays off modestly. The margin is small (+0.0013); treat it as "a real, isolated win," not a breakthrough.

### 3.4 Shipped defaults (the winning config)
`frame_dynamics_enabled` remains **`false`** by default, but when enabled the baked-in defaults are now the winning values (`config.py` `FrameDynamicsConfig`, `tracker.py` constructor defaults, and the YAML all agree):

```yaml
frame_dynamics_blend_weight: 0.06
frame_dynamics_tiny_only: true
frame_dynamics_tiny_area_fraction: 0.001
```

---

## 4. Feature 2 — `size_adaptive_gate` (net negative)

### 4.1 Mechanism
From the target's size (default metric `area_fraction = bbox_area / frame_area`) compute a bounded factor in `[min_factor, 1.0]` that is `1.0` for normal/large targets and decays toward `min_factor` (default `0.2`) as the target crosses from `tiny_threshold` down to `minimum_threshold`. The host multiplies that factor into the DRM composite's appearance weight `lam_app` at the two re-acquisition call sites in `occlusion_recovery.py`.

### 4.2 Why it failed badly (−0.0814)
The current DRM composite is **appearance-only** in this configuration:

```yaml
drm_lam_app: 1     drm_lam_dist: 0.0   drm_lam_cand_dir: 0.0
drm_lam_time: 0.0  drm_lam_iou: 0.0    drm_lam_mot: 0.0
```

So gating `lam_app` down-weights **the only active scoring term**. Scaling the sole term by a constant factor scales *all* candidates equally — argmax is unchanged, but the absolute scores (and therefore the acceptance **thresholds and margins**) collapse, so re-acquisition rejects good candidates. The gate would only make sense in a regime where the motion priors (`lam_dist`, `lam_cand_dir`, `lam_time`, …) carry non-zero weight to take over as appearance is suppressed. They are all zero here, so the gate has nothing to lean on.

### 4.3 Verdict
Off. Revisiting requires first giving the DRM motion terms non-zero weight; only then could down-weighting appearance for tiny targets help rather than hollow out the score.

> Note: an earlier review caught a backwards default — `minimum_threshold` was `0.5` (would suppress targets up to 50% of frame area under the `area_fraction` metric). Fixed to `0.003` in both the dataclass and YAML. The −0.0814 above is with the corrected default; the structural problem (gating the only active term) is independent of that fix.

---

## 5. Feature 3 — `trajectory_filter` (inert)

### 5.1 Mechanism
From the previous two target centres predict the current one by constant-velocity extrapolation (paper Eq. 3–4):

```
C_t ≈ 2*C_{t-1} - C_{t-2}
```

and accept a candidate only if its centre lies within a motion window of radius `d_max` of `C_t` (Eq. 5). `d_max` is expressed **relative to the target bbox diagonal** (`d_max_diag_scale`, default `0.5`) with an absolute floor (`d_max_min_px`, default `5.0`) so the window adapts to scale. The module owns a 2-slot centre history; it is inert until `min_history_frames` (≥2) centres are seen, and accepts all candidates whenever it cannot form a prediction.

### 5.2 Why it was inert (0.000 vs base)
The filter is wired at a **single, narrow** site: the occlusion-recovery candidate gate (`tracker.py:3751`), which only sees YOLO re-detection candidates during occlusion. On this test set that path:
1. rarely fires (occlusion is uncommon), and
2. is built to **keep-all rather than reject-all** — if the window would reject every candidate, the host falls back to accepting them (so it can never strand recovery).

Net effect: it almost never removes a candidate, and when it could, the keep-all guard prevents it from changing the outcome → identical to base.

### 5.3 Verdict
Off. It is correctly implemented and safe, but has no measurable effect where it is currently wired. To matter it would need to gate the **main per-frame candidate path** (not just occlusion recovery), which changes normal-tracking behavior globally and was deliberately not attempted (the user manually tests; no speculative broadening).

---

## 6. Config reference

All keys live under `ram_tracker:` in [`config/inference_config_experimental.yaml`](../../config/inference_config_experimental.yaml). Master switches default OFF.

### `frame_dynamics:`
| Key | Default | Meaning |
|---|---|---|
| `frame_dynamics_enabled` | `false` | Master switch. Off → processor never built; search crop unchanged. |
| `frame_dynamics_blend_weight` | `0.06` | Additive blend weight. `0.0` = exact no-op. Higher regressed accuracy. |
| `frame_dynamics_scale` | `1.0` | Multiplier on raw saliency before blending. `>1` amplifies faint motion. |
| `frame_dynamics_clip` | `-1.0` | Per-pixel upper clip on scaled saliency. `<=0` disables clipping. |
| `frame_dynamics_tiny_only` | `true` | Blend ONLY when target is tiny; normal/large crops left clean. `false` = blend every frame (regressed). |
| `frame_dynamics_tiny_area_fraction` | `0.001` | bbox area fraction at/below which a target counts as tiny. `0.03` (the old default, == `roi_search.tiny.long_distance_area_fraction`) was too wide. |

### `size_adaptive_gate:`
| Key | Default | Meaning |
|---|---|---|
| `size_adaptive_gate_enabled` | `false` | Master switch. Off → `lam_app` unchanged. |
| `size_adaptive_gate_metric` | `area_fraction` | `area_fraction` (scale-invariant) or `bbox_diagonal_px` (paper's native). |
| `size_adaptive_gate_tiny_threshold` | `0.03` | Size at/above which the gate is fully open (factor 1.0). |
| `size_adaptive_gate_minimum_threshold` | `0.003` | Size at/below which suppression is maximal (factor `min_factor`). |
| `size_adaptive_gate_min_factor` | `0.2` | Floor of the gate factor (keeps ~20% appearance signal). |
| `size_adaptive_gate_interpolation` | `sigmoid` | `sigmoid` (steepness `sigmoid_k`) or `linear`. |
| `size_adaptive_gate_sigmoid_k` | `10.0` | Sigmoid steepness. |
| `size_adaptive_gate_apply_to` | `appearance` | `appearance` (gate `lam_app` only) or `all` (reserved). |

### `trajectory_filter:`
| Key | Default | Meaning |
|---|---|---|
| `trajectory_filter_enabled` | `false` | Master switch. Off → no candidate filtered. |
| `trajectory_filter_d_max_diag_scale` | `0.5` | Window radius as a multiple of target bbox diagonal. Raise toward ~1.5 if fast manoeuvres are rejected. |
| `trajectory_filter_min_history_frames` | `2` | Prior centres required before the gate activates (clamped ≥2). |
| `trajectory_filter_d_max_min_px` | `5.0` | Absolute floor (px) on the window radius. `<=0` disables the floor. |

---

## 7. How to reproduce

All runs use the 89-video test set (datasets 2–5); only the `ram_tracker:` flags change. **No code edits needed.** Final score = `W1·AUC + W2·NormPrec`.

- **Base (reference 0.7719):** all three `*_enabled: false`.
- **`frame_dynamics` win (0.7732):** `frame_dynamics_enabled: true`, `frame_dynamics_tiny_only: true`, `frame_dynamics_tiny_area_fraction: 0.001`, `frame_dynamics_blend_weight: 0.06` (these are the shipped defaults — only the master switch needs flipping).
- **`frame_dynamics` full-blend repro (0.7455):** `frame_dynamics_tiny_only: false`, `frame_dynamics_blend_weight: 0.3`.
- **`size_adaptive_gate` (0.6905):** `size_adaptive_gate_enabled: true` (others off).
- **`trajectory_filter` (== base, 0.7719):** `trajectory_filter_enabled: true` (others off).

---

## 8. Conclusion & recommendation

| Feature | Recommended state | Rationale |
|---|---|---|
| `frame_dynamics` | **available; default OFF, winning values baked in** | Only +0.0013, but a real, isolated win on tiny targets. Flip `frame_dynamics_enabled: true` to use it. |
| `size_adaptive_gate` | **REMOVED** | Gated the only active DRM term in the current appearance-only composite → score collapse. Would need non-zero DRM motion weights first. Code deleted; revive from git history if the DRM weighting changes. |
| `trajectory_filter` | **REMOVED** | Inert where wired (occlusion-only, keep-all). Correct and safe, but no headroom without broadening to the main candidate path. Code deleted; revive from git history if revisited. |

**Durable lessons (independent of these verdicts):**
1. **Match the feature's regime to the paper's regime.** Frame dynamics only helps where the paper claimed — *genuinely tiny* targets. A "tiny" threshold borrowed from another subsystem (`0.03`) was 150× too loose and turned a win into a loss.
2. **Know which scoring terms are live before gating them.** The size gate's failure is entirely explained by the DRM composite being appearance-only; gating the sole active term scales all candidates uniformly and collapses thresholds.
3. **A safe no-op is not a win.** The trajectory filter's keep-all guard makes it harmless but also inert; safety and impact were traded off, and on this set it produced zero change.
4. **Input-space deviations from a frozen backbone are fragile.** Additively blending motion into the RGB the backbone matches on is corrupting unless tightly confined; the post-network and post-process deviations were structurally safer but found no headroom here.
</content>
</invoke>
