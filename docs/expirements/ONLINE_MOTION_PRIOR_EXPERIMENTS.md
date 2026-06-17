# Online Kalman Motion Prior — Experiment Report

**Author:** Claude (Fable 5), code-driven
**Date:** 2026-06-10
**Status:** Concluded.
- **`kf_motion` score fusion** (SAMURAI-style KF-IoU blend at peak selection) — **WIN (+0.0054)**. Shipped ON.
- **`kf_motion_center_search`** (KF-predicted search-crop centering) — **WIN (+0.0047)**. Shipped ON.
- **`drm_lam_dist: 0.3`** (motion-distance bias in occlusion-recovery DRM) — **net negative (−0.0114)**. Reverted to 0.0; documented below.
- Weight/engagement sweep (`w=0.2, stable=8`) — tie (0.7838 vs 0.7836). Kept conservative defaults.

**Source paper:** Yang et al., *"SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory"* (arXiv:2411.11922), Sec. 4.1 — KF-IoU candidate-score fusion, `α_kf ≈ 0.15–0.2`, stability-gated. Search-crop centering additionally motivated by FocusTrack (arXiv:2504.13604) and standard MOT predict-then-associate.

---

## 0. TL;DR

| Configuration | Final score | Δ | Verdict |
|---|---|---|---|
| Baseline (held-box + frame_dynamics, online-legal) | 0.7735 | — | reference (AUC 0.7365 / NP 0.8291) |
| + KF-IoU score fusion (`w=0.15`, stable=15) | **0.7789** | +0.0054 | **WIN. Shipped.** |
| + DRM `lam_dist: 0.3` (on top of fusion) | 0.7675 | −0.0114 | **FAIL. Reverted.** |
| + KF-centered search crop (on top of fusion) | **0.7836** | +0.0047 | **WIN. Shipped.** |
| fusion `w=0.2`, stable=8 (aggressiveness probe) | 0.7838 | +0.0002 | tie/noise — kept w=0.15, stable=15 |

(Final score = `0.6·AUC + 0.4·NormPrec` over datasets 2–5, 79 scored sequences. All changes are **online/causal** and cost ~0 compute: 119–146 fps unchanged, the fusion is 256 vectorized IoUs/frame plus one 8×8 Kalman step.)

**Cumulative shipped result: 0.7735 → 0.7836 (+0.0101), fully real-time-legal.**

## 1. The gap these features fill

With `bbox_smoothing_enabled: false` (the tuned competition config), `_confidence_postprocess()` early-returns the raw sigmoid score map — **no cosine window, no scale/aspect penalty is ever applied**. Peak selection therefore had *zero* spatial or motion prior: any off-center cell whose appearance score edged out the true target's won instantly. This is exactly the identity-switch failure class (car16_3's row of white parked cars, duck2, crowds).

## 2. Mechanism

New module [`models/SiamABC/tracker/kalman_motion.py`](../../models/SiamABC/tracker/kalman_motion.py): constant-velocity Kalman filter over `[cx, cy, w, h, vx, vy, vw, vh]` with MOT-standard h-scaled noise. Hooks in [`SiamABC_Tracker.py`](../../models/SiamABC/tracker/SiamABC_Tracker.py):

1. **`update()`**: `_kf_motion_begin_frame()` advances the filter (or re-seeds it when something external — e.g. a reacquisition commit — rewrote `tracking_state.bbox`); after `run_track()`, `_kf_motion_observe()` feeds the selected box back as a measurement when `score ≥ kf_motion_score_threshold`, else the stability streak resets and the filter coasts.
2. **`_postprocess()`**: before `box_coder.decode()`, the KF-predicted box (mapped into search-crop coordinates) is IoU'd against all 256 grid-cell candidate boxes and blended: `fused = (1−w)·cls_map + w·KF_IoU` (SAMURAI Eq. 7). The *reported* peak score still reads from the raw appearance map, so occlusion-entry thresholds see unchanged semantics.
3. **`run_track()`**: with `kf_motion_center_search`, the search crop is centered at the KF-predicted position (size untouched), keeping fast movers inside the crop.
4. **`run_track_for_candidate()`**: fusion and centering are **suppressed** — recovery hypothesis verification must be appearance-only, since the correct reacquisition is often far from the stale trajectory.

Gating: active only after `kf_motion_stable_frames` consecutive confident frames; a single low-score frame disengages it. Re-seed on innovation > `kf_motion_reseed_dist`× box diagonal.

Config (all under `tracker:` in [`inference_config_experimental.yaml`](../../config/inference_config_experimental.yaml); `kf_motion_enabled: false` restores byte-for-byte legacy behaviour):

```yaml
kf_motion_enabled: true
kf_motion_weight: 0.15
kf_motion_stable_frames: 15
kf_motion_score_threshold: 0.55
kf_motion_reseed_dist: 2.0
kf_motion_center_search: true
```

## 3. Per-sequence effects (cumulative, vs 0.7735 baseline)

Wins: **car16_3 0.28 → 0.85** (white-cars identity switch eliminated), duck2 0.57 → 0.77, human 0.72 → 0.81, air_conditioning_box1 +0.03, bus1-n +0.03, car6_2 +0.03, excavator +0.01. person19_3 transiently regressed in the fusion-only run (−0.29; recovery committed to a wrong person) and fully recovered with crop centering (back to 0.66). Residual losses ≤ 0.027 (parterre2, group2_1).

## 4. Failed branch: DRM motion-distance bias (don't repeat)

person19_3 diagnosis showed the occlusion recovery committing at 0.99 confidence to a person 290 px from the EKF prediction while the true target reappeared ~20 px from it. Enabling the existing-but-zeroed `drm_lam_dist: 0.3` (penalty `−λ·(1−exp(−½(d/σ)²))`, EKF-uncertainty-adaptive σ) fixed person19_3 (+0.29) and group2_1 (+0.06) **but broke RcCar4 (−0.52), truck (−0.29), jogging2 (−0.28)** — sequences whose targets genuinely reappear far away. Net −0.0114, reverted.

**The durable lesson (now confirmed three ways with BackTrack and DRM-introspection):** motion priors are safe where their per-frame influence is bounded and reversible (normal-tracking peak selection — a wrong nudge costs one frame), and toxic where a single decision is irreversible (recovery commitment). Bias the steering, never the teleporter.

## 5. Remaining headroom (census from this session's diagnostics)

- **Animal3 (0.21)** — *not* a center switch: gradual scale bloat to ~2.5× GT height absorbing neighbouring llamas. With `smooth: false` no scale/aspect penalty exists, and the per-frame growth (~0.5%/frame) is too slow for any change-rate penalty to catch. Needs a size-anchored regression prior; no clean causal fix found.
- **uav7 (0.25) / uav2 (0.11)** — tracked cleanly until an abrupt ~1-box-width/frame lateral jump (camera pan), then frozen on background. Constant-velocity prediction cannot anticipate the reversal; the crop centering doesn't fire (score stays high on the wrong lock). Needs a tiny-target motion detector (frame-difference peaks gated to empty backgrounds) — note the offline STLFD attempt already failed on beach clutter.
- **group2_1 (0.44–0.52)** — fluctuates between identical people; saturated for appearance+motion.

## 6. How to reproduce

```
# shipped result (0.7836):
.venv/Scripts/python.exe run_inference.py
.venv/Scripts/python.exe submission_to_sequence_metrics.py --submission submission.csv

# baseline (0.7735): set tracker.kf_motion_enabled: false
# fusion-only (0.7789): kf_motion_enabled: true, kf_motion_center_search: false
# DRM failure repro (0.7675): + drm_lam_dist: 0.3, distractor_occ_drm_lam_dist: 0.3
```
