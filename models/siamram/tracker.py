"""
SiamRAM Tracker: A robust hybrid tracker with occlusion recovery.

This module implements the SiamRAMTracker, which combines Siamese visual
tracking (SiamABC) with a YOLO-based re-detection system and an Extended
Kalman Filter (EKF) for motion modelling. It features a Dynamic Reference
Memory (DRM) for reliable re-acquisition after long-term occlusion.
"""

import atexit
import os
import weakref
from collections import deque
from typing import List, Optional, Tuple, TypedDict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.console import siamram_log
from utils.utils import (
    _cos_sim,
    _extract_descriptor,
    _extract_descriptor_async,
    _iou,
    _resolve_descriptor,
    configure_descriptor_backend,
    descriptor_async_supported,
)

from ..SiamABC.tracker.SiamABC_Tracker import SiamABCTracker
from .camera_motion import CameraMotionSubsystem
from .distractor_mode import DistractorModeSubsystem
from .memory import AppearanceMemory
from .motion import BBoxEKF
from .occlusion_recovery import OcclusionRecoverySubsystem
from .spike_watcher import SpikeWatcher
from .tracker_state import (
    CandidateRecord,
    FrameHistory,
    FrameRecord,
    VisualState,
)


# ---------------------------------------------------------------------------
# Research-telemetry aggregation (DRM introspection).
#
# run_inference.py builds a FRESH tracker per clip by default (only
# --reuse_tracker reuses a single wrapper across clips), so a per-instance
# cumulative tally cannot span videos and per-instance atexit would emit one
# fragmented line per clip. These module-level globals accumulate counts across
# every tracker instance in BOTH modes, so a single aggregate prints once at
# process exit. Live instances are held by weakref so fresh per-clip trackers
# garbage-collect normally (no leak) after they emit their video at the clip
# boundary; only the still-alive tracker(s) are flushed for residual at exit.
# ---------------------------------------------------------------------------
_TLM_KEYS: Tuple[str, ...] = (
    "occlusion_entries_total",
    "drm_secondary_peak_frames",
    "drm_divergence_below_theta",
    "drm_should_update_true",
    "drm_anchors_added",
    "drm_distractor_anchors_added",
    "drm_gate_calls",
    "drm_gate_block_not_present",
    "drm_gate_block_not_due",
    "drm_gate_block_reliability",
    "drm_gate_block_size",
    "drm_gate_block_divergence",
    "drm_gate_block_secondary",
    "drm_gate_pass",
)
_TLM_GLOBAL_TOTAL: Optional[dict] = None
_TLM_GLOBAL_VIDEOS: int = 0
_TLM_LIVE_INSTANCES: "weakref.WeakSet" = weakref.WeakSet()
_TLM_DUMP_REGISTERED: bool = False
# Compact echo of which research features were actually active (and their key
# params), captured once from the first telemetry-enabled tracker so the
# module-level process-exit aggregate (_tlm_global_dump) can print it even though
# it holds no tracker reference. Stays None until set; rendered as "flags(?)" if
# never captured so the dump never crashes.
_TLM_ACTIVE_FLAGS: Optional[str] = None


def _tlm_flags_str() -> str:
    """Render the active-feature-flag echo, robust to an uncaptured value."""
    return _TLM_ACTIVE_FLAGS if _TLM_ACTIVE_FLAGS else "flags(?)"


def _tlm_format(counts: dict) -> str:
    """Render one counter dict as the compact DRM telemetry string."""
    c = counts
    return (
        f"{_tlm_flags_str()} "
        f"occ_entries={c['occlusion_entries_total']} "
        f"DRM(gate_calls={c.get('drm_gate_calls', 0)} "
        f"pass={c.get('drm_gate_pass', 0)} "
        f"[not_present={c.get('drm_gate_block_not_present', 0)} "
        f"not_due={c.get('drm_gate_block_not_due', 0)} "
        f"reliab={c.get('drm_gate_block_reliability', 0)} "
        f"size={c.get('drm_gate_block_size', 0)} "
        f"diverg={c.get('drm_gate_block_divergence', 0)} "
        f"sec={c.get('drm_gate_block_secondary', 0)}] "
        f"sec_peak={c['drm_secondary_peak_frames']} "
        f"div<theta={c['drm_divergence_below_theta']} "
        f"should_update={c['drm_should_update_true']} "
        f"anchors={c['drm_anchors_added']} "
        f"dist_anchors={c.get('drm_distractor_anchors_added', 0)})"
    )


def _tlm_add_video(counts: dict) -> int:
    """Fold one finished video's counts into the module-global total.

    Returns the running global video index (1-based) for per-video labelling, so
    numbering stays consistent whether one tracker is reused or one is built per
    clip.
    """
    global _TLM_GLOBAL_TOTAL, _TLM_GLOBAL_VIDEOS
    if _TLM_GLOBAL_TOTAL is None:
        _TLM_GLOBAL_TOTAL = {k: 0 for k in _TLM_KEYS}
    for k in _TLM_KEYS:
        _TLM_GLOBAL_TOTAL[k] += int(counts.get(k, 0))
    _TLM_GLOBAL_VIDEOS += 1
    return _TLM_GLOBAL_VIDEOS


def _tlm_global_dump() -> None:
    """atexit hook: flush any residual per-instance video, then print the total.

    Registered exactly once, when the first telemetry-enabled tracker is built.
    First it asks every still-live tracker (the reused wrapper, or the final
    fresh per-clip wrapper) to emit a last unflushed video; earlier per-clip
    trackers already emitted at their clip boundary and have been GC'd, so they
    are correctly excluded. Then it prints one aggregate line. Never raises
    (atexit handlers must not during interpreter shutdown).
    """
    try:
        for inst in list(_TLM_LIVE_INSTANCES):
            try:
                inst._emit_video_telemetry()
            except Exception:
                pass
        if _TLM_GLOBAL_TOTAL is None or _TLM_GLOBAL_VIDEOS == 0:
            return
        siamram_log(
            f"[research telemetry] FINAL cumulative over {_TLM_GLOBAL_VIDEOS} "
            f"video(s) {_tlm_format(_TLM_GLOBAL_TOTAL)}"
        )
    except Exception:
        pass


class DRMKwargs(TypedDict):
    """
    Keyword arguments for the Dynamic Reference Memory (DRM) matching logic.

    Attributes:
        lam_iou (float): Weight for Intersection-over-Union similarity.
        lam_app (float): Weight for appearance (cosine) similarity.
        lam_mot (float): Weight for motion consistency.
        lam_time (float): Weight for temporal decay.
        alpha (float): Temporal decay rate.
        gamma (float): Distractor penalty weight.
        margin (float): Minimum score margin over distractors.
        top_k (int): Number of top candidates to consider.
        skip_threshold (float): Score above which re-verification is skipped.
        lam_dist (float): Weight for spatial distance penalty.
        lam_cand_dir (float): Weight for candidate direction consistency.
    """

    lam_iou: float
    lam_app: float
    lam_mot: float
    lam_time: float
    alpha: float
    gamma: float
    margin: float
    top_k: int
    skip_threshold: float
    lam_dist: float
    lam_cand_dir: float


class SiamRAMExperimentTracker:
    """
    SiamRAM Tracker.

    A robust hybrid tracker that combines SiamABC for short-term tracking
    with a YOLO-based re-detection system and an Extended Kalman Filter (EKF)
    for occlusion recovery and motion modeling.

    This class orchestrates tracking by calling the underlying SiamABCTracker
    methods like `update()` and `run_track_for_candidate()`, while managing
    its own appearance memory and states to handle object disappearance.
    """
    _MAX_PROC_LONG_EDGE: int = 1280

    def __init__(
        self,
        siam_tracker: SiamABCTracker,
        yolo_weights: str = "yolo11n.pt",
        descriptor_backend: str = "osnet",
        osnet_model_name: str = "osnet_x1_0",
        osnet_model_path: str = "",
        osnet_pretrained_checkpoint: str = "imagenet",
        osnet_device: str = "auto",
        osnet_fp16: bool = False,
        trt_compile_osnet: bool = False,
        osnet_async_overlap: bool = False,
        trt_cache_dir: str = "",
        trt_rebuild_cache: bool = False,
        siamese_feature_source: str = "neck",
        siamese_comparison_mode: str = "xcorr",
        descriptor_stride: int = 1,
        conf_threshold: float = 0.60,
        occ_siam_reacq_threshold=0.8,
        reacq_threshold: float = 0.55,
        reacq_confirm_frames: int = 1,
        yolo_conf: float = 0.30,
        yolo_iou: float = 0.45,
        yolo_imgsz: int = 320,
        yolo_augment: bool = False,
        app_match_threshold: float = 0.72,
        occ_siam_margin=0.2,
        nudge_alpha: float = 0.30,
        tau_occ: float = 0.40,
        beta: float = 0.06,
        mem_capacity: int = 20,
        tau_iou: float = 0.40,
        tau_area: float = 0.25,
        ncc_threshold: float = 0.70,
        ncc_expand: float = 2.5,
        conf_history_len: int = 200,
        history_decay: float = 0.5,
        history_skip_last: int = 2,
        yolo_search_expand: float = 5.0,
        roi_start_expand: float = 1.5,
        size_history_len: int = 40,
        drm_capacity: int = 8,
        drm_tau_sim: float = 0.85,
        drm_window_W: int = 10,
        drm_mmin: int = 3,
        drm_lam_iou: float = 0.40,
        drm_lam_app: float = 0.30,
        drm_lam_mot: float = 0.20,
        drm_lam_time: float = 0.10,
        drm_alpha: float = 0.05,
        drm_gamma: float = 0.30,
        drm_margin: "float | str" = 0.35,
        drm_top_k: int = 3,
        drm_skip_threshold: float = 0.80,
        drm_lam_dist: float = 0.15,
        drm_dist_sigma_factor: float = 2.5,
        drm_lam_cand_dir: float = 0.15,
        drm_lam_cand_vel: float = 0.20,
        # Adaptive DRM acceptance margin, active only when drm_margin == "auto".
        # All defaults reproduce a fixed margin until enough samples accrue, and
        # a numeric drm_margin ignores every one of these knobs.
        drm_margin_auto_min: float = 0.55,
        drm_margin_auto_max: float = 0.80,
        drm_margin_auto_delta: float = 0.10,
        drm_margin_auto_ema_alpha: float = 0.2,
        drm_margin_auto_n_frames: int = 10,
        drm_margin_auto_warmup: float = 0.70,
        drm_margin_auto_min_samples: int = 5,
        # Separate occlusion-recovery DRM weights used only when occlusion was
        # entered via the distractor below-gate force path. Defaults mirror the
        # normal occlusion DRM params, so an unset block reproduces old behavior.
        distractor_occ_drm_lam_iou: float = 0.40,
        distractor_occ_drm_lam_app: float = 0.30,
        distractor_occ_drm_lam_mot: float = 0.20,
        distractor_occ_drm_lam_time: float = 0.10,
        distractor_occ_drm_alpha: float = 0.05,
        distractor_occ_drm_gamma: float = 0.30,
        distractor_occ_drm_margin: float = 0.35,
        distractor_occ_drm_top_k: int = 3,
        distractor_occ_drm_skip_threshold: float = 0.80,
        distractor_occ_drm_lam_dist: float = 0.15,
        distractor_occ_drm_lam_cand_dir: float = 0.15,
        distractor_occ_drm_dist_sigma_factor: float = 2.5,
        vel_score_min_speed: float = 0.5,
        ekf_process_noise: float = 2.0,
        ekf_meas_noise: float = 5.0,
        homo_max_corners: int = 200,
        homo_inlier_threshold: float = 0.50,
        homography_mode: str = "classic",
        shrinkage_min_drop_frac: float = 0.06,
        shrinkage_max_lookback: int = 60,
        velocity_lookback: int = 3,
        velocity_smooth_alpha: float = 0.4,
        distractor_bank_maxlen: int = 50,
        velocity_decay: float = 0.95,
        search_expand_growth_factor: float = 1.2,
        search_expand_growth_every: int = 5,
        search_expand_max: float = 15.0,
        long_distance_conf_threshold: float = 0.35,
        long_distance_area_fraction: float = 0.004,
        long_distance_mode: bool = False,
        enter_occlusion_on_loss: bool = True,
        no_occlusion_first_n_frames: int = 0,
        # Introspection-based DRM anchor update (arXiv:2411.17576, Sec. 3.2.2).
        # When enabled, a DRM distractor-context anchor is written when the
        # SiamABC response map reveals a competing secondary peak during reliable
        # tracking. All defaults are inert: when drm_introspection_enabled is
        # False the updater object is never built, the inner SiamABC tracker never
        # gets the introspection flag, and the legacy DRM write path is untouched.
        drm_introspection_enabled: bool = False,
        drm_introspection_theta_anc: float = 0.7,
        drm_introspection_theta_iou: float = 0.8,
        drm_introspection_theta_area: float = 0.2,
        drm_introspection_theta_M: int = 10,
        drm_introspection_delta: int = 5,
        drm_introspection_secondary_min_ratio: float = 0.5,
        drm_introspection_scale_aware_divergence: bool = True,
        drm_introspection_distractor_bank_enabled: bool = False,
        # Flag-gated diagnostics for the research features (DRM
        # introspection). When False (default) there is zero behavioural change
        # and effectively zero overhead; when True the tracker maintains per-video
        # and cumulative integer counters and prints a compact summary per video.
        research_telemetry_enabled: bool = False,
        # Frame-difference motion-saliency input augmentation (arXiv:2505.04917,
        # Sec. 3.2). When enabled, a short-term frame-difference motion-saliency
        # signal (|x_t - x_{t-1}| / |x_t - x_{t-2}|) is computed between
        # consecutive full frames and blended ADDITIVELY into the search region
        # the frozen SiamABC backbone consumes. All defaults are inert: when
        # frame_dynamics_enabled is False the processor object is never built and
        # the search crop is byte-for-byte unchanged (no frame-difference work).
        frame_dynamics_enabled: bool = False,
        frame_dynamics_blend_weight: float = 0.06,
        frame_dynamics_scale: float = 1.0,
        frame_dynamics_clip: float = -1.0,
        frame_dynamics_tiny_only: bool = True,
        frame_dynamics_tiny_area_fraction: float = 0.001,
        block_distractor_mode_on_camera_motion: bool = True,
        block_occlusion_on_camera_motion: bool = True,
        camera_motion_heavy_disp_threshold: float = 18.0,
        camera_motion_heavy_norm_threshold: float = 0.35,
        # Optionally swap the inner SiamABC dynamic-template update params
        # (re-selection cadence N + memory window length) while camera motion is
        # heavy. Disabled by default; high-motion values default to the base
        # tracker values (resolved lazily) so enabling is a no-op until tuned.
        camera_motion_template_adapt_enabled: bool = False,
        camera_motion_high_N: int = 15,
        camera_motion_high_memory_window_size: int = 15,
        # Dynamic-update admission gates under heavy motion; < 0 = leave unchanged.
        camera_motion_high_dynamic_update_threshold: float = -1.0,
        camera_motion_high_iou_threshold: float = -1.0,
        # Keep high-motion template behavior for this many frames after motion
        # calms before reverting to base (0 = revert immediately).
        camera_motion_template_adapt_linger_frames: int = 0,
        # Adaptive dynamic-template update rate. When template_rate_auto_enabled
        # is True, N and memory_window_size glide between the slow-end and
        # fast-end values below based on an EMA of combined camera+target motion,
        # and this path TAKES OVER from the binary camera_motion_template_adapt.
        # All defaults are inert until the flag is set, so legacy behavior holds.
        template_rate_auto_enabled: bool = False,
        template_rate_auto_N_fast: int = 3,
        template_rate_auto_N_slow: int = 40,
        template_rate_auto_window_fast: int = 8,
        template_rate_auto_window_slow: int = 40,
        template_rate_auto_ema_alpha: float = 0.2,
        # Camera motion (in bbox-diagonal units) that counts as "fully fast".
        # <= 0 reuses camera_motion_heavy_norm_threshold as the reference.
        template_rate_auto_cam_ref_norm: float = -1.0,
        # Target speed (bbox-diagonals per frame) that counts as "fully fast".
        template_rate_auto_target_ref_norm: float = 0.15,
        # Binary dynamic-template adapt on the TARGET's own motion, measured in
        # full-frame px/frame -- the same on-screen object speed shown in the
        # target-motion pill. When target_motion_template_adapt_enabled and that
        # speed exceeds target_motion_high_px_threshold, N (re-selection cadence)
        # and memory_window_size switch to the high-motion values below, reverting
        # (after an optional linger) once the object slows. Standalone binary mode,
        # mutually exclusive with template_rate_auto / camera_motion_template_adapt
        # (precedence resolved in _normal_update). Inert by default.
        target_motion_template_adapt_enabled: bool = False,
        target_motion_high_px_threshold: float = 18.0,
        target_motion_high_N: int = 20,
        target_motion_high_memory_window_size: int = 20,
        target_motion_template_adapt_linger_frames: int = 0,
        early_occlusion_entry_n_frames: int = -1,
        early_occlusion_max_frames: int = -1,
        occlusion_use_ram_until_drm_full: bool = True,
        velocity_window_average: int = 80,
        occlusion_patience: int = 5,
        occlusion_hysteresis: float = 0.10,
        tiny_roi_start_expand: float = 3.0,
        tiny_yolo_search_expand: float = 20.0,
        tiny_search_expand_growth_factor: float = 1.3,
        tiny_search_expand_growth_every: int = 5,
        tiny_search_expand_max: float = 40.0,
        out_of_frame_roi_edge_pin_enabled: bool = True,
        out_of_frame_roi_use_tiny_params: bool = False,
        out_of_frame_roi_skip_when_past_edge_mult: float = 2.0,
        out_of_frame_roi_start_expand: float = -1.0,
        out_of_frame_yolo_search_expand: float = -1.0,
        out_of_frame_search_expand_growth_factor: float = -1.0,
        out_of_frame_search_expand_growth_every: int = -1,
        max_proc_long_edge: int = 1280,
        entry_patience: int = 3,
        entry_patience_high_motion: int = -1,
        cand_collection_frames: int = 3,
        osnet_max_candidate_batch: int = 0,
        yolo_warmup_frames: int = 60,
        yolo_warmup_center_scale: float = 0.4,
        vel_dir_hard_gate: float = 0.5,
        yolo_filter_class: bool = False,
        yolo_class_detect_frames: int = 5,
        yolo_detectability_enabled: bool = False,
        yolo_detectability_probe_attempts: int = 6,
        yolo_detectability_probe_stride: int = 3,
        yolo_detectability_min_hits: int = 1,
        yolo_detectability_iou_thr: float = 0.3,
        jump_reject_enabled: bool = True,
        jump_reject_min_frames: int = 3,
        jump_reject_min_score: float = 0.50,
        jump_reject_norm_threshold: float = 1.25,
        jump_reject_max_iou: float = 0.15,
        jump_reject_use_velocity_residual: bool = True,
        jump_reject_use_appearance: bool = True,
        jump_reject_max_sim: float = 0.60,
        jump_reject_force_occlusion: bool = True,
        jump_reject_watch_min_frames: int = 2,
        jump_reject_watch_max_frames: int = 12,
        jump_reject_stable_norm_threshold: float = 0.35,
        jump_reject_stable_frames: int = 2,
        jump_reject_distractor_mode_frames: int = 12,
        jump_reject_distractor_penalty_enabled: bool = True,
        jump_reject_distractor_penalty_weight: float = 0.25,
        jump_reject_distractor_penalty_sim_floor: float = 0.55,
        jump_reject_distractor_penalty_bank_topk: int = 5,
        distractor_focus_dist_penalty_enabled: bool = True,
        distractor_focus_dist_penalty_weight: float = 0.35,
        distractor_focus_dist_soft_radius: float = 0.60,
        distractor_focus_dist_hard_radius: float = 2.00,
        distractor_mode_min_similarity: float = 0.50,
        distractor_mode_selected_min_similarity: float = 0.0,
        distractor_mode_selected_max_focus_dist_frac: float = -1.0,
        distractor_mode_yolo_topk: int = 2,
        distractor_mode_history_limit: int = 80,
        distractor_mode_use_tracker_mapping: bool = True,
        distractor_mode_roi_expand: float = 1.20,
        distractor_mode_roi_min_side: int = 32,
        distractor_drm_lam_iou: float = 0.25,
        distractor_drm_lam_app: float = 1.00,
        distractor_drm_lam_dist: float = 0.35,
        distractor_drm_gamma: float = 0.30,
        distractor_drm_dist_sigma_factor: float = 2.5,
        distractor_drm_bank_topk: int = 5,
        distractor_compare_mode: str = "drm",
        distractor_mode_update_memory: bool = False,
        distractor_mode_exit_reinit_enabled: bool = True,
        distractor_mode_exit_stable_frames: int = 4,
        distractor_mode_exit_same_iou: float = 0.60,
        distractor_mode_anchor_ekf_enabled: bool = True,
        distractor_mode_anchor_uncertainty_roi_scale: float = 2.0,
        distractor_mode_anchor_uncertainty_roi_cap: float = 96.0,
        distractor_mode_mahalanobis_gate_enabled: bool = True,
        distractor_mode_mahalanobis_threshold: float = 9.21,
        distractor_mode_mahalanobis_meas_var: float = 25.0,
        distractor_mode_switch_margin: float = 0.08,
        distractor_mode_ambiguity_hold_frames: int = 2,
        distractor_mode_selected_below_gate_hold_frames: int = 0,
        distractor_mode_selected_below_gate_force_occlusion: bool = False,
        distractor_mode_reentry_cooldown_frames: int = 12,
        distractor_mode_post_exit_memory_freeze_frames: int = 8,
        distractor_mode_post_exit_template_freeze_frames: int = 8,
        distractor_mode_behavior_mode: str = "standard",
        distractor_mode_overlap_motion_lock_enabled: bool = False,
        distractor_mode_overlap_iou_enter: float = 0.35,
        distractor_mode_overlap_iou_exit: float = 0.15,
        distractor_mode_overlap_clear_frames: int = 3,
        distractor_mode_overlap_lock_max_frames: int = 24,
        spike_anchor_history_window: int = 20,
        spike_anchor_trigger_norm: float = 0.60,
        spike_anchor_update_norm_max: float = 0.45,
        spike_reject_enabled: bool = True,
        spike_reject_min_frames: int = 5,
        spike_reject_min_score: float = 0.0,
        spike_reject_history_window: int = 40,
        spike_reject_min_history: int = 6,
        spike_reject_ratio: float = 3.5,
        spike_reject_abs_norm_min: float = 0.90,
        spike_reject_settle_ratio: float = 1.4,
        spike_reject_settle_abs_norm_max: float = 0.35,
        spike_reject_settle_frames: int = 2,
        spike_reject_watch_max_frames: int = 12,
        spike_reject_settle_from_spike_frac: float = 0.45,
        spike_reject_use_appearance: bool = True,
        spike_reject_max_sim: float = 0.65,
        spike_reject_confirm_frames: int = 1,
        spike_reject_long_distance_abs_norm_min: float = 0.0,
        spike_reject_camera_residual_min_ratio: float = 0.0,
        spike_reject_disable_in_tiny_mode: bool = False,
        copile_yolo=False,
        debug=True,
        disable_camera_motion: bool = False,
        gmc_prior_enabled: bool = True,
        gmc_prior_require_reliable_h: bool = True,
        gmc_prior_skip_in_distractor_mode: bool = True,
        gmc_prior_max_translation_frac: float = 0.25,
        gmc_prior_min_scale: float = 0.70,
        gmc_prior_max_scale: float = 1.40,
        gmc_prior_max_rotation_deg: float = 25.0,
        gmc_prior_max_corner_displacement_frac: float = 0.25,
    ):
        """
        Stores every configuration parameter and builds all the runtime state
                    containers — histories, the DRM kwargs dict, the appearance memory,
                    the EKF placeholder, and the phase/hysteresis counters. No computation
                    happens here beyond storing values and constructing empty deques.

        This is the single configuration point for the entire tracker. Every
                    behavioural knob lives here so callers can tune the tracker without
                    touching the implementation. Keeping all defaults sane means it works
                    out of the box for most scenarios while still being fully adjustable.
        Args:
            siam_tracker (any): the underlying SiamABC tracker instance (ORTrack)
            yolo_weights (any): path to YOLO weights file used for re-detection
            conf_threshold (any): tracker score below this triggers the entry streak counter
            reacq_threshold (any): minimum tracker score to accept a candidate as the target during phase 0
            reacq_confirm_frames (any): consecutive frames required after Stage-3 verify before leaving occlusion
            yolo_conf (any): minimum YOLO detection confidence
            yolo_iou (any): NMS IoU threshold passed to YOLO
            app_match_threshold (any): minimum DRM score required to accept a reacquisition
            nudge_alpha (any): interpolation weight used when nudging held_box toward a nearby detection
            tau_occ (any): IoU threshold above which a detection overlaps held_box enough to be flagged as a distractor
            beta (any): unused decay constant reserved for future use
            mem_capacity (any): max number of frames stored in the short-term RAM appearance memory
            tau_iou (any): IoU gate inside AppearanceMemory for admission
            tau_area (any): area ratio gate inside AppearanceMemory for admission
            drm_capacity (any): max number of reference descriptors kept in the DRM bank
            drm_tau_sim (any): cosine similarity threshold for DRM bank admission
            drm_window_W (any): temporal window used by DRM when computing motion consistency
            drm_mmin (any): minimum number of DRM bank entries required before DRM is trusted
            drm_lam_iou/app/mot/time (any): per-component weights of the DRM scoring formula
            drm_alpha (any): DRM bank update learning rate
            drm_gamma (any): DRM temporal decay exponent
            drm_margin (any): DRM acceptance margin above the distractor bank score
            drm_top_k (any): number of top DRM candidates forwarded to tracker verification
            drm_skip_threshold (any): DRM score above which a candidate is accepted without verification
            drm_lam_dist (any): weight of the Gaussian distance penalty in the DRM score
            drm_dist_sigma_factor (any): scales the Gaussian sigma by max(obj_w, obj_h)
            drm_lam_cand_dir (any): weight of direction consistency in DRM scoring (phase 0 and final)
            drm_lam_cand_vel (any): weight of full velocity consistency term in the final DRM phase
            vel_score_min_speed (any): if the EKF velocity magnitude is below this (px/frame) velocity scoring is skipped (returns neutral 0.5)
            ekf_process_noise (any): EKF process noise covariance scalar
            ekf_meas_noise (any): EKF measurement noise covariance scalar
            homo_max_corners (any): max GFTT feature points used by the accurate homography mode
            homo_inlier_threshold (any): minimum RANSAC inlier ratio to mark a homography as reliable
            homography_mode (any): camera-motion mode; "classic" uses grid+affine RANSAC, "accurate" uses feature tracking + full homography RANSAC
            shrinkage_min_drop_frac (any): minimum fractional area drop to trigger shrinkage detection
            shrinkage_max_lookback (any): how many frames back shrinkage and drift detection looks
            velocity_lookback (any): how many frames back finite-difference velocity is computed over
            velocity_smooth_alpha (any): EMA weight for smoothing velocity updates
            distractor_bank_maxlen (any): max number of distractor descriptors kept
            velocity_decay (any): unused decay constant reserved for future use
            search_expand_growth_factor (any): how much the ROI grows per growth interval during occlusion
            search_expand_growth_every (any): number of occlusion frames between each ROI growth step
            search_expand_max (any): maximum ROI expansion factor for normal-size objects
            long_distance_conf_threshold (any): score threshold used instead of conf_threshold when target is tiny/far
            long_distance_area_fraction (any): object area / frame area ratio below which long-distance mode activates
            long_distance_mode (any): force long-distance mode on regardless of area fraction
            enter_occlusion_on_loss (any): if False the tracker never enters the occlusion recovery path
            block_distractor_mode_on_camera_motion (any): if True, suppresses distractor-mode entry during heavy camera motion
            block_occlusion_on_camera_motion (any): if True, suppresses occlusion entry during heavy camera motion
            camera_motion_heavy_disp_threshold (any): camera displacement threshold in px for heavy-motion gating
            camera_motion_heavy_norm_threshold (any): normalized camera displacement threshold (disp / bbox_diag) for heavy-motion gating, useful for tiny objects
            early_occlusion_entry_n_frames (any): if non-negative, entering occlusion at frame_idx <= this value enables early-occlusion mode for that occlusion episode
            early_occlusion_max_frames (any): max occlusion duration (frames) still considered "early"; negative keeps legacy behavior (always early on successful reacquisition)
            occlusion_use_ram_until_drm_full (any): if True, occlusion recovery uses RAM matching until DRM reaches its configured capacity
            velocity_window_average (any): window length for robust velocity estimation at occlusion entry
            occlusion_patience (any): unused legacy parameter (replaced by entry_patience)
            occlusion_hysteresis (any): unused legacy parameter reserved for future use
            tiny_roi_start_expand (any): starting ROI expansion factor when the object is tiny/far
            tiny_yolo_search_expand (any): maximum ROI expansion for tiny objects
            tiny_search_expand_growth_factor (any): ROI growth factor per interval for tiny objects
            tiny_search_expand_growth_every (any): growth interval (frames) for tiny object ROI
            tiny_search_expand_max (any): hard cap on tiny-object ROI expansion
            max_proc_long_edge (any): processing resolution cap on the long frame edge
            entry_patience (any): consecutive low-score frames required before declaring occlusion
            cand_collection_frames (any): number of YOLO collection frames before the final DRM phase
            osnet_max_candidate_batch (any): max detections per frame sent to OSNet; 0 disables capping
            osnet_fp16 (any): run the OSNet descriptor backbone in FP16 (CUDA only; ignored on CPU)
            trt_compile_osnet (any): TRT-compile the OSNet descriptor backbone (CUDA only; needs torch_tensorrt)
            yolo_warmup_frames (any): initial frame count that uses centered warm-up ROI
            yolo_warmup_center_scale (any): warm-up ROI side ratio relative to frame size
            vel_dir_hard_gate (any): if cosine similarity between candidate and expected velocity is below -this, velocity score is floored to 0.05
            yolo_filter_class (any): if True, YOLO detections are filtered to the target's class after warm-up
            yolo_class_detect_frames (any): stride used during the class warm-up period
    """
        self.tracker: SiamABCTracker = siam_tracker
        tracker_cfg = getattr(siam_tracker, "tracking_config", {}) or {}
        self._initial_template_N = max(
            1, int(tracker_cfg.get("N", getattr(siam_tracker, "N", 1)))
        )
        self._initial_template_window = max(
            1,
            int(
                tracker_cfg.get(
                    "memory_window_size",
                    getattr(siam_tracker, "memory_window_size", 50),
                )
                or 50
            ),
        )
        self._initial_dynamic_update_threshold = float(
            tracker_cfg.get(
                "dynamic_update_threshold",
                getattr(siam_tracker, "dynamic_update_threshold", 0.8),
            )
        )
        self._initial_iou_threshold = float(tracker_cfg.get("iou_threshold", 0.3))
        configure_descriptor_backend(
            descriptor_backend=descriptor_backend,
            osnet_model_name=osnet_model_name,
            osnet_model_path=osnet_model_path,
            osnet_pretrained_checkpoint=osnet_pretrained_checkpoint,
            osnet_device=osnet_device,
            siam_tracker=siam_tracker,
            siamese_feature_source=siamese_feature_source,
            siamese_comparison_mode=siamese_comparison_mode,
            osnet_fp16=osnet_fp16,
            trt_compile_osnet=trt_compile_osnet,
            osnet_max_candidate_batch=osnet_max_candidate_batch,
            trt_cache_dir=trt_cache_dir,
            trt_rebuild_cache=trt_rebuild_cache,
        )

        # How often the descriptor backend (OSNet / siamese) is run on the
        # per-frame normal-tracking memory-admission path. 1 = every confident
        # frame (default). N>1 only extracts/admits a descriptor every Nth frame
        # to save compute during stable tracking; the event-driven occlusion /
        # distractor-recovery paths still run the descriptor every time they need
        # it, so appearance fidelity is preserved exactly where it matters.
        self._descriptor_stride = max(1, int(descriptor_stride))

        # Tier-3 GPU overlap: run OSNet for frame N on a side stream concurrently
        # with SiamABC for frame N+1. Only active for the OSNet/CUDA backend;
        # otherwise everything stays fully synchronous (no behavior change).
        # _overlap_active is resolved lazily on the first update so the OSNet
        # extractor (and thus its CUDA device) has been created.
        self._osnet_async_overlap = bool(osnet_async_overlap)
        self._overlap_active: Optional[bool] = None
        # Dedicated non-default stream for the SiamABC forward; OSNet uses its own
        # side stream inside the extractor. Both must be non-default so the legacy
        # default stream does not serialize them.
        self._siam_stream = (
            torch.cuda.Stream()
            if (self._osnet_async_overlap and torch.cuda.is_available())
            else None
        )
        # Holds the deferred descriptor for the previous confident frame:
        # (handle, pred_bbox, ref_bbox, velocity, distractor_bank, sample_drm).
        self._pending_desc: Optional[dict] = None

        if copile_yolo:
            self.yolo = self.load_yolo_compiled(yolo_weights)
        else:
            self.yolo = YOLO(yolo_weights)

        self.debug = debug
        self.conf_threshold = conf_threshold
        self.reacq_threshold = reacq_threshold
        self.yolo_conf = yolo_conf
        self.yolo_iou_thr = yolo_iou
        self.yolo_imgsz = max(32, int(yolo_imgsz))
        self.yolo_augment = bool(yolo_augment)
        self.app_match_threshold = app_match_threshold
        self.nudge_alpha = nudge_alpha
        self.tau_occ = tau_occ
        self.beta = beta
        self.ncc_threshold = ncc_threshold
        self.ncc_expand = ncc_expand
        self.conf_history_len = conf_history_len
        self.history_decay = history_decay
        self.history_skip_last = history_skip_last
        self.yolo_search_expand = yolo_search_expand
        self.roi_start_expand = roi_start_expand
        self.size_history_len = size_history_len

        self.ekf_process_noise = ekf_process_noise
        self.ekf_meas_noise = ekf_meas_noise
        self.homo_max_corners = homo_max_corners
        self.homo_inlier_threshold = homo_inlier_threshold
        mode = str(homography_mode).strip().lower()
        if mode not in {"classic", "accurate"} and self.debug:
            print(
                f"[homography] unsupported mode '{homography_mode}', "
                "falling back to 'classic'"
            )
        self._homography_mode = mode if mode in {"classic", "accurate"} else "classic"

        self.shrinkage_min_drop_frac = shrinkage_min_drop_frac
        self.shrinkage_max_lookback = shrinkage_max_lookback

        self.velocity_lookback = velocity_lookback
        self.velocity_smooth_alpha = velocity_smooth_alpha

        self._distractor_bank_maxlen = distractor_bank_maxlen
        self.velocity_window_average = velocity_window_average

        self.occ_siam_margin = occ_siam_margin

        self.search_expand_growth_factor = search_expand_growth_factor
        self.search_expand_growth_every = search_expand_growth_every
        self.search_expand_max = search_expand_max
        self._occ_frames: int = 0
        self.occ_siam_reacq_threshold = occ_siam_reacq_threshold
        self._drm_dist_sigma_factor = drm_dist_sigma_factor
        self._size_history: deque = deque(maxlen=size_history_len)
        self._cam_disp_history: deque = deque(maxlen=conf_history_len)
        self._vel_history: deque = deque(maxlen=200)

        self.long_distance_conf_threshold = long_distance_conf_threshold
        self.long_distance_area_fraction = long_distance_area_fraction
        self.long_distance_mode = long_distance_mode
        self.recovered_early_occlusion = True
        self.enter_occlusion_on_loss = enter_occlusion_on_loss
        self._block_distractor_mode_on_camera_motion = bool(
            block_distractor_mode_on_camera_motion
        )
        self._block_occlusion_on_camera_motion = bool(
            block_occlusion_on_camera_motion
        )
        self._camera_motion_heavy_disp_threshold = max(
            0.0, float(camera_motion_heavy_disp_threshold)
        )
        self._camera_motion_heavy_norm_threshold = max(
            0.0, float(camera_motion_heavy_norm_threshold)
        )
        self._camera_motion_template_adapt_enabled = bool(
            camera_motion_template_adapt_enabled
        )
        self._camera_motion_high_N = max(1, int(camera_motion_high_N))
        self._camera_motion_high_memory_window_size = max(
            1, int(camera_motion_high_memory_window_size)
        )
        self._camera_motion_high_dynamic_update_threshold = float(
            camera_motion_high_dynamic_update_threshold
        )
        self._camera_motion_high_iou_threshold = float(
            camera_motion_high_iou_threshold
        )
        self._camera_motion_template_adapt_linger_frames = max(
            0, int(camera_motion_template_adapt_linger_frames)
        )
        # Frames of high-motion behavior still owed after motion calmed.
        self._camera_motion_adapt_linger_left: int = 0
        # Base (calm-motion) values captured lazily from the inner tracker on the
        # first adapt call, so we restore exactly what the tracker config set.
        self._base_template_N: Optional[int] = None
        self._base_template_window: Optional[int] = None
        self._base_dynamic_update_threshold: Optional[float] = None
        self._base_iou_threshold: Optional[float] = None

        # Adaptive dynamic-template update rate (template_rate_auto). None unless
        # the mode is enabled, so _normal_update's adapt block and every helper
        # below short-circuit and the legacy / binary-adapt paths are unchanged.
        self._template_rate_auto = None
        self._template_rate_auto_cam_ref_norm = float(template_rate_auto_cam_ref_norm)
        self._template_rate_auto_target_ref_norm = max(
            0.0, float(template_rate_auto_target_ref_norm)
        )
        if bool(template_rate_auto_enabled):
            from .auto_template_rate import AutoTemplateRate

            self._template_rate_auto = AutoTemplateRate(
                n_fast=template_rate_auto_N_fast,
                n_slow=template_rate_auto_N_slow,
                window_fast=template_rate_auto_window_fast,
                window_slow=template_rate_auto_window_slow,
                ema_alpha=template_rate_auto_ema_alpha,
            )
        # Last frame's camera-motion magnitude in bbox-diagonal units, published
        # by is_heavy_camera_motion for the auto-rate controller (and debug).
        self._last_cam_motion_norm: float = 0.0

        # Binary target-motion template adapt (target_motion_template_adapt).
        # Gated on the object's own speed in full-frame px/frame (the pill value).
        # Shares the lazily-captured base N/window with the camera-motion path
        # (only one adapt branch runs per frame, see _normal_update) but keeps its
        # own linger counter. All inert unless the flag is set.
        self._target_motion_template_adapt_enabled = bool(
            target_motion_template_adapt_enabled
        )
        self._target_motion_high_px_threshold = max(
            0.0, float(target_motion_high_px_threshold)
        )
        self._target_motion_high_N = max(1, int(target_motion_high_N))
        self._target_motion_high_memory_window_size = max(
            1, int(target_motion_high_memory_window_size)
        )
        self._target_motion_template_adapt_linger_frames = max(
            0, int(target_motion_template_adapt_linger_frames)
        )
        # Frames of high-motion behavior still owed after the object slowed.
        self._target_motion_adapt_linger_left: int = 0
        # Last frame's target speed in full-frame px/frame (pill units), published
        # for the viz fast/slow verdict and the rate box.
        self._last_target_motion_px: float = 0.0
        self._early_occlusion_entry_n_frames = int(early_occlusion_entry_n_frames)
        self._early_occlusion_max_frames = int(early_occlusion_max_frames)
        self._occlusion_use_ram_until_drm_full = bool(
            occlusion_use_ram_until_drm_full
        )
        self._drm_kwargs: DRMKwargs = {
            "lam_iou": drm_lam_iou,
            "lam_app": drm_lam_app,
            "lam_mot": drm_lam_mot,
            "lam_time": drm_lam_time,
            "alpha": drm_alpha,
            "gamma": drm_gamma,
            "margin": drm_margin,
            "top_k": drm_top_k,
            "skip_threshold": drm_skip_threshold,
            "lam_dist": drm_lam_dist,
            "lam_cand_dir": drm_lam_cand_dir,
        }

        # Adaptive DRM acceptance margin (drm_margin == "auto"). A numeric
        # drm_margin leaves this None, so _active_drm_kwargs and the per-frame
        # estimator hook below are both no-ops and legacy behaviour is identical.
        self._drm_margin_auto = None
        if isinstance(drm_margin, str) and drm_margin.strip().lower() == "auto":
            from .auto_margin import AutoDrmMargin

            self._drm_margin_auto = AutoDrmMargin(
                min_margin=drm_margin_auto_min,
                max_margin=drm_margin_auto_max,
                delta=drm_margin_auto_delta,
                ema_alpha=drm_margin_auto_ema_alpha,
                n_frames=drm_margin_auto_n_frames,
                warmup=drm_margin_auto_warmup,
                min_samples=drm_margin_auto_min_samples,
            )
            # Seed the stored dict with a concrete float so any reader touching
            # _drm_kwargs["margin"] directly still gets a number, not "auto".
            self._drm_kwargs["margin"] = self._drm_margin_auto.value

        # Introspection-based DRM anchor update (arXiv:2411.17576, Sec. 3.2.2).
        # Built only when enabled; otherwise the updater stays None, the inner
        # SiamABC tracker never gets the introspection flag (so the response-map
        # secondary-peak signal is never computed), and the legacy DRM write path
        # in AppearanceMemory runs untouched. Mirrors the auto-rate / auto-margin
        # on/off convention -- the object is never constructed when disabled.
        self._drm_introspection_enabled = bool(drm_introspection_enabled)
        # Scale the divergence box + non-max suppression to the target's grid-cell
        # extent (de-degenerates theta_anc on the coarse 16x16 map). Stored
        # unconditionally so the attribute always exists, even when DRM is off.
        self._drm_introspection_scale_aware = bool(
            drm_introspection_scale_aware_divergence
        )
        # Write the detected secondary-peak (distractor) descriptor into the
        # NEGATIVE distractor bank during reliable tracking. Sub-feature of DRM
        # introspection; inert unless a recovery gamma is raised. Stored
        # unconditionally so the attribute always exists.
        self._drm_introspection_distractor_bank_enabled = bool(
            drm_introspection_distractor_bank_enabled
        )
        self._drm_introspection = None
        if self._drm_introspection_enabled:
            from .drm_introspection import DrmIntrospectionUpdater

            self._drm_introspection = DrmIntrospectionUpdater(
                theta_anc=drm_introspection_theta_anc,
                theta_iou=drm_introspection_theta_iou,
                theta_area=drm_introspection_theta_area,
                theta_M=drm_introspection_theta_M,
                delta=drm_introspection_delta,
                secondary_min_ratio=drm_introspection_secondary_min_ratio,
            )
            # The inner SiamABC tracker's introspection flag is flipped via
            # _attach_inner_feature_state() below (single source of truth, re-run
            # per clip) so its track() computes the secondary-peak divergence
            # signal; when disabled the flag is never set, keeping the legacy
            # response-map path inert.

        # Flag-gated research telemetry (DRM introspection).
        # When OFF every increment site is guarded by self._research_telemetry_enabled
        # so there is zero behavioural change and effectively zero overhead. When ON
        # the tracker keeps per-video counters (_tlm_video, reset each clip) and folds
        # each finished clip into the module-global cumulative tally
        # (_TLM_GLOBAL_TOTAL), printing one compact per-video line at the clip
        # boundary plus one aggregate line at process exit. The DRM secondary-peak
        # counts live on the inner SiamABC tracker and are folded in here.
        self._research_telemetry_enabled = bool(research_telemetry_enabled)
        # Per-video counters for THIS instance's current clip. The cumulative
        # tally lives at module scope (_TLM_GLOBAL_TOTAL) so it spans clips even
        # when run_inference.py builds a fresh tracker per clip (the default).
        self._tlm_video = self._new_telemetry_counters()
        if self._research_telemetry_enabled:
            # The inner tracker's telemetry flag + counter dict (so it can stash
            # DRM-introspection counts observed inside its own track()) are
            # attached via _attach_inner_feature_state() below.
            # Capture the active-feature-flag echo once at module scope so the
            # process-exit aggregate dump (which holds no tracker reference) can
            # print it. Robust: only set if not already captured.
            global _TLM_ACTIVE_FLAGS
            if _TLM_ACTIVE_FLAGS is None:
                _TLM_ACTIVE_FLAGS = self._research_flag_echo()
            # Register this instance for residual flushing at exit (weakref, so a
            # fresh per-clip tracker GCs normally once it has emitted its video at
            # the clip boundary), and register the single process-exit aggregate
            # dump exactly once across all instances.
            global _TLM_DUMP_REGISTERED
            _TLM_LIVE_INSTANCES.add(self)
            if not _TLM_DUMP_REGISTERED:
                atexit.register(_tlm_global_dump)
                _TLM_DUMP_REGISTERED = True

        # Frame-difference motion-saliency input augmentation (arXiv:2505.04917,
        # Sec. 3.2). Built only when enabled; otherwise the processor stays None,
        # no frame-difference work runs, and the search crop fed to the frozen
        # SiamABC backbone is byte-for-byte unchanged. Mirrors the auto-margin
        # on/off convention -- the object is never constructed when disabled.
        self._frame_dynamics_enabled = bool(frame_dynamics_enabled)
        self._frame_dynamics_processor = None
        if self._frame_dynamics_enabled:
            from .frame_dynamics import FrameDynamicsProcessor

            self._frame_dynamics_processor = FrameDynamicsProcessor(
                enabled=True,
                blend_weight=frame_dynamics_blend_weight,
                scale=frame_dynamics_scale,
                clip=frame_dynamics_clip,
                tiny_only=frame_dynamics_tiny_only,
                tiny_area_fraction=frame_dynamics_tiny_area_fraction,
            )

        # Single source of truth for attaching the research-feature state
        # bundles (DRM introspection flag, telemetry flag + inner counter dict)
        # onto the inner SiamABC tracker. The inner tracker is
        # reset / re-initialized per clip and its own __init__ defaults these back
        # to inert values, so this MUST also be re-run from initialize() (see the
        # call there). Idempotent and attaches nothing for a disabled feature, so
        # the all-off path stays byte-for-byte inert.
        self._attach_inner_feature_state()

        self._drm_lam_cand_dir = drm_lam_cand_dir

        # Parallel occlusion-DRM params, selected by _active_drm_* when the
        # active occlusion episode was triggered by the distractor below-gate
        # force-occlusion path (_distractor_occlusion_active).
        self._distractor_occ_drm_kwargs: DRMKwargs = {
            "lam_iou": distractor_occ_drm_lam_iou,
            "lam_app": distractor_occ_drm_lam_app,
            "lam_mot": distractor_occ_drm_lam_mot,
            "lam_time": distractor_occ_drm_lam_time,
            "alpha": distractor_occ_drm_alpha,
            "gamma": distractor_occ_drm_gamma,
            "margin": distractor_occ_drm_margin,
            "top_k": distractor_occ_drm_top_k,
            "skip_threshold": distractor_occ_drm_skip_threshold,
            "lam_dist": distractor_occ_drm_lam_dist,
            "lam_cand_dir": distractor_occ_drm_lam_cand_dir,
        }
        self._distractor_occ_drm_lam_cand_dir = distractor_occ_drm_lam_cand_dir
        self._distractor_occ_drm_dist_sigma_factor = max(
            0.1, float(distractor_occ_drm_dist_sigma_factor)
        )
        self._use_distractor_bank = (
            self._distractor_bank_maxlen > 0
        )
        self._vel_score_min_speed = vel_score_min_speed
        self._entry_patience = max(1, entry_patience)
        # Grace period: refuse to enter occlusion for this many initial frames
        # (warm-up). 0 = no restriction.
        self._no_occlusion_first_n_frames = max(0, int(no_occlusion_first_n_frames))
        # Alternate occlusion-entry patience used while camera motion is heavy.
        # < 1 disables it (always use _entry_patience). Meant as a softer
        # alternative to block_occlusion_on_camera_motion: set that False and set
        # a larger patience here to wait longer before declaring loss during pans
        # (with blocking ON the streak resets each heavy frame, so this is moot).
        self._entry_patience_high_motion = (
            max(1, int(entry_patience_high_motion))
            if int(entry_patience_high_motion) >= 1
            else -1
        )
        self._cand_collection_frames = max(1, cand_collection_frames)
        self._reacq_confirm_frames = max(1, int(reacq_confirm_frames))

        self.tiny_roi_start_expand = tiny_roi_start_expand
        self.tiny_yolo_search_expand = tiny_yolo_search_expand
        self.tiny_search_expand_growth_factor = tiny_search_expand_growth_factor
        self.tiny_search_expand_growth_every = tiny_search_expand_growth_every
        self.tiny_search_expand_max = tiny_search_expand_max
        self._out_of_frame_roi_edge_pin_enabled = bool(out_of_frame_roi_edge_pin_enabled)
        self._out_of_frame_roi_use_tiny_params = bool(out_of_frame_roi_use_tiny_params)
        self._out_of_frame_roi_skip_when_past_edge_mult = max(
            0.0, float(out_of_frame_roi_skip_when_past_edge_mult)
        )
        self._out_of_frame_roi_start_expand = float(out_of_frame_roi_start_expand)
        self._out_of_frame_yolo_search_expand = float(out_of_frame_yolo_search_expand)
        self._out_of_frame_search_expand_growth_factor = float(
            out_of_frame_search_expand_growth_factor
        )
        self._out_of_frame_search_expand_growth_every = int(
            out_of_frame_search_expand_growth_every
        )
        self._max_proc_long_edge = max(1, int(max_proc_long_edge))
        self._osnet_max_candidate_batch = max(0, int(osnet_max_candidate_batch))
        self._yolo_warmup_frames = max(0, int(yolo_warmup_frames))
        self._yolo_warmup_center_scale = float(
            np.clip(yolo_warmup_center_scale, 0.05, 1.0)
        )

        self.disable_camera_motion = disable_camera_motion
        self._gmc_prior_enabled = bool(gmc_prior_enabled)
        self._gmc_prior_require_reliable_h = bool(gmc_prior_require_reliable_h)
        self._gmc_prior_skip_in_distractor_mode = bool(
            gmc_prior_skip_in_distractor_mode
        )
        self._gmc_prior_max_translation_frac = max(
            0.0, float(gmc_prior_max_translation_frac)
        )
        self._gmc_prior_min_scale = max(0.01, float(gmc_prior_min_scale))
        self._gmc_prior_max_scale = max(
            self._gmc_prior_min_scale + 1e-3, float(gmc_prior_max_scale)
        )
        self._gmc_prior_max_rotation_deg = max(0.0, float(gmc_prior_max_rotation_deg))
        self._gmc_prior_max_corner_displacement_frac = max(
            0.0, float(gmc_prior_max_corner_displacement_frac)
        )

        self.memory = AppearanceMemory(
            capacity=mem_capacity,
            tau_iou=tau_iou,
            tau_area=tau_area,
            drm_capacity=drm_capacity,
            tau_sim=drm_tau_sim,
            window_W=drm_window_W,
            mmin=drm_mmin,
            distractor_bank_maxlen=distractor_bank_maxlen,
        )

        self.current_bbox: Optional[np.ndarray] = None
        self.held_box: Optional[np.ndarray] = None
        self.in_occlusion: bool = False
        self.frame_idx: int = 0
        # Snapshot of the most recent successful recovery (set by occlusion exit
        # and by distractor-mode resolved exit). Consumed by the visualiser.
        self._last_recovery_patch: Optional[np.ndarray] = None
        self._last_recovery_info: Optional[dict] = None
        self.velocity: np.ndarray = np.zeros(2)
        self.prev_gray: Optional[np.ndarray] = None
        self.init_frame: Optional[np.ndarray] = None
        self.init_bbox: Optional[np.ndarray] = None
        self._last_yolo: List = []
        self._yolo_cache: List = []
        self._distractor_bank: deque = deque(maxlen=distractor_bank_maxlen)
        self._out_of_frame: bool = False
        self._exit_edge: Optional[str] = None
        self._search_cx: Optional[float] = None
        self._search_cy: Optional[float] = None
        self._conf_history: deque = deque(maxlen=conf_history_len)
        self._center_history: deque = deque(maxlen=200)
        self._cam_vel_history: deque = deque(maxlen=200)

        self._heavy_motion_cache_frame_idx: int = -1
        self._heavy_motion_cache_weighted_disp: float = 0.0
        self._heavy_motion_cache_inst_disp: float = 0.0
        # Latest heavy-camera-motion verdict for this frame (for the viz marker
        # and any consumer that wants the gate's current state).
        self._last_heavy_cam_motion: bool = False
        self._last_heavy_cam_disp: float = 0.0

        self._spike_step_norms: deque = deque(
            maxlen=max(2, int(spike_reject_history_window))
        )
        self._history = FrameHistory(
            conf_history=self._conf_history,
            size_history=self._size_history,
            center_history=self._center_history,
            vel_history=self._vel_history,
            cam_vel_history=self._cam_vel_history,
            cam_disp_history=self._cam_disp_history,
            spike_step_norms=self._spike_step_norms,
        )

        self._vel_dir_hard_gate = vel_dir_hard_gate
        self._yolo_filter_class = yolo_filter_class
        self._yolo_class_detect_frames = yolo_class_detect_frames
        self._target_class_id: Optional[int] = None
        self._class_warmup_done: bool = False
        self._class_votes: dict[int, int] = {}

        # --- Adaptive YOLO-detectability probe -----------------------------
        # During the first frames of healthy tracking we run YOLO on the
        # tracker's own search ROI a few times. If YOLO ever lands a box on the
        # target, the object is "YOLO-detectable" and occlusion recovery leans
        # on YOLO+DRM (the SiamABC-alone phase-0 commit is shut off). If YOLO
        # never finds it, we keep relying solely on phase=siam during occlusion.
        # See _maybe_run_detectability_probe / _occ_phase_siam.
        self._yolo_detectability_enabled = bool(yolo_detectability_enabled)
        self._yolo_detectability_probe_attempts = max(1, int(yolo_detectability_probe_attempts))
        self._yolo_detectability_probe_stride = max(1, int(yolo_detectability_probe_stride))
        self._yolo_detectability_min_hits = max(1, int(yolo_detectability_min_hits))
        self._yolo_detectability_iou_thr = float(yolo_detectability_iou_thr)
        self._yolo_detectable: bool = False
        self._detectability_probe_done: bool = False
        self._detectability_runs: int = 0
        self._detectability_hits: int = 0
        self._jump_reject_enabled = bool(jump_reject_enabled)
        self._jump_reject_min_frames = max(0, int(jump_reject_min_frames))
        self._jump_reject_min_score = float(jump_reject_min_score)
        self._jump_reject_norm_threshold = float(jump_reject_norm_threshold)
        self._jump_reject_max_iou = float(jump_reject_max_iou)
        self._jump_reject_use_velocity_residual = bool(jump_reject_use_velocity_residual)
        self._jump_reject_use_appearance = bool(jump_reject_use_appearance)
        self._jump_reject_max_sim = float(jump_reject_max_sim)
        self._jump_reject_force_occlusion = bool(jump_reject_force_occlusion)
        self._jump_reject_watch_min_frames = max(1, int(jump_reject_watch_min_frames))
        self._jump_reject_watch_max_frames = max(
            self._jump_reject_watch_min_frames, int(jump_reject_watch_max_frames)
        )
        self._jump_reject_stable_norm_threshold = float(
            jump_reject_stable_norm_threshold
        )
        self._jump_reject_stable_frames = max(1, int(jump_reject_stable_frames))
        self._jump_reject_distractor_mode_frames = max(
            0, int(jump_reject_distractor_mode_frames)
        )
        self._jump_reject_distractor_penalty_enabled = bool(
            jump_reject_distractor_penalty_enabled
        )
        self._jump_reject_distractor_penalty_weight = max(
            0.0, float(jump_reject_distractor_penalty_weight)
        )
        self._jump_reject_distractor_penalty_sim_floor = float(
            np.clip(jump_reject_distractor_penalty_sim_floor, 0.0, 0.999)
        )
        self._jump_reject_distractor_penalty_bank_topk = max(
            1, int(jump_reject_distractor_penalty_bank_topk)
        )
        self._distractor_focus_dist_penalty_enabled = bool(
            distractor_focus_dist_penalty_enabled
        )
        self._distractor_focus_dist_penalty_weight = max(
            0.0, float(distractor_focus_dist_penalty_weight)
        )
        self._distractor_focus_dist_soft_radius = max(
            0.0, float(distractor_focus_dist_soft_radius)
        )
        self._distractor_focus_dist_hard_radius = max(
            self._distractor_focus_dist_soft_radius + 1e-3,
            float(distractor_focus_dist_hard_radius),
        )
        self._distractor_mode_min_similarity = float(
            np.clip(distractor_mode_min_similarity, 0.0, 1.0)
        )
        self._distractor_mode_selected_min_similarity = float(
            np.clip(distractor_mode_selected_min_similarity, 0.0, 1.0)
        )
        self._distractor_mode_selected_max_focus_dist_frac = float(
            distractor_mode_selected_max_focus_dist_frac
        )
        # 0 means "use all ROI detections" (bounded by osnet_max_candidate_batch if set).
        self._distractor_mode_yolo_topk = max(0, int(distractor_mode_yolo_topk))
        self._distractor_mode_history_limit = max(4, int(distractor_mode_history_limit))
        self._distractor_mode_use_tracker_mapping = bool(
            distractor_mode_use_tracker_mapping
        )
        self._distractor_mode_roi_expand = max(1.0, float(distractor_mode_roi_expand))
        self._distractor_mode_roi_min_side = max(8, int(distractor_mode_roi_min_side))
        self._distractor_drm_lam_iou = float(distractor_drm_lam_iou)
        self._distractor_drm_lam_app = float(distractor_drm_lam_app)
        self._distractor_drm_lam_dist = float(distractor_drm_lam_dist)
        self._distractor_drm_gamma = max(0.0, float(distractor_drm_gamma))
        self._distractor_drm_dist_sigma_factor = max(
            0.1, float(distractor_drm_dist_sigma_factor)
        )
        self._distractor_drm_bank_topk = max(1, int(distractor_drm_bank_topk))
        mode = str(distractor_compare_mode).strip().lower()
        self._distractor_compare_mode = mode if mode in {"drm", "ram"} else "drm"
        self._distractor_mode_update_memory = bool(distractor_mode_update_memory)
        self._distractor_mode_exit_reinit_enabled = bool(
            distractor_mode_exit_reinit_enabled
        )
        self._distractor_mode_exit_stable_frames = max(
            1, int(distractor_mode_exit_stable_frames)
        )
        self._distractor_mode_exit_same_iou = float(
            np.clip(distractor_mode_exit_same_iou, 0.0, 1.0)
        )
        self._distractor_mode_anchor_ekf_enabled = bool(
            distractor_mode_anchor_ekf_enabled
        )
        self._distractor_mode_anchor_uncertainty_roi_scale = max(
            0.0, float(distractor_mode_anchor_uncertainty_roi_scale)
        )
        self._distractor_mode_anchor_uncertainty_roi_cap = max(
            0.0, float(distractor_mode_anchor_uncertainty_roi_cap)
        )
        self._distractor_mode_mahalanobis_gate_enabled = bool(
            distractor_mode_mahalanobis_gate_enabled
        )
        self._distractor_mode_mahalanobis_threshold = max(
            0.0, float(distractor_mode_mahalanobis_threshold)
        )
        self._distractor_mode_mahalanobis_meas_var = max(
            1e-6, float(distractor_mode_mahalanobis_meas_var)
        )
        self._distractor_mode_switch_margin = max(
            0.0, float(distractor_mode_switch_margin)
        )
        self._distractor_mode_ambiguity_hold_frames = max(
            0, int(distractor_mode_ambiguity_hold_frames)
        )
        self._distractor_mode_selected_below_gate_hold_frames = max(
            0, int(distractor_mode_selected_below_gate_hold_frames)
        )
        self._distractor_mode_selected_below_gate_force_occlusion = bool(
            distractor_mode_selected_below_gate_force_occlusion
        )
        self._distractor_mode_reentry_cooldown_frames = max(
            0, int(distractor_mode_reentry_cooldown_frames)
        )
        self._distractor_mode_post_exit_memory_freeze_frames = max(
            0, int(distractor_mode_post_exit_memory_freeze_frames)
        )
        self._distractor_mode_post_exit_template_freeze_frames = max(
            0, int(distractor_mode_post_exit_template_freeze_frames)
        )
        self._distractor_mode_behavior_mode = str(
            distractor_mode_behavior_mode
        ).strip().lower()
        if self._distractor_mode_behavior_mode not in {
            "standard",
            "overlap_motion_lock",
        }:
            self._distractor_mode_behavior_mode = "standard"
        self._distractor_mode_overlap_motion_lock_enabled = bool(
            distractor_mode_overlap_motion_lock_enabled
        ) or (self._distractor_mode_behavior_mode == "overlap_motion_lock")
        self._distractor_mode_overlap_iou_enter = float(
            np.clip(distractor_mode_overlap_iou_enter, 0.0, 1.0)
        )
        self._distractor_mode_overlap_iou_exit = float(
            np.clip(distractor_mode_overlap_iou_exit, 0.0, 1.0)
        )
        if self._distractor_mode_overlap_iou_exit > self._distractor_mode_overlap_iou_enter:
            self._distractor_mode_overlap_iou_exit = self._distractor_mode_overlap_iou_enter
        self._distractor_mode_overlap_clear_frames = max(
            1, int(distractor_mode_overlap_clear_frames)
        )
        self._distractor_mode_overlap_lock_max_frames = max(
            self._distractor_mode_overlap_clear_frames,
            int(distractor_mode_overlap_lock_max_frames),
        )
        self._spike_anchor_history_window = max(3, int(spike_anchor_history_window))
        self._spike_anchor_trigger_norm = max(0.05, float(spike_anchor_trigger_norm))
        self._spike_anchor_update_norm_max = max(0.01, float(spike_anchor_update_norm_max))
        self._spike_reject_enabled = bool(spike_reject_enabled)
        self._spike_reject_min_frames = max(0, int(spike_reject_min_frames))
        self._spike_reject_min_score = float(spike_reject_min_score)
        self._spike_reject_history_window = max(2, int(spike_reject_history_window))
        self._spike_reject_min_history = max(2, int(spike_reject_min_history))
        self._spike_reject_ratio = float(spike_reject_ratio)
        self._spike_reject_abs_norm_min = float(spike_reject_abs_norm_min)
        self._spike_reject_settle_ratio = float(spike_reject_settle_ratio)
        self._spike_reject_settle_abs_norm_max = float(
            spike_reject_settle_abs_norm_max
        )
        self._spike_reject_settle_frames = max(1, int(spike_reject_settle_frames))
        self._spike_reject_watch_max_frames = max(2, int(spike_reject_watch_max_frames))
        self._spike_reject_settle_from_spike_frac = float(
            np.clip(spike_reject_settle_from_spike_frac, 0.05, 0.95)
        )
        self._spike_reject_use_appearance = bool(spike_reject_use_appearance)
        self._spike_reject_max_sim = float(spike_reject_max_sim)
        self._spike_reject_confirm_frames = max(1, int(spike_reject_confirm_frames))
        self._spike_reject_long_distance_abs_norm_min = max(
            0.0, float(spike_reject_long_distance_abs_norm_min)
        )
        self._spike_reject_camera_residual_min_ratio = max(
            0.0, float(spike_reject_camera_residual_min_ratio)
        )
        self._spike_reject_disable_in_tiny_mode = bool(
            spike_reject_disable_in_tiny_mode
        )

        self._jump_watch_active: bool = False
        self._jump_watch_anchor_bbox: Optional[np.ndarray] = None
        self._jump_watch_prev_bbox: Optional[np.ndarray] = None
        self._jump_watch_frames: int = 0
        self._jump_watch_stable_count: int = 0
        self._jump_watch_baseline_norm: float = 0.0
        self._jump_watch_last_speed_norm: float = 0.0
        self._jump_watch_last_ratio: float = 1.0
        self._jump_watch_last_sim: Optional[float] = None
        self._jump_reject_distractor_timer: int = 0
        self._jump_watch_last_step_norm: float = 0.0
        self._spike_confirm_streak: int = 0
        self._spike_debug_speed_norm: float = float("nan")
        self._spike_debug_baseline_norm: float = float("nan")
        self._spike_debug_ratio: float = float("nan")
        self._spike_debug_trigger: bool = False
        self._stable_anchor_bbox: Optional[np.ndarray] = None
        self._distractor_focus_bbox: Optional[np.ndarray] = None
        self._distractor_mode_active: bool = False
        self._distractor_mode_visual_reals: List[np.ndarray] = []
        self._distractor_mode_visual_distractors: List[np.ndarray] = []
        self._distractor_mode_roi: Optional[np.ndarray] = None
        self._distractor_mode_roi_size: Optional[Tuple[int, int]] = None
        self._distractor_mode_stable_count: int = 0
        self._distractor_mode_ambiguous_count: int = 0
        self._distractor_mode_below_gate_count: int = 0
        # True while the current occlusion episode was entered via the distractor
        # below-gate force path (selects _distractor_occ_drm_* ranking params).
        self._distractor_occlusion_active: bool = False
        # Per-frame request set by the force path, consumed at occlusion entry.
        self._pending_distractor_occlusion: bool = False
        self._distractor_mode_reentry_cooldown: int = 0
        self._distractor_mode_memory_freeze_left: int = 0
        self._distractor_mode_template_freeze_left: int = 0
        self._distractor_anchor_ekf: Optional[BBoxEKF] = None
        self._distractor_anchor_pred_bbox: Optional[np.ndarray] = None
        self._distractor_anchor_uncertainty: float = 0.0
        self._distractor_mode_overlap_lock_active: bool = False
        self._distractor_mode_overlap_clear_count: int = 0
        self._distractor_mode_overlap_lock_frames: int = 0

        self._occ_phase: int = 0
        self._pending_candidates: List[CandidateRecord] = []
        self._reacq_confirm_active: bool = False
        self._reacq_confirm_streak: int = 0

        self._cand_frames: List[List[CandidateRecord]] = []
        self._occ_cam_vels: List = []

        self._entry_streak: int = 0

        self.ekf: Optional[BBoxEKF] = None
        self._last_H: Optional[np.ndarray] = None
        self._last_H_reliable: bool = False
        self._FLOW_LONG_EDGE = 300

        self._cached_pts: Optional[np.ndarray] = None
        self._cached_shape: Optional[tuple[int, int]] = None

        self._low_score_streak = 0
        self._occlusion_patience = occlusion_patience
        self._occlusion_hysteresis = occlusion_hysteresis
        self._gated_score = 1.0
        self.visual_mode: str = "tracking"
        self.visual_reason: str = ""
        self.visual_details: str = ""
        self._visual_state = VisualState(
            mode=self.visual_mode,
            reason=self.visual_reason,
            details=self.visual_details,
        )
        self._spike_watcher = SpikeWatcher(self)
        self._distractor_mode_subsystem = DistractorModeSubsystem(self)
        self._camera_motion_subsystem = CameraMotionSubsystem(self)
        self._occlusion_subsystem = OcclusionRecoverySubsystem(self)

    def _attach_inner_feature_state(self) -> None:
        """(Re)attach the research-feature state onto the live inner SiamABC tracker.

        Two research features attach STATE onto the inner tracker (``self.tracker``):
        the DRM-introspection enable flag (so its track() computes the response-map
        secondary-peak signal) and the telemetry flag + inner counter dict. The
        inner tracker is reset / re-initialized per clip and its own __init__
        restores these to inert defaults (``_drm_introspection_enabled=False``,
        ``_research_telemetry_enabled=False``, ``_tlm_inner=None``). Attaching once in
        __init__ is therefore not enough -- from the 2nd clip on the live inner
        tracker loses them. This method is the single source of truth, called from
        __init__ AND from initialize() (the per-clip entry) so the inner tracker used
        to track each clip always carries the state.

        Idempotent and safe to call repeatedly. Attaches NOTHING for a disabled
        feature (leaving the attribute absent so the OFF path is byte-for-byte
        inert). Crucially, it never blows away un-folded inner telemetry counts: a
        fresh ``_tlm_inner`` is created only when none currently exists (i.e. it was
        lost on a per-clip inner re-init); an existing dict is kept as-is so counts
        that ``_emit_video_telemetry`` has not yet folded survive.
        """
        # DRM introspection: only set the flag when the feature is on; otherwise
        # leave it absent so the inner track() skips the block entirely.
        if self._drm_introspection is not None:
            self.tracker._drm_introspection_enabled = True
            self.tracker._drm_scale_aware_divergence = bool(
                self._drm_introspection_scale_aware
            )
            self.tracker._drm_write_distractor_bank = bool(
                self._drm_introspection_distractor_bank_enabled
            )

        # Research telemetry: flip the inner flag and ensure a valid counter dict
        # exists -- but preserve any existing dict so counts not yet folded by
        # _emit_video_telemetry() are not discarded. Only mint a fresh one when the
        # inner tracker lost it (None) on a per-clip re-init.
        if self._research_telemetry_enabled:
            self.tracker._research_telemetry_enabled = True
            if getattr(self.tracker, "_tlm_inner", None) is None:
                self.tracker._tlm_inner = self._new_inner_telemetry_counters()

    def _restore_inner_tracker_sequence_defaults(self) -> None:
        """
        Restore SiamABC fields that SiamRAM may adapt during a sequence.

        The expensive SiamABC/TRT object is intentionally reused across videos,
        but its template cadence/window/gates are sequence-local state once any
        adaptive mode has touched them.
        """
        tr = self.tracker
        tr.N = self._initial_template_N
        tr.memory_window_size = self._initial_template_window
        tr.dynamic_update_threshold = self._initial_dynamic_update_threshold
        tr.tracking_config["iou_threshold"] = self._initial_iou_threshold

        if hasattr(tr, "all_memory_imgs"):
            tr.all_memory_imgs = deque(
                tr.all_memory_imgs, maxlen=self._initial_template_window
            )
        if hasattr(tr, "classification_scores"):
            tr.classification_scores = deque(
                tr.classification_scores, maxlen=self._initial_template_window
            )
        tracking_state = getattr(tr, "tracking_state", None)
        if tracking_state is not None:
            tracking_state.mapping = None
            tracking_state.prev_size = None
            if hasattr(tracking_state, "paths"):
                tracking_state.paths.clear()

    def _discard_pending_descriptor(self) -> None:
        """
        Drain an unfinished async descriptor from the previous sequence.

        Clearing the Python handle is not enough when OSNet ran on a side CUDA
        stream: the kernel may still be in flight and overlap the next video's
        initialization. Resolve/synchronize it, then intentionally drop the
        descriptor so no old-video appearance enters the fresh memory bank.
        """
        # This is the one hook that fires at EVERY clip boundary in both run
        # modes: the run loop's per-clip `finally` calls it after each clip
        # (fresh-tracker mode), and initialize() calls it before each clip
        # (reused-tracker mode). Emit the just-finished video's telemetry here so
        # aggregation works regardless of mode. Idempotent + no-op when disabled.
        self._emit_video_telemetry()

        pending = self._pending_desc
        self._pending_desc = None
        if pending is None:
            return
        try:
            _resolve_descriptor(pending["handle"])
        except Exception:
            # This is only best-effort cleanup at a sequence boundary. The fresh
            # initialize below fully resets target state even if cleanup fails.
            pass

    def initialize(
        self,
        frame: np.ndarray,
        bbox,
    ) -> None:
        """
        Same as the original except:
        1. _frame_scale is computed from the first frame so all subsequent
            processing is capped at the configured long-edge limit.
        2. The frame passed to the tracker and all internal state is in
            proc-frame (scaled) coordinates from the start.
        3. The caller-supplied bbox (full-frame pixels) is scaled down before
            being stored or passed to the underlying tracker.

        After this call every piece of internal state — EKF, conf_history,
        size_history, current_bbox, held_box, tracker.tracking_state.bbox —
        is in proc-frame pixel coordinates.  update() scales the output back.
        """

        # A single tracker instance is reused across videos to avoid rebuilding
        # heavy models/TRT engines. Keep that warm state, but reset every adaptive
        # per-sequence accumulator so statistics cannot leak video-to-video.
        # _discard_pending_descriptor() (above) already emitted the telemetry for
        # the video that just finished (reused-tracker mode) before this fresh
        # per-sequence reset; no separate flush is needed here.
        self._discard_pending_descriptor()
        self._overlap_active = None
        if self._drm_margin_auto is not None:
            self._drm_margin_auto.reset()
            self._drm_kwargs["margin"] = self._drm_margin_auto.value
        if self._template_rate_auto is not None:
            self._template_rate_auto.reset()
        if self._drm_introspection is not None:
            self._drm_introspection.reset()
        if self._frame_dynamics_processor is not None:
            self._frame_dynamics_processor.reset()
        self._camera_motion_adapt_linger_left = 0
        self._target_motion_adapt_linger_left = 0
        self._last_cam_motion_norm = 0.0
        self._last_target_motion_px = 0.0
        self._heavy_motion_cache_frame_idx = -1
        self._heavy_motion_cache_weighted_disp = 0.0
        self._heavy_motion_cache_inst_disp = 0.0
        self._last_heavy_cam_motion = False
        self._last_heavy_cam_disp = 0.0
        self._base_template_N = None
        self._base_template_window = None
        self._base_dynamic_update_threshold = None
        self._base_iou_threshold = None
        self._restore_inner_tracker_sequence_defaults()

        h_fr, w_fr = frame.shape[:2]
        long_edge = max(h_fr, w_fr)
        self._frame_scale = (
            self._max_proc_long_edge / long_edge
            if long_edge > self._max_proc_long_edge
            else 1.0
        )

        proc_frame = self._prescale_frame(frame)

        bbox = np.round(np.array(bbox, dtype=float) * self._frame_scale).astype(int)

        self.tracker.enable_tta()
        self.tracker.initialize(proc_frame, bbox)
        # The inner SiamABC tracker is reset / re-initialized per clip and its
        # __init__ restores the research-feature state to inert defaults, so
        # re-attach it onto the LIVE inner tracker now -- AFTER its (re)init for
        # this clip and AFTER _discard_pending_descriptor() above already emitted
        # (folded + zeroed) the previous clip's inner telemetry counts. This
        # emit-then-reattach ordering means a freshly-minted _tlm_inner can never
        # discard un-folded counts. No-op / attaches nothing when all features off.
        self._attach_inner_feature_state()
        self.current_bbox = bbox.copy()
        self.held_box = bbox.copy()
        self._stable_anchor_bbox = bbox.copy()
        self._distractor_focus_bbox = bbox.copy()
        self.in_occlusion = False
        self._distractor_occlusion_active = False
        self._pending_distractor_occlusion = False
        self.frame_idx = 0
        self.velocity = np.zeros(2)

        h_p, w_p = proc_frame.shape[:2]
        self._flow_scale = min(
            0.5,
            self._FLOW_LONG_EDGE / max(h_p, w_p),
        )
        small_init = cv2.resize(
            proc_frame,
            (int(w_p * self._flow_scale), int(h_p * self._flow_scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        self.prev_gray = cv2.cvtColor(small_init, cv2.COLOR_BGR2GRAY)

        self.init_frame = proc_frame.copy()
        self.init_bbox = bbox.copy()

        self._distractor_bank = deque(maxlen=self._distractor_bank_maxlen)
        self._out_of_frame = False
        self._exit_edge = None
        self._search_cx = None
        self._search_cy = None
        self._clear_all_frame_histories()
        self.memory.reset()
        self._yolo_cache = []
        self._occ_frames = 0
        self._occ_phase = 0
        self._pending_candidates = []
        self._reacq_confirm_active = False
        self._reacq_confirm_streak = 0
        self.recovered_early_occlusion = True
        self._last_H = None
        self._last_H_reliable = False
        self._last_yolo = []

        self._entry_streak = 0
        self._cand_frames = []
        self._occ_cam_vels = []

        self._target_class_id = None
        self._class_warmup_done = False
        self._class_votes = {}
        self._yolo_detectable = False
        self._detectability_probe_done = False
        self._detectability_runs = 0
        self._detectability_hits = 0
        self._last_recovery_patch = None
        self._last_recovery_info = None
        self._clear_jump_watch_state()
        self._jump_watch_last_step_norm = 0.0
        self._spike_debug_speed_norm = float("nan")
        self._spike_debug_baseline_norm = float("nan")
        self._spike_debug_ratio = float("nan")
        self._spike_debug_trigger = False
        self._stable_anchor_bbox = bbox.copy()
        self._distractor_focus_bbox = bbox.copy()
        self._distractor_mode_active = False
        self._distractor_mode_visual_reals = []
        self._distractor_mode_visual_distractors = []
        self._distractor_mode_roi = None
        self._distractor_mode_roi_size = None
        self._distractor_mode_stable_count = 0
        self._distractor_mode_ambiguous_count = 0
        self._distractor_mode_below_gate_count = 0
        self._distractor_mode_reentry_cooldown = 0
        self._distractor_mode_memory_freeze_left = 0
        self._distractor_mode_template_freeze_left = 0
        self._distractor_anchor_ekf = None
        self._distractor_anchor_pred_bbox = None
        self._distractor_mode_overlap_lock_active = False
        self._distractor_mode_overlap_clear_count = 0
        self._distractor_mode_overlap_lock_frames = 0
        self._distractor_anchor_uncertainty = 0.0
        self._jump_reject_distractor_timer = 0
        if self._distractor_mode_active or self._jump_reject_distractor_timer > 0:
            self.visual_mode = "distractor"
            self.visual_reason = "Distractor suppression active"
            self.visual_details = ""
        else:
            self.visual_mode = "tracking"
            self.visual_reason = "Normal tracking"
            self.visual_details = ""
        self._sync_visual_state()

        self.ekf = BBoxEKF(
            bbox,
            process_noise=self.ekf_process_noise,
            meas_noise=self.ekf_meas_noise,
        )

        from utils.utils import _extract_descriptor

        desc = _extract_descriptor(proc_frame, bbox)

        if desc is not None:
            self.memory.try_admit(bbox, desc, bbox)

    def _overlap_enabled(self) -> bool:
        """
        Whether the Tier-3 OSNet/SiamABC GPU overlap is active. Resolved lazily on
        first use so the OSNet extractor (and its CUDA device) already exists.
        """
        if self._overlap_active is None:
            self._overlap_active = bool(
                self._osnet_async_overlap
                and self._siam_stream is not None
                and descriptor_async_supported()
            )
            if self._overlap_active:
                # Make the SiamABC stream wait for any init work still pending on
                # the default stream (template features built during initialize),
                # so the first streamed forward can't race those tensors.
                self._siam_stream.wait_stream(torch.cuda.default_stream())
        return self._overlap_active

    def update(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, List]:
        """
        Thin wrapper around `_update_impl`. When the OSNet/SiamABC overlap is
        active, the whole per-frame pipeline runs on a dedicated non-default CUDA
        stream so SiamABC(frame N+1) overlaps the OSNet forward queued for
        frame N on the extractor's side stream. Otherwise it's a direct call.
        """
        if self._overlap_enabled():
            with torch.cuda.stream(self._siam_stream):
                return self._update_impl(frame)
        return self._update_impl(frame)

    def _update_impl(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, List]:
        """
        Same as the original except:
        • The raw frame is prescaled to proc resolution at the very top.
            Every function called from here — _estimate_homography, _normal_update,
            _occlusion_update, tracker.update, _extract_descriptor, _yolo_detect —
            receives proc_frame.  None of them ever touch the full-res frame.
        • The returned bbox and _last_yolo detections are scaled back to
            full-frame coordinates before being handed to the caller.

        Internally all coordinates remain in proc-frame space.
        """

        proc_frame = self._prescale_frame(frame)

        self.frame_idx += 1
        self._last_yolo = []
        self._yolo_cache = []
        self._spike_debug_speed_norm = float("nan")
        self._spike_debug_baseline_norm = float("nan")
        self._spike_debug_ratio = float("nan")
        self._spike_debug_trigger = False
        ekf = self.ekf
        assert ekf is not None

        H, H_reliable, current_gray = self._estimate_homography(proc_frame)
        self._last_H = H
        self._last_H_reliable = H_reliable

        # Refresh the heavy-camera-motion verdict every frame (incl. occlusion)
        # so the viz marker stays live. Cheap: displacement is cached per frame,
        # and _normal_update re-runs this with the fresh pred bbox afterwards.
        self._is_heavy_camera_motion(proc_frame)

        if self.in_occlusion and self._out_of_frame:
            ekf.P = ekf.P + ekf.Q
        else:
            ekf.predict(H=H, H_reliable=H_reliable)

        if self.in_occlusion:
            bbox, score = self._occlusion_update(proc_frame)
            self.visual_mode = "occluded"
            if not self.visual_reason:
                self.visual_reason = "Occlusion recovery"
            self.visual_details = ""
        else:
            bbox, score = self._normal_update(proc_frame)

        self.prev_gray = current_gray

        scale_inv = 1.0 / self._frame_scale

        if self.in_occlusion:
            self._sync_visual_state()
            return np.zeros(4, dtype=int), 0.0, True, self._last_yolo

        bbox_out = np.round(np.array(bbox, dtype=float) * scale_inv).astype(int)

        yolo_out = [
            np.round(np.array(b, dtype=float) * scale_inv).astype(int)
            for b in self._last_yolo
        ]

        self._sync_visual_state()
        return bbox_out, float(score), self.in_occlusion, yolo_out

    def _evaluate_hard_jump_candidate(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> Tuple[bool, Optional[np.ndarray], float, float, float, Optional[float]]:
        """
        Backward-compatible wrapper delegated to SpikeWatcher.
        """
        return self._spike_watcher.evaluate_hard_jump_candidate(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
        )

    @staticmethod
    def _camera_displacement_from_h_at_point(
        H: Optional[np.ndarray],
        px: float,
        py: float,
    ) -> Optional[np.ndarray]:
        """
        Estimate camera-induced displacement at a specific image point using H.

        Returns None when homography cannot be safely evaluated.
        """
        if H is None:
            return None
        denom = H[2, 0] * px + H[2, 1] * py + H[2, 2]
        if abs(float(denom)) < 1e-8:
            return None
        npx = (H[0, 0] * px + H[0, 1] * py + H[0, 2]) / denom
        npy = (H[1, 0] * px + H[1, 1] * py + H[1, 2]) / denom
        if not np.isfinite(npx) or not np.isfinite(npy):
            return None
        return np.array([float(npx - px), float(npy - py)], dtype=float)

    def _record_spike_step_norm(
        self,
        new_bbox: np.ndarray,
        H: Optional[np.ndarray],
        H_reliable: bool,
        *,
        append: bool = True,
    ) -> Optional[float]:
        """
        Computes an incremental camera-compensated step norm so that
        _evaluate_hard_jump_candidate does not have to recompute the entire
        baseline window from _conf_history each frame.
        """
        if not self._conf_history:
            return None
        prev_bbox = self._conf_history[-1].bbox
        h_step = H if bool(H_reliable) else None
        step = self._camera_compensated_step_norm(
            prev_bbox=np.asarray(prev_bbox, dtype=float),
            curr_bbox=np.asarray(new_bbox, dtype=float),
            H=h_step,
        )
        if append:
            self._spike_step_norms.append(float(step))
        return float(step)

    def _camera_compensated_step_norm(
        self,
        prev_bbox: np.ndarray,
        curr_bbox: np.ndarray,
        H: Optional[np.ndarray] = None,
    ) -> float:
        """
        Normalized bbox-centre displacement after subtracting camera motion.

        Uses H whenever available. Falls back to raw displacement only when
        homography is unavailable or cannot be safely evaluated.
        """
        prev = np.asarray(prev_bbox, dtype=float)
        curr = np.asarray(curr_bbox, dtype=float)
        p0x = float(prev[0] + prev[2] / 2.0)
        p0y = float(prev[1] + prev[3] / 2.0)
        p1x = float(curr[0] + curr[2] / 2.0)
        p1y = float(curr[1] + curr[3] / 2.0)

        rel_dx = p1x - p0x
        rel_dy = p1y - p0y
        if H is not None:
            cam_delta = self._camera_displacement_from_h_at_point(H, p0x, p0y)
            if cam_delta is not None:
                rel_dx -= float(cam_delta[0])
                rel_dy -= float(cam_delta[1])

        diag = float(np.hypot(prev[2], prev[3])) + 1e-8
        return float(np.hypot(rel_dx, rel_dy) / diag)

    @staticmethod
    def _normalized_center_distance(
        ref_bbox: np.ndarray,
        test_bbox: np.ndarray,
    ) -> float:
        ref = np.asarray(ref_bbox, dtype=float)
        tst = np.asarray(test_bbox, dtype=float)
        rcx = float(ref[0] + ref[2] / 2.0)
        rcy = float(ref[1] + ref[3] / 2.0)
        tcx = float(tst[0] + tst[2] / 2.0)
        tcy = float(tst[1] + tst[3] / 2.0)
        diag = float(np.hypot(ref[2], ref[3])) + 1e-8
        return float(np.hypot(tcx - rcx, tcy - rcy) / diag)

    def _select_pre_spike_anchor_bbox(
        self,
    ) -> Optional[np.ndarray]:
        """
        Backward-compatible wrapper delegated to SpikeWatcher.
        """
        return self._spike_watcher.select_pre_spike_anchor_bbox()

    def _target_descriptor_history(
        self,
    ) -> List[np.ndarray]:
        """
        Get descriptor history for the real target from memory buffers.
        """
        descs: List[np.ndarray] = []
        if hasattr(self.memory, "_buf"):
            buf = list(getattr(self.memory, "_buf"))
            for _bbox, d in buf[-self._distractor_mode_history_limit:]:
                if d is not None:
                    descs.append(np.asarray(d, dtype=float))
        if not descs and hasattr(self.memory, "_drm"):
            drm = list(getattr(self.memory, "_drm"))
            for _bbox, d, _t in drm[-self._distractor_mode_history_limit:]:
                if d is not None:
                    descs.append(np.asarray(d, dtype=float))
        best = self.memory.best_descriptor()
        if best is not None:
            descs.append(np.asarray(best, dtype=float))
        return descs

    def _get_active_distractor_bank(
        self,
    ) -> List[np.ndarray]:
        """
        Return distractor descriptors currently stored for scoring/penalties.

        Preference is the memory-owned bank; local legacy bank is used as a
        fallback for compatibility.
        """
        mem_bank: List[np.ndarray] = []
        if hasattr(self.memory, "get_distractor_bank"):
            mem_bank = list(getattr(self.memory, "get_distractor_bank")())
        if mem_bank:
            return mem_bank
        return list(self._distractor_bank)

    def _add_distractor_descriptor(
        self,
        desc: Optional[np.ndarray],
    ) -> None:
        """
        Insert one descriptor into both local and memory distractor banks.
        """
        if desc is None or self._distractor_bank_maxlen <= 0:
            return
        desc_arr = np.asarray(desc, dtype=float).copy()
        self._distractor_bank.append(desc_arr)
        if hasattr(self.memory, "add_distractor"):
            getattr(self.memory, "add_distractor")(desc_arr)

    def _extend_distractor_descriptors(
        self,
        descs: List[Optional[np.ndarray]],
    ) -> None:
        """
        Insert multiple descriptors into both local and memory distractor banks.
        """
        if self._distractor_bank_maxlen <= 0 or not descs:
            return
        valid = [np.asarray(d, dtype=float).copy() for d in descs if d is not None]
        if not valid:
            return
        self._distractor_bank.extend(valid)
        if hasattr(self.memory, "extend_distractor_bank"):
            getattr(self.memory, "extend_distractor_bank")(valid)

    def _get_distractor_mode_roi(
        self,
        frame: np.ndarray,
        focus_override: Optional[np.ndarray] = None,
        uncertainty_pad_px: float = 0.0,
    ) -> Tuple[int, int, int, int]:
        return self._distractor_mode_subsystem.get_distractor_mode_roi(
            frame=frame,
            focus_override=focus_override,
            uncertainty_pad_px=uncertainty_pad_px,
        )

    @staticmethod
    def _mahalanobis_distance_sq(
        point_xy: np.ndarray,
        mean_xy: np.ndarray,
        cov: np.ndarray,
    ) -> Optional[float]:
        try:
            delta = np.asarray(point_xy, dtype=float) - np.asarray(mean_xy, dtype=float)
            cov_arr = np.asarray(cov, dtype=float)
            inv_cov = np.linalg.pinv(cov_arr)
            d2 = float(delta.T @ inv_cov @ delta)
            if not np.isfinite(d2):
                return None
            return d2
        except Exception:
            return None

    def _yolo_detect_in_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> List[np.ndarray]:
        rx, ry, rw, rh = roi
        crop = frame[ry: ry + rh, rx: rx + rw]
        if crop.size == 0:
            return []
        results = self.yolo.predict(
            crop,
            conf=self.yolo_conf,
            iou=self.yolo_iou_thr,
            verbose=False,
            imgsz=self.yolo_imgsz,
            augment=self.yolo_augment,
        )
        boxes: List[np.ndarray] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            result_boxes = results[0].boxes
            xyxy_all = result_boxes.xyxy.detach().cpu().numpy()
            cls_all = result_boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
            if self._yolo_filter_class and self._target_class_id is not None:
                keep = cls_all == int(self._target_class_id)
                xyxy_all = xyxy_all[keep]
            if xyxy_all.size > 0:
                xywh = np.empty((xyxy_all.shape[0], 4), dtype=np.int32)
                xywh[:, 0] = xyxy_all[:, 0].astype(np.int32, copy=False) + rx
                xywh[:, 1] = xyxy_all[:, 1].astype(np.int32, copy=False) + ry
                xywh[:, 2] = (xyxy_all[:, 2] - xyxy_all[:, 0]).astype(np.int32, copy=False)
                xywh[:, 3] = (xyxy_all[:, 3] - xyxy_all[:, 1]).astype(np.int32, copy=False)
                boxes = [xywh[i].copy() for i in range(xywh.shape[0])]
        return boxes

    def _reinit_dynamic_template_only(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        score_hint: float = 1.0,
    ) -> np.ndarray:
        """
        Re-anchor SiamABC dynamic template/search state without touching static template.

        This keeps frame-0 static template features intact and only refreshes
        dynamic features + dynamic memory around the provided bbox.
        """
        bb = self._clamp_bbox_to_frame(np.array(bbox, dtype=int).copy(), frame)
        self.tracker.tracking_state.bbox = bb.copy()

        dyn_template = self.tracker.get_template_features(frame, bb)
        dyn_search, _, _ = self.tracker.get_search_features(frame, bb)
        self.tracker.dynamic_template_features = dyn_template
        self.tracker.dynamic_search_features = dyn_search
        self.tracker.running_dynamic_image = frame.copy()
        self.tracker.running_dynamic_bbox = bb.copy()

        mem_cap = max(1, int(getattr(self.tracker, "memory_window_size", 1)))
        thr = float(getattr(self.tracker, "dynamic_update_threshold", 0.87))
        seed_score = float(np.clip(score_hint, 0.0, 1.0))
        seed_score = max(seed_score, min(1.0, thr + 1e-3))

        self.tracker.all_memory_imgs = deque([[frame.copy(), bb.copy()]], maxlen=mem_cap)
        self.tracker.classification_scores = deque([seed_score], maxlen=mem_cap)
        self.tracker._best_idx = 0
        self.tracker._best_score = seed_score
        self.tracker._is_full = False

        if hasattr(self.tracker, "_prev_bbox"):
            self.tracker._prev_bbox = bb.copy()
        if hasattr(self.tracker, "prev_good_bbox"):
            self.tracker.prev_good_bbox = bb.copy()
        if hasattr(self.tracker.tracking_state, "pred_score"):
            self.tracker.tracking_state.pred_score = seed_score
        if hasattr(self.tracker.net, "invalidate_template_cache"):
            self.tracker.net.invalidate_template_cache()

        return bb

    def _enter_distractor_mode(
        self,
        anchor_bbox: np.ndarray,
    ) -> None:
        self._distractor_mode_subsystem.enter_distractor_mode(
            anchor_bbox=anchor_bbox,
        )

    def _exit_distractor_mode(
        self,
        reason: str,
        resolved: bool = False,
    ) -> None:
        self._distractor_mode_subsystem.exit_distractor_mode(
            reason=reason,
            resolved=resolved,
        )

    def _maybe_apply_ambiguity_hold(
        self,
        *,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        best_sim: float,
        focus_override: Optional[np.ndarray],
        cands: List[Tuple],
        top_margin: float,
        roi: Tuple[int, int, int, int],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Updates the ambiguity counter and, while the top-2 score margin is
        below the switch threshold for fewer than the allowed hold frames,
        returns a (hold_bbox, score) tuple to keep identity locked on the
        previous focus instead of switching to the slightly-higher-scoring
        candidate. Returns None when the caller should fall through to the
        ranking-driven commit path.
        """
        ambiguous = (
            len(cands) > 1
            and top_margin < self._distractor_mode_switch_margin
        )
        if ambiguous:
            self._distractor_mode_ambiguous_count += 1
        else:
            self._distractor_mode_ambiguous_count = 0

        if not (
            ambiguous
            and self._distractor_mode_ambiguous_count
            <= self._distractor_mode_ambiguity_hold_frames
        ):
            return None

        hold_source = (
            focus_override
            if focus_override is not None
            else (
                self._distractor_focus_bbox
                if self._distractor_focus_bbox is not None
                else pred_bbox
            )
        )
        hold_bbox = self._clamp_bbox_to_frame(
            np.array(hold_source, dtype=int).copy(), frame
        )
        self._distractor_mode_visual_reals = [hold_bbox.copy()]
        self._distractor_mode_visual_distractors = [
            np.array(bb, dtype=int).copy() for bb, *_ in cands
        ]
        self._distractor_mode_stable_count = 0
        self._distractor_focus_bbox = hold_bbox.copy()
        self.tracker.tracking_state.bbox = hold_bbox.copy()
        self.visual_mode = "distractor"
        self.visual_reason = "Distractor mode: ambiguity hold"
        self.visual_details = (
            f"margin={top_margin:.3f}<{self._distractor_mode_switch_margin:.3f}  "
            f"hold={self._distractor_mode_ambiguous_count}/{self._distractor_mode_ambiguity_hold_frames}  "
            f"best_app={best_sim:.2f}  cands={len(cands)}  roi={roi[2]}x{roi[3]}"
        )
        self._jump_reject_distractor_timer = max(
            self._jump_reject_distractor_timer,
            self._jump_reject_distractor_mode_frames,
        )
        adjusted_score = max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
        adjusted_score = self._apply_distractor_mode_penalty(
            frame, hold_bbox, adjusted_score
        )
        return hold_bbox.copy(), float(adjusted_score)

    def _maybe_engage_overlap_motion_lock(
        self,
        *,
        frame: np.ndarray,
        focus_bbox_arr: np.ndarray,
        best_sim: float,
        max_overlap_iou: float,
        distractor_entries: List[Tuple],
        cands: List[Tuple],
        roi: Tuple[int, int, int, int],
        score: float,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Handles the overlap motion-lock branch of distractor-mode update.

        When the best ROI candidate overlaps a distractor heavily (entry
        threshold), engages a motion-only lock that holds the EKF/anchor box
        instead of switching identity until the overlap clears or the lock
        times out.

        Returns (motion_bbox, adjusted_score) when the lock is engaged and the
        caller should commit that result and return now. Returns None when
        the lock did not engage (or just released cleanly), so the caller
        should fall through to the normal ranking path.
        """
        if not self._distractor_mode_overlap_motion_lock_enabled:
            return None

        if (
            not self._distractor_mode_overlap_lock_active
            and distractor_entries
            and max_overlap_iou >= self._distractor_mode_overlap_iou_enter
        ):
            self._distractor_mode_overlap_lock_active = True
            self._distractor_mode_overlap_clear_count = 0
            self._distractor_mode_overlap_lock_frames = 0

        if not self._distractor_mode_overlap_lock_active:
            return None

        self._distractor_mode_overlap_lock_frames += 1
        if (
            not distractor_entries
            or max_overlap_iou <= self._distractor_mode_overlap_iou_exit
        ):
            self._distractor_mode_overlap_clear_count += 1
        else:
            self._distractor_mode_overlap_clear_count = 0

        cleared = (
            self._distractor_mode_overlap_clear_count
            >= self._distractor_mode_overlap_clear_frames
        )
        timed_out = (
            self._distractor_mode_overlap_lock_frames
            >= self._distractor_mode_overlap_lock_max_frames
        )

        if cleared or timed_out:
            self._distractor_mode_overlap_lock_active = False
            self._distractor_mode_overlap_clear_count = 0
            self._distractor_mode_overlap_lock_frames = 0
            return None

        motion_source = self._distractor_anchor_pred_bbox
        if motion_source is None and self.ekf is not None:
            motion_source = self.ekf.get_bbox()
        if motion_source is None:
            motion_source = np.array(focus_bbox_arr, dtype=int)
        motion_bbox = self._clamp_bbox_to_frame(
            np.array(motion_source, dtype=int).copy(),
            frame,
        )
        self._distractor_mode_visual_reals = [motion_bbox.copy()]
        self._distractor_mode_visual_distractors = [
            np.array(bb, dtype=int).copy() for bb, *_ in cands
        ]
        self._distractor_mode_stable_count = 0
        self._distractor_mode_ambiguous_count = 0
        self._distractor_focus_bbox = motion_bbox.copy()
        self.tracker.tracking_state.bbox = motion_bbox.copy()
        self.visual_mode = "distractor"
        self.visual_reason = "Distractor mode: overlap motion lock"
        self.visual_details = (
            f"ov={max_overlap_iou:.2f}  "
            f"clear={self._distractor_mode_overlap_clear_count}/{self._distractor_mode_overlap_clear_frames}  "
            f"frames={self._distractor_mode_overlap_lock_frames}/{self._distractor_mode_overlap_lock_max_frames}  "
            f"best_app={best_sim:.2f}  roi={roi[2]}x{roi[3]}"
        )
        self._jump_reject_distractor_timer = max(
            self._jump_reject_distractor_timer,
            self._jump_reject_distractor_mode_frames,
        )
        adjusted_score = max(
            float(score),
            float(np.clip(best_sim, 0.0, 1.0)),
        )
        adjusted_score = self._apply_distractor_mode_penalty(
            frame,
            motion_bbox,
            adjusted_score,
        )
        return motion_bbox.copy(), float(adjusted_score)

    def _score_distractor_candidates(
        self,
        *,
        dets_for_desc: List[np.ndarray],
        det_descs: List[Optional[np.ndarray]],
        refs: List[np.ndarray],
        focus_bbox_arr: np.ndarray,
        focus_cx: float,
        focus_cy: float,
        dist_sigma: float,
        distractor_bank: List[np.ndarray],
        anchor_mean: Optional[np.ndarray],
        anchor_cov: Optional[np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]]:
        """
        Builds the per-candidate DRM score tuples for distractor-mode ranking.

        For each (bbox, descriptor) pair this scores:
        - appearance similarity vs the target descriptor history (max over refs);
        - IoU with the focus bbox;
        - Mahalanobis distance vs the anchor EKF mean (gated, candidate dropped
          if it exceeds the chi-squared threshold);
        - spatial Gaussian distance penalty from the focus centre;
        - cosine similarity vs the recent distractor bank (negative term).

        Returns a list of tuples shaped exactly like the legacy inline path:
        (bbox_int, desc, app_sim, drm_score, iou_focus, dist_term, dist_sim, maha_d2).
        """
        cands: List[
            Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]
        ] = []
        for bb, dd in zip(dets_for_desc, det_descs):
            if dd is None:
                continue
            bb_arr = np.asarray(bb, dtype=float)
            app_sim = max(float(_cos_sim(dd, r)) for r in refs)
            iou_focus = float(_iou(focus_bbox_arr, bb_arr))

            cand_cx = float(bb_arr[0] + bb_arr[2] / 2.0)
            cand_cy = float(bb_arr[1] + bb_arr[3] / 2.0)
            maha_d2 = float("nan")
            if (
                self._distractor_mode_mahalanobis_gate_enabled
                and anchor_mean is not None
                and anchor_cov is not None
            ):
                cov_eff = anchor_cov + np.eye(2, dtype=float) * self._distractor_mode_mahalanobis_meas_var
                maha_val = self._mahalanobis_distance_sq(
                    np.array([cand_cx, cand_cy], dtype=float),
                    anchor_mean,
                    cov_eff,
                )
                if maha_val is not None:
                    maha_d2 = float(maha_val)
                    if maha_d2 > self._distractor_mode_mahalanobis_threshold:
                        continue
            d_center = float(np.hypot(cand_cx - focus_cx, cand_cy - focus_cy))
            dist_term = 1.0 - float(np.exp(-0.5 * (d_center / (dist_sigma + 1e-8)) ** 2))

            dist_sim = 0.0
            if distractor_bank and self._distractor_drm_gamma > 0.0:
                dist_sim = max(float(_cos_sim(dd, nu)) for nu in distractor_bank)

            drm_score = (
                self._distractor_drm_lam_app * app_sim
                + self._distractor_drm_lam_iou * iou_focus
                - self._distractor_drm_lam_dist * dist_term
                - self._distractor_drm_gamma * dist_sim
            )

            cands.append(
                (
                    np.array(bb, dtype=int),
                    dd,
                    app_sim,
                    float(drm_score),
                    float(iou_focus),
                    float(dist_term),
                    float(dist_sim),
                    float(maha_d2),
                )
            )
        return cands

    def _advance_distractor_anchor_ekf(
        self,
        frame: np.ndarray,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
    ]:
        """
        Predicts the distractor-mode anchor EKF forward one frame using the
        latest homography, then derives the focus-override bbox, the EKF
        position mean/covariance used for Mahalanobis gating, and an additional
        ROI padding scaled by the EKF position uncertainty.

        Returns (focus_override, anchor_mean, anchor_cov, uncertainty_pad_px).
        When the anchor EKF is disabled or not yet initialised, returns the
        no-op (None, None, None, 0.0) tuple.
        """
        if not (
            self._distractor_mode_anchor_ekf_enabled
            and self._distractor_anchor_ekf is not None
        ):
            return None, None, None, 0.0

        self._distractor_anchor_ekf.predict(
            H=self._last_H,
            H_reliable=self._last_H_reliable,
        )
        anchor_bbox = self._clamp_bbox_to_frame(
            self._distractor_anchor_ekf.get_bbox(), frame
        )
        self._distractor_anchor_pred_bbox = anchor_bbox.copy()
        self._distractor_anchor_uncertainty = float(
            self._distractor_anchor_ekf.get_uncertainty()
        )
        focus_override = anchor_bbox.copy()
        uncertainty_pad_px = min(
            self._distractor_mode_anchor_uncertainty_roi_cap,
            self._distractor_mode_anchor_uncertainty_roi_scale
            * self._distractor_anchor_uncertainty,
        )
        anchor_mean = np.asarray(self._distractor_anchor_ekf.x[:2], dtype=float).copy()
        anchor_cov = np.asarray(self._distractor_anchor_ekf.P[:2, :2], dtype=float).copy()
        return focus_override, anchor_mean, anchor_cov, uncertainty_pad_px

    def _distractor_mode_update(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        effective_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        return self._distractor_mode_subsystem.distractor_mode_update(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
            effective_threshold=effective_threshold,
        )

    def _clear_jump_watch_state(
        self,
    ) -> None:
        self._spike_watcher.clear_watch_state()

    def _start_jump_watch(
        self,
        prev_bbox: np.ndarray,
        first_switched_bbox: np.ndarray,
        spike_speed_norm: float,
        spike_ratio: float,
        baseline_norm: float,
        sim: Optional[float],
    ) -> None:
        self._spike_watcher.start_watch(
            prev_bbox=prev_bbox,
            first_switched_bbox=first_switched_bbox,
            spike_speed_norm=spike_speed_norm,
            spike_ratio=spike_ratio,
            baseline_norm=baseline_norm,
            sim=sim,
        )

    def _apply_distractor_mode_penalty(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> float:
        return self._distractor_mode_subsystem.apply_distractor_mode_penalty(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
        )

    def _apply_hard_jump_rejection(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        effective_threshold: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Backward-compatible wrapper delegated to SpikeWatcher.
        """
        return self._spike_watcher.apply_hard_jump_rejection(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
            effective_threshold=effective_threshold,
        )

    @staticmethod
    def _camera_velocity_from_h(
        H: Optional[np.ndarray],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Projects the frame centre through H and returns the 2-vector camera
        velocity (px) at the centre. Returns zeros when H is None. Shared by
        _normal_update and _commit_reacquisition for cam_vel_history bookkeeping.
        """
        if H is None:
            return np.zeros(2)
        h_fr, w_fr = frame_shape[:2]
        cx, cy = w_fr / 2.0, h_fr / 2.0
        denom = H[2, 0] * cx + H[2, 1] * cy + H[2, 2] + 1e-8
        ncx = (H[0, 0] * cx + H[0, 1] * cy + H[0, 2]) / denom
        ncy = (H[1, 0] * cx + H[1, 1] * cy + H[1, 2]) / denom
        return np.array([ncx - cx, ncy - cy])

    def _sync_visual_state(self) -> None:
        self._visual_state = VisualState(
            mode=self.visual_mode,
            reason=self.visual_reason,
            details=self.visual_details,
        )

    def _set_visual_mode_for_normal_update(self) -> None:
        """
        Picks the visual_mode/reason/details strings shown to the UI overlay
        for normal-mode frames. Distractor-suppression timer wins over normal.
        """
        if self._jump_reject_distractor_timer > 0:
            self.visual_mode = "distractor"
            self.visual_reason = "Distractor suppression active"
        else:
            self.visual_mode = "tracking"
            self.visual_reason = "Normal tracking"
        self.visual_details = ""

    def _maybe_run_class_warmup(self, frame: np.ndarray) -> None:
        """
        Runs the YOLO class-vote warmup during the first few frames so that
        downstream YOLO calls can class-filter detections to the inferred target
        class. Once the warmup window closes, commits the majority class.
        """
        if not (
            self._yolo_filter_class
            and not self._class_warmup_done
            and self.frame_idx <= self._yolo_class_detect_frames * 3
            and self.frame_idx % max(1, self._yolo_class_detect_frames) == 0
        ):
            return
        self._try_detect_target_class(frame)
        if self.frame_idx >= self._yolo_class_detect_frames * 2:
            self._maybe_commit_target_class()

    def _compute_effective_threshold(self, frame: np.ndarray) -> float:
        """
        Returns the score threshold used to decide if the current frame is
        a tracking loss. Also adjusts the visual tracker search-context offset
        to widen the search when the object is far away ("long-distance" mode).
        """
        if self.frame_idx <= 0:
            return 0.0
        long_distanced_object = self._is_long_distance(frame)
        effective_threshold = (
            self.long_distance_conf_threshold
            if long_distanced_object
            else self.conf_threshold
        )
        self.tracker.offset = (
            self.tracker.tracking_config["search_context"]
            if not long_distanced_object
            else self.tracker.tracking_config["search_context"] + 0.5
        )
        return effective_threshold

    def _record_camera_motion_history(self, frame: np.ndarray) -> None:
        """
        Appends the per-frame scalar camera displacement magnitude and the
        2-vector camera velocity to their respective histories. Called once
        per successful normal-mode frame.
        """
        cam_disp = self._h_translation_magnitude(self._last_H, frame)
        self._cam_disp_history.append(cam_disp)
        self._cam_vel_history.append(
            self._camera_velocity_from_h(self._last_H, frame.shape)
        )

    def _maybe_update_stable_anchor(self, pred_bbox: np.ndarray) -> None:
        """
        Promotes the latest pred_bbox to be the new stable anchor (used as the
        pre-spike fallback) when motion is calm and no jump-watch / distractor
        timer is active. Anchor stays put otherwise so we keep a clean pre-jump
        reference for spike recovery.
        """
        if (
            self._jump_reject_distractor_timer > 0
            or self._jump_watch_active
            or self.current_bbox is None
        ):
            return
        step_norm_anchor = self._camera_compensated_step_norm(
            prev_bbox=np.asarray(self.current_bbox, dtype=float),
            curr_bbox=np.asarray(pred_bbox, dtype=float),
            H=self._last_H,
        )
        if step_norm_anchor <= self._spike_anchor_update_norm_max:
            self._stable_anchor_bbox = pred_bbox.copy()
            self._distractor_focus_bbox = pred_bbox.copy()

    def _clear_all_frame_histories(self) -> None:
        """
        Resets every per-frame history deque at once. Centralising this in one
        place avoids the previous pattern of scattered `.clear()` calls that
        drifted out of sync as new histories were added.
        """
        self._history.clear()

    def _commit_normal_frame_history(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
    ) -> None:
        """
        Atomically appends one frame's worth of state to every per-frame
        history deque consumed downstream (occlusion recovery, spike detection,
        shrinkage analysis, stable-anchor selection, velocity smoothing).

        Order matters: _record_spike_step_norm reads the previous
        _conf_history bbox, so it MUST run before the new _conf_history append.
        """
        cx = float(pred_bbox[0] + pred_bbox[2] / 2.0)
        cy = float(pred_bbox[1] + pred_bbox[3] / 2.0)
        cam_disp = self._h_translation_magnitude(self._last_H, frame)
        cam_vel = self._camera_velocity_from_h(self._last_H, frame.shape)
        spike_step_norm = self._record_spike_step_norm(
            new_bbox=pred_bbox,
            H=self._last_H,
            H_reliable=self._last_H_reliable,
            append=False,
        )
        self._history.commit_normal(
            record=FrameRecord(
                bbox=pred_bbox.copy(),
                velocity=self.velocity.copy(),
                H=self._last_H,
                H_reliable=self._last_H_reliable,
            ),
            center=np.array([cx, cy], dtype=float),
            velocity=self.velocity,
            size_wh=(int(pred_bbox[2]), int(pred_bbox[3])),
            cam_velocity=cam_vel,
            cam_displacement=cam_disp,
            spike_step_norm=spike_step_norm,
        )

    def _commit_recovery_frame_history(
        self,
        frame: np.ndarray,
        ekf_bbox: np.ndarray,
    ) -> None:
        """
        Lighter-weight history commit used by _commit_reacquisition. Writes
        centre, cam-vel, spike step-norm, and the FrameRecord — does NOT touch
        size_history, vel_history, or cam_disp_history, matching the legacy
        behaviour at the post-recovery commit point.
        """
        cx = float(ekf_bbox[0] + ekf_bbox[2] / 2.0)
        cy = float(ekf_bbox[1] + ekf_bbox[3] / 2.0)
        spike_step_norm = self._record_spike_step_norm(
            new_bbox=ekf_bbox,
            H=self._last_H,
            H_reliable=self._last_H_reliable,
            append=False,
        )
        self._history.commit_recovery(
            record=FrameRecord(
                bbox=ekf_bbox.copy(),
                velocity=self.velocity.copy(),
                H=self._last_H,
                H_reliable=self._last_H_reliable,
            ),
            center=np.array([cx, cy], dtype=float),
            cam_velocity=self._camera_velocity_from_h(self._last_H, frame.shape),
            spike_step_norm=spike_step_norm,
        )

    def _apply_camera_motion_template_adapt(self, heavy: bool) -> None:
        """
        Swap the inner SiamABC dynamic-template update params to the high-motion
        set while camera motion is heavy, and restore the base values otherwise.

        - N is the select_representatives() cadence (read fresh each frame, so a
          plain reassignment takes effect immediately).
        - memory_window_size is the maxlen of the fixed-size memory deques, so a
          change requires rebuilding them (oldest frames drop when shrinking),
          mirroring the warmup rebuild and _reinit_dynamic_template_only. The
          rebuild only runs on an actual size change, so a flickering motion
          signal does not thrash the deques every frame.
        - The dynamic-update admission gates (dynamic_update_threshold score gate
          and the iou_threshold continuity gate) are also swapped when their
          high-motion values are set (>= 0); each is read fresh per frame by the
          inner tracker, so a plain reassignment takes effect immediately.

        No-op unless camera_motion_template_adapt_enabled.
        """
        if not self._camera_motion_template_adapt_enabled:
            return
        tr = self.tracker
        if self._base_template_N is None:
            self._base_template_N = int(getattr(tr, "N", self._camera_motion_high_N))
            self._base_template_window = int(
                getattr(
                    tr,
                    "memory_window_size",
                    self._camera_motion_high_memory_window_size,
                )
            )
            self._base_dynamic_update_threshold = float(
                getattr(tr, "dynamic_update_threshold", 0.8)
            )
            self._base_iou_threshold = float(
                tr.tracking_config.get("iou_threshold", 0.3)
            )

        desired_n = self._camera_motion_high_N if heavy else self._base_template_N
        desired_w = (
            self._camera_motion_high_memory_window_size
            if heavy
            else self._base_template_window
        )

        self._set_template_N_window(desired_n, desired_w)

        # Dynamic-update admission gates (sentinel < 0 => leave the gate alone).
        if self._camera_motion_high_dynamic_update_threshold >= 0.0:
            desired_dut = (
                self._camera_motion_high_dynamic_update_threshold
                if heavy
                else self._base_dynamic_update_threshold
            )
            if float(getattr(tr, "dynamic_update_threshold", desired_dut)) != desired_dut:
                tr.dynamic_update_threshold = desired_dut

        if self._camera_motion_high_iou_threshold >= 0.0:
            desired_iou = (
                self._camera_motion_high_iou_threshold
                if heavy
                else self._base_iou_threshold
            )
            if float(tr.tracking_config.get("iou_threshold", desired_iou)) != desired_iou:
                tr.tracking_config["iou_threshold"] = desired_iou

    def _current_target_speed_px(self) -> float:
        """
        Current target speed in full-frame px/frame -- the same on-screen value
        the target-motion pill shows. self.velocity is in proc-frame px/frame, so
        dividing by the frame downscale maps it back to full-frame pixels (the
        pill multiplies by 1/_frame_scale, which is identical). Uses the previous
        frame's velocity (one-frame-stale, as in _apply_auto_template_rate).
        """
        scale = self._frame_scale if self._frame_scale else 1.0
        return float(np.linalg.norm(self.velocity)) / scale

    def _apply_target_motion_template_adapt(self, fast: bool) -> None:
        """
        Swap the inner SiamABC dynamic-template params (re-selection cadence N +
        memory window length) to the high-motion set while the TARGET is moving
        fast, and restore the base values otherwise.

        "Fast" is decided in _normal_update by comparing the object's speed in
        full-frame px/frame (the value shown in the target-motion pill) against
        target_motion_high_px_threshold. Reuses _set_template_N_window (memory
        deques rebuilt only on an actual window change) and the same lazily
        captured base N/window as the camera-motion path; only N and window are
        touched (the admission gates are left alone).

        No-op unless target_motion_template_adapt_enabled.
        """
        if not self._target_motion_template_adapt_enabled:
            return
        tr = self.tracker
        if self._base_template_N is None:
            self._base_template_N = int(
                getattr(tr, "N", self._target_motion_high_N)
            )
            self._base_template_window = int(
                getattr(
                    tr,
                    "memory_window_size",
                    self._target_motion_high_memory_window_size,
                )
            )

        desired_n = self._target_motion_high_N if fast else self._base_template_N
        desired_w = (
            self._target_motion_high_memory_window_size
            if fast
            else self._base_template_window
        )
        self._set_template_N_window(desired_n, desired_w)

    def _set_template_N_window(self, desired_n: int, desired_w: int) -> None:
        """
        Apply a dynamic-template cadence (N) and memory window size to the inner
        tracker, rebuilding the fixed-size memory deques only when the window
        actually changes (oldest frames drop when shrinking). Shared by the
        binary camera-motion adapt and the adaptive template_rate_auto path.
        """
        tr = self.tracker
        desired_n = max(1, int(desired_n))
        desired_w = max(1, int(desired_w))

        if int(getattr(tr, "N", desired_n)) != desired_n:
            tr.N = desired_n

        if int(getattr(tr, "memory_window_size", desired_w)) != desired_w:
            tr.memory_window_size = desired_w
            tr.all_memory_imgs = deque(tr.all_memory_imgs, maxlen=desired_w)
            tr.classification_scores = deque(
                tr.classification_scores, maxlen=desired_w
            )
            if tr.classification_scores:
                scores = np.array(tr.classification_scores, dtype=np.float16)
                tr._best_idx = int(np.argmax(scores))
                tr._best_score = float(scores[tr._best_idx])
            else:
                tr._best_idx = 0
                tr._best_score = 0.5
            tr._is_full = len(tr.classification_scores) == desired_w

    def _apply_auto_template_rate(self, frame: np.ndarray) -> None:
        """
        Adaptive dynamic-template update rate (template_rate_auto). Builds a 0..1
        "fastness" from this frame's camera motion and the (one-frame-stale)
        target velocity -- combined with max() so either being fast drives fast
        updating -- smooths it (EMA, inside the controller), and interpolates N
        and memory_window_size between the slow-end and fast-end values.

        Runs BEFORE self.tracker.update(), where N / memory_window_size are
        consumed. No-op unless the mode is enabled.
        """
        if self._template_rate_auto is None:
            return

        ref_diag = self._motion_gate_reference_diag(frame, bbox_hint=self.current_bbox)

        # Camera fastness: refresh + read the published bbox-relative magnitude.
        self._is_heavy_camera_motion(frame, bbox_hint=self.current_bbox)
        cam_ref = self._template_rate_auto_cam_ref_norm
        if cam_ref <= 0.0:
            cam_ref = self._camera_motion_heavy_norm_threshold
        cam_fast = (
            min(1.0, max(0.0, self._last_cam_motion_norm / cam_ref))
            if cam_ref > 0.0
            else 0.0
        )

        # Target fastness: speed (px/frame) over a bbox-diagonal reference.
        tgt_speed = float(np.linalg.norm(self.velocity))
        tgt_ref = self._template_rate_auto_target_ref_norm * ref_diag
        tgt_fast = (
            min(1.0, max(0.0, tgt_speed / tgt_ref)) if tgt_ref > 0.0 else 0.0
        )

        motion = max(cam_fast, tgt_fast)
        desired_n, desired_w = self._template_rate_auto.observe(motion)
        self._set_template_N_window(desired_n, desired_w)

    def _research_flag_echo(self) -> str:
        """Compact echo of which research features are ACTUALLY active + key params.

        Sourced from the tracker's stored feature objects/attributes so a reader
        can instantly tell, from any telemetry line, whether each feature was on
        (and its key resolved param) -- even on a run where a feature is ON but
        never fires. Pure read-only; never raises.
        """
        drm = "on" if self._drm_introspection is not None else "off"
        return f"flags(drm={drm})"

    @staticmethod
    def _new_telemetry_counters() -> dict:
        """Fresh zeroed research-telemetry counter dict (per-video or cumulative)."""
        return {
            "occlusion_entries_total": 0,
            # DRM introspection.
            "drm_secondary_peak_frames": 0,
            "drm_divergence_below_theta": 0,
            "drm_should_update_true": 0,
            "drm_anchors_added": 0,
            "drm_distractor_anchors_added": 0,
            # DRM should_update gate funnel (which gate blocks, on confident frames).
            "drm_gate_calls": 0,
            "drm_gate_block_not_present": 0,
            "drm_gate_block_not_due": 0,
            "drm_gate_block_reliability": 0,
            "drm_gate_block_size": 0,
            "drm_gate_block_divergence": 0,
            "drm_gate_block_secondary": 0,
            "drm_gate_pass": 0,
        }

    @staticmethod
    def _new_inner_telemetry_counters() -> dict:
        """Fresh zeroed counters owned by the inner SiamABC tracker.

        These cover signals only observable inside the inner tracker's own
        ``track()`` (DRM secondary-peak signal). The outer SiamRAM tracker folds
        them into its per-video / cumulative tallies at each per-video reset.
        """
        return {
            "drm_secondary_peak_frames": 0,
            "drm_divergence_below_theta": 0,
        }

    def _emit_video_telemetry(self) -> None:
        """Emit the just-finished clip's telemetry into the module-global tally.

        Folds the inner SiamABC tracker's counts in, prints one compact per-video
        line (only when something actually fired), adds the counts to the
        process-wide cumulative total (``_TLM_GLOBAL_TOTAL``), then zeroes this
        instance's per-video counters and the inner tracker's.

        Called at every clip boundary via ``_discard_pending_descriptor`` (the run
        loop's per-clip ``finally`` in fresh-tracker mode, and ``initialize`` in
        reused-tracker mode) and once more for any residual at process exit. After
        a clip is emitted its counters are zeroed, so a repeat call is a harmless
        no-op (no double-count). No-op unless research telemetry is enabled.
        """
        if not self._research_telemetry_enabled:
            return

        # Fold the inner SiamABC tracker's per-video counts in, then zero them so a
        # repeat call cannot re-add the same counts.
        inner = getattr(self.tracker, "_tlm_inner", None)
        if isinstance(inner, dict):
            for key, val in inner.items():
                if key in self._tlm_video:
                    self._tlm_video[key] += int(val)
                inner[key] = 0

        v = self._tlm_video
        # Nothing fired this clip -> stay quiet and skip the global increment.
        if not any(int(x) for x in v.values()):
            return

        idx = _tlm_add_video(v)
        siamram_log(f"[research telemetry] video #{idx} {_tlm_format(v)}")

        # Reset per-video counters for the next clip on this instance.
        self._tlm_video = self._new_telemetry_counters()

    _DRM_GATE_REASON_KEY = {
        "not_present": "drm_gate_block_not_present",
        "not_due": "drm_gate_block_not_due",
        "reliability": "drm_gate_block_reliability",
        "size": "drm_gate_block_size",
        "divergence": "drm_gate_block_divergence",
        "secondary": "drm_gate_block_secondary",
        "pass": "drm_gate_pass",
    }

    def _tlm_record_drm_gate(self, reason: str) -> None:
        """Fold one DRM should_update gate consultation into the funnel counters.

        Increments the total ``drm_gate_calls``, the per-reason bucket, and -- on
        a passing gate -- the legacy ``drm_should_update_true`` (so it is driven
        off the same single ``evaluate`` call rather than re-evaluated). Telemetry
        only; never affects the DRM write decision.
        """
        v = self._tlm_video
        v["drm_gate_calls"] += 1
        key = self._DRM_GATE_REASON_KEY.get(reason)
        if key is not None:
            v[key] += 1
        if reason == "pass":
            v["drm_should_update_true"] += 1

    def _maybe_update_drm_introspection(
        self,
        pred_bbox: np.ndarray,
        desc: np.ndarray,
        score: float,
        frame_idx: int,
    ) -> None:
        """
        Introspection-based DRM anchor update (arXiv:2411.17576, Sec. 3.2.2).

        Consumes the response-map secondary-peak signal stashed by the inner
        SiamABC tracker plus the per-frame reliability score and target size, and
        -- when the paper's gates fire -- writes the current confident frame into
        DRM as a distractor-context anchor (alongside, not replacing, the existing
        agreement-based DRM write path). No-op unless the feature is enabled.

        Called on confident frames (target present) for both the immediate and the
        deferred (overlap) descriptor-admission paths.
        """
        updater = self._drm_introspection
        if updater is None or desc is None:
            return

        area = float(pred_bbox[2] * pred_bbox[3])
        # Keep the size-stability median current on every confident frame.
        updater.observe_size(area)

        ratio = float(getattr(self.tracker, "latest_secondary_peak_ratio", 1.0))
        prim = float(getattr(self.tracker, "latest_primary_peak_score", 0.0))
        sec = float(getattr(self.tracker, "latest_secondary_peak_score", 0.0))

        if self._research_telemetry_enabled:
            reason = updater.evaluate(
                frame_idx=frame_idx,
                target_present=True,
                divergence_ratio=ratio,
                primary_score=prim,
                secondary_score=sec,
                reliability_score=float(score),
                area=area,
            )
            self._tlm_record_drm_gate(reason)
            passed = reason == "pass"
        else:
            passed = updater.should_update(
                frame_idx=frame_idx,
                target_present=True,
                divergence_ratio=ratio,
                primary_score=prim,
                secondary_score=sec,
                reliability_score=float(score),
                area=area,
            )

        if passed:
            if self.memory.add_drm_anchor(pred_bbox, desc):
                updater.mark_updated(frame_idx)
                if self._research_telemetry_enabled:
                    self._tlm_video["drm_anchors_added"] += 1
                if self.debug:
                    print(
                        f"[drm introspection] frame={frame_idx} wrote DRM anchor "
                        f"ratio={ratio:.3f} prim={prim:.3f} sec={sec:.3f} "
                        f"score={float(score):.3f}"
                    )

    def _maybe_write_distractor_anchor(self, frame, score: float, frame_idx: int) -> None:
        """Write the secondary-peak (distractor) descriptor to the NEGATIVE bank.

        The introspection-based DRM update (arXiv:2411.17576) detects a competing
        secondary peak during reliable tracking. This records that distractor's
        appearance in the negative distractor bank so the gamma-weighted suppression
        term in drm_match / distractor-mode scoring can penalize it on re-acquisition.
        Independent of the positive-anchor delta gate (the negative bank benefits from
        every genuine competitor sample). No-op unless the sub-feature is enabled and
        a competing distractor was detected this frame during reliable tracking.
        """
        if not self._drm_introspection_distractor_bank_enabled:
            return
        if frame is None or not self._use_distractor_bank:
            return
        updater = self._drm_introspection
        if updater is None:
            return
        sec_bbox = getattr(self.tracker, "latest_secondary_peak_bbox", None)
        if sec_bbox is None:
            return
        ratio = float(getattr(self.tracker, "latest_secondary_peak_ratio", 1.0))
        prim = float(getattr(self.tracker, "latest_primary_peak_score", 0.0))
        sec = float(getattr(self.tracker, "latest_secondary_peak_score", 0.0))
        # Distractor detected during reliable tracking (reuse the updater thresholds;
        # NO delta spacing / mark_updated -- the positive-anchor cadence is separate).
        if float(score) <= updater.theta_iou:
            return
        if not (ratio < updater.theta_anc):
            return
        if (sec / max(1e-8, prim)) < updater.secondary_min_ratio:
            return
        bb = np.asarray(sec_bbox, dtype=float)
        if bb.shape[0] < 4 or bb[2] <= 1.0 or bb[3] <= 1.0:
            return
        h, w = frame.shape[:2]
        x = float(np.clip(bb[0], 0.0, w - 2.0))
        y = float(np.clip(bb[1], 0.0, h - 2.0))
        ww = float(np.clip(bb[2], 1.0, w - x))
        hh = float(np.clip(bb[3], 1.0, h - y))
        desc = _extract_descriptor(frame, np.array([x, y, ww, hh], dtype=float))
        if desc is None:
            return
        self._add_distractor_descriptor(desc)
        if self._research_telemetry_enabled and self._tlm_video is not None:
            self._tlm_video["drm_distractor_anchors_added"] += 1

    def _resolve_pending_descriptor(self) -> None:
        """
        Finish the OSNet descriptor deferred from the previous confident frame
        (Tier-3 overlap) and admit it to the appearance memory, replaying the
        adaptive-drm_margin sample with the snapshot captured at launch.

        Called right after the SiamABC forward — so the deferred OSNet forward
        has overlapped it on the side stream — and before any memory consumer.
        No-op when nothing is pending (the common non-overlap case).
        """
        pending = self._pending_desc
        if pending is None:
            return
        self._pending_desc = None

        desc = _resolve_descriptor(pending["handle"])
        if desc is None:
            return

        self.memory.try_admit(pending["pred_bbox"], desc, pending["ref_bbox"])
        # Introspection-based DRM anchor update (arXiv:2411.17576) using the
        # signal snapshotted when this descriptor was launched (the inner
        # tracker's live signal now reflects a later frame). No-op when disabled.
        if self._drm_introspection is not None and "intro_score" in pending:
            updater = self._drm_introspection
            area = float(pending["pred_bbox"][2] * pending["pred_bbox"][3])
            updater.observe_size(area)
            if self._research_telemetry_enabled:
                reason = updater.evaluate(
                    frame_idx=int(pending["frame_idx"]),
                    target_present=True,
                    divergence_ratio=float(pending["intro_ratio"]),
                    primary_score=float(pending["intro_prim"]),
                    secondary_score=float(pending["intro_sec"]),
                    reliability_score=float(pending["intro_score"]),
                    area=area,
                )
                self._tlm_record_drm_gate(reason)
                passed = reason == "pass"
            else:
                passed = updater.should_update(
                    frame_idx=int(pending["frame_idx"]),
                    target_present=True,
                    divergence_ratio=float(pending["intro_ratio"]),
                    primary_score=float(pending["intro_prim"]),
                    secondary_score=float(pending["intro_sec"]),
                    reliability_score=float(pending["intro_score"]),
                    area=area,
                )
            if passed:
                if self.memory.add_drm_anchor(pending["pred_bbox"], desc):
                    updater.mark_updated(int(pending["frame_idx"]))
                    if self._research_telemetry_enabled:
                        self._tlm_video["drm_anchors_added"] += 1
        if pending["sample_drm"]:
            self_score = self.memory.score_target_against_drm(
                target_bbox=pending["pred_bbox"],
                target_desc=desc,
                ref_bbox=pending["ref_bbox"],
                velocity=pending["velocity"],
                distractor_bank=pending["distractor_bank"],
                lam_iou=self._drm_kwargs["lam_iou"],
                lam_app=self._drm_kwargs["lam_app"],
                lam_mot=self._drm_kwargs["lam_mot"],
                lam_time=self._drm_kwargs["lam_time"],
                alpha=self._drm_kwargs["alpha"],
                gamma=self._drm_kwargs["gamma"],
            )
            if self_score is not None:
                self._drm_margin_auto.observe(self_score, pending["frame_idx"])

    def _normal_update(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Runs the SiamABC tracker to get a predicted bbox and score, optionally
                    runs class warm-up detection during the first few frames, then decides
                    whether to enter occlusion. Occlusion entry requires the score to stay
                    below the effective threshold for `entry_patience` consecutive frames
                    (hysteresis). When the streak is long enough, this method computes how
                    many corrupted history frames to skip, rebuilds the EKF from clean
                    history, sets up the search centre, and immediately hands off to
                    _occlusion_update for the first recovery frame.

                    If tracking is healthy, it updates the EKF, records camera displacement,
                    appends the current centre and velocity to their histories, conditionally
                    admits a new appearance descriptor to memory, and stores the full
                    history entry.

        This is the hot path — it runs every frame when the target is visible.
                    It owns all the bookkeeping that makes occlusion recovery reliable:
                    clean history, accurate velocity, up-to-date appearance memory, and
                    correct EKF state. If this bookkeeping is wrong, recovery will fail.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            pred_bbox: np.ndarray [x, y, w, h] from SiamABC
            score: float confidence from SiamABC
    """
        # Per-frame request flag; only the distractor below-gate force path sets
        # it (later this frame), and occlusion entry consumes it. Reset here so a
        # blocked/declined force does not leak into a later occlusion episode.
        self._pending_distractor_occlusion = False
        if self._distractor_mode_template_freeze_left > 0:
            self.tracker.dynamic_update = False
            self._distractor_mode_template_freeze_left = max(
                0, self._distractor_mode_template_freeze_left - 1
            )
        else:
            self.tracker.dynamic_update = bool(
                self.tracker.tracking_config["dynamic_update"]
            )
        if not self._distractor_mode_active and self._distractor_mode_memory_freeze_left > 0:
            self._distractor_mode_memory_freeze_left = max(
                0, self._distractor_mode_memory_freeze_left - 1
            )

        self._set_visual_mode_for_normal_update()

        if self._gmc_prior_enabled:
            self._apply_gmc_search_prior(frame)
        if self._template_rate_auto is not None:
            # Adaptive template rate takes over from the binary adapts when
            # enabled (they would otherwise fight over N/window).
            self._apply_auto_template_rate(frame)
        elif self._target_motion_template_adapt_enabled:
            # Binary swap on the object's own speed (full-frame px/frame, the
            # value shown in the target-motion pill). Linger keeps the high-motion
            # params for a window after the object slows, mirroring the camera
            # path so they don't snap back the instant it decelerates.
            speed_px = self._current_target_speed_px()
            self._last_target_motion_px = speed_px
            fast_now = speed_px >= self._target_motion_high_px_threshold
            if fast_now:
                self._target_motion_adapt_linger_left = (
                    self._target_motion_template_adapt_linger_frames
                )
                effective_fast = True
            elif self._target_motion_adapt_linger_left > 0:
                self._target_motion_adapt_linger_left -= 1
                effective_fast = True
            else:
                effective_fast = False
            self._apply_target_motion_template_adapt(effective_fast)
        elif self._camera_motion_template_adapt_enabled:
            heavy_now, _ = self._is_heavy_camera_motion(
                frame, bbox_hint=self.current_bbox
            )
            # Keep the high-motion behavior lingering for a window of frames after
            # motion calms, so the template settings don't snap back the instant a
            # pan ends (and during the brief blur that follows).
            if heavy_now:
                self._camera_motion_adapt_linger_left = (
                    self._camera_motion_template_adapt_linger_frames
                )
                effective_heavy = True
            elif self._camera_motion_adapt_linger_left > 0:
                self._camera_motion_adapt_linger_left -= 1
                effective_heavy = True
            else:
                effective_heavy = False
            self._apply_camera_motion_template_adapt(effective_heavy)
        # Frame-difference motion-saliency injection (arXiv:2505.04917). When the
        # processor exists (master switch on) it blends short-term motion saliency
        # into the search region of the frame before SiamABC extracts its search
        # crop; when None this is a no-op and `frame` is passed through untouched,
        # so the legacy crop is byte-for-byte unchanged.
        frame = self._apply_frame_dynamics(frame)
        pred_bbox, score, _ = self.tracker.update(frame)
        pred_bbox = np.array(pred_bbox, dtype=int)

        # Tier-3: the SiamABC forward above just overlapped last frame's OSNet
        # forward on the side stream. Finish + admit that deferred descriptor now,
        # before any appearance-memory consumer (jump-reject / spike) runs below.
        self._resolve_pending_descriptor()

        self._maybe_run_class_warmup(frame)
        self._maybe_run_detectability_probe(frame)

        effective_threshold = self._compute_effective_threshold(frame)

        if self._distractor_mode_active:
            pred_bbox, score = self._distractor_mode_update(
                frame=frame,
                pred_bbox=pred_bbox,
                score=float(score),
                effective_threshold=float(effective_threshold),
            )
        else:
            pred_bbox, score = self._apply_hard_jump_rejection(
                frame=frame,
                pred_bbox=pred_bbox,
                score=float(score),
                effective_threshold=float(effective_threshold),
            )

        heavy_cam_motion, heavy_cam_disp = self._is_heavy_camera_motion(
            frame, bbox_hint=pred_bbox
        )

        if self._distractor_mode_active:
            self._entry_streak = 0
        elif self.frame_idx < self._no_occlusion_first_n_frames:
            # Grace period: never enter occlusion during the first N frames
            # (warm-up). Low scores while the dynamic template is still settling
            # must not be treated as target loss, and the entry streak is held
            # at 0 so it cannot fire the instant the window ends.
            self._entry_streak = 0
            if self.debug and score < effective_threshold:
                print(
                    f"[occlusion guard] frame={self.frame_idx} in no-occlusion "
                    f"grace window (< {self._no_occlusion_first_n_frames})"
                )
        elif (
            score < effective_threshold
            and self.frame_idx >= 0
            and self.enter_occlusion_on_loss
        ):
            if (
                self._block_occlusion_on_camera_motion
                and heavy_cam_motion
            ):
                self._entry_streak = 0
                if self.debug:
                    print(
                        f"[occlusion guard] frame={self.frame_idx} blocked streak by camera motion "
                        f"disp={heavy_cam_disp:.2f} thr={self._camera_motion_heavy_disp_threshold:.2f}"
                    )
            else:
                self._entry_streak += 1
        else:
            self._entry_streak = 0

        effective_entry_patience = self._entry_patience
        if self._entry_patience_high_motion >= 1 and heavy_cam_motion:
            effective_entry_patience = self._entry_patience_high_motion

        if (
            self._entry_streak >= effective_entry_patience
            and self.frame_idx >= self._no_occlusion_first_n_frames
            and self.enter_occlusion_on_loss
        ):

            entry_streak_val = self._entry_streak
            is_exiting, exit_edge = self._detect_exit_direction(frame)
            if (
                self._block_occlusion_on_camera_motion
                and heavy_cam_motion
                and not is_exiting
            ):
                self._entry_streak = 0
                if self.debug:
                    print(
                        f"[occlusion guard] frame={self.frame_idx} blocked entry by camera motion "
                        f"disp={heavy_cam_disp:.2f} thr={self._camera_motion_heavy_disp_threshold:.2f}"
                    )
                return pred_bbox, float(score)

            if self._research_telemetry_enabled:
                self._tlm_video["occlusion_entries_total"] += 1
            self._entry_streak = 0
            self._set_early_occlusion_mode_on_entry()
            self.in_occlusion = True
            self._distractor_mode_active = False
            self._distractor_mode_visual_reals = []
            self._distractor_mode_visual_distractors = []
            self._distractor_mode_roi = None
            self._distractor_mode_roi_size = None
            self._distractor_mode_stable_count = 0
            self._distractor_mode_ambiguous_count = 0
            self._distractor_mode_below_gate_count = 0
            # Tag this occlusion episode as distractor-origin iff the below-gate
            # force path requested it this frame (selects _distractor_occ_drm_*).
            self._distractor_occlusion_active = bool(self._pending_distractor_occlusion)
            self._distractor_mode_overlap_lock_active = False
            self._distractor_mode_overlap_clear_count = 0
            self._distractor_mode_overlap_lock_frames = 0
            self._distractor_anchor_ekf = None
            self._distractor_anchor_pred_bbox = None
            self._distractor_anchor_uncertainty = 0.0
            self._clear_jump_watch_state()
            self._jump_reject_distractor_timer = 0
            if self.visual_mode != "distractor":
                self.visual_mode = "occluded"
                self.visual_reason = "Occlusion recovery"
                self.visual_details = ""

            self._out_of_frame = is_exiting
            self._exit_edge = exit_edge
            loss_cause = self._classify_loss_cause()
            if is_exiting:
                loss_cause = "out_of_frame"
            if self.debug:
                iou_now = float(getattr(self.tracker, "latest_iou_score", 1.0))
                iou_ema = float(getattr(self.tracker, "latest_iou_score_ema", iou_now))
                iou_gate_pass = bool(getattr(self.tracker, "_last_iou_gate_pass", True))
                print(
                    f"[occlusion entry] frame={self.frame_idx}  "
                    f"score={score:.3f}  iou={iou_now:.3f}  iou_ema={iou_ema:.3f}  "
                    f"iou_gate_pass={iou_gate_pass}  "
                    f"loss_cause={loss_cause}  "
                    f"out_of_frame={self._out_of_frame}  exit_edge={self._exit_edge}  "
                    f"entry_streak={entry_streak_val}"
                )

            area_skip = self._detect_shrinkage_onset(
                max_lookback=self.shrinkage_max_lookback,
                min_drop_frac=self.shrinkage_min_drop_frac,
            )
            drift_skip = self._detect_center_drift_skip(
                max_lookback=self.shrinkage_max_lookback
            )
            dynamic_skip = max(area_skip, drift_skip)

            if loss_cause in ("camera_motion", "out_of_frame"):
                effective_skip = 0
            else:
                effective_skip = max(dynamic_skip, entry_streak_val)
            if self.debug:
                print(
                    f"[occlusion entry] frame={self.frame_idx}  "
                    f"loss_cause={loss_cause}  dynamic_skip={dynamic_skip}  "
                    f"entry_streak_skip={entry_streak_val}  "
                    f"effective_skip={effective_skip}  "
                    f"history_len={len(self._conf_history)}"
                )

            self._occ_phase = 0
            self._pending_candidates = []
            self._cand_frames = []
            self._occ_cam_vels = []

            self.ekf = self._rebuild_ekf_from_clean_history(
                skip_override=effective_skip
            )
            ekf = self.ekf
            ekf.predict(H=self._last_H, H_reliable=self._last_H_reliable)
            self._init_search_centre_from_history(skip_override=effective_skip)
            self.tracker.dynamic_update = False
            self._occ_frames = 0
            self.tracker.disable_tta()
            return self._occlusion_update(frame)

        ekf = self.ekf
        assert ekf is not None
        ekf.update(pred_bbox)

        self.velocity = self._compute_velocity_from_history(pred_bbox)

        descriptor_due = (self.frame_idx % self._descriptor_stride) == 0
        if (
            score >= effective_threshold
            and self._distractor_mode_memory_freeze_left <= 0
            and descriptor_due
        ):
            # Whether to sample the adaptive drm_margin estimator for this frame.
            sample_drm = (
                self._drm_margin_auto is not None
                and self._drm_margin_auto.due(self.frame_idx)
            )
            if self._overlap_enabled():
                # Tier-3: queue OSNet on the side stream now and defer admission
                # to the next frame (resolved after that frame's SiamABC forward,
                # so this OSNet forward overlaps it). Snapshot every input the
                # deferred admit / drm sample needs, since tracker state advances.
                self._pending_desc = {
                    "handle": _extract_descriptor_async(frame, pred_bbox),
                    "pred_bbox": pred_bbox.copy(),
                    "ref_bbox": (
                        self.current_bbox.copy()
                        if self.current_bbox is not None
                        else None
                    ),
                    "velocity": (
                        self.velocity.copy()
                        if self.velocity is not None
                        else self.velocity
                    ),
                    "distractor_bank": (
                        self._get_active_distractor_bank()
                        if self._use_distractor_bank
                        else ()
                    ),
                    "sample_drm": bool(sample_drm),
                    "frame_idx": self.frame_idx,
                    # Introspection-based DRM update: snapshot this frame's score
                    # and response-map secondary-peak signal, since by the time the
                    # deferred admit resolves (next frame) the inner tracker's live
                    # signal reflects a different frame. No-op when disabled.
                    "intro_score": float(score),
                    "intro_ratio": float(
                        getattr(self.tracker, "latest_secondary_peak_ratio", 1.0)
                    ),
                    "intro_prim": float(
                        getattr(self.tracker, "latest_primary_peak_score", 0.0)
                    ),
                    "intro_sec": float(
                        getattr(self.tracker, "latest_secondary_peak_score", 0.0)
                    ),
                }
                # DRM-introspection distractor-bank write-back (sub-feature). Uses
                # the inner tracker's live secondary-peak bbox from THIS frame, so it
                # runs immediately (not deferred). No-op unless the sub-feature is on.
                self._maybe_write_distractor_anchor(frame, float(score), self.frame_idx)
            else:
                desc = _extract_descriptor(frame, pred_bbox)
                if desc is not None:
                    self.memory.try_admit(pred_bbox, desc, self.current_bbox)
                    # Introspection-based DRM anchor update (arXiv:2411.17576).
                    # No-op unless drm_introspection_enabled; sits alongside the
                    # legacy DRM write inside try_admit.
                    self._maybe_update_drm_introspection(
                        pred_bbox, desc, float(score), self.frame_idx
                    )
                    self._maybe_write_distractor_anchor(
                        frame, float(score), self.frame_idx
                    )
                    # Adaptive drm_margin: on confident frames, sample how well
                    # the genuine target scores against its own DRM anchors and
                    # feed the estimator. No-op unless drm_margin == "auto".
                    if sample_drm:
                        self_score = self.memory.score_target_against_drm(
                            target_bbox=pred_bbox,
                            target_desc=desc,
                            ref_bbox=self.current_bbox,
                            velocity=self.velocity,
                            distractor_bank=(
                                self._get_active_distractor_bank()
                                if self._use_distractor_bank
                                else ()
                            ),
                            lam_iou=self._drm_kwargs["lam_iou"],
                            lam_app=self._drm_kwargs["lam_app"],
                            lam_mot=self._drm_kwargs["lam_mot"],
                            lam_time=self._drm_kwargs["lam_time"],
                            alpha=self._drm_kwargs["alpha"],
                            gamma=self._drm_kwargs["gamma"],
                        )
                        if self_score is not None:
                            self._drm_margin_auto.observe(
                                self_score, self.frame_idx
                            )

        self._maybe_update_stable_anchor(pred_bbox)

        self.current_bbox = pred_bbox.copy()
        self.held_box = pred_bbox.copy()
        self._commit_normal_frame_history(frame, pred_bbox)

        return pred_bbox, score

    def _compute_velocity_from_history(
        self,
        current_bbox: np.ndarray,
        lookback: Optional[int] = None,
        smooth_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Looks `lookback` frames back in _conf_history, measures the centre
                    displacement between that frame and the current bbox, divides by
                    lookback to get px/frame, then blends into the running velocity
                    estimate with an exponential moving average. Handles the case where
                    the history is shorter than lookback gracefully.

        Called every frame in _normal_update. The velocity it produces is used
                    by the EKF as a prior when occlusion starts, by DRM for motion
                    consistency scoring, and by the exit direction detector. Getting this
                    right matters a lot — bad velocity estimates cause the EKF to
                    extrapolate the wrong direction during occlusion.
        Args:
            current_bbox (any): np.ndarray [x, y, w, h] of the bbox at the current frame
            lookback (any): how many frames back to measure displacement over;
            defaults to self.velocity_lookback
            smooth_alpha (any): EMA weight for blending new measurement into current velocity;
            defaults to self.velocity_smooth_alpha
        Returns:
            np.ndarray of shape (2,) — (vx, vy) in pixels per frame
    """
        lb = lookback if lookback is not None else self.velocity_lookback
        alpha = smooth_alpha if smooth_alpha is not None else self.velocity_smooth_alpha

        if not self._conf_history:
            return np.zeros(2)

        history = list(self._conf_history)
        n_back = min(lb, len(history))
        if n_back == 0:
            return np.zeros(2)

        curr_cx = current_bbox[0] + current_bbox[2] / 2.0
        curr_cy = current_bbox[1] + current_bbox[3] / 2.0

        ref_bbox = history[-n_back].bbox
        ref_cx = ref_bbox[0] + ref_bbox[2] / 2.0
        ref_cy = ref_bbox[1] + ref_bbox[3] / 2.0

        raw_vx = (curr_cx - ref_cx) / n_back
        raw_vy = (curr_cy - ref_cy) / n_back

        vx = alpha * raw_vx + (1.0 - alpha) * self.velocity[0]
        vy = alpha * raw_vy + (1.0 - alpha) * self.velocity[1]
        return np.array([vx, vy])

    def _init_search_centre_from_history(
        self,
        skip_override=None,
    ) -> None:
        """
        Reads the last clean history entry (after skipping corrupted tail frames)
                    and stores its centre as _search_cx / _search_cy. If the history is
                    entirely consumed by the skip, falls back to the current bbox centre.

        Called at the moment occlusion is declared, immediately before phase 0
                    starts. The search centre is the anchor for _get_yolo_search_roi —
                    it is where the EKF says the target should be and is what every
                    subsequent ROI is built around. Planting it at a clean position rather
                    than at the last (possibly corrupted) bbox is what keeps the search
                    area honest at the start of recovery.
        Args:
            skip_override (any): int or None; if given, this many frames are dropped
            from the tail of _conf_history before reading the last
            clean bbox. If None, self.history_skip_last is used.
    """
        history = list(self._conf_history)
        skip = (
            min(self.history_skip_last, len(history) - 1)
            if skip_override is None
            else skip_override
        )
        clean = history[: len(history) - skip] if skip > 0 else history

        if clean:
            last_clean_bbox = np.asarray(clean[-1].bbox, dtype=float)
            self._search_cx = last_clean_bbox[0] + last_clean_bbox[2] / 2.0
            self._search_cy = last_clean_bbox[1] + last_clean_bbox[3] / 2.0
        else:
            current_bbox = self.current_bbox
            assert current_bbox is not None
            self._search_cx = float(current_bbox[0] + current_bbox[2] / 2.0)
            self._search_cy = float(current_bbox[1] + current_bbox[3] / 2.0)

    def _occlusion_update(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        # Flush any deferred descriptor before occlusion recovery runs its own
        # (synchronous) OSNet calls, so the side stream is drained and the
        # appearance memory is consistent. Normally a no-op (pending is cleared
        # on occlusion entry).
        self._resolve_pending_descriptor()
        return self._occlusion_subsystem.occlusion_update(
            frame=frame,
        )

    def _begin_reacq_confirmation(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        score: float,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.begin_reacq_confirmation(
            frame=frame,
            bbox=bbox,
            score=score,
        )

    def _reset_reacq_confirmation(
        self,
    ) -> None:
        self._occlusion_subsystem.reset_reacq_confirmation()

    def _occ_phase_reacq_confirm(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.occ_phase_reacq_confirm(
            frame=frame,
        )

    def _occ_phase_siam(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.occ_phase_siam(
            frame=frame,
        )

    def _occ_phase_collect(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.occ_phase_collect(
            frame=frame,
        )

    def _set_recovered_early_occlusion_flag(
        self,
    ) -> None:
        """
        Mark whether the just-finished occlusion was "early" based on duration.
        Negative threshold keeps legacy behavior (always early on success).
        """
        if self._early_occlusion_entry_n_frames >= 0:
            return
        if self._early_occlusion_max_frames < 0:
            self.recovered_early_occlusion = True
            return
        self.recovered_early_occlusion = (
            self._occ_frames <= self._early_occlusion_max_frames
        )

    def _set_early_occlusion_mode_on_entry(
        self,
    ) -> None:
        """
        Decide early-occlusion mode exactly at occlusion entry from frame_idx.
        """
        if self._early_occlusion_entry_n_frames < 0:
            return
        self.recovered_early_occlusion = (
            self.frame_idx <= self._early_occlusion_entry_n_frames
        )

    def _should_use_ram_for_occlusion_recovery(
        self,
    ) -> bool:
        """
        Returns True when occlusion matching should use RAM instead of DRM.
        """
        if not self._occlusion_use_ram_until_drm_full:
            return False
        drm_cap = max(1, int(getattr(self.memory, "drm_capacity", 1)))
        return self.memory.drm_size() < drm_cap

    def _occlusion_memory_match(
        self,
        frame: np.ndarray,
        candidates: List[np.ndarray],
        ref_bbox: np.ndarray,
        velocity: np.ndarray,
        margin: float,
        search_cx: Optional[float],
        search_cy: Optional[float],
        dist_sigma: Optional[float],
        lam_iou: float,
        lam_app: float,
        lam_mot: float,
        lam_time: float,
        alpha: float,
        gamma: float,
        top_k: int,
        skip_threshold: float,
        lam_dist: float,
        lam_cand_dir: float,
    ) -> Tuple[List[Tuple[np.ndarray, float]], str]:
        return self._occlusion_subsystem.occlusion_memory_match(
            frame=frame,
            candidates=candidates,
            ref_bbox=ref_bbox,
            velocity=velocity,
            margin=margin,
            search_cx=search_cx,
            search_cy=search_cy,
            dist_sigma=dist_sigma,
            lam_iou=lam_iou,
            lam_app=lam_app,
            lam_mot=lam_mot,
            lam_time=lam_time,
            alpha=alpha,
            gamma=gamma,
            top_k=top_k,
            skip_threshold=skip_threshold,
            lam_dist=lam_dist,
            lam_cand_dir=lam_cand_dir,
        )

    def _occ_phase_final_drm(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.occ_phase_final_drm(
            frame=frame,
        )

    def _commit_reacquisition(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        desc: Optional[np.ndarray],
        score: float,
        drm_score: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        return self._occlusion_subsystem.commit_reacquisition(
            frame=frame,
            bbox=bbox,
            desc=desc,
            score=score,
            drm_score=drm_score,
        )

    def _record_recovery(
        self,
        mode: str,
        frame: np.ndarray,
        bbox: np.ndarray,
        score: float,
        sim: Optional[float] = None,
        iou: Optional[float] = None,
        desc: Optional[np.ndarray] = None,
        drm_score: Optional[float] = None,
    ) -> None:
        """
        Snapshot the patch + stats of the latest successful recovery so the
        visualiser can render what we recovered onto. Called from the
        occlusion-recovery commit and the distractor-mode resolved exit.

        `score` is the tracker (SiamABC) confidence at the recovered box;
        `drm_score` is the DRM composite ranking score of the chosen candidate
        (occlusion only — None/NaN elsewhere). Both are surfaced in the viz panel.
        """
        bb = np.asarray(bbox, dtype=int).reshape(-1)
        if bb.size < 4:
            self._last_recovery_patch = None
            self._last_recovery_info = None
            return
        x, y, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        h_fr, w_fr = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_fr, x + bw)
        y2 = min(h_fr, y + bh)
        if x2 > x1 and y2 > y1:
            self._last_recovery_patch = frame[y1:y2, x1:x2].copy()
        else:
            self._last_recovery_patch = None

        iou_held = float(iou) if iou is not None else float("nan")
        if not np.isfinite(iou_held) and self.held_box is not None:
            iou_held = float(_iou(self.held_box, bbox))

        sim_val = float(sim) if sim is not None else float("nan")
        if not np.isfinite(sim_val) and desc is not None:
            best = self.memory.best_descriptor()
            if best is not None:
                sim_val = float(_cos_sim(desc, best))

        drm_val = float(drm_score) if drm_score is not None else float("nan")

        self._last_recovery_info = {
            "mode": str(mode),
            "bbox": np.asarray(bbox, dtype=int).copy(),
            "score": float(score),
            "drm_score": drm_val,
            "sim": sim_val,
            "iou_held": iou_held,
            "frame_idx": int(self.frame_idx),
            "thresholds": {
                "reacq": float(self.reacq_threshold),
                "app_match": float(self.app_match_threshold),
                "min_sim": float(self._distractor_mode_min_similarity),
                "sel_min_sim": float(self._distractor_mode_selected_min_similarity),
            },
        }

    def _get_median_size(
        self,
    ) -> Tuple[int, int]:
        """
        Computes the median width and height of the tracked object across all
                    entries in _size_history. Falls back to held_box dimensions if the
                    history is empty.

        Used in _occ_phase_siam to build the seed bbox, and in
                    _get_yolo_search_roi to scale the ROI. Using the median rather than
                    the most recent size makes both the seed and the ROI robust to frames
                    where the tracker was drifting or locking onto something of the wrong
                    scale.
        Args:
            None. Reads self._size_history internally.
        Returns:
            (w, h): tuple of two ints, median object size in pixels, minimum 1px each
    """
        if self._size_history:
            size_arr = np.asarray(self._size_history, dtype=np.float32)
            med_wh = np.median(size_arr, axis=0)
            w = max(1, int(med_wh[0]))
            h = max(1, int(med_wh[1]))
        else:
            held_box = self.held_box
            assert held_box is not None
            w = max(1, int(held_box[2]))
            h = max(1, int(held_box[3]))
        return w, h

    def _cam_vel_from_H(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        return self._occlusion_subsystem.cam_vel_from_h(
            frame=frame,
        )

    @property
    def _active_drm_kwargs(self) -> DRMKwargs:
        """Occlusion-DRM ranking weights for the current episode (distractor-origin → separate set)."""
        if self._distractor_occlusion_active:
            return self._distractor_occ_drm_kwargs
        if self._drm_margin_auto is not None:
            # Inject the learned margin at read time. A shallow copy keeps the
            # stored dict (and its seeded numeric margin) unmutated.
            active = dict(self._drm_kwargs)
            active["margin"] = self._drm_margin_auto.value
            return active  # type: ignore[return-value]
        return self._drm_kwargs

    @property
    def _active_drm_lam_cand_dir(self) -> float:
        if self._distractor_occlusion_active:
            return self._distractor_occ_drm_lam_cand_dir
        return self._drm_lam_cand_dir

    @property
    def _active_drm_dist_sigma_factor(self) -> float:
        if self._distractor_occlusion_active:
            return self._distractor_occ_drm_dist_sigma_factor
        return self._drm_dist_sigma_factor

    def _effective_dist_sigma(
        self,
        frame: np.ndarray,
    ) -> float:
        return self._occlusion_subsystem.effective_dist_sigma(
            frame=frame,
        )

    def _build_candidate_velocities(
        self,
        last_idx: int,
    ) -> List[Optional[np.ndarray]]:
        return self._occlusion_subsystem.build_candidate_velocities(
            last_idx=last_idx,
        )

    def _compute_velocity_score(
        self,
        cand_vel,
        expected_vel,
    ) -> float:
        """
        Computes a scalar score in [0, 1] measuring how well the candidate's
                    observed motion matches the expected target motion:

                    - If expected speed < vel_score_min_speed → returns 0.5 (neutral;
                      the EKF is essentially stationary, so velocity is unreliable).
                    - Computes cosine similarity between the two vectors, maps it to [0,1]
                      as dir_score = (cos + 1) / 2.
                    - Computes speed_score = clip(1 - |speed_ratio - 1|, 0, 1); set to 0
                      if cosine < 0 (wrong direction, speed match is a coincidence).
                    - Combines as 0.80 * dir_score + 0.20 * speed_score.
                    - Applies a hard gate: if cos < -vel_dir_hard_gate, clamps the score
                      to 0.05, ensuring strongly opposite candidates cannot slip through.

        Called in _occ_phase_siam (for the single-frame direction augmentation)
                    and in _occ_phase_final_drm (for each fully-tracked candidate). Its
                    output is scaled by lam_cand_dir and added to the DRM score, biasing
                    selection toward candidates that actually move with the target rather
                    than just look like it.
        Args:
            cand_vel (any): np.ndarray (2,) camera-compensated velocity of a candidate
            in pixels per frame
            expected_vel (any): np.ndarray (2,) EKF-predicted target velocity in px/frame
        Returns:
            float in [0.05, 1.0] — higher means the candidate moves more like the target
    """
        expected_speed = float(np.linalg.norm(expected_vel))
        cand_speed = float(np.linalg.norm(cand_vel))
        if expected_speed < self._vel_score_min_speed:
            return 0.5

        cos = float(
            np.dot(expected_vel, cand_vel) / (expected_speed * (cand_speed + 1e-8))
        )
        cos = float(np.clip(cos, -1.0, 1.0))
        dir_score = (cos + 1.0) / 2.0

        speed_ratio = cand_speed / (expected_speed + 1e-8)
        speed_score = float(np.clip(1.0 - abs(speed_ratio - 1.0), 0.0, 1.0))
        if cos < 0.0:
            speed_score = 0.0

        raw = 0.80 * dir_score + 0.20 * speed_score

        if cos < -self._vel_dir_hard_gate:
            raw = min(raw, 0.05)

        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _rescale_homography_from_scaled(
        H_small: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """
        Convert a homography estimated in scaled-image coordinates to the
        original frame coordinates.
        """
        s = float(scale)
        S = np.array(
            [
                [s, 0.0, 0.0],
                [0.0, s, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        S_inv = np.array(
            [
                [1.0 / (s + 1e-8), 0.0, 0.0],
                [0.0, 1.0 / (s + 1e-8), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H_full = S_inv @ H_small @ S
        if abs(float(H_full[2, 2])) > 1e-8:
            H_full /= float(H_full[2, 2])
        return H_full

    def _estimate_homography_classic(
        self,
        prev_gray_scaled: np.ndarray,
        gray: np.ndarray,
        scale: float,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Original fast homography path: grid points + affine partial RANSAC.
        """
        lk_params = dict(
            winSize=(20, 20),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                8,
                0.04,
            ),
        )

        H = None
        reliable = False

        if self._cached_shape != gray.shape:
            step = 50
            yg, xg = np.mgrid[
                step // 2: gray.shape[0]: step,
                step // 2: gray.shape[1]: step,
            ]
            self._cached_pts = np.column_stack((xg.ravel(), yg.ravel())).astype(
                np.float32
            )
            self._cached_shape = gray.shape

        grid = self._cached_pts
        assert grid is not None

        ref_box = self.held_box if self.held_box is not None else self.current_bbox
        if ref_box is not None:
            x, y, w, h = (v * scale for v in map(int, ref_box))
            pad = max(4, int(max(w, h) * 0.15))
            inside = (
                (grid[:, 0] >= x - pad)
                & (grid[:, 0] < x + w + pad)
                & (grid[:, 1] >= y - pad)
                & (grid[:, 1] < y + h + pad)
            )
            pts = grid[~inside].reshape(-1, 1, 2)
        else:
            pts = grid.reshape(-1, 1, 2)

        if len(pts) < 6:
            return H, reliable

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_scaled, gray, pts, None, **lk_params
        )
        if status is None:
            return H, reliable

        ok = status.ravel() == 1
        good_old = pts[ok]
        good_new = new_pts[ok]
        if len(good_old) < 6:
            return H, reliable

        A, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=500,
            confidence=0.99,
        )
        if A is None or inliers is None:
            return H, reliable

        inlier_ratio = float(inliers.sum()) / len(good_old)
        reliable = inlier_ratio >= self.homo_inlier_threshold
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = A
        H[0, 2] /= scale
        H[1, 2] /= scale
        return H, reliable

    def _estimate_homography_accurate(
        self,
        prev_gray_scaled: np.ndarray,
        gray: np.ndarray,
        scale: float,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Higher-accuracy path: GFTT features + LK tracking + forward/backward
        consistency + full homography RANSAC.
        """
        lk_params = dict(
            winSize=(24, 24),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        max_corners = max(32, int(self.homo_max_corners))
        mask = np.full_like(prev_gray_scaled, 255, dtype=np.uint8)
        ref_box = self.held_box if self.held_box is not None else self.current_bbox
        if ref_box is not None:
            x, y, w, h = (v * scale for v in map(int, ref_box))
            pad = max(6, int(max(w, h) * 0.20))
            x1 = max(0, int(x - pad))
            y1 = max(0, int(y - pad))
            x2 = min(mask.shape[1], int(x + w + pad))
            y2 = min(mask.shape[0], int(y + h + pad))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0

        pts0 = cv2.goodFeaturesToTrack(
            prev_gray_scaled,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=False,
            mask=mask,
        )
        if pts0 is None or len(pts0) < 8:
            return None, False

        pts1, st01, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_scaled, gray, pts0, None, **lk_params
        )
        if pts1 is None or st01 is None:
            return None, False

        pts0_back, st10, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray_scaled, pts1, None, **lk_params
        )
        if pts0_back is None or st10 is None:
            return None, False

        ok01 = st01.ravel() == 1
        ok10 = st10.ravel() == 1
        fb_err = np.linalg.norm(pts0_back - pts0, axis=2).ravel()
        fb_ok = fb_err <= 1.5
        keep = ok01 & ok10 & fb_ok
        if int(np.count_nonzero(keep)) < 8:
            return None, False

        good_old = pts0[keep].reshape(-1, 2)
        good_new = pts1[keep].reshape(-1, 2)

        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H_small, inliers = cv2.findHomography(
            good_old,
            good_new,
            method=method,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.995,
        )
        if H_small is None or inliers is None:
            return None, False

        inlier_ratio = float(inliers.sum()) / max(1, len(good_old))
        reliable = inlier_ratio >= self.homo_inlier_threshold
        H = self._rescale_homography_from_scaled(
            np.asarray(H_small, dtype=np.float64), scale
        )
        return H, reliable

    def _estimate_homography(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], bool, np.ndarray]:
        return self._camera_motion_subsystem.estimate_homography(
            frame=frame,
        )

    def _limit_osnet_candidates(
        self,
        detections: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Cap detections routed to OSNet descriptor extraction.

        A non-positive cap means "no limit".
        """
        cap = self._osnet_max_candidate_batch
        if cap <= 0 or len(detections) <= cap:
            return detections
        return detections[:cap]

    @staticmethod
    def _iou_many_to_one(
        boxes: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized IoU between N boxes and one reference box.
        """
        if boxes.size == 0:
            return np.empty((0,), dtype=np.float64)

        a = np.asarray(boxes, dtype=np.float64)
        b = np.asarray(box, dtype=np.float64)

        ax2 = a[:, 0] + a[:, 2]
        ay2 = a[:, 1] + a[:, 3]
        bx2 = b[0] + b[2]
        by2 = b[1] + b[3]

        ix1 = np.maximum(a[:, 0], b[0])
        iy1 = np.maximum(a[:, 1], b[1])
        ix2 = np.minimum(ax2, bx2)
        iy2 = np.minimum(ay2, by2)

        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter = inter_w * inter_h
        union = a[:, 2] * a[:, 3] + b[2] * b[3] - inter
        return inter / (union + 1e-8)

    def _nudge_toward_nearest(
        self,
        frame: np.ndarray,
        detections: List[np.ndarray],
    ) -> np.ndarray:
        """
        When no candidate survives to DRM matching, this prevents held_box
                    from freezing completely. It finds the detection closest to held_box,
                    weighted by appearance similarity to the best memory descriptor if
                    one is available, then moves held_box a small step (nudge_alpha)
                    toward that detection while keeping the box dimensions unchanged.
                    The result is clamped to the frame.

        Called in _occ_phase_final_drm when no fully-tracked candidates exist
                    but raw YOLO detections are available. Without this the EKF would keep
                    extrapolating uncorrected. A gentle nudge toward the most plausible
                    nearby detection biases the EKF search centre in roughly the right
                    direction while keeping the tracker conservative enough not to commit
                    to an unverified candidate.
        Args:
            frame (any): current video frame as a numpy BGR array
            detections (any): list of np.ndarray [x, y, w, h] YOLO detections
        Returns:
            np.ndarray [x, y, w, h] — nudged held_box position
    """
        ref = self.memory.best_descriptor()
        held_box = self.held_box
        assert held_box is not None
        if not detections:
            return held_box.copy()

        hcx = held_box[0] + held_box[2] / 2.0
        hcy = held_box[1] + held_box[3] / 2.0

        det_descs = []
        if ref is not None and detections:
            dets_for_desc = self._limit_osnet_candidates(detections)
            det_descs = _extract_descriptor(frame, dets_for_desc) if dets_for_desc else []

        det_arr = np.asarray(detections, dtype=np.float64)
        centers = np.column_stack(
            (det_arr[:, 0] + det_arr[:, 2] / 2.0, det_arr[:, 1] + det_arr[:, 3] / 2.0)
        )
        dists = np.hypot(centers[:, 0] - hcx, centers[:, 1] - hcy)

        if ref is not None:
            sims = np.zeros((len(detections),), dtype=np.float64)
            n_desc = min(len(det_descs), len(detections))
            if n_desc > 0:
                sims[:n_desc] = np.asarray(
                    [
                        _cos_sim(ref, det_descs[i]) if det_descs[i] is not None else 0.0
                        for i in range(n_desc)
                    ],
                    dtype=np.float64,
                )
            ranks = dists * (1.0 - 0.5 * sims)
        else:
            ranks = dists

        best_det = det_arr[int(np.argmin(ranks))]
        dcx = float(best_det[0] + best_det[2] / 2.0)
        dcy = float(best_det[1] + best_det[3] / 2.0)
        new_cx = hcx + self.nudge_alpha * (dcx - hcx)
        new_cy = hcy + self.nudge_alpha * (dcy - hcy)

        hw, hh = held_box[2], held_box[3]
        h_fr, w_fr = frame.shape[:2]
        nx = int(np.clip(new_cx - hw / 2.0, 0, w_fr - 1))
        ny = int(np.clip(new_cy - hh / 2.0, 0, h_fr - 1))
        return np.array(
            [nx, ny, int(np.clip(hw, 1, w_fr - nx)), int(np.clip(hh, 1, h_fr - ny))],
            dtype=int,
        )

    def _get_yolo_search_roi(
        self,
        frame: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
        Computes a [x, y, w, h] search region in which YOLO will be run.
                    The region grows over time during occlusion based on _occ_frames and
                    the growth parameters. Two parameter sets exist:
                        normal objects   → roi_start_expand, yolo_search_expand, etc.
                        tiny/far objects → tiny_roi_start_expand, tiny_yolo_search_expand, etc.

                    During the configured warm-up window or after a failed recovery
                    (recovered_early_occlusion=False), a centred region is used to
                    give the tracker time to stabilise.

                    For out-of-frame targets, the ROI is a strip pinned to the relevant edge.
                    For in-frame targets, the ROI is a square centred on the EKF search centre,
                    expanding geometrically with time. Everything is clamped to the frame.

        Called in _occ_phase_siam and _occ_phase_collect to crop the frame
                    before running YOLO. Running YOLO on a cropped region rather than the
                    full frame reduces false detections (fewer distractors in view), speeds
                    up inference, and keeps the search area semantically relevant. The
                    growing ROI ensures that even after many occlusion frames the true
                    target can eventually fall back inside the search window.
        Args:
            frame (any): current video frame (used for its shape and _is_long_distance)
        Returns:
            (x, y, w, h): tuple of ints defining the YOLO search region
    """
        h_fr, w_fr = frame.shape[:2]
        max_side = min(w_fr, h_fr)

        if self.frame_idx <= self._yolo_warmup_frames or not self.recovered_early_occlusion:
            scale = self._yolo_warmup_center_scale
            bw = int(w_fr * scale)
            bh = int(h_fr * scale)
            x = (w_fr - bw) // 2
            y = (h_fr - bh) // 2
            return x, y, bw, bh

        tiny_now = self._is_long_distance(frame)
        if self._out_of_frame:
            is_tiny = self._out_of_frame_roi_use_tiny_params and tiny_now
        else:
            is_tiny = tiny_now
        if is_tiny:
            _roi_start_expand = self.tiny_roi_start_expand
            _yolo_search_expand = self.tiny_yolo_search_expand
            _search_expand_growth_factor = self.tiny_search_expand_growth_factor
            _search_expand_growth_every = self.tiny_search_expand_growth_every
        else:
            _roi_start_expand = self.roi_start_expand
            _yolo_search_expand = self.yolo_search_expand
            _search_expand_growth_factor = self.search_expand_growth_factor
            _search_expand_growth_every = self.search_expand_growth_every

        if self._out_of_frame:
            if self._out_of_frame_roi_start_expand > 0.0:
                _roi_start_expand = self._out_of_frame_roi_start_expand
            if self._out_of_frame_yolo_search_expand > 0.0:
                _yolo_search_expand = self._out_of_frame_yolo_search_expand
            if self._out_of_frame_search_expand_growth_factor > 0.0:
                _search_expand_growth_factor = self._out_of_frame_search_expand_growth_factor
            if self._out_of_frame_search_expand_growth_every > 0:
                _search_expand_growth_every = self._out_of_frame_search_expand_growth_every

        obj_w, obj_h = self._get_median_size()
        obj_size = (obj_w + obj_h) // 2

        steps = self._occ_frames // max(1, _search_expand_growth_every)
        time_expand = float(_search_expand_growth_factor ** steps)
        effective_expand = min(_roi_start_expand * time_expand, _yolo_search_expand)

        if (
            self._out_of_frame
            and self._exit_edge is not None
            and self._out_of_frame_roi_edge_pin_enabled
        ):
            side = max(1, int(obj_size * effective_expand))

            if self._exit_edge == "right":
                if (self._search_cx - w_fr) > obj_w * self._out_of_frame_roi_skip_when_past_edge_mult:
                    return 0, 0, 0, 0
                scy = float(np.clip(self._search_cy, side // 2, h_fr - side // 2))
                x1, y1 = w_fr - side, int(scy - side // 2)
                rw, rh = side, side

            elif self._exit_edge == "left":
                scy = float(np.clip(self._search_cy, side // 2, h_fr - side // 2))
                x1, y1 = 0, int(scy - side // 2)
                rw, rh = side, side

            elif self._exit_edge == "bottom":
                scx = float(np.clip(self._search_cx, side // 2, w_fr - side // 2))
                x1, y1 = int(scx - side // 2), h_fr - side
                rw, rh = side, side

            else:
                scx = float(np.clip(self._search_cx, side // 2, w_fr - side // 2))
                x1, y1 = int(scx - side // 2), 0
                rw, rh = side, side

        else:
            half = min((obj_size * effective_expand) / 2.0, max_side / 2.0)
            cx_lo = min(half, w_fr / 2.0)
            cy_lo = min(half, h_fr / 2.0)
            scx = float(np.clip(self._search_cx, cx_lo, w_fr - cx_lo))
            scy = float(np.clip(self._search_cy, cy_lo, h_fr - cy_lo))
            side = max(1, int(half * 2))
            x1, y1 = int(scx - half), int(scy - half)
            rw, rh = side, side

        x1 = max(0, x1)
        y1 = max(0, y1)
        rw = min(rw, w_fr - x1)
        rh = min(rh, h_fr - y1)
        rw = max(1, rw)
        rh = max(1, rh)
        return x1, y1, rw, rh

    def _yolo_detect(
        self,
        frame: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Calls _get_yolo_search_roi to get the search crop, runs YOLO predict
                    on that crop at imgsz=self.yolo_imgsz, then translates all resulting
                    bounding boxes back to full-frame coordinates. Optionally filters
                    detections to _target_class_id when yolo_filter_class=True and the
                    class has been committed. Stores results in _yolo_cache so
                    _yolo_detect_cached can skip a redundant call within the same frame.

        The only place in the system that runs YOLO. Called during candidate
                    collection (phases 1…N) and referenced by the phase 0 ROI logic.
                    Keeping detection isolated here means the ROI logic, class filtering,
                    and coordinate translation all live in one place. The configurable
                    imgsz lets callers trade detection accuracy for speed.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            list of np.ndarray [x, y, w, h] in full: frame pixel coordinates.
            Empty list if the crop is invalid or YOLO finds nothing.
    """
        rx, ry, rw, rh = self._get_yolo_search_roi(frame)
        crop = frame[ry: ry + rh, rx: rx + rw]
        if crop.size == 0:
            self._yolo_cache = []
            return []
        results = self.yolo.predict(
            crop,
            conf=self.yolo_conf,
            iou=self.yolo_iou_thr,
            verbose=False,
            imgsz=self.yolo_imgsz,
            augment=self.yolo_augment,
        )
        boxes = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            result_boxes = results[0].boxes
            xyxy_all = result_boxes.xyxy.detach().cpu().numpy()
            cls_all = result_boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
            if self._yolo_filter_class and self._target_class_id is not None:
                keep = cls_all == int(self._target_class_id)
                xyxy_all = xyxy_all[keep]
            if xyxy_all.size > 0:
                xywh = np.empty((xyxy_all.shape[0], 4), dtype=np.int32)
                xywh[:, 0] = xyxy_all[:, 0].astype(np.int32, copy=False) + rx
                xywh[:, 1] = xyxy_all[:, 1].astype(np.int32, copy=False) + ry
                xywh[:, 2] = (xyxy_all[:, 2] - xyxy_all[:, 0]).astype(
                    np.int32, copy=False
                )
                xywh[:, 3] = (xyxy_all[:, 3] - xyxy_all[:, 1]).astype(
                    np.int32, copy=False
                )
                boxes = [xywh[i].copy() for i in range(xywh.shape[0])]
        self._yolo_cache = boxes
        return boxes

    def _clamp_bbox_to_frame(
        self,
        bbox: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Clips width and height to [1, frame_dimension], then clips x and y
                    to [-w, frame_w] and [-h, frame_h] respectively. This allows the
                    box to be partially off-screen (which is intentional for out-of-frame
                    targets) while preventing completely degenerate zero-size boxes.

        Called throughout the occlusion code — after seed bbox construction,
                    after EKF output, and when building the motion-compensated adjusted
                    bbox for tracker verification. Without this, negative or zero-size
                    bounding boxes would crash OpenCV crop operations downstream.
        Args:
            bbox (any): np.ndarray [x, y, w, h] potentially out of frame bounds
            frame (any): current video frame (used only for its shape)
        Returns:
            np.ndarray [x, y, w, h] dtype int, all values within the permissive
            bounds described above
    """
        h_fr, w_fr = frame.shape[:2]
        x, y, w, h = bbox
        w = int(np.clip(w, 1, w_fr))
        h = int(np.clip(h, 1, h_fr))
        x = int(np.clip(x, -w, w_fr))
        y = int(np.clip(y, -h, h_fr))
        return np.array([x, y, w, h], dtype=int)

    def _clip_bbox_inside_frame(
        self,
        bbox: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Strict frame clip used by GMC prior injection.
        """
        h_fr, w_fr = frame.shape[:2]
        x, y, w, h = np.asarray(bbox, dtype=float).reshape(-1)[:4]
        w = max(1.0, min(float(w), float(w_fr)))
        h = max(1.0, min(float(h), float(h_fr)))
        x = min(max(0.0, float(x)), max(0.0, float(w_fr) - w))
        y = min(max(0.0, float(y)), max(0.0, float(h_fr) - h))
        return np.array([round(x), round(y), round(w), round(h)], dtype=int)

    def _to_homography_matrix(
        self,
        transform: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if transform is None:
            return None
        mat = np.asarray(transform, dtype=np.float64)
        if mat.shape == (3, 3):
            out = mat.copy()
        elif mat.shape == (2, 3):
            out = np.eye(3, dtype=np.float64)
            out[:2, :] = mat
        else:
            return None
        if not np.all(np.isfinite(out)):
            return None
        return out

    def _gmc_motion_stats(
        self,
        transform: Optional[np.ndarray],
        frame: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        hmat = self._to_homography_matrix(transform)
        if hmat is None:
            return None
        dx = float(hmat[0, 2])
        dy = float(hmat[1, 2])
        a = float(hmat[0, 0])
        b = float(hmat[1, 0])
        scale = float(np.sqrt(max(a * a + b * b, 1e-9)))
        rotation_deg = float(np.degrees(np.arctan2(b, a)))

        h_fr, w_fr = frame.shape[:2]
        corners = np.array(
            [
                [0.0, 0.0],
                [float(w_fr - 1), 0.0],
                [float(w_fr - 1), float(h_fr - 1)],
                [0.0, float(h_fr - 1)],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, hmat).reshape(-1, 2)
        except cv2.error:
            return None
        base = corners.reshape(-1, 2)
        max_corner_disp = float(np.linalg.norm(warped - base, axis=1).max())
        if not np.isfinite(max_corner_disp):
            return None
        return dx, dy, scale, rotation_deg, max_corner_disp

    def _gmc_motion_is_valid(
        self,
        transform: Optional[np.ndarray],
        frame: np.ndarray,
    ) -> Tuple[bool, str]:
        return self._camera_motion_subsystem.gmc_motion_is_valid(
            transform=transform,
            frame=frame,
        )

    def _warp_bbox_with_homography(
        self,
        bbox: np.ndarray,
        transform: Optional[np.ndarray],
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        hmat = self._to_homography_matrix(transform)
        if hmat is None:
            return None

        arr = np.asarray(bbox, dtype=float).reshape(-1)
        if arr.size < 4:
            return None
        x, y, w, h = arr[:4]
        if w <= 0 or h <= 0:
            return None

        corners = np.array(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, hmat).reshape(-1, 2)
        except cv2.error:
            return None

        if not np.all(np.isfinite(warped)):
            return None

        x1 = float(np.min(warped[:, 0]))
        y1 = float(np.min(warped[:, 1]))
        x2 = float(np.max(warped[:, 0]))
        y2 = float(np.max(warped[:, 1]))
        warped_bbox = np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])
        return self._clip_bbox_inside_frame(warped_bbox, frame)

    def _innermost_visual_tracker(self):
        tracker = self.tracker
        if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "tracking_state"):
            return tracker.tracker
        if hasattr(tracker, "base_tracker"):
            base = tracker.base_tracker
            if hasattr(base, "tracker") and hasattr(base.tracker, "tracking_state"):
                return base.tracker
            if hasattr(base, "tracking_state"):
                return base
        if hasattr(tracker, "tracking_state"):
            return tracker
        return None

    def _set_tracker_search_prior(
        self,
        prior_bbox: np.ndarray,
    ) -> bool:
        visual_tracker = self._innermost_visual_tracker()
        if visual_tracker is None:
            return False
        prior = np.asarray(prior_bbox, dtype=int).copy()
        visual_tracker.tracking_state.bbox = prior.copy()
        if hasattr(visual_tracker, "_ot"):
            visual_tracker._ot.state = list(prior.astype(float))
        return True

    def _apply_gmc_search_prior(
        self,
        frame: np.ndarray,
    ) -> None:
        self._camera_motion_subsystem.apply_gmc_search_prior(
            frame=frame,
        )

    def _h_translation_magnitude(
        self,
        H,
        frame: np.ndarray,
    ) -> float:
        """
        Projects the frame centre through H and returns the Euclidean distance
                    between the original and projected centres as a scalar pixel magnitude.
                    Returns 0.0 if H is None.

        Called every frame in _normal_update and stored in _cam_disp_history.
                    _classify_loss_cause uses the weighted average of this history to
                    decide whether loss was caused by camera motion (fast camera pan) or
                    by the target being occluded. A high and sustained displacement suggests
                    the camera moved faster than the tracker could follow.
        Args:
            H (any): 3×3 homography matrix or None
            frame (any): current video frame (used only for its shape)
        Returns:
            float — camera displacement magnitude in pixels this frame
    """
        if H is None:
            return 0.0
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        denom = H[2, 0] * cx + H[2, 1] * cy + H[2, 2] + 1e-8
        new_cx = (H[0, 0] * cx + H[0, 1] * cy + H[0, 2]) / denom
        new_cy = (H[1, 0] * cx + H[1, 1] * cy + H[1, 2]) / denom
        return float(np.hypot(new_cx - cx, new_cy - cy))

    def _camera_motion_weighted_disp(
        self,
        frame: Optional[np.ndarray] = None,
    ) -> float:
        """
        Exponentially weighted camera displacement mean. If frame is provided,
        includes the current-frame displacement in addition to history.
        """
        cam_hist = list(self._cam_disp_history)
        if frame is not None:
            cam_hist.append(self._h_translation_magnitude(self._last_H, frame))
        if not cam_hist:
            return 0.0

        vals = np.asarray(cam_hist, dtype=float)
        weights = np.exp(np.linspace(-1.0, 0.0, len(vals)))
        denom = float(np.sum(weights))
        if denom <= 1e-8:
            return float(vals[-1])
        weights /= denom
        return float(np.dot(weights, vals))

    def _motion_gate_reference_diag(
        self,
        frame: np.ndarray,
        bbox_hint: Optional[np.ndarray] = None,
    ) -> float:
        """
        Reference bbox diagonal used to normalize camera displacement for
        tiny-object-aware heavy-motion gating.
        """
        ref_bbox = bbox_hint
        if ref_bbox is None:
            ref_bbox = self.current_bbox
        if ref_bbox is None:
            ref_bbox = self.held_box

        if ref_bbox is not None:
            arr = np.asarray(ref_bbox, dtype=float).reshape(-1)
            if arr.size >= 4:
                w = float(arr[2])
                h = float(arr[3])
                if w > 0.0 and h > 0.0:
                    return max(1e-6, float(np.hypot(w, h)))

        h_fr, w_fr = frame.shape[:2]
        return max(1e-6, float(np.hypot(float(w_fr), float(h_fr))))

    def _is_heavy_camera_motion(
        self,
        frame: np.ndarray,
        bbox_hint: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        return self._camera_motion_subsystem.is_heavy_camera_motion(
            frame=frame,
            bbox_hint=bbox_hint,
        )

    def _classify_loss_cause(
        self,
        cam_disp_threshold: float = 18.0,
        area_shrink_threshold: float = -0.005,
    ) -> str:
        """
        Casts two independent votes — one from area trend, one from camera
                    displacement history — and returns 'camera_motion' only when both agree.
                    Area vote: fits a linear slope to the bbox area history; a non-negative
                    slope means the object was not shrinking before loss, suggesting camera
                    motion rather than occlusion. Camera vote: computes an exponentially
                    weighted mean of _cam_disp_history; high displacement confirms camera
                    motion.

        Called once at occlusion entry from _normal_update. Its return value
                    controls effective_skip: camera_motion and out_of_frame losses get
                    skip=0 (the history is clean), while occlusion losses get
                    skip=max(dynamic_skip, entry_streak). Using the wrong skip wastes good
                    history entries or retains corrupted ones, both of which hurt EKF
                    reconstruction accuracy.
        Args:
            cam_disp_threshold (any): weighted mean camera displacement (px) above
            which motion is classified as camera_motion
            area_shrink_threshold (any): normalised area slope below which object is
            classified as shrinking (i.e. not camera motion)
        Returns:
            str — either 'camera_motion' or 'occlusion'
    """
        history = list(self._conf_history)
        cam_hist = list(self._cam_disp_history)

        area_vote = "occlusion"
        if len(history) >= 3:
            areas = np.array([float(e.bbox[2] * e.bbox[3]) for e in history])
            med_area = float(np.median(areas))
            n = len(areas)
            t = np.arange(n, dtype=float)
            slope = float(np.polyfit(t, areas, 1)[0])
            norm_slope = slope / (med_area + 1e-6)
            if norm_slope >= area_shrink_threshold:
                area_vote = "camera_motion"

        cam_vote = "occlusion"
        if cam_hist:
            mean_disp = self._camera_motion_weighted_disp(frame=None)
            if mean_disp >= cam_disp_threshold:
                cam_vote = "camera_motion"

        if area_vote == "camera_motion" and cam_vote == "camera_motion":
            return "camera_motion"
        return "occlusion"

    def _is_long_distance(
        self,
        frame: np.ndarray,
    ) -> bool:
        """
        Returns True if the object is classified as tiny or far away.
                    This is the case when long_distance_mode is explicitly set to True,
                    or when the object's median area from _size_history is below
                    long_distance_area_fraction of the total frame area. Returns False
                    if _size_history is empty (can't judge yet).

        Called in _normal_update (to switch confidence threshold and search
                    context), in _get_yolo_search_roi (to pick the ROI parameter set),
                    and in _occ_phase_final_drm (to halve lam_cand_dir). Tiny objects
                    have unreliable velocity measurements and need wider ROIs, so
                    separating them here avoids polluting the normal-scale parameters.
        Args:
            frame (any): current video frame (used only for its shape)
        Returns:
            bool — True means the target is tiny/far and tiny: object parameter
            sets should be used
    """
        if self.long_distance_mode:
            return True
        if not self._size_history:
            return False
        h_fr, w_fr = frame.shape[:2]
        frame_area = float(h_fr * w_fr)
        size_arr = np.asarray(self._size_history, dtype=np.float32)
        med_wh = np.median(size_arr, axis=0)
        obj_w = float(med_wh[0])
        obj_h = float(med_wh[1])
        obj_area = obj_w * obj_h
        return (obj_area / (frame_area + 1e-8)) < self.long_distance_area_fraction

    def _rebuild_ekf_from_clean_history(
        self,
        skip_override=None,
    ) -> BBoxEKF:
        """
        Creates a brand-new BBoxEKF seeded at the oldest clean history entry,
                    then replays all clean history entries through predict + update to get
                    a physically consistent state estimate at the last clean frame. After
                    replay, injects the robust velocity estimate from _robust_velocity_from_history
                    directly into the EKF state vector with a moderately high covariance
                    so the filter has a sensible prior but is not overconfident about
                    velocity at the start of occlusion.

        Called at occlusion entry from _normal_update. The existing EKF has
                    been running on live SiamABC output which may have been noisy or
                    drifting during the entry_streak frames. Replaying only the clean
                    portion from scratch gives a much cleaner velocity estimate and
                    eliminates EKF state contamination from bad tracker outputs.
        Args:
            skip_override (any): int or None; how many tail entries to drop before
            replaying history. If None, uses self.history_skip_last.
        Returns:
            BBoxEKF instance — fresh, consistent, velocity: seeded, ready to predict
    """
        history = list(self._conf_history)
        raw_skip = self.history_skip_last if skip_override is None else skip_override
        skip = min(raw_skip, max(0, len(history) - 2))
        clean = history[: len(history) - skip] if skip > 0 else history

        if len(clean) == 0:
            ekf = self.ekf
            assert ekf is not None
            return ekf

        first_bbox = clean[0].bbox
        fresh_ekf = BBoxEKF(
            first_bbox,
            process_noise=self.ekf_process_noise,
            meas_noise=self.ekf_meas_noise,
        )

        for entry in clean[1:]:
            fresh_ekf.predict(H=entry.H, H_reliable=(entry.H is not None))
            fresh_ekf.update(entry.bbox)

        robust_vel = self._robust_velocity_from_history(
            skip=skip,
            window=self.velocity_window_average,
            decay=0.97,
            clip_percentile=96.0,
        )
        fresh_ekf.x[2] = float(robust_vel[0])
        fresh_ekf.x[3] = float(robust_vel[1])
        fresh_ekf.P[2, 2] = 40.0
        fresh_ekf.P[3, 3] = 40.0

        return fresh_ekf

    def _detect_shrinkage_onset(
        self,
        max_lookback: int = 10,
        min_drop_frac: float = 0.002,
        smooth_k: int = 3,
    ) -> int:
        """
        Scans _conf_history from newest to oldest looking for frames where the
                    smoothed bounding box area fell below (1 - min_drop_frac) × ref_area,
                    where ref_area is the 95th percentile area across all history. Counts
                    a contiguous block of shrinking frames (with a small gap budget of 3
                    to allow isolated glitches) and returns the count as the recommended
                    skip depth. Caps at max_lookback and runs a sanity check — if the
                    average area of the flagged frames is not actually below threshold,
                    returns 0.

        Called at occlusion entry from _normal_update alongside
                    _detect_center_drift_skip. When a target is being occluded gradually,
                    the tracker bbox often shrinks before the score drops below threshold.
                    Skipping those shrinking frames prevents the EKF from being seeded with
                    a velocity that points into the occluder instead of tracking the target.
        Args:
            max_lookback (any): maximum number of tail frames to scan backward
            min_drop_frac (any): minimum fractional area drop relative to 95th
            percentile area to count a frame as shrinking
            smooth_k (any): rolling median window applied to areas before analysis
        Returns:
            int — number of tail history frames that appear corrupted by shrinkage;
            0 if no shrinkage detected
    """
        history = list(self._conf_history)
        n = len(history)
        if n < 4:
            return 0

        areas = np.array([float(e.bbox[2] * e.bbox[3]) for e in history], dtype=float)
        smoothed = np.array(
            [
                float(np.median(areas[max(0, i - smooth_k + 1): i + 1]))
                for i in range(n)
            ]
        )

        ref_area = float(np.percentile(smoothed, 95))
        if ref_area <= 0:
            return 0

        threshold = ref_area * (1.0 - min_drop_frac)
        skip = 0
        gap_budget = 3
        gaps_used = 0

        for i in range(n - 1, max(n - 1 - max_lookback, -1), -1):
            if smoothed[i] < threshold:
                skip += 1
                gaps_used = 0
            elif skip > 0 and gaps_used < gap_budget:
                skip += 1
                gaps_used += 1
            else:
                break

        if skip >= max_lookback:
            full_drop = (ref_area - float(np.mean(smoothed[n - skip:]))) / (
                ref_area + 1e-6
            )
            if full_drop < min_drop_frac:
                return 0

        return max(skip, 5)

    def _detect_center_drift_skip(
        self,
        max_lookback: int = 20,
        spike_factor: float = 2.5,
    ) -> int:
        """
        Computes per-frame centre speeds from _center_history, estimates a
                    reference speed from the first two thirds of the history, then scans
                    backward from the newest frame looking for speeds above
                    spike_factor × reference. Counts a contiguous spike block (with a
                    small gap budget of 2) and returns the count. Unlike shrinkage
                    detection this does not apply a sanity cap — any spike block is
                    returned as-is.

        Called at occlusion entry alongside _detect_shrinkage_onset. A centre
                    drift spike usually means the tracker jumped to a distractor just
                    before losing the target — those frames will have bad velocity that
                    would corrupt the EKF rebuild if not skipped. Together, shrinkage and
                    drift detection give the EKF the cleanest possible initial conditions.
        Args:
            max_lookback (any): maximum number of tail frames to scan backward
            spike_factor (any): how many times the median speed a frame must exceed
            to be flagged as a drift spike
        Returns:
            int — number of tail history frames flagged as drift spikes; 0 if none
    """
        hist = list(self._center_history)
        if len(hist) < 4:
            return 0

        centers = np.stack(hist)
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        n = len(speeds)

        ref_n = max(2, n * 2 // 3)
        ref_mag = float(np.median(speeds[:ref_n])) + 1e-6
        thresh = ref_mag * spike_factor

        skip = 0
        gap_budget = 2
        gaps_used = 0

        for i in range(n - 1, max(n - 1 - max_lookback, -1), -1):
            if speeds[i] > thresh:
                skip += 1
                gaps_used = 0
            elif skip > 0 and gaps_used < gap_budget:
                skip += 1
                gaps_used += 1
            else:
                break

        return skip

    def _detect_exit_direction(
        self,
        frame: np.ndarray,
        lookahead_frames: int = 6,
        trend_frames: int = 10,
        margin_factor: float = 0.5,
    ) -> Tuple[bool, Optional[str]]:
        """
        Runs three parallel tests to decide if the target is exiting the frame
                    and in which direction:
                    1. Proximity + velocity: is the bbox near an edge and moving toward it?
                    2. Velocity extrapolation: does current_bbox + velocity × lookahead go off-frame?
                    3. Trend extrapolation: does a linear fit to recent history × lookahead go off-frame?
                    If two out of three tests agree on an edge, that edge is returned as
                    the exit direction. Also accepts the case where the bbox is literally
                    already off-screen on any side.

        Called at occlusion entry from _normal_update. If exit is detected,
                    loss_cause is overridden to 'out_of_frame', effective_skip is set to 0,
                    and _out_of_frame + _exit_edge are set. The occlusion dispatcher then
                    pins the search centre to that edge and switches to edge-strip ROIs.
                    Without this, a target that walked off frame would be searched for in
                    the centre of the image — almost certainly finding the wrong object.
        Args:
            frame (any): current video frame (used only for its shape)
            lookahead_frames (any): how many frames forward velocity/trend is extrapolated
            trend_frames (any): how many history frames are used to fit the linear trend
            margin_factor (any): fraction of max(obj_w, obj_h) used as proximity margin
        Returns:
            (bool, Optional[str]) — (is_exiting, edge)
            edge is one of 'right', 'left', 'bottom', 'top', or None
    """
        h_fr, w_fr = frame.shape[:2]

        current_bbox = self.current_bbox
        assert current_bbox is not None
        last_bbox = current_bbox.astype(float)
        lx1, ly1, lw, lh = last_bbox
        lx2, ly2 = lx1 + lw, ly1 + lh
        lcx = lx1 + lw / 2.0
        lcy = ly1 + lh / 2.0

        vx = float(self.velocity[0])
        vy = float(self.velocity[1])

        margin = max(lw, lh) * margin_factor
        prox_right = lx2 > w_fr - margin and vx > 0
        prox_left = lx1 < margin and vx < 0
        prox_bottom = ly2 > h_fr - margin and vy > 0
        prox_top = ly1 < margin and vy < 0

        fut_cx = lcx + vx * lookahead_frames
        fut_cy = lcy + vy * lookahead_frames
        extrap_right = fut_cx >= w_fr
        extrap_left = fut_cx < 0
        extrap_bottom = fut_cy >= h_fr
        extrap_top = fut_cy < 0

        history = list(self._conf_history)
        n_use = min(trend_frames, len(history))
        trend_right = trend_left = trend_bottom = trend_top = False

        if n_use >= 3:
            recent = history[-n_use:]
            xs = np.array([float(e.bbox[0] + e.bbox[2] / 2) for e in recent])
            ys = np.array([float(e.bbox[1] + e.bbox[3] / 2) for e in recent])
            t = np.arange(n_use, dtype=float)

            vx_trend = float(np.polyfit(t, xs, 1)[0])
            vy_trend = float(np.polyfit(t, ys, 1)[0])

            fut_tx = lcx + vx_trend * lookahead_frames
            fut_ty = lcy + vy_trend * lookahead_frames
            trend_right = fut_tx >= w_fr and vx_trend > 0
            trend_left = fut_tx < 0 and vx_trend < 0
            trend_bottom = fut_ty >= h_fr and vy_trend > 0
            trend_top = fut_ty < 0 and vy_trend < 0

        def _votes(
            right,
            left,
            bottom,
            top,
        ):
            return {"right": right, "left": left, "bottom": bottom, "top": top}

        e1 = _votes(prox_right, prox_left, prox_bottom, prox_top)
        e2 = _votes(extrap_right, extrap_left, extrap_bottom, extrap_top)
        e3 = _votes(trend_right, trend_left, trend_bottom, trend_top)

        for edge in ("right", "left", "bottom", "top"):
            if sum([e1[edge], e2[edge], e3[edge]]) >= 2:
                return True, edge

        if lx2 >= w_fr:
            return True, "right"
        if lx1 <= 0:
            return True, "left"
        if ly2 >= h_fr:
            return True, "bottom"
        if ly1 <= 0:
            return True, "top"

        return False, None

    def _robust_velocity_from_history(
        self,
        skip=0,
        window=80,
        decay=0.97,
        clip_percentile=80.0,
    ) -> np.ndarray:
        """
        Computes a weighted average of per-frame centre displacements from
                    _center_history, subtracts the correspondingly weighted camera velocity
                    to get ego-motion-compensated target velocity, then caps the magnitude
                    at the clip_percentile of all per-frame speeds to suppress outlier
                    frames (e.g. a single frame where the tracker jumped). Returns the
                    current self.velocity if fewer than 2 history points exist after
                    trimming.

        Called inside _rebuild_ekf_from_clean_history to seed the EKF
                    velocity state at occlusion entry. A simple finite-difference velocity
                    would be corrupted by any tracker jump at the end of the sequence.
                    The decay weighting and magnitude clipping together produce a velocity
                    that reflects stable long-term motion rather than the last noisy frame.
        Args:
            skip (any): number of tail entries to discard from both
            _center_history and _cam_vel_history before computing
            window (any): maximum number of frames to look back
            decay (any): per-frame weight decay; recent frames get higher weight
            clip_percentile (any): per-frame speed percentile used as a magnitude cap
        Returns:
            np.ndarray (2,) — robust camera: compensated velocity in px/frame
    """
        hist = list(self._center_history)
        if len(hist) < 2:
            return self.velocity.copy()

        conf_len = len(list(self._conf_history))
        hist = hist[-conf_len:] if len(hist) > conf_len else hist

        if skip > 0:
            hist = hist[: max(2, len(hist) - skip)]

        hist = hist[-window:] if len(hist) > window else hist
        if len(hist) < 2:
            return self.velocity.copy()

        centers = np.stack(hist)
        frame_vels = np.diff(centers, axis=0)
        n = len(frame_vels)

        weights = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
        weights /= weights.sum()

        avg = (weights[:, None] * frame_vels).sum(axis=0)

        cam_hist = list(self._cam_vel_history)
        cam_hist = cam_hist[-conf_len:] if len(cam_hist) > conf_len else cam_hist
        if skip > 0:
            cam_hist = cam_hist[: max(2, len(cam_hist) - skip)]
        cam_hist = cam_hist[-window:] if len(cam_hist) > window else cam_hist

        if len(cam_hist) >= n:
            cam_vels = np.stack(cam_hist[-n:])
            avg_cam = (weights[:, None] * cam_vels).sum(axis=0)
            avg -= avg_cam

        magnitudes = np.linalg.norm(frame_vels, axis=1)
        mag_cap = float(np.percentile(magnitudes, clip_percentile))
        avg_mag = float(np.linalg.norm(avg))
        if avg_mag > mag_cap and avg_mag > 0:
            avg = avg * (mag_cap / avg_mag)

        return avg

    def _try_detect_target_class(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Runs YOLO on a padded crop around the current bbox, finds the
                    detection with the highest IoU overlap with that bbox, and adds a
                    vote for that detection's class ID into self._class_votes. Does
                    nothing if no detection clears 0.3 IoU.

        Called every yolo_class_detect_frames frames during the first
                    yolo_class_detect_frames × 3 frames when yolo_filter_class=True.
                    It runs before _maybe_commit_target_class. The idea is to observe
                    which class YOLO consistently assigns to the target area during
                    healthy tracking, then lock the filter to that class before entering
                    any occlusion — so YOLO won't waste DRM budget on detections from
                    unrelated classes during recovery.
        Args:
            frame (any): current video frame as a numpy BGR array
    """
        if self.current_bbox is None:
            return
        x, y, w, h = self.current_bbox
        pad = max(10, int(max(w, h) * 0.2))
        h_fr, w_fr = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_fr, x + w + pad)
        y2 = min(h_fr, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        results = self.yolo.predict(
            crop,
            conf=self.yolo_conf,
            iou=self.yolo_iou_thr,
            verbose=False,
            imgsz=self.yolo_imgsz,
            augment=self.yolo_augment,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return

        result_boxes = results[0].boxes
        xyxy_all = result_boxes.xyxy.detach().cpu().numpy()
        cls_all = result_boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)

        det_xywh = np.column_stack(
            (
                xyxy_all[:, 0] + x1,
                xyxy_all[:, 1] + y1,
                xyxy_all[:, 2] - xyxy_all[:, 0],
                xyxy_all[:, 3] - xyxy_all[:, 1],
            )
        ).astype(np.int32, copy=False)
        ious = self._iou_many_to_one(
            det_xywh.astype(np.float64, copy=False), self.current_bbox
        )
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        best_cls = int(cls_all[best_idx])

        if best_iou >= 0.3 and best_cls is not None:
            self._class_votes[best_cls] = self._class_votes.get(best_cls, 0) + 1

    def _maybe_commit_target_class(
        self,
    ) -> None:
        """
        Checks whether any single class has received at least 60% of all votes
                    in _class_votes. If yes, commits that class as _target_class_id and
                    sets _class_warmup_done=True so the warm-up loop stops. If no class
                    has 60% consensus, does nothing and leaves _target_class_id as None
                    so YOLO detections remain unfiltered until consensus is reached.

        Called from _normal_update after every _try_detect_target_class call,
                    starting at frame yolo_class_detect_frames × 2. The 60% threshold
                    prevents premature commitment on sequences where YOLO is inconsistent
                    about the target's class in the first few frames, while still locking
                    quickly enough to be useful before the first occlusion typically occurs.
        Args:
            None. Reads self._class_votes internally.
    """
        votes = self._class_votes
        if not votes:
            return
        best_cls = max(votes, key=lambda cls_id: votes[cls_id])
        total = sum(votes.values())
        if votes[best_cls] / total >= 0.6:
            self._target_class_id = best_cls
            self._class_warmup_done = True
            if self.debug:
                print(
                    f"[class filter] target class locked: {best_cls}  "
                    f"(votes={votes})"
                )

    def _apply_frame_dynamics(self, frame: np.ndarray) -> np.ndarray:
        """Blend frame-difference motion saliency into the search region.

        Frame-dynamics input augmentation (arXiv:2505.04917, Sec. 3.2). When the
        processor is built (master switch on), compute the short-term
        frame-difference motion saliency between consecutive full frames and blend
        it ADDITIVELY into the search region the frozen SiamABC backbone is about
        to crop. The processor owns the previous-frame buffer, so we always feed
        it the current full frame (even before a bbox exists) to keep its history
        warm. Returns a NEW frame with the search region emphasised, or the input
        frame unchanged when the feature is disabled / no motion is available yet.

        When the processor is None (feature off) this is a no-op and the original
        frame is returned, so the legacy search crop is byte-for-byte unchanged.
        """
        proc = self._frame_dynamics_processor
        if proc is None:
            return frame

        # Resolve the search window (bbox padded by the SiamABC search_context
        # ratio) so the saliency registers with the crop. Before a bbox exists
        # (frame 0 paths) fall back to the whole frame.
        if self.current_bbox is not None:
            x, y, w, h = self._tracker_search_roi(frame, self.current_bbox)
            x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        else:
            h_fr, w_fr = frame.shape[:2]
            x0, y0, x1, y1 = 0, 0, int(w_fr), int(h_fr)

        h_fr, w_fr = frame.shape[:2]
        x0 = max(0, min(w_fr, x0))
        y0 = max(0, min(h_fr, y0))
        x1 = max(0, min(w_fr, x1))
        y1 = max(0, min(h_fr, y1))
        if x1 <= x0 or y1 <= y0:
            # Degenerate window: still advance the buffer so motion is available
            # next frame, but make no spatial edit.
            proc.process(frame, frame, None, None)
            return frame

        search_crop = frame[y0:y1, x0:x1]
        modified = proc.process(
            search_crop,
            frame,
            self.current_bbox,
            (x0, y0, x1, y1),
        )
        # No change (disabled / zero weight / no motion yet) -> return original.
        if modified is search_crop:
            return frame

        out = frame.copy()
        out[y0:y1, x0:x1] = modified
        return out

    def _tracker_search_roi(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
        Build the YOLO probe ROI from the tracker's own search region: the bbox
        padded on every side by the SiamABC `search_context` ratio, mirroring how
        the tracker crops its search window (see SiamABC_Tracker.run_track_candidate).
        Clamped to the frame.
        """
        ctx = float(self.tracker.tracking_config["search_context"])
        x, y, w, h = (float(v) for v in bbox)
        pad_w = w * ctx
        pad_h = h * ctx
        h_fr, w_fr = frame.shape[:2]
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(w_fr, int(x + w + pad_w))
        y2 = min(h_fr, int(y + h + pad_h))
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _maybe_run_detectability_probe(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Class-agnostic YOLO-detectability probe.

        For the first `yolo_detectability_probe_attempts` strided frames (one
        attempt every `yolo_detectability_probe_stride` frames), run YOLO on the
        tracker's search ROI and count it as a "hit" when any detection overlaps
        the current bbox by at least `yolo_detectability_iou_thr` IoU. As soon as
        the hit count reaches `yolo_detectability_min_hits` we lock
        `_yolo_detectable = True` and stop. If the attempts are exhausted without
        enough hits, we lock `_yolo_detectable = False`.

        The resulting flag drives the occlusion-recovery policy in _occ_phase_siam.
        Does nothing unless `yolo_detectability_enabled` is set.
        """
        if not self._yolo_detectability_enabled or self._detectability_probe_done:
            return
        if self.frame_idx <= 0 or self.frame_idx % self._yolo_detectability_probe_stride != 0:
            return
        if self.current_bbox is None:
            return

        roi = self._tracker_search_roi(frame, self.current_bbox)
        detections = self._yolo_detect_in_roi(frame, roi)
        self._detectability_runs += 1

        if detections:
            ious = self._iou_many_to_one(
                np.asarray(detections, dtype=np.float64), self.current_bbox
            )
            if float(np.max(ious)) >= self._yolo_detectability_iou_thr:
                self._detectability_hits += 1

        if self._detectability_hits >= self._yolo_detectability_min_hits:
            self._yolo_detectable = True
            self._detectability_probe_done = True
            if self.debug:
                print(
                    f"[detectability] YOLO-detectable=True after {self._detectability_runs} "
                    f"probe(s), hits={self._detectability_hits} (frame {self.frame_idx})"
                )
            return

        if self._detectability_runs >= self._yolo_detectability_probe_attempts:
            self._yolo_detectable = False
            self._detectability_probe_done = True
            if self.debug:
                print(
                    f"[detectability] YOLO-detectable=False after {self._detectability_runs} "
                    f"probe(s), hits={self._detectability_hits} (frame {self.frame_idx})"
                )

    def _detectability_policy_active(
        self,
    ) -> bool:
        """
        The adaptive occlusion policy only kicks in once the probe has finished
        and produced a verdict. Until then (or when disabled) recovery behaves
        exactly as before.
        """
        return self._yolo_detectability_enabled and self._detectability_probe_done

    def _phase_after_failed_siam(
        self,
    ) -> int:
        """
        Occlusion phase to advance to after a failed phase=siam attempt.

        Normally → 1 (begin YOLO candidate collection). But when the policy is
        active and the target was found NOT YOLO-detectable, YOLO collection is
        futile, so stay at phase 0 and keep retrying SiamABC every frame.
        """
        if self._detectability_policy_active() and not self._yolo_detectable:
            return 0
        return 1

    def _is_near_exit_edge(
        self,
        bbox: np.ndarray,
        frame: np.ndarray,
        fraction: float = 0.5,
    ) -> bool:
        """
        During out-of-frame occlusion, returns True only when the candidate's
        centre lies within `fraction` of the frame dimension measured from the
        exit edge.  Always returns True when not in out-of-frame mode.

        fraction=0.5  →  only the half of the frame nearest the exit edge is valid.
        fraction=0.33 →  only the nearest third.
        """
        if not self._out_of_frame or self._exit_edge is None:
            return True

        h_fr, w_fr = frame.shape[:2]
        cx = float(bbox[0] + bbox[2] / 2.0)
        cy = float(bbox[1] + bbox[3] / 2.0)

        if self._exit_edge == "right":
            return cx >= w_fr * (1.0 - fraction)
        elif self._exit_edge == "left":
            return cx <= w_fr * fraction
        elif self._exit_edge == "bottom":
            return cy >= h_fr * (1.0 - fraction)
        elif self._exit_edge == "top":
            return cy <= h_fr * fraction

        return True

    def load_yolo_compiled(
        self,
        weights_path,
        force_recompile=False,
    ):
        """
        Load a TensorRT compiled YOLO engine, compiling it if necessary.

        Args:
            weights_path (str): Path to the PyTorch YOLO weights (.pt).
            force_recompile (bool, optional): Whether to force recompilation. Defaults to False.

        Returns:
            YOLO: The loaded YOLO model using the TensorRT engine.
        """
        engine_path = weights_path.replace(".pt", ".engine")

        if not os.path.exists(engine_path) or force_recompile:
            siamram_log(
                f"Compiling YOLO engine at {self.yolo_imgsz}x{self.yolo_imgsz} "
                "(may take a minute)",
                phase="YOLO",
                status="build",
            )

            model = YOLO(weights_path)

            model.export(format="engine", half=False, device=0, imgsz=self.yolo_imgsz)

        return YOLO(engine_path)

    def _prescale_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Downscale frame to at most the configured long-edge cap.
        Uses self._frame_scale set once during initialize().

        Returns the original frame unchanged if _frame_scale == 1.0 (no copy,
        no allocation — zero cost for inputs already at or below the cap).
        """
        if self._frame_scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (int(w * self._frame_scale), int(h * self._frame_scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    @property
    def running_dynamic_bbox(
        self,
    ):
        """
        Return the bounding box used for the running dynamic template.
        """
        return self.tracker.running_dynamic_bbox

    @property
    def running_dynamic_image(
        self,
    ):
        """
        Get the image used for the dynamic template.

        Returns:
            np.ndarray: The running dynamic image from the underlying tracker.
        """
        return self.tracker.running_dynamic_image

    @property
    def tracking_config(
        self,
    ):
        """
        Return the underlying tracker configuration.
        """
        return self.tracker.tracking_config

    @property
    def tracking_state(
        self,
    ):
        """
        Return the current internal tracking state.
        """
        return self.tracker.tracking_state
