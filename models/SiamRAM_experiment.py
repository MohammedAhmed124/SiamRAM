"""
SiamRAM Tracker: A robust hybrid tracker with occlusion recovery.

This module implements the SiamRAMTracker, which combines Siamese visual
tracking (SiamABC) with a YOLO-based re-detection system and an Extended
Kalman Filter (EKF) for motion modelling. It features a Dynamic Reference
Memory (DRM) for reliable re-acquisition after long-term occlusion.
"""

import os
from collections import deque
from typing import List, Optional, Tuple, TypedDict, cast

import cv2
import numpy as np
from numpy._typing import NDArray
from ultralytics import YOLO

from utils.utils import (
    _cos_sim,
    _extract_descriptor,
    _iou,
    configure_descriptor_backend,
)

from .SiamABC.tracker.SiamABC_Tracker import SiamABCTracker
from .camera_motion_botsort import (
    CameraMotionEstimator as BotSortMotionEstimator,
)
from .camera_motion_botsort import MotionConfig as BotSortMotionConfig
from .motion_model import BBoxEKF
from .ram_memory import AppearanceMemory


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
        osnet_device: str = "auto",
        conf_threshold: float = 0.60,
        occ_siam_reacq_threshold=0.8,
        reacq_threshold: float = 0.55,
        yolo_conf: float = 0.30,
        yolo_iou: float = 0.45,
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
        drm_margin: float = 0.35,
        drm_top_k: int = 3,
        drm_skip_threshold: float = 0.80,
        drm_lam_dist: float = 0.15,
        drm_dist_sigma_factor: float = 2.5,
        drm_lam_cand_dir: float = 0.15,
        drm_lam_cand_vel: float = 0.20,
        vel_score_min_speed: float = 0.5,
        ekf_process_noise: float = 2.0,
        ekf_meas_noise: float = 5.0,
        homo_max_corners: int = 200,
        homo_inlier_threshold: float = 0.50,
        homography_mode: str = "classic",
        botsort_motion_model: str = "partial_affine",
        botsort_motion_work_scale: float = 0.75,
        botsort_motion_bbox_pad: float = 0.50,
        botsort_motion_grid_rows: int = 6,
        botsort_motion_grid_cols: int = 8,
        botsort_motion_max_corners_per_cell: int = 24,
        botsort_motion_quality_level: float = 0.01,
        botsort_motion_min_distance: int = 7,
        botsort_motion_block_size: int = 7,
        botsort_motion_lk_win_size: int = 21,
        botsort_motion_lk_max_level: int = 3,
        botsort_motion_fb_max_error: float = 1.5,
        botsort_motion_ransac_reproj_threshold: float = 3.0,
        botsort_motion_ransac_confidence: float = 0.99,
        botsort_motion_ransac_max_iters: int = 3000,
        botsort_motion_min_inliers: int = 30,
        botsort_motion_min_inlier_ratio: float = 0.35,
        botsort_motion_max_median_residual: float = 4.0,
        botsort_motion_min_coverage_cells: int = 6,
        botsort_motion_max_translation_frac: float = 0.25,
        botsort_motion_min_scale: float = 0.70,
        botsort_motion_max_scale: float = 1.40,
        botsort_motion_max_rotation_deg: float = 25.0,
        botsort_motion_residual_mad_multiplier: float = 4.0,
        botsort_motion_residual_min_threshold: int = 18,
        botsort_motion_residual_max_fraction: float = 0.22,
        botsort_motion_residual_min_area: int = 12,
        botsort_motion_dynamic_dilate: int = 9,
        botsort_motion_feature_detect_interval: int = 2,
        botsort_motion_min_points_to_redetect: int = 120,
        botsort_motion_max_track_points: int = 320,
        botsort_motion_enable_dynamic_mask: bool = True,
        botsort_motion_residual_update_interval: int = 2,
        botsort_motion_use_fallback_transform: bool = True,
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
        block_distractor_mode_on_camera_motion: bool = True,
        block_occlusion_on_camera_motion: bool = True,
        camera_motion_heavy_disp_threshold: float = 18.0,
        camera_motion_heavy_norm_threshold: float = 0.35,
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
        max_proc_long_edge: int = 1280,
        entry_patience: int = 3,
        cand_collection_frames: int = 3,
        osnet_max_candidate_batch: int = 0,
        yolo_warmup_frames: int = 60,
        yolo_warmup_center_scale: float = 0.4,
        vel_dir_hard_gate: float = 0.5,
        yolo_filter_class: bool = False,
        yolo_class_detect_frames: int = 5,
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
            homography_mode (any): camera-motion mode; "classic" uses grid+affine RANSAC, "accurate" uses feature tracking + full homography RANSAC, "botsort" uses BoT-SORT-inspired GMC gating/fallback
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
            yolo_warmup_frames (any): initial frame count that uses centered warm-up ROI
            yolo_warmup_center_scale (any): warm-up ROI side ratio relative to frame size
            vel_dir_hard_gate (any): if cosine similarity between candidate and expected velocity is below -this, velocity score is floored to 0.05
            yolo_filter_class (any): if True, YOLO detections are filtered to the target's class after warm-up
            yolo_class_detect_frames (any): stride used during the class warm-up period
    """
        self.tracker: SiamABCTracker = siam_tracker
        configure_descriptor_backend(
            descriptor_backend=descriptor_backend,
            osnet_model_name=osnet_model_name,
            osnet_model_path=osnet_model_path,
            osnet_device=osnet_device,
        )

        if copile_yolo:
            self.yolo = self.load_yolo_compiled(yolo_weights)
        else:
            self.yolo = YOLO(yolo_weights)

        self.debug = debug
        self.conf_threshold = conf_threshold
        self.reacq_threshold = reacq_threshold
        self.yolo_conf = yolo_conf
        self.yolo_iou_thr = yolo_iou
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
        if mode in {"bot_sort", "gmc", "botsort_gmc"}:
            mode = "botsort"
        if mode not in {"classic", "accurate", "botsort"} and self.debug:
            print(
                f"[homography] unsupported mode '{homography_mode}', "
                "falling back to 'classic'"
            )
        self._homography_mode = (
            mode if mode in {"classic", "accurate", "botsort"} else "classic"
        )
        bot_model = str(botsort_motion_model).strip().lower()
        if bot_model not in {"partial_affine", "full_affine", "homography"}:
            if self.debug:
                print(
                    f"[homography:botsort] unsupported model '{botsort_motion_model}', "
                    "falling back to 'partial_affine'"
                )
            bot_model = "partial_affine"
        self._botsort_motion_cfg = BotSortMotionConfig(
            model=bot_model,
            work_scale=float(max(0.10, botsort_motion_work_scale)),
            bbox_pad=float(max(0.0, botsort_motion_bbox_pad)),
            grid_rows=max(1, int(botsort_motion_grid_rows)),
            grid_cols=max(1, int(botsort_motion_grid_cols)),
            max_corners_per_cell=max(1, int(botsort_motion_max_corners_per_cell)),
            quality_level=float(max(1e-6, botsort_motion_quality_level)),
            min_distance=int(max(1, botsort_motion_min_distance)),
            block_size=int(max(3, botsort_motion_block_size)),
            lk_win_size=int(max(5, botsort_motion_lk_win_size)),
            lk_max_level=int(max(0, botsort_motion_lk_max_level)),
            fb_max_error=float(max(0.0, botsort_motion_fb_max_error)),
            ransac_reproj_threshold=float(max(0.1, botsort_motion_ransac_reproj_threshold)),
            ransac_confidence=float(np.clip(botsort_motion_ransac_confidence, 0.50, 0.9999)),
            ransac_max_iters=int(max(50, botsort_motion_ransac_max_iters)),
            min_inliers=int(max(4, botsort_motion_min_inliers)),
            min_inlier_ratio=float(np.clip(botsort_motion_min_inlier_ratio, 0.0, 1.0)),
            max_median_residual=float(max(0.0, botsort_motion_max_median_residual)),
            min_coverage_cells=int(max(1, botsort_motion_min_coverage_cells)),
            max_translation_frac=float(max(0.0, botsort_motion_max_translation_frac)),
            min_scale=float(max(0.01, botsort_motion_min_scale)),
            max_scale=float(max(0.02, botsort_motion_max_scale)),
            max_rotation_deg=float(max(0.0, botsort_motion_max_rotation_deg)),
            residual_mad_multiplier=float(max(0.1, botsort_motion_residual_mad_multiplier)),
            residual_min_threshold=int(max(1, botsort_motion_residual_min_threshold)),
            residual_max_fraction=float(np.clip(botsort_motion_residual_max_fraction, 0.0, 1.0)),
            residual_min_area=int(max(1, botsort_motion_residual_min_area)),
            dynamic_dilate=int(max(1, botsort_motion_dynamic_dilate)),
            feature_detect_interval=int(max(1, botsort_motion_feature_detect_interval)),
            min_points_to_redetect=int(max(4, botsort_motion_min_points_to_redetect)),
            max_track_points=int(max(8, botsort_motion_max_track_points)),
            enable_dynamic_mask=bool(botsort_motion_enable_dynamic_mask),
            residual_update_interval=int(max(1, botsort_motion_residual_update_interval)),
        )
        if self._botsort_motion_cfg.max_scale <= self._botsort_motion_cfg.min_scale:
            self._botsort_motion_cfg.max_scale = self._botsort_motion_cfg.min_scale + 1e-3
        if (
            self._botsort_motion_cfg.min_points_to_redetect
            > self._botsort_motion_cfg.max_track_points
        ):
            self._botsort_motion_cfg.min_points_to_redetect = (
                self._botsort_motion_cfg.max_track_points
            )
        self._botsort_motion_use_fallback_transform = bool(
            botsort_motion_use_fallback_transform
        )
        self._botsort_motion_estimator: Optional[BotSortMotionEstimator] = None

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

        self._drm_lam_cand_dir = drm_lam_cand_dir
        self._use_distractor_bank = (
            self._distractor_bank_maxlen > 0
        )
        self._vel_score_min_speed = vel_score_min_speed
        self._entry_patience = max(1, entry_patience)
        self._cand_collection_frames = max(1, cand_collection_frames)

        self.tiny_roi_start_expand = tiny_roi_start_expand
        self.tiny_yolo_search_expand = tiny_yolo_search_expand
        self.tiny_search_expand_growth_factor = tiny_search_expand_growth_factor
        self.tiny_search_expand_growth_every = tiny_search_expand_growth_every
        self.tiny_search_expand_max = tiny_search_expand_max
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

        self._vel_dir_hard_gate = vel_dir_hard_gate
        self._yolo_filter_class = yolo_filter_class
        self._yolo_class_detect_frames = yolo_class_detect_frames
        self._target_class_id: Optional[int] = None
        self._class_warmup_done: bool = False
        self._class_votes: dict[int, int] = {}
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
        self._pending_candidates: List = []

        self._cand_frames: List = []
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
        self.current_bbox = bbox.copy()
        self.held_box = bbox.copy()
        self._stable_anchor_bbox = bbox.copy()
        self._distractor_focus_bbox = bbox.copy()
        self.in_occlusion = False
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
        if self._homography_mode == "botsort":
            self._botsort_motion_estimator = BotSortMotionEstimator(
                first_frame=proc_frame,
                config=self._botsort_motion_cfg,
            )
        else:
            self._botsort_motion_estimator = None

        self.init_frame = proc_frame.copy()
        self.init_bbox = bbox.copy()

        self._distractor_bank = deque(maxlen=self._distractor_bank_maxlen)
        self._out_of_frame = False
        self._exit_edge = None
        self._search_cx = None
        self._search_cy = None
        self._size_history.clear()
        self.memory.reset()
        self._conf_history.clear()
        self._cam_disp_history.clear()
        self._yolo_cache = []
        self._occ_frames = 0
        self._occ_phase = 0
        self._pending_candidates = []
        self.recovered_early_occlusion = True
        self._last_H = None
        self._last_H_reliable = False
        self._last_yolo = []

        self._entry_streak = 0
        self._cand_frames = []
        self._occ_cam_vels = []

        self._target_class_id = None
        self._class_warmup_done = False
        self._jump_watch_active = False
        self._jump_watch_anchor_bbox = None
        self._jump_watch_prev_bbox = None
        self._jump_watch_frames = 0
        self._jump_watch_stable_count = 0
        self._jump_watch_baseline_norm = 0.0
        self._jump_watch_last_speed_norm = 0.0
        self._jump_watch_last_ratio = 1.0
        self._jump_watch_last_sim = None
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

        self.ekf = BBoxEKF(
            bbox,
            process_noise=self.ekf_process_noise,
            meas_noise=self.ekf_meas_noise,
        )

        from utils.utils import _extract_descriptor

        desc = _extract_descriptor(proc_frame, bbox)

        self._vel_history.clear()
        self._center_history.clear()
        self._cam_vel_history.clear()
        if desc is not None:
            self.memory.try_admit(bbox, desc, bbox)

    def update(
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
            return np.zeros(4, dtype=int), 0.0, True, self._last_yolo

        bbox_out = np.round(np.array(bbox, dtype=float) * scale_inv).astype(int)

        yolo_out = [
            np.round(np.array(b, dtype=float) * scale_inv).astype(int)
            for b in self._last_yolo
        ]

        return bbox_out, float(score), self.in_occlusion, yolo_out

    def _evaluate_hard_jump_candidate(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> Tuple[bool, Optional[np.ndarray], float, float, float, Optional[float]]:
        """
        Spike-only distractor switch trigger.

        Jump magnitude is measured using camera-compensated bbox motion when
        homography is reliable, so fast camera pans are less likely to trigger
        false distractor-mode entry.

        Returns:
            trigger, candidate_descriptor, speed_norm, baseline_norm, speed_ratio, sim
        """
        self._spike_debug_speed_norm = float("nan")
        self._spike_debug_baseline_norm = float("nan")
        self._spike_debug_ratio = float("nan")
        self._spike_debug_trigger = False

        if not self._spike_reject_enabled:
            return False, None, 0.0, 0.0, 1.0, None
        if self.current_bbox is None:
            return False, None, 0.0, 0.0, 1.0, None
        if self.frame_idx < self._spike_reject_min_frames:
            return False, None, 0.0, 0.0, 1.0, None
        if score < self._spike_reject_min_score:
            return False, None, 0.0, 0.0, 1.0, None

        prev_bbox = np.asarray(self.current_bbox, dtype=float)
        cand_bbox = np.asarray(pred_bbox, dtype=float)
        h_curr = self._last_H if bool(self._last_H_reliable) else None
        speed_norm = self._camera_compensated_step_norm(
            prev_bbox=prev_bbox,
            curr_bbox=cand_bbox,
            H=h_curr,
        )
        self._spike_debug_speed_norm = float(speed_norm)

        hist = list(self._conf_history)
        if len(hist) < self._spike_reject_min_history + 1:
            return False, None, speed_norm, 0.0, 1.0, None

        lookback = min(len(hist), self._spike_reject_history_window + 1)
        hist = hist[-lookback:]
        speed_norms = []
        for i in range(1, len(hist)):
            prev_entry = hist[i - 1]
            curr_entry = hist[i]
            b0 = np.asarray(prev_entry[0], dtype=float)
            b1 = np.asarray(curr_entry[0], dtype=float)
            h_step = curr_entry[2] if len(curr_entry) > 2 else None
            h_rel = bool(curr_entry[3]) if len(curr_entry) > 3 else False
            if not h_rel:
                h_step = None
            step_speed_norm = self._camera_compensated_step_norm(
                prev_bbox=b0,
                curr_bbox=b1,
                H=h_step,
            )
            speed_norms.append(step_speed_norm)

        if len(speed_norms) < self._spike_reject_min_history:
            return False, None, speed_norm, 0.0, 1.0, None

        baseline_norm = float(np.median(np.asarray(speed_norms, dtype=float)))
        ratio = speed_norm / (baseline_norm + 1e-6)
        self._spike_debug_baseline_norm = float(baseline_norm)
        self._spike_debug_ratio = float(ratio)

        if speed_norm < self._spike_reject_abs_norm_min:
            return False, None, speed_norm, baseline_norm, ratio, None
        if ratio < self._spike_reject_ratio:
            return False, None, speed_norm, baseline_norm, ratio, None

        cand_desc = _extract_descriptor(frame, pred_bbox)
        if self._spike_reject_use_appearance:
            ref_desc = self.memory.best_descriptor()
            if ref_desc is not None and cand_desc is not None:
                sim = float(_cos_sim(ref_desc, cand_desc))
                if sim > self._spike_reject_max_sim:
                    return False, cand_desc, speed_norm, baseline_norm, ratio, sim
                self._spike_debug_trigger = True
                return True, cand_desc, speed_norm, baseline_norm, ratio, sim

        self._spike_debug_trigger = True
        return True, cand_desc, speed_norm, baseline_norm, ratio, None

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
        Choose a robust pre-spike anchor box.

        Priority:
        1) Explicit stable anchor tracked during normal stable motion.
        2) Box right before the largest recent normalized step jump in history.
        3) Current bbox fallback.
        """
        anchor = None
        if self._stable_anchor_bbox is not None:
            anchor = np.array(self._stable_anchor_bbox, dtype=int).copy()

        hist = list(self._conf_history)
        if not hist:
            if anchor is not None:
                return anchor
            if self.current_bbox is not None:
                return np.array(self.current_bbox, dtype=int).copy()
            return None

        boxes = [
            np.asarray(entry[0], dtype=float)
            for entry in hist[-self._spike_anchor_history_window:]
        ]
        if self.current_bbox is not None:
            boxes.append(np.asarray(self.current_bbox, dtype=float))

        if len(boxes) >= 2:
            step_norms = [
                self._normalized_center_distance(boxes[i], boxes[i + 1])
                for i in range(len(boxes) - 1)
            ]
            if step_norms:
                j = int(np.argmax(step_norms))
                max_step = float(step_norms[j])
                if max_step >= self._spike_anchor_trigger_norm:
                    hist_anchor = np.array(np.round(boxes[j]), dtype=int)
                    anchor = hist_anchor

        if anchor is None and self.current_bbox is not None:
            anchor = np.array(self.current_bbox, dtype=int).copy()
        return anchor

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
        """
        ROI for distractor mode centered on the last reliable/real target focus.

        The center always follows the currently selected real target bbox
        (`_distractor_focus_bbox`), while ROI size can optionally inherit the
        tracker search mapping dimensions captured at distractor-mode entry.
        """
        h_fr, w_fr = frame.shape[:2]
        focus = (
            focus_override
            if focus_override is not None
            else (
            self._distractor_focus_bbox
            if self._distractor_focus_bbox is not None
            else self.current_bbox
            )
        )
        if focus is None:
            return 0, 0, w_fr, h_fr

        fx, fy, fw, fh = map(float, focus)
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0

        # Base ROI uses target scale first (2x box side), then optionally expands
        # with tracker mapping size captured when distractor mode started.
        base_w = max(1.0, fw * 2.0)
        base_h = max(1.0, fh * 2.0)

        if self._distractor_mode_use_tracker_mapping:
            roi_size = self._distractor_mode_roi_size
            if roi_size is None:
                mapping = getattr(self.tracker.tracking_state, "mapping", None)
                if mapping is not None and len(mapping) >= 4:
                    _mx, _my, mw, mh = map(float, mapping[:4])
                    if mw > 1.0 and mh > 1.0:
                        roi_size = (int(round(mw)), int(round(mh)))
            if roi_size is not None:
                base_w = max(base_w, float(roi_size[0]))
                base_h = max(base_h, float(roi_size[1]))

        unc_pad = max(0.0, float(uncertainty_pad_px))
        base_w += 2.0 * unc_pad
        base_h += 2.0 * unc_pad

        ew = max(
            self._distractor_mode_roi_min_side,
            int(round(base_w * self._distractor_mode_roi_expand)),
        )
        eh = max(
            self._distractor_mode_roi_min_side,
            int(round(base_h * self._distractor_mode_roi_expand)),
        )
        rx = int(round(cx - ew / 2.0))
        ry = int(round(cy - eh / 2.0))
        rw = ew
        rh = eh
        rx = int(np.clip(rx, 0, w_fr - 1))
        ry = int(np.clip(ry, 0, h_fr - 1))
        rw = int(np.clip(rw, 1, w_fr - rx))
        rh = int(np.clip(rh, 1, h_fr - ry))
        return rx, ry, rw, rh

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
            imgsz=320,
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
        self._distractor_mode_active = True
        self._distractor_focus_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._stable_anchor_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._distractor_mode_roi_size = None
        self._distractor_mode_stable_count = 0
        self._distractor_mode_ambiguous_count = 0
        self._distractor_mode_overlap_lock_active = False
        self._distractor_mode_overlap_clear_count = 0
        self._distractor_mode_overlap_lock_frames = 0
        self._distractor_anchor_pred_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._distractor_anchor_uncertainty = 0.0
        self._distractor_anchor_ekf = None
        if self._distractor_mode_anchor_ekf_enabled:
            self._distractor_anchor_ekf = BBoxEKF(
                anchor_bbox,
                process_noise=self.ekf_process_noise,
                meas_noise=self.ekf_meas_noise,
            )
            src_ekf = self.ekf
            if src_ekf is not None:
                self._distractor_anchor_ekf.reseed(
                    bbox=anchor_bbox,
                    velocity=src_ekf.get_velocity(),
                )
                try:
                    self._distractor_anchor_ekf.P = np.array(
                        src_ekf.P, dtype=float
                    ).copy()
                except Exception:
                    pass
            self._distractor_anchor_uncertainty = float(
                self._distractor_anchor_ekf.get_uncertainty()
            )
        if self._distractor_mode_use_tracker_mapping:
            mapping = getattr(self.tracker.tracking_state, "mapping", None)
            if mapping is not None and len(mapping) >= 4:
                _mx, _my, mw, mh = map(float, mapping[:4])
                if mw > 1.0 and mh > 1.0:
                    self._distractor_mode_roi_size = (
                        int(round(mw)),
                        int(round(mh)),
                    )
        self._jump_reject_distractor_timer = max(
            self._jump_reject_distractor_timer,
            self._jump_reject_distractor_mode_frames,
        )

    def _exit_distractor_mode(
        self,
        reason: str,
        resolved: bool = False,
    ) -> None:
        self._distractor_mode_active = False
        self._distractor_mode_visual_reals = []
        self._distractor_mode_visual_distractors = []
        self._distractor_mode_roi = None
        self._distractor_mode_roi_size = None
        self._distractor_mode_stable_count = 0
        self._distractor_mode_ambiguous_count = 0
        self._distractor_mode_overlap_lock_active = False
        self._distractor_mode_overlap_clear_count = 0
        self._distractor_mode_overlap_lock_frames = 0
        self._distractor_anchor_ekf = None
        self._distractor_anchor_pred_bbox = None
        self._distractor_anchor_uncertainty = 0.0
        self._jump_reject_distractor_timer = 0
        if resolved:
            self._distractor_mode_reentry_cooldown = (
                self._distractor_mode_reentry_cooldown_frames
            )
            self._distractor_mode_memory_freeze_left = (
                self._distractor_mode_post_exit_memory_freeze_frames
            )
            self._distractor_mode_template_freeze_left = (
                self._distractor_mode_post_exit_template_freeze_frames
            )
        self.visual_mode = "tracking"
        self.visual_reason = reason
        self.visual_details = ""

    def _distractor_mode_update(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> Tuple[np.ndarray, float]:
        """
        During distractor mode:
        1) run YOLO in ROI around last reliable/focus region,
        2) evaluate ROI hypotheses with OSNet against target descriptor history,
        3) pick most similar as real target and mark all other hypotheses as distractors.
        Exit only when no similar candidate exists in ROI.
        """
        focus_override: Optional[np.ndarray] = None
        anchor_mean: Optional[np.ndarray] = None
        anchor_cov: Optional[np.ndarray] = None
        uncertainty_pad_px = 0.0

        if self._distractor_mode_anchor_ekf_enabled and self._distractor_anchor_ekf is not None:
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

        roi = self._get_distractor_mode_roi(
            frame,
            focus_override=focus_override,
            uncertainty_pad_px=uncertainty_pad_px,
        )
        self._distractor_mode_roi = np.array(roi, dtype=int)

        detections = self._yolo_detect_in_roi(frame, roi)
        self._last_yolo = detections

        refs = self._target_descriptor_history()
        if not refs and self._distractor_focus_bbox is not None:
            focus_desc = _extract_descriptor(frame, self._distractor_focus_bbox)
            if focus_desc is not None:
                refs = [np.asarray(focus_desc, dtype=float)]

        if not detections or not refs:
            self._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score

        desired_hyps = int(self._distractor_mode_yolo_topk)
        if desired_hyps <= 0:
            desired_hyps = len(detections)
        if self._osnet_max_candidate_batch > 0:
            desired_hyps = min(desired_hyps, int(self._osnet_max_candidate_batch))
        n_eval = min(len(detections), max(1, desired_hyps))
        dets_for_desc = detections[:n_eval]
        det_descs = _extract_descriptor(frame, dets_for_desc) if dets_for_desc else []

        focus_bbox = (
            focus_override
            if focus_override is not None
            else (
                self._distractor_focus_bbox
                if self._distractor_focus_bbox is not None
                else self.current_bbox
            )
        )
        if focus_bbox is None:
            focus_bbox = pred_bbox
        focus_bbox_arr = np.asarray(focus_bbox, dtype=float)
        focus_cx = float(focus_bbox_arr[0] + focus_bbox_arr[2] / 2.0)
        focus_cy = float(focus_bbox_arr[1] + focus_bbox_arr[3] / 2.0)
        dist_sigma = (
            self._distractor_drm_dist_sigma_factor
            * max(1.0, float(max(focus_bbox_arr[2], focus_bbox_arr[3])))
        )

        distractor_bank = self._get_active_distractor_bank()
        if distractor_bank:
            distractor_bank = distractor_bank[-self._distractor_drm_bank_topk:]

        cands = []
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

        if not cands:
            self._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score

        max_app_sim = max(float(c[2]) for c in cands)
        if max_app_sim < self._distractor_mode_min_similarity:
            self._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score

        compare_mode = self._distractor_compare_mode
        score_idx = 3 if compare_mode == "drm" else 2
        cands.sort(key=lambda t: t[score_idx], reverse=True)
        best_bbox, best_desc, best_sim, best_score, best_iou, best_dist, best_neg, best_maha = cands[0]
        if best_sim < self._distractor_mode_selected_min_similarity:
            self._exit_distractor_mode(
                "Best ROI candidate similarity below distractor-mode gate"
            )
            return pred_bbox, score

        distractor_entries = cands[1:]
        max_overlap_iou = 0.0
        if distractor_entries:
            max_overlap_iou = max(
                float(_iou(best_bbox, bb)) for bb, *_ in distractor_entries
            )

        if self._distractor_mode_overlap_motion_lock_enabled:
            if (
                not self._distractor_mode_overlap_lock_active
                and distractor_entries
                and max_overlap_iou >= self._distractor_mode_overlap_iou_enter
            ):
                self._distractor_mode_overlap_lock_active = True
                self._distractor_mode_overlap_clear_count = 0
                self._distractor_mode_overlap_lock_frames = 0

            if self._distractor_mode_overlap_lock_active:
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
                if not (cleared or timed_out):
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

                self._distractor_mode_overlap_lock_active = False
                self._distractor_mode_overlap_clear_count = 0
                self._distractor_mode_overlap_lock_frames = 0

        top_margin = float("inf")
        if len(cands) > 1:
            top_margin = float(cands[0][score_idx] - cands[1][score_idx])
        ambiguous = (
            len(cands) > 1
            and top_margin < self._distractor_mode_switch_margin
        )
        if ambiguous:
            self._distractor_mode_ambiguous_count += 1
        else:
            self._distractor_mode_ambiguous_count = 0

        if (
            ambiguous
            and self._distractor_mode_ambiguous_count <= self._distractor_mode_ambiguity_hold_frames
        ):
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

        same_iou = float(_iou(focus_bbox_arr, best_bbox))
        if same_iou >= self._distractor_mode_exit_same_iou:
            self._distractor_mode_stable_count += 1
        else:
            self._distractor_mode_stable_count = 1

        self._distractor_mode_visual_reals = [best_bbox.copy()]
        self._distractor_mode_visual_distractors = [
            np.array(bb, dtype=int).copy() for bb, *_ in distractor_entries
        ]
        if self._distractor_bank_maxlen > 0:
            self._extend_distractor_descriptors([ddesc for _bb, ddesc, *_ in distractor_entries])

        self._distractor_focus_bbox = best_bbox.copy()
        self._stable_anchor_bbox = best_bbox.copy()
        self.tracker.tracking_state.bbox = best_bbox.copy()
        if self._distractor_mode_anchor_ekf_enabled and self._distractor_anchor_ekf is not None:
            try:
                self._distractor_anchor_ekf.update(best_bbox)
                self._distractor_anchor_pred_bbox = best_bbox.copy()
                self._distractor_anchor_uncertainty = float(
                    self._distractor_anchor_ekf.get_uncertainty()
                )
            except Exception:
                pass
        self.visual_mode = "distractor"
        if compare_mode == "ram":
            self.visual_reason = "Distractor mode: RAM appearance ROI ranking"
        else:
            self.visual_reason = "Distractor mode: DRM-style ROI re-ranking"
        maha_txt = "n/a" if not np.isfinite(best_maha) else f"{best_maha:.2f}"
        self.visual_details = (
            f"cmp={compare_mode}  app={best_sim:.2f}  score={best_score:.2f}  iou={best_iou:.2f}  "
            f"dist_pen={best_dist:.2f}  neg={best_neg:.2f}  maha={maha_txt}  "
            f"margin={top_margin:.3f}  same_iou={same_iou:.2f}  "
            f"stable={self._distractor_mode_stable_count}/{self._distractor_mode_exit_stable_frames}  "
            f"cands={len(cands)}  distractors={len(distractor_entries)}  roi={roi[2]}x{roi[3]}"
        )
        self._jump_reject_distractor_timer = max(
            self._jump_reject_distractor_timer,
            self._jump_reject_distractor_mode_frames,
        )

        if (
            self._distractor_mode_exit_reinit_enabled
            and self._distractor_mode_stable_count >= self._distractor_mode_exit_stable_frames
        ):
            reinit_bbox = self._clamp_bbox_to_frame(best_bbox.copy(), frame)
            try:
                reinit_bbox = self._reinit_dynamic_template_only(
                    frame=frame,
                    bbox=reinit_bbox.copy(),
                    score_hint=float(np.clip(best_sim, 0.0, 1.0)),
                )
                self.current_bbox = reinit_bbox.copy()
                self.held_box = reinit_bbox.copy()
                self._stable_anchor_bbox = reinit_bbox.copy()
                self._distractor_focus_bbox = reinit_bbox.copy()
                self._exit_distractor_mode(
                    f"Distractor resolved; dynamic template reinit after {self._distractor_mode_stable_count} stable frames",
                    resolved=True,
                )
                return reinit_bbox.copy(), max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
            except Exception as exc:
                if self.debug:
                    print(f"[distractor-exit:reinit-fail] frame={self.frame_idx} err={exc}")

        if (
            self._distractor_mode_update_memory
            and best_desc is not None
            and self.current_bbox is not None
        ):
            self.memory.try_admit(best_bbox, best_desc, self.current_bbox)

        adjusted_score = max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
        adjusted_score = self._apply_distractor_mode_penalty(frame, best_bbox, adjusted_score)
        return best_bbox.copy(), float(adjusted_score)

    def _clear_jump_watch_state(
        self,
    ) -> None:
        self._jump_watch_active = False
        self._jump_watch_anchor_bbox = None
        self._jump_watch_prev_bbox = None
        self._jump_watch_frames = 0
        self._jump_watch_stable_count = 0
        self._jump_watch_baseline_norm = 0.0
        self._jump_watch_last_speed_norm = 0.0
        self._jump_watch_last_ratio = 1.0
        self._jump_watch_last_sim = None

    def _start_jump_watch(
        self,
        prev_bbox: np.ndarray,
        first_switched_bbox: np.ndarray,
        spike_speed_norm: float,
        spike_ratio: float,
        baseline_norm: float,
        sim: Optional[float],
    ) -> None:
        self._jump_watch_active = True
        self._jump_watch_anchor_bbox = np.array(prev_bbox, dtype=int).copy()
        self._jump_watch_prev_bbox = np.array(first_switched_bbox, dtype=int).copy()
        self._jump_watch_frames = 0
        self._jump_watch_stable_count = 0
        self._jump_watch_baseline_norm = float(baseline_norm)
        self._jump_watch_last_speed_norm = float(spike_speed_norm)
        self._jump_watch_last_ratio = float(spike_ratio)
        self._jump_watch_last_sim = sim
        self._jump_watch_last_step_norm = float(spike_speed_norm)
        if self.debug:
            sim_txt = "n/a" if sim is None else f"{sim:.3f}"
            print(
                f"[spike-watch:start] frame={self.frame_idx}  "
                f"speed_norm={spike_speed_norm:.3f}  ratio={spike_ratio:.2f}  "
                f"baseline={baseline_norm:.3f}  app_sim={sim_txt}"
            )

    def _apply_distractor_mode_penalty(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> float:
        if not self._distractor_mode_active and self._jump_reject_distractor_timer <= 0:
            return float(score)

        self.visual_mode = "distractor"
        self.visual_reason = "Distractor suppression active"
        self.visual_details = ""

        adjusted = float(score)
        top_sim: Optional[float] = None
        app_penalty = 0.0
        dist_penalty = 0.0
        dist_norm: Optional[float] = None

        bank = self._get_active_distractor_bank()
        if (
            self._jump_reject_distractor_penalty_enabled
            and bank
            and self._jump_reject_distractor_penalty_weight > 0.0
        ):
            desc = _extract_descriptor(frame, pred_bbox)
            if desc is not None:
                recent = bank[-self._jump_reject_distractor_penalty_bank_topk:]
                sims = [float(_cos_sim(desc, d)) for d in recent if d is not None]
                if sims:
                    top_sim = max(sims)
                    sim_floor = self._jump_reject_distractor_penalty_sim_floor
                    frac = max(0.0, (top_sim - sim_floor) / (1.0 - sim_floor + 1e-8))
                    app_penalty = self._jump_reject_distractor_penalty_weight * frac
                    adjusted = max(0.0, adjusted - app_penalty)

        if (
            self._distractor_focus_dist_penalty_enabled
            and self._distractor_focus_bbox is not None
            and self._distractor_focus_dist_penalty_weight > 0.0
        ):
            dist_norm = self._normalized_center_distance(
                self._distractor_focus_bbox, pred_bbox
            )
            soft_r = self._distractor_focus_dist_soft_radius
            hard_r = self._distractor_focus_dist_hard_radius
            if dist_norm > soft_r:
                frac = float(np.clip((dist_norm - soft_r) / (hard_r - soft_r + 1e-8), 0.0, 1.0))
                dist_penalty = self._distractor_focus_dist_penalty_weight * frac
                adjusted = max(0.0, adjusted - dist_penalty)

        sim_txt = "n/a" if top_sim is None else f"{top_sim:.2f}"
        dist_txt = "n/a" if dist_norm is None else f"{dist_norm:.2f}"
        self.visual_details = (
            f"sim={sim_txt}  app_pen={app_penalty:.2f}  "
            f"dist={dist_txt}  dist_pen={dist_penalty:.2f}  "
            f"frames_left={self._jump_reject_distractor_timer}"
        )

        if self._distractor_mode_active:
            self._jump_reject_distractor_timer = max(
                1, self._jump_reject_distractor_timer
            )
        else:
            self._jump_reject_distractor_timer = max(0, self._jump_reject_distractor_timer - 1)
        if not self._distractor_mode_active and self._jump_reject_distractor_timer == 0:
            self._distractor_focus_bbox = (
                self.current_bbox.copy() if self.current_bbox is not None else None
            )
        return adjusted

    def _apply_hard_jump_rejection(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        effective_threshold: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Multi-frame jump handling for distractor swaps in experimental mode.
        """
        if self._distractor_mode_reentry_cooldown > 0:
            self._distractor_mode_reentry_cooldown = max(
                0, self._distractor_mode_reentry_cooldown - 1
            )
            if self._jump_watch_active:
                self._clear_jump_watch_state()
            score = self._apply_distractor_mode_penalty(frame, pred_bbox, score)
            return pred_bbox, float(score)

        heavy_cam_motion, heavy_cam_disp = self._is_heavy_camera_motion(
            frame, bbox_hint=pred_bbox
        )
        if (
            self._block_distractor_mode_on_camera_motion
            and heavy_cam_motion
        ):
            if self._jump_watch_active:
                self._clear_jump_watch_state()
            if self.debug:
                print(
                    f"[distractor guard] frame={self.frame_idx} blocked by camera motion "
                    f"disp={heavy_cam_disp:.2f} thr={self._camera_motion_heavy_disp_threshold:.2f}"
                )
            score = self._apply_distractor_mode_penalty(frame, pred_bbox, score)
            return pred_bbox, float(score)

        spike_trigger, cand_desc, speed_norm, baseline_norm, speed_ratio, sim = (
            self._evaluate_hard_jump_candidate(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
            )
        )
        if (
            spike_trigger
            and not self._jump_watch_active
        ):
            anchor_bbox = self._select_pre_spike_anchor_bbox()
            if anchor_bbox is None:
                anchor_bbox = (
                    np.array(self.current_bbox, dtype=int).copy()
                    if self.current_bbox is not None
                    else None
                )
            if anchor_bbox is None:
                score = self._apply_distractor_mode_penalty(frame, pred_bbox, score)
                return pred_bbox, float(score)
            self._start_jump_watch(
                prev_bbox=anchor_bbox,
                first_switched_bbox=pred_bbox,
                spike_speed_norm=speed_norm,
                spike_ratio=speed_ratio,
                baseline_norm=baseline_norm,
                sim=sim,
            )

        if self._jump_watch_active:
            if spike_trigger:
                self._jump_watch_last_speed_norm = float(speed_norm)
                self._jump_watch_last_ratio = float(speed_ratio)
                self._jump_watch_last_sim = sim

            self._jump_watch_frames += 1

            prev_watch = self._jump_watch_prev_bbox
            if prev_watch is not None:
                step_speed_norm = self._camera_compensated_step_norm(
                    prev_bbox=np.asarray(prev_watch, dtype=float),
                    curr_bbox=np.asarray(pred_bbox, dtype=float),
                    H=self._last_H,
                )
            else:
                step_speed_norm = 0.0

            self._jump_watch_prev_bbox = pred_bbox.copy()
            settle_thresh = max(
                self._spike_reject_settle_abs_norm_max,
                self._jump_watch_baseline_norm * self._spike_reject_settle_ratio,
                self._jump_watch_last_speed_norm * self._spike_reject_settle_from_spike_frac,
            )
            self._jump_watch_last_step_norm = float(step_speed_norm)
            if step_speed_norm <= settle_thresh:
                self._jump_watch_stable_count += 1
            else:
                self._jump_watch_stable_count = 0

            anchor = self._jump_watch_anchor_bbox
            if anchor is not None and _iou(pred_bbox, anchor) >= 0.50:
                if self.debug:
                    print(
                        f"[spike-watch:cancel] frame={self.frame_idx}  "
                        "candidate returned near anchor"
                    )
                self._clear_jump_watch_state()
                score = self._apply_distractor_mode_penalty(frame, pred_bbox, score)
                return pred_bbox, score

            stable_ready = (
                self._jump_watch_stable_count >= self._spike_reject_settle_frames
            )
            timed_out = self._jump_watch_frames >= self._spike_reject_watch_max_frames

            should_commit = False
            if stable_ready and anchor is not None:
                should_commit = True
            elif (
                timed_out
                and anchor is not None
                and self._jump_watch_stable_count > 0
            ):
                should_commit = True
                if self.debug:
                    print(
                        f"[spike-watch:fallback-commit] frame={self.frame_idx}  "
                        f"watch_frames={self._jump_watch_frames}  "
                        f"stable_count={self._jump_watch_stable_count}"
                    )

            if should_commit and anchor is not None:
                switched_desc = cand_desc
                if switched_desc is None:
                    switched_desc = _extract_descriptor(frame, pred_bbox)
                if switched_desc is not None and self._distractor_bank_maxlen > 0:
                    self._add_distractor_descriptor(switched_desc.copy())

                self.visual_mode = "distractor"
                self.visual_reason = "Distractor switch detected; snapped to anchor"
                self.visual_details = (
                    f"spike_norm={self._jump_watch_last_speed_norm:.2f}  "
                    f"ratio={self._jump_watch_last_ratio:.2f}  "
                    f"stable_frames={self._jump_watch_stable_count}  "
                    f"settle<={settle_thresh:.2f}"
                )

                if self.debug:
                    print(
                        f"[spike-watch:commit] frame={self.frame_idx}  "
                        f"watch_frames={self._jump_watch_frames}  "
                        f"stable_count={self._jump_watch_stable_count}  "
                        f"step_norm={step_speed_norm:.3f}  settle_thr={settle_thresh:.3f}"
                    )

                snap_bbox = np.array(anchor, dtype=int).copy()
                self.tracker.tracking_state.bbox = snap_bbox.copy()
                self._enter_distractor_mode(snap_bbox)
                self._distractor_mode_visual_reals = [snap_bbox.copy()]
                self._distractor_mode_visual_distractors = (
                    [np.array(pred_bbox, dtype=int).copy()]
                )
                self._clear_jump_watch_state()

                if self._jump_reject_force_occlusion:
                    score = min(float(score), float(effective_threshold) - 1e-6)
                return snap_bbox, float(score)

            if timed_out:
                if self.debug:
                    print(
                        f"[spike-watch:timeout] frame={self.frame_idx}  "
                        f"watch_frames={self._jump_watch_frames}"
                    )
                self._clear_jump_watch_state()

        score = self._apply_distractor_mode_penalty(frame, pred_bbox, score)
        return pred_bbox, float(score)

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

        if self._jump_reject_distractor_timer > 0:
            self.visual_mode = "distractor"
            self.visual_reason = "Distractor suppression active"
            self.visual_details = ""
        else:
            self.visual_mode = "tracking"
            self.visual_reason = "Normal tracking"
            self.visual_details = ""

        if self._gmc_prior_enabled:
            self._apply_gmc_search_prior(frame)
        pred_bbox, score, _ = self.tracker.update(frame)
        pred_bbox = np.array(pred_bbox, dtype=int)

        if (
            self._yolo_filter_class
            and not self._class_warmup_done
            and self.frame_idx <= self._yolo_class_detect_frames * 3
            and self.frame_idx % max(1, self._yolo_class_detect_frames) == 0
        ):
            self._try_detect_target_class(frame)
            if self.frame_idx >= self._yolo_class_detect_frames * 2:
                self._maybe_commit_target_class()

        if self.frame_idx > 0:
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
        else:
            effective_threshold = 0.0

        if self._distractor_mode_active:
            pred_bbox, score = self._distractor_mode_update(
                frame=frame,
                pred_bbox=pred_bbox,
                score=float(score),
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

        if (
            self._entry_streak >= self._entry_patience
            and self.frame_idx >= 0
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

        cam_disp = self._h_translation_magnitude(self._last_H, frame)
        self._cam_disp_history.append(cam_disp)

        h_fr, w_fr = frame.shape[:2]
        if self._last_H is not None:
            cx, cy = w_fr / 2.0, h_fr / 2.0
            denom = (
                self._last_H[2, 0] * cx
                + self._last_H[2, 1] * cy
                + self._last_H[2, 2]
                + 1e-8
            )
            ncx = (
                      self._last_H[0, 0] * cx + self._last_H[0, 1] * cy + self._last_H[0, 2]
                  ) / denom
            ncy = (
                      self._last_H[1, 0] * cx + self._last_H[1, 1] * cy + self._last_H[1, 2]
                  ) / denom
            self._cam_vel_history.append(np.array([ncx - cx, ncy - cy]))
        else:
            self._cam_vel_history.append(np.zeros(2))

        cx = float(pred_bbox[0] + pred_bbox[2] / 2.0)
        cy = float(pred_bbox[1] + pred_bbox[3] / 2.0)
        self._center_history.append(np.array([cx, cy]))

        self.velocity = self._compute_velocity_from_history(pred_bbox)
        self._vel_history.append(self.velocity.copy())

        if (
            score >= effective_threshold
            and self._distractor_mode_memory_freeze_left <= 0
        ):
            desc = _extract_descriptor(frame, pred_bbox)
            if desc is not None:
                self.memory.try_admit(pred_bbox, desc, self.current_bbox)

        if (
            self._jump_reject_distractor_timer <= 0
            and not self._jump_watch_active
            and self.current_bbox is not None
        ):
            step_norm_anchor = self._camera_compensated_step_norm(
                prev_bbox=np.asarray(self.current_bbox, dtype=float),
                curr_bbox=np.asarray(pred_bbox, dtype=float),
                H=self._last_H,
            )
            if step_norm_anchor <= self._spike_anchor_update_norm_max:
                self._stable_anchor_bbox = pred_bbox.copy()
                self._distractor_focus_bbox = pred_bbox.copy()

        self.current_bbox = pred_bbox.copy()
        self.held_box = pred_bbox.copy()
        self._size_history.append((int(pred_bbox[2]), int(pred_bbox[3])))
        self._conf_history.append(
            (
                pred_bbox.copy(),
                self.velocity.copy(),
                self._last_H,
                self._last_H_reliable,
            )
        )

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

        ref_bbox = history[-n_back][0]
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
            last_clean_bbox = clean[-1][0].astype(float)
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
        """
        Dispatcher for all three occlusion recovery phases. At each call it
                    pulls the latest EKF prediction to update the search centre and
                    held_box, increments the occlusion frame counter, and handles all
                    out-of-frame edge pinning and re-entry logic (e.g. if the target
                    exited right, the search centre is pinned to the right edge until
                    the EKF velocity turns inward). Then routes to:
                        phase 0       → _occ_phase_siam (SiamABC fast attempt)
                        phase 1…N     → _occ_phase_collect (YOLO candidate collection)
                        phase N+1+    → _occ_phase_final_drm (velocity-scored DRM + verify)

        Called every frame while in_occlusion is True. It owns the
                    out-of-frame state machine and the phase routing. Centralising both
                    here means neither individual phase method has to worry about edge
                    pinning or out-of-frame transitions — they just run their own logic
                    and return.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            held_box: np.ndarray [x, y, w, h] last known / EKF-predicted position
            score: float; 0.0 when no reacquisition happened
    """
        h_fr, w_fr = frame.shape[:2]
        ekf = self.ekf
        assert ekf is not None

        ekf_raw = ekf.get_bbox()
        self._search_cx = float(ekf_raw[0] + ekf_raw[2] / 2.0)
        self._search_cy = float(ekf_raw[1] + ekf_raw[3] / 2.0)
        self.held_box = self._clamp_bbox_to_frame(ekf_raw, frame)
        self.velocity = ekf.get_velocity()
        self._occ_frames += 1

        if self._out_of_frame and self._exit_edge is not None:
            if self._exit_edge == "right":
                self._search_cx = float(w_fr - 1)
            elif self._exit_edge == "left":
                self._search_cx = 0.0
            elif self._exit_edge == "bottom":
                self._search_cy = float(h_fr - 1)
            elif self._exit_edge == "top":
                self._search_cy = 0.0

        if self._out_of_frame:
            ekf_inside = 0 <= self._search_cx < w_fr and 0 <= self._search_cy < h_fr
            vel_inward = False
            if ekf_inside and self._exit_edge is not None:
                vel_inward = {
                    "right": float(self.velocity[0]) < 0,
                    "left": float(self.velocity[0]) > 0,
                    "bottom": float(self.velocity[1]) < 0,
                    "top": float(self.velocity[1]) > 0,
                }.get(self._exit_edge, True)
            if ekf_inside and vel_inward:
                self._out_of_frame = False
                self._exit_edge = None

        else:
            obj_w, obj_h = self._get_median_size()
            oof_margin = float(max(obj_w, obj_h)) * 0.5
            if (
                self._search_cx < -oof_margin
                or self._search_cx >= w_fr + oof_margin
                or self._search_cy < -oof_margin
                or self._search_cy >= h_fr + oof_margin
            ):
                if self._search_cx >= w_fr + oof_margin:
                    self._exit_edge = "right"
                elif self._search_cx < -oof_margin:
                    self._exit_edge = "left"
                elif self._search_cy >= h_fr + oof_margin:
                    self._exit_edge = "bottom"
                else:
                    self._exit_edge = "top"
                self._out_of_frame = True

        if self._occ_phase == 0:
            return self._occ_phase_siam(frame)
        elif 1 <= self._occ_phase <= self._cand_collection_frames:
            return self._occ_phase_collect(frame)
        else:
            return self._occ_phase_final_drm(frame)

    def _occ_phase_siam(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Phase 0 of occlusion recovery — the fast path. Computes the search
                    ROI, plants a seed bbox there at the object's median size, runs
                    SiamABC, and if the score clears reacq_threshold, runs a DRM check
                    augmented with a single-frame direction score. If both pass, calls
                    _commit_reacquisition and exits occlusion immediately. If either
                    fails, resets the SiamABC state to held_box and advances to phase 1
                    (candidate collection).

        This is the cheapest recovery attempt — one tracker forward pass,
                    one DRM call. When the target reappears quickly (e.g. brief occlusion
                    by a thin object), this phase catches it without ever invoking YOLO,
                    keeping latency low. Only if this fails do we pay the cost of YOLO
                    candidate collection.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            held_box: np.ndarray [x, y, w, h] if recovery failed this frame
            score: float SiamABC confidence
            OR, on success, returns the output of _commit_reacquisition.
    """
        held_box = self.held_box
        assert held_box is not None

        rx, ry, rw, rh = self._get_yolo_search_roi(frame=frame)

        obj_w, obj_h = self._get_median_size()

        roi_cx = rx + rw / 2.0
        roi_cy = ry + rh / 2.0
        seed_bbox = np.array(
            [
                int(roi_cx - obj_w / 2.0),
                int(roi_cy - obj_h / 2.0),
                obj_w,
                obj_h,
            ],
            dtype=int,
        )
        seed_bbox = self._clamp_bbox_to_frame(seed_bbox, frame)

        self.tracker.tracking_state.bbox = seed_bbox
        pred_bbox, score, _ = self.tracker.update(frame)
        pred_bbox = np.array(pred_bbox, dtype=int)

        if score >= self.occ_siam_reacq_threshold:

            if (
                not self._is_near_exit_edge(pred_bbox, frame, fraction=0.50)
                and self.recovered_early_occlusion
            ):
                if self.debug:
                    print(
                        f"[occ frame {self._occ_frames}] phase=siam  "
                        f"score={score:.3f}  REJECTED — too far from exit edge "
                        f"({self._exit_edge})"
                    )
                self.tracker.tracking_state.bbox = held_box.copy()
                self._cand_frames = []
                self._occ_cam_vels = []
                self._occ_phase = 1
                return self.held_box, score
            pred_desc = _extract_descriptor(frame, pred_bbox)

            cand_vel_phase0 = None
            if self._conf_history:
                last_bbox = self._conf_history[-1][0]
                dx = (pred_bbox[0] + pred_bbox[2] / 2.0) - (
                    last_bbox[0] + last_bbox[2] / 2.0
                )
                dy = (pred_bbox[1] + pred_bbox[3] / 2.0) - (
                    last_bbox[1] + last_bbox[3] / 2.0
                )
                cam = self._cam_vel_from_H(frame)
                cand_vel_phase0 = np.array([dx - cam[0], dy - cam[1]])

            occ_match_results, occ_match_mode = self._occlusion_memory_match(
                frame=frame,
                candidates=[pred_bbox],
                ref_bbox=held_box,
                velocity=self.velocity,
                margin=self.occ_siam_margin,
                search_cx=self._search_cx,
                search_cy=self._search_cy,
                dist_sigma=self._effective_dist_sigma(frame),
                lam_iou=self._drm_kwargs["lam_iou"],
                lam_app=self._drm_kwargs["lam_app"],
                lam_mot=self._drm_kwargs["lam_mot"],
                lam_time=self._drm_kwargs["lam_time"],
                alpha=self._drm_kwargs["alpha"],
                gamma=self._drm_kwargs["gamma"],
                top_k=self._drm_kwargs["top_k"],
                skip_threshold=self._drm_kwargs["skip_threshold"],
                lam_dist=self._drm_kwargs["lam_dist"],
                lam_cand_dir=self._drm_kwargs["lam_cand_dir"],
            )

            drm_score = occ_match_results[0][1] if occ_match_results else -1.0

            lam_dir = self._drm_lam_cand_dir
            if lam_dir > 0 and cand_vel_phase0 is not None:
                dir_score = self._compute_velocity_score(cand_vel_phase0, self.velocity)
                drm_score += lam_dir * (2.0 * dir_score - 1.0)

            drm_ok = drm_score >= self.app_match_threshold
            if self.debug:
                print(
                    f"[occ frame {self._occ_frames}] phase=siam  "
                    f"score={score:.3f}  occ_match={occ_match_mode}  "
                    f"drm={drm_score:.3f}  pass={drm_ok}"
                )

            if drm_ok:
                self._set_recovered_early_occlusion_flag()
                return self._commit_reacquisition(frame, pred_bbox, pred_desc, score)

            held_box = self.held_box
            assert held_box is not None
            self.tracker.tracking_state.bbox = held_box.copy()

        self._cand_frames = []
        self._occ_cam_vels = []
        self._occ_phase = 1
        return self.held_box, score

    def _occ_phase_collect(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Runs YOLO on the search ROI for this frame, extracts appearance
                    descriptors for detections (optionally capped by
                    osnet_max_candidate_batch), and appends (bbox, desc) pairs to
                    _cand_frames. Also records the camera velocity vector for this frame
                    in _occ_cam_vels so the final phase can camera-compensate each
                    per-step displacement. Updates the distractor bank with any detection
                    that heavily overlaps held_box. Advances _occ_phase by 1 each call;
                    when _occ_phase exceeds cand_collection_frames, the dispatcher will
                    automatically route to the final phase next frame.

        Runs for `cand_collection_frames` consecutive frames (default 3) after
                    phase 0 fails. Gathering candidates across multiple frames rather than
                    a single frame is what makes velocity scoring possible — without a
                    multi-frame track we can't measure how fast and in what direction each
                    candidate is moving.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            held_box: np.ndarray [x, y, w, h] — no position update this frame
            0.0: score is always 0.0 during collection; no reacquisition here
    """
        cam_vel = self._cam_vel_from_H(frame)
        self._occ_cam_vels.append(cam_vel)

        detections = self._yolo_detect(frame)
        self._last_yolo = detections

        dets_for_desc = self._limit_osnet_candidates(detections)
        det_descs = _extract_descriptor(frame, dets_for_desc) if dets_for_desc else []
        frame_cands = [
            (np.array(bbox, dtype=int), desc.copy())
            for bbox, desc in zip(dets_for_desc, det_descs)
            if desc is not None
        ]

        self._cand_frames.append(frame_cands)

        if self._use_distractor_bank and dets_for_desc:
            held_box = self.held_box
            assert held_box is not None
            ious = self._iou_many_to_one(
                np.asarray(dets_for_desc, dtype=np.float64), held_box
            )
            new_distractors = [
                det_desc
                for det_desc, iou in zip(det_descs, ious)
                if det_desc is not None and iou >= self.tau_occ
            ]
            self._extend_distractor_descriptors(new_distractors)

        collection_phase_num = self._occ_phase
        if self.debug:
            print(
                f"[occ frame {self._occ_frames}] "
                f"phase=collect({collection_phase_num}/{self._cand_collection_frames})  "
                f"detections={len(detections)}  stored={len(frame_cands)}"
            )

        self._occ_phase += 1

        return self.held_box, 0.0

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
        """
        Unified occlusion matcher:
        - RAM when configured and DRM bank not yet full.
        - DRM otherwise.
        """
        if self._should_use_ram_for_occlusion_recovery():
            box, score = self.memory.match(frame, candidates, threshold=margin)
            if box is None:
                return [], "ram"
            return [(box, float(score))], "ram"

        drm_results = self.memory.drm_match(
            frame=frame,
            candidates=candidates,
            ref_bbox=ref_bbox,
            velocity=velocity,
            distractor_bank=self._get_active_distractor_bank() if self._use_distractor_bank else (),
            search_cx=search_cx,
            search_cy=search_cy,
            dist_sigma=dist_sigma,
            lam_iou=lam_iou,
            lam_app=lam_app,
            lam_mot=lam_mot,
            lam_time=lam_time,
            alpha=alpha,
            gamma=gamma,
            margin=margin,
            top_k=top_k,
            skip_threshold=skip_threshold,
            lam_dist=lam_dist,
            lam_cand_dir=lam_cand_dir,
        )
        return drm_results, "drm"

    def _occ_phase_final_drm(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        The full reacquisition phase. Works through these steps:

                    1. Finds the last non-empty collection frame and uses its detections
                       as the candidate pool.
                    2. Calls _build_candidate_velocities to trace each candidate back
                       through all earlier collection frames. Candidates that cannot be
                       matched across every prior frame are dropped — they have no
                       reliable velocity measurement.
                    3. Runs DRM matching on the surviving fully-tracked candidates using
                       an EKF-uncertainty-aware dist_sigma.
                    4. Augments each DRM score with a direction/velocity consistency term
                       (lam_cand_dir). If the target is out-of-frame or tiny, this term
                       is reduced or zeroed.
                    5. Sorts candidates by augmented score and verifies the top-k with
                       the SiamABC tracker (motion-compensated by one frame of EKF velocity
                       to account for the candidates being from a prior frame).
                    6. The first candidate that clears reacq_threshold is committed via
                       _commit_reacquisition. If all fail, resets to phase 0 and returns
                       held_box.

        This is the expensive but high-quality recovery path. It only runs
                    once after the collection window closes, so its cost is amortised
                    across the collection frames. Velocity scoring is the key advantage
                    over a single-frame DRM match — a candidate that moves in the same
                    direction and at roughly the same speed as the EKF-predicted target
                    gets a boost; one moving the wrong way gets penalised, reducing
                    false reacquisitions on distractors.
        Args:
            frame (any): current video frame as a numpy BGR array
        Returns:
            On success  → output of _commit_reacquisition (ekf_bbox, verify_score)
            On failure  → held_box, 0.0  and phase reset to 0
    """

        def _reset():
            self._occ_phase = 0
            self._cand_frames = []
            self._occ_cam_vels = []
            held_box = self.held_box
            assert held_box is not None
            self.tracker.tracking_state.bbox = held_box.copy()

        last_idx = -1
        for i in range(len(self._cand_frames) - 1, -1, -1):
            if self._cand_frames[i]:
                last_idx = i
                break

        if last_idx == -1:
            if self.debug:
                print(
                    f"[occ frame {self._occ_frames}] phase=final_drm  "
                    f"no candidates in any collection frame — resetting"
                )
            _reset()
            return self.held_box, 0.0

        last_frame_cands = self._cand_frames[last_idx]
        last_cand_bboxes = [b for (b, _) in last_frame_cands]
        last_cand_arr = (
            np.asarray(last_cand_bboxes, dtype=np.float64)
            if last_cand_bboxes
            else np.empty((0, 4), dtype=np.float64)
        )

        cand_vels = self._build_candidate_velocities(last_idx)

        single_frame_mode = last_idx == 0

        fully_tracked_bboxes = [
            bbox
            for bbox, vel in zip(last_cand_bboxes, cand_vels)
            if (vel is not None or single_frame_mode)
               and self._is_near_exit_edge(bbox, frame, fraction=0.50)
        ]

        _n_edge_rejected = sum(
            1
            for bbox, vel in zip(last_cand_bboxes, cand_vels)
            if (vel is not None or single_frame_mode)
            and not self._is_near_exit_edge(bbox, frame, fraction=0.50)
        )
        if self.debug:
            print(
                f"[occ frame {self._occ_frames}] phase=final_drm  "
                f"last_cands={len(last_cand_bboxes)}  "
                f"fully_tracked={len(fully_tracked_bboxes)}  "
                f"drm_size={self.memory.drm_size()}  ram={len(self.memory)}  "
                f"ekf_unc={cast(BBoxEKF, self.ekf).get_uncertainty():.1f}px"
            )

        if not fully_tracked_bboxes:
            if self.debug:
                print(
                    f"[occ frame {self._occ_frames}] phase=final_drm  "
                    f"no fully-tracked candidates — resetting"
                )
            if last_cand_bboxes:
                self.held_box = self._nudge_toward_nearest(frame, last_cand_bboxes)
                ekf = self.ekf
                assert ekf is not None
                ekf.nudge_position(self.held_box)
            _reset()
            return self.held_box, 0.0

        dist_sigma = self._effective_dist_sigma(frame)

        occ_match_results, occ_match_mode = self._occlusion_memory_match(
            frame=frame,
            candidates=fully_tracked_bboxes,
            ref_bbox=self.held_box,
            velocity=self.velocity,
            margin=self._drm_kwargs["margin"],
            search_cx=self._search_cx,
            search_cy=self._search_cy,
            dist_sigma=dist_sigma,
            lam_iou=self._drm_kwargs["lam_iou"],
            lam_app=self._drm_kwargs["lam_app"],
            lam_mot=self._drm_kwargs["lam_mot"],
            lam_time=self._drm_kwargs["lam_time"],
            alpha=self._drm_kwargs["alpha"],
            gamma=self._drm_kwargs["gamma"],
            top_k=self._drm_kwargs["top_k"],
            skip_threshold=self._drm_kwargs["skip_threshold"],
            lam_dist=self._drm_kwargs["lam_dist"],
            lam_cand_dir=self._drm_kwargs["lam_cand_dir"],
        )

        if self.debug:
            print(
                f"[occ frame {self._occ_frames}] phase=final_drm  "
                f"occ_match={occ_match_mode}"
            )

        if not occ_match_results:
            if last_cand_bboxes:
                self.held_box = self._nudge_toward_nearest(frame, last_cand_bboxes)
                ekf = self.ekf
                assert ekf is not None
                ekf.nudge_position(self.held_box)
            _reset()
            return self.held_box, 0.0

        lam_dir = self._drm_lam_cand_dir
        if self._out_of_frame:
            lam_dir = 0.0
        elif self._is_long_distance(frame):
            lam_dir *= 0.5

        expected_vel = self.velocity

        def _find_cand_idx(
            drm_bbox,
        ):
            if last_cand_arr.size == 0:
                return None
            ious = self._iou_many_to_one(last_cand_arr, drm_bbox)
            best_idx = int(np.argmax(ious))
            return best_idx if float(ious[best_idx]) > 0.3 else None

        final_scored = []
        for drm_bbox, drm_score in occ_match_results:
            cand_idx = _find_cand_idx(drm_bbox)

            vel = (
                cand_vels[cand_idx]
                if cand_idx is not None and cand_idx < len(cand_vels)
                else None
            )
            dir_score = (
                self._compute_velocity_score(vel, expected_vel)
                if vel is not None
                else 0.5
            )

            augmented = drm_score + lam_dir * (2.0 * dir_score - 1.0)
            final_scored.append((drm_bbox, augmented, dir_score))

        final_scored.sort(key=lambda x: x[1], reverse=True)

        vx = float(self.velocity[0])
        vy = float(self.velocity[1])
        top_k = self._drm_kwargs.get("top_k", 3)

        for match_bbox, match_score, vel_score in final_scored[:top_k]:
            adjusted = match_bbox.astype(float).copy()
            adjusted[0] += vx
            adjusted[1] += vy
            adjusted = self._clamp_bbox_to_frame(np.array(adjusted, dtype=int), frame)

            self.tracker.dynamic_update = False
            verify_bbox, verify_score, _ = self.tracker.run_track_for_candidate(
                frame, adjusted
            )
            verify_bbox = np.array(verify_bbox, dtype=int)
            if self.debug:
                print(
                    f"[occ frame {self._occ_frames}] phase=final_drm_verify  "
                    f"drm={match_score:.3f}  vel={vel_score:.3f}  "
                    f"verify={verify_score:.3f}  "
                    f"pass={verify_score >= self.reacq_threshold}"
                )

            if verify_score >= self.reacq_threshold:
                self._set_recovered_early_occlusion_flag()
                desc = _extract_descriptor(frame, verify_bbox)
                self._cand_frames = []
                self._occ_cam_vels = []
                return self._commit_reacquisition(
                    frame, verify_bbox, desc, verify_score
                )

        _reset()
        return self.held_box, 0.0

    def _commit_reacquisition(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        desc: Optional[np.ndarray],
        score: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Shared exit point for all three recovery phases. Updates the EKF
                    with the confirmed bbox, then resets every occlusion-related flag
                    and counter — in_occlusion, out_of_frame, exit_edge, occ_frames,
                    occ_phase, cand_frames, entry_streak. Re-enables TTA and dynamic
                    update on the SiamABC tracker. Admits the new descriptor to memory
                    if available, syncs current_bbox / held_box / tracker state to the
                    EKF output, and appends a history entry so _normal_update starts
                    with a clean slate on the very next frame.

        Every successful reacquisition path funnels through here. Centralising
                    the teardown means none of the three phase methods has to individually
                    worry about leaving stale state — one call cleans everything up. Any
                    bug in the reset logic would affect all three phases equally, making
                    it easy to test and audit.
        Args:
            frame (any): current video frame as a numpy BGR array
            bbox (any): np.ndarray [x, y, w, h] of the verified reacquired position
            desc (any): appearance descriptor for the reacquired bbox, or None
            score (any): tracker confidence at the reacquired position
        Returns:
            ekf_bbox: np.ndarray [x, y, w, h] EKF-smoothed position after update
            score: float passed through from the caller unchanged
    """
        ekf = self.ekf
        assert ekf is not None
        ekf.update(bbox)
        ekf_bbox = ekf.get_bbox()
        self.velocity = ekf.get_velocity()

        self.in_occlusion = False
        self._out_of_frame = False
        self._exit_edge = None
        self._occ_frames = 0
        self._occ_phase = 0
        self._pending_candidates = []
        self._cand_frames = []
        self._occ_cam_vels = []
        self._entry_streak = 0
        self._distractor_mode_active = False
        self._distractor_mode_visual_reals = []
        self._distractor_mode_visual_distractors = []
        self._distractor_mode_roi = None
        self._distractor_mode_roi_size = None
        self._distractor_mode_stable_count = 0
        self._distractor_mode_ambiguous_count = 0
        self._distractor_mode_reentry_cooldown = 0
        self._distractor_mode_memory_freeze_left = 0
        self._distractor_mode_template_freeze_left = 0
        self._distractor_anchor_ekf = None
        self._distractor_anchor_pred_bbox = None
        self._distractor_mode_overlap_lock_active = False
        self._distractor_mode_overlap_clear_count = 0
        self._distractor_mode_overlap_lock_frames = 0
        self._distractor_anchor_uncertainty = 0.0
        self._clear_jump_watch_state()
        self._jump_reject_distractor_timer = 0
        self.tracker.enable_tta()
        self.tracker.dynamic_update = self.tracker.tracking_config["dynamic_update"]

        if desc is not None:
            self.memory.try_admit(ekf_bbox, desc, self.held_box)
        self.current_bbox: NDArray = ekf_bbox.copy()
        self.held_box = ekf_bbox.copy()
        self._stable_anchor_bbox = ekf_bbox.copy()
        self._distractor_focus_bbox = ekf_bbox.copy()
        self.tracker.tracking_state.bbox = ekf_bbox.copy()
        self._search_cx = float(ekf_bbox[0] + ekf_bbox[2] / 2.0)
        self._search_cy = float(ekf_bbox[1] + ekf_bbox[3] / 2.0)

        cx = float(ekf_bbox[0] + ekf_bbox[2] / 2.0)
        cy = float(ekf_bbox[1] + ekf_bbox[3] / 2.0)
        self._center_history.append(np.array([cx, cy]))

        cam_disp = np.zeros(2)
        if self._last_H is not None:
            h_fr, w_fr = frame.shape[:2]
            cx, cy = w_fr / 2.0, h_fr / 2.0
            denom = (
                self._last_H[2, 0] * cx
                + self._last_H[2, 1] * cy
                + self._last_H[2, 2]
                + 1e-8
            )
            ncx = (
                      self._last_H[0, 0] * cx + self._last_H[0, 1] * cy + self._last_H[0, 2]
                  ) / denom
            ncy = (
                      self._last_H[1, 0] * cx + self._last_H[1, 1] * cy + self._last_H[1, 2]
                  ) / denom
            cam_disp = np.array([ncx - cx, ncy - cy])
        self._cam_vel_history.append(cam_disp)

        self._conf_history.append(
            (
                ekf_bbox.copy(),
                self.velocity.copy(),
                self._last_H,
                self._last_H_reliable,
            )
        )
        return ekf_bbox, score

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
        """
        Projects the frame centre through self._last_H and returns the
                    displacement (dx, dy) as the camera velocity vector for this frame.
                    Returns zeros if _last_H is None.

        Called in _occ_phase_collect and _build_candidate_velocities to
                    camera-compensate candidate displacements. Without this, a candidate
                    that is stationary but appears to move because the camera panned would
                    receive an undeserved velocity score and might beat a genuinely moving
                    true target.
        Args:
            frame (any): current video frame (used only for its shape)
        Returns:
            np.ndarray of shape (2,) — (dx, dy) camera motion in pixels this frame
    """
        if self._last_H is None:
            return np.zeros(2)
        h_fr, w_fr = frame.shape[:2]
        cx, cy = w_fr / 2.0, h_fr / 2.0
        denom = (
            self._last_H[2, 0] * cx
            + self._last_H[2, 1] * cy
            + self._last_H[2, 2]
            + 1e-8
        )
        ncx = (
                  self._last_H[0, 0] * cx + self._last_H[0, 1] * cy + self._last_H[0, 2]
              ) / denom
        ncy = (
                  self._last_H[1, 0] * cx + self._last_H[1, 1] * cy + self._last_H[1, 2]
              ) / denom
        return np.array([ncx - cx, ncy - cy])

    def _effective_dist_sigma(
        self,
        frame: np.ndarray,
    ) -> float:
        """
        Computes the Gaussian distance penalty sigma for the DRM scoring
                    function. Takes the larger of two candidates:
                        size_sigma = drm_dist_sigma_factor × max(obj_w, obj_h)
                        ekf_sigma  = ekf.get_uncertainty() × 1.5
                    When the EKF is very uncertain (large covariance = long occlusion),
                    ekf_sigma dominates and the penalty softens, allowing correct
                    candidates that have drifted far from the stale EKF prediction to
                    still score well.

        Passed to drm_match in both _occ_phase_siam and _occ_phase_final_drm.
                    A fixed sigma would over-penalise spatially displaced candidates after
                    a long occlusion. This adaptive version ensures the spatial penalty
                    stays meaningful early (tight sigma) and relaxes late (wide sigma)
                    automatically, without any manual tuning per sequence.
        Args:
            frame (any): current video frame (used only for its shape via _get_median_size
            and _is_long_distance)
        Returns:
            float: sigma in pixels to pass as dist_sigma to drm_match
    """
        obj_w, obj_h = self._get_median_size()
        size_sigma = self._drm_dist_sigma_factor * float(max(obj_w, obj_h))
        ekf = self.ekf
        assert ekf is not None
        ekf_sigma = ekf.get_uncertainty() * 1.5
        return max(size_sigma, ekf_sigma)

    def _build_candidate_velocities(
        self,
        last_idx: int,
    ) -> List[Optional[np.ndarray]]:
        """
        For each candidate in _cand_frames[last_idx], tries to match it back
                    through every prior collection frame (0 … last_idx-1) using a combined
                    IoU + cosine similarity score. If the candidate cannot be matched in
                    any one prior frame, it is marked as None (will be excluded from DRM).

                    For candidates that survive all frames, computes per-step displacements,
                    subtracts the camera velocity at each step from _occ_cam_vels, and
                    returns the mean camera-compensated velocity vector over the full track.

                    Special case: if last_idx == 0 (only one collection frame), no prior
                    frames exist so no matching is attempted — every candidate gets None
                    (neutral score, not excluded).

        Called once per final DRM phase from _occ_phase_final_drm. The
                    velocities it produces are what separate "this detection is moving
                    like the target" from "this is a distractor that just happens to look
                    similar." Without cross-frame tracking, the velocity score would be
                    meaningless single-frame noise.
        Args:
            last_idx (any): index into _cand_frames pointing to the last non-empty
            collection frame; candidates from this frame are the ones
            being scored
        Returns:
            List of Optional[np.ndarray] — one entry per candidate in last_frame.
            Each is either a (2,) velocity vector [vx, vy] in px/frame,
            or None if the candidate was not found in every prior frame.
    """
        last_frame = self._cand_frames[last_idx]
        if not last_frame:
            return []

        n_prior = last_idx

        results: List[Optional[np.ndarray]] = []

        for bbox_last, desc_last in last_frame:
            cx_last = float(bbox_last[0] + bbox_last[2] / 2.0)
            cy_last = float(bbox_last[1] + bbox_last[3] / 2.0)

            if n_prior == 0:
                results.append(None)
                continue

            per_frame_cx = []
            per_frame_cy = []
            all_found = True

            for j in range(last_idx):
                early_frame = self._cand_frames[j]
                if not early_frame:
                    all_found = False
                    break

                best_score = 0.35
                best_cx_e = None
                best_cy_e = None

                for bbox_e, desc_e in early_frame:
                    match = 0.55 * _iou(bbox_last, bbox_e) + 0.45 * _cos_sim(
                        desc_last, desc_e
                    )
                    if match > best_score:
                        best_score = match
                        best_cx_e = float(bbox_e[0] + bbox_e[2] / 2.0)
                        best_cy_e = float(bbox_e[1] + bbox_e[3] / 2.0)

                if best_cx_e is None:
                    all_found = False
                    break

                per_frame_cx.append(best_cx_e)
                per_frame_cy.append(best_cy_e)

            if not all_found:
                results.append(None)
                continue

            per_frame_cx.append(cx_last)
            per_frame_cy.append(cy_last)

            step_vels = []
            for step in range(last_idx):
                raw_dx = per_frame_cx[step + 1] - per_frame_cx[step]
                raw_dy = per_frame_cy[step + 1] - per_frame_cy[step]

                cam_idx = step + 1
                if cam_idx < len(self._occ_cam_vels):
                    cam = self._occ_cam_vels[cam_idx]
                else:
                    cam = np.zeros(2)

                step_vels.append(
                    np.array([raw_dx - cam[0], raw_dy - cam[1]], dtype=float)
                )

            found_vel = np.mean(step_vels, axis=0) if step_vels else np.zeros(2)
            results.append(found_vel)

        return results

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
        """
        Estimate background motion homography between consecutive frames.

        Mode selection:
        - classic: original grid+affine RANSAC path (fast).
        - accurate: feature+full-homography path with fallback to classic.
        - botsort: BoT-SORT-inspired GMC estimator with stronger validity gates.
        """
        scale = self._flow_scale
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.disable_camera_motion:
            return None, False, gray

        if self._homography_mode == "botsort":
            if self._botsort_motion_estimator is None:
                self._botsort_motion_estimator = BotSortMotionEstimator(
                    first_frame=frame,
                    config=self._botsort_motion_cfg,
                )
                return None, False, gray

            ref_bbox = self.held_box if self.held_box is not None else self.current_bbox
            prev_target_bbox = None
            if ref_bbox is not None:
                ref = np.asarray(ref_bbox, dtype=float).reshape(-1)
                if ref.size >= 4:
                    prev_target_bbox = (
                        float(ref[0]),
                        float(ref[1]),
                        float(ref[2]),
                        float(ref[3]),
                    )

            step_res = self._botsort_motion_estimator.step(
                current_frame=frame,
                prev_target_bbox=prev_target_bbox,
            )
            H_raw = np.asarray(step_res.transform_original, dtype=np.float64)
            if H_raw.shape == (3, 3):
                H = H_raw
            elif H_raw.shape == (2, 3):
                H = np.eye(3, dtype=np.float64)
                H[:2, :] = H_raw
            else:
                H = None

            if H is not None and abs(float(H[2, 2])) > 1e-8:
                H = H / float(H[2, 2])

            if step_res.used_fallback and not self._botsort_motion_use_fallback_transform:
                return None, False, gray

            reliable = bool(step_res.confidence.valid and not step_res.used_fallback)
            if self.debug and (not step_res.confidence.valid):
                print(
                    f"[homography:botsort] frame={self.frame_idx} "
                    f"valid=0 reason={step_res.confidence.reason} "
                    f"fallback={int(step_res.used_fallback)} "
                    f"runtime_ms={step_res.runtime_ms:.1f}"
                )
            return H, reliable, gray

        if self.prev_gray is None:
            return None, False, gray

        if self.prev_gray.shape != gray.shape:
            prev_gray_scaled = cv2.resize(
                self.prev_gray,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            prev_gray_scaled = self.prev_gray

        if self._homography_mode == "accurate":
            H, reliable = self._estimate_homography_accurate(
                prev_gray_scaled=prev_gray_scaled,
                gray=gray,
                scale=scale,
            )
            if H is None:
                H, reliable = self._estimate_homography_classic(
                    prev_gray_scaled=prev_gray_scaled,
                    gray=gray,
                    scale=scale,
                )
            return H, reliable, gray

        H, reliable = self._estimate_homography_classic(
            prev_gray_scaled=prev_gray_scaled,
            gray=gray,
            scale=scale,
        )
        return H, reliable, gray

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

        is_tiny = (not self._out_of_frame) and self._is_long_distance(frame)
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

        obj_w, obj_h = self._get_median_size()
        obj_size = (obj_w + obj_h) // 2

        steps = self._occ_frames // max(1, _search_expand_growth_every)
        time_expand = float(_search_expand_growth_factor ** steps)
        effective_expand = min(_roi_start_expand * time_expand, _yolo_search_expand)

        if self._out_of_frame and self._exit_edge is not None:
            side = max(1, int(obj_size * effective_expand))

            if self._exit_edge == "right":
                if (self._search_cx - w_fr) > obj_w * 2:
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
                    on that crop at imgsz=320, then translates all resulting bounding
                    boxes back to full-frame coordinates. Optionally filters detections
                    to _target_class_id when yolo_filter_class=True and the class has
                    been committed. Stores results in _yolo_cache so _yolo_detect_cached
                    can skip a redundant call within the same frame.

        The only place in the system that runs YOLO. Called during candidate
                    collection (phases 1…N) and referenced by the phase 0 ROI logic.
                    Keeping detection isolated here means the ROI logic, class filtering,
                    and coordinate translation all live in one place. The 320px imgsz
                    keeps YOLO inference fast enough to run on embedded/edge hardware.
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
            crop, conf=self.yolo_conf, iou=self.yolo_iou_thr, verbose=False, imgsz=320
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
        if transform is None:
            return False, "no_h"
        if self._gmc_prior_require_reliable_h and not bool(self._last_H_reliable):
            return False, "unreliable_h"

        stats = self._gmc_motion_stats(transform, frame)
        if stats is None:
            return False, "invalid_h"

        dx, dy, scale, rotation_deg, max_corner_disp = stats
        h_fr, w_fr = frame.shape[:2]

        max_translation = self._gmc_prior_max_translation_frac * max(w_fr, h_fr)
        if abs(dx) > max_translation or abs(dy) > max_translation:
            return False, "large_translation"
        if scale < self._gmc_prior_min_scale or scale > self._gmc_prior_max_scale:
            return False, "bad_scale"
        if abs(rotation_deg) > self._gmc_prior_max_rotation_deg:
            return False, "large_rotation"

        max_corner_allowed = (
            self._gmc_prior_max_corner_displacement_frac * float(np.hypot(w_fr, h_fr))
        )
        if max_corner_disp > max_corner_allowed:
            return False, "large_corner_warp"

        return True, "ok"

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
        if self._gmc_prior_skip_in_distractor_mode and self._distractor_mode_active:
            return
        if self.current_bbox is None:
            return

        valid, reason = self._gmc_motion_is_valid(self._last_H, frame)
        if not valid:
            if self.debug and reason not in {"no_h", "unreliable_h"}:
                print(f"[gmc-prior] frame={self.frame_idx} skip reason={reason}")
            return

        prior_bbox = self._warp_bbox_with_homography(
            np.asarray(self.current_bbox, dtype=float),
            self._last_H,
            frame,
        )
        if prior_bbox is None:
            if self.debug:
                print(f"[gmc-prior] frame={self.frame_idx} skip reason=warp_failed")
            return

        used = self._set_tracker_search_prior(prior_bbox)
        if self.debug and used:
            print(
                f"[gmc-prior] frame={self.frame_idx} used=1 "
                f"prior={prior_bbox.tolist()}"
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
        """
        Returns (is_heavy, max_disp_px) used by camera-motion guards.
        """
        weighted_disp = self._camera_motion_weighted_disp(frame=None)
        inst_disp = self._h_translation_magnitude(self._last_H, frame)
        max_disp = max(weighted_disp, inst_disp)

        ref_diag = self._motion_gate_reference_diag(frame, bbox_hint=bbox_hint)
        weighted_norm = weighted_disp / ref_diag
        inst_norm = inst_disp / ref_diag
        max_norm = max(weighted_norm, inst_norm)

        has_px_gate = self._camera_motion_heavy_disp_threshold > 0.0
        has_norm_gate = self._camera_motion_heavy_norm_threshold > 0.0
        if not has_px_gate and not has_norm_gate:
            return False, max_disp

        heavy_px = (
            has_px_gate and max_disp >= self._camera_motion_heavy_disp_threshold
        )
        heavy_norm = (
            has_norm_gate and max_norm >= self._camera_motion_heavy_norm_threshold
        )
        return (heavy_px or heavy_norm), max_disp

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
            areas = np.array([float(b[2] * b[3]) for b, _, _, _ in history])
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

        first_bbox, _, _, _ = clean[0]
        fresh_ekf = BBoxEKF(
            first_bbox,
            process_noise=self.ekf_process_noise,
            meas_noise=self.ekf_meas_noise,
        )

        for bbox, _vel, h, h_rel in clean[1:]:
            fresh_ekf.predict(H=h, H_reliable=(h is not None))
            fresh_ekf.update(bbox)

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

        areas = np.array([float(b[2] * b[3]) for b, _, _, _ in history], dtype=float)
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
            xs = np.array([float(b[0] + b[2] / 2) for b, *_ in recent])
            ys = np.array([float(b[1] + b[3] / 2) for b, *_ in recent])
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
            crop, conf=self.yolo_conf, iou=self.yolo_iou_thr, verbose=False, imgsz=320
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
            print(
                "Compiling YOLO model using TensorRT at 320x320 (may take a minute)..."
            )

            model = YOLO(weights_path)

            model.export(format="engine", half=False, device=0, imgsz=320)

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
