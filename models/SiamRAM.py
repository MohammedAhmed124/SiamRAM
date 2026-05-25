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


class SiamRAMTracker:
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
        copile_yolo=False,
        debug=True,
        disable_camera_motion: bool = False,
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
            homography_mode (any): homography estimator mode; "classic" keeps old grid+affine RANSAC, "accurate" uses feature tracking + full homography RANSAC
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
            self._distractor_bank_maxlen > 0 and self._drm_kwargs["gamma"] > 0.0
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

        self.memory = AppearanceMemory(
            capacity=mem_capacity,
            tau_iou=tau_iou,
            tau_area=tau_area,
            drm_capacity=drm_capacity,
            tau_sim=drm_tau_sim,
            window_W=drm_window_W,
            mmin=drm_mmin,
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

        if (
            score < effective_threshold
            and self.frame_idx >= 0
            and self.enter_occlusion_on_loss
        ):
            self._entry_streak += 1
        else:
            self._entry_streak = 0

        if (
            self._entry_streak >= self._entry_patience
            and self.frame_idx >= 0
            and self.enter_occlusion_on_loss
        ):

            entry_streak_val = self._entry_streak
            self._entry_streak = 0
            self.in_occlusion = True

            is_exiting, exit_edge = self._detect_exit_direction(frame)
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

        if score >= effective_threshold:
            desc = _extract_descriptor(frame, pred_bbox)
            if desc is not None:
                self.memory.try_admit(pred_bbox, desc, self.current_bbox)

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

            drm_results = self.memory.drm_match(
                frame=frame,
                candidates=[pred_bbox],
                ref_bbox=held_box,
                velocity=self.velocity,
                distractor_bank=self._distractor_bank if self._use_distractor_bank else (),
                search_cx=self._search_cx,
                search_cy=self._search_cy,
                dist_sigma=self._effective_dist_sigma(frame),
                lam_iou=self._drm_kwargs["lam_iou"],
                lam_app=self._drm_kwargs["lam_app"],
                lam_mot=self._drm_kwargs["lam_mot"],
                lam_time=self._drm_kwargs["lam_time"],
                alpha=self._drm_kwargs["alpha"],
                gamma=self._drm_kwargs["gamma"],
                margin=self.occ_siam_margin,
                top_k=self._drm_kwargs["top_k"],
                skip_threshold=self._drm_kwargs["skip_threshold"],
                lam_dist=self._drm_kwargs["lam_dist"],
                lam_cand_dir=self._drm_kwargs["lam_cand_dir"],
            )

            drm_score = drm_results[0][1] if drm_results else -1.0

            lam_dir = self._drm_lam_cand_dir
            if lam_dir > 0 and cand_vel_phase0 is not None:
                dir_score = self._compute_velocity_score(cand_vel_phase0, self.velocity)
                drm_score += lam_dir * (2.0 * dir_score - 1.0)

            drm_ok = drm_score >= self.app_match_threshold
            if self.debug:
                print(
                    f"[occ frame {self._occ_frames}] phase=siam  "
                    f"score={score:.3f}  drm={drm_score:.3f}  pass={drm_ok}"
                )

            if drm_ok:
                self.recovered_early_occlusion = True
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
            self._distractor_bank.extend(
                det_desc
                for det_desc, iou in zip(det_descs, ious)
                if det_desc is not None and iou >= self.tau_occ
            )

        collection_phase_num = self._occ_phase
        if self.debug:
            print(
                f"[occ frame {self._occ_frames}] "
                f"phase=collect({collection_phase_num}/{self._cand_collection_frames})  "
                f"detections={len(detections)}  stored={len(frame_cands)}"
            )

        self._occ_phase += 1

        return self.held_box, 0.0

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

        drm_results = self.memory.drm_match(
            frame=frame,
            candidates=fully_tracked_bboxes,
            ref_bbox=self.held_box,
            velocity=self.velocity,
            distractor_bank=self._distractor_bank if self._use_distractor_bank else (),
            search_cx=self._search_cx,
            search_cy=self._search_cy,
            dist_sigma=dist_sigma,
            **self._drm_kwargs,
        )

        if not drm_results:
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
        for drm_bbox, drm_score in drm_results:
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
                self.recovered_early_occlusion = True
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
        self.tracker.enable_tta()
        self.tracker.dynamic_update = self.tracker.tracking_config["dynamic_update"]

        if desc is not None:
            self.memory.try_admit(ekf_bbox, desc, self.held_box)
        self.current_bbox: NDArray = ekf_bbox.copy()
        self.held_box = ekf_bbox.copy()
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
        """
        scale = self._flow_scale
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.disable_camera_motion:
            return None, False, gray
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
            self.recovered_early_occlusion = False
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
            weights = np.exp(np.linspace(-1, 0, len(cam_hist)))
            weights /= weights.sum()
            mean_disp = float(np.dot(weights, cam_hist))
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
