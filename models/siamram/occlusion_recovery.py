"""Occlusion-recovery subsystem extracted from SiamRAM experiment tracker."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

import numpy as np
from numpy._typing import NDArray

from utils.utils import _cos_sim, _extract_descriptor, _iou

from .motion import BBoxEKF
from .tracker_state import CandidateRecord


class OcclusionRecoverySubsystem:
    def __init__(self, host: Any) -> None:
        self._host = host

    def occlusion_update(
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
        ekf = self._host.ekf
        assert ekf is not None
    
        ekf_raw = ekf.get_bbox()
        self._host._search_cx = float(ekf_raw[0] + ekf_raw[2] / 2.0)
        self._host._search_cy = float(ekf_raw[1] + ekf_raw[3] / 2.0)
        self._host.held_box = self._host._clamp_bbox_to_frame(ekf_raw, frame)
        self._host.velocity = ekf.get_velocity()
        self._host._occ_frames += 1
    
        if self._host._out_of_frame and self._host._exit_edge is not None:
            if self._host._exit_edge == "right":
                self._host._search_cx = float(w_fr - 1)
            elif self._host._exit_edge == "left":
                self._host._search_cx = 0.0
            elif self._host._exit_edge == "bottom":
                self._host._search_cy = float(h_fr - 1)
            elif self._host._exit_edge == "top":
                self._host._search_cy = 0.0
    
        if self._host._out_of_frame:
            ekf_inside = 0 <= self._host._search_cx < w_fr and 0 <= self._host._search_cy < h_fr
            vel_inward = False
            if ekf_inside and self._host._exit_edge is not None:
                vel_inward = {
                    "right": float(self._host.velocity[0]) < 0,
                    "left": float(self._host.velocity[0]) > 0,
                    "bottom": float(self._host.velocity[1]) < 0,
                    "top": float(self._host.velocity[1]) > 0,
                }.get(self._host._exit_edge, True)
            if ekf_inside and vel_inward:
                self._host._out_of_frame = False
                self._host._exit_edge = None
    
        else:
            obj_w, obj_h = self._host._get_median_size()
            oof_margin = float(max(obj_w, obj_h)) * 0.5
            if (
                self._host._search_cx < -oof_margin
                or self._host._search_cx >= w_fr + oof_margin
                or self._host._search_cy < -oof_margin
                or self._host._search_cy >= h_fr + oof_margin
            ):
                if self._host._search_cx >= w_fr + oof_margin:
                    self._host._exit_edge = "right"
                elif self._host._search_cx < -oof_margin:
                    self._host._exit_edge = "left"
                elif self._host._search_cy >= h_fr + oof_margin:
                    self._host._exit_edge = "bottom"
                else:
                    self._host._exit_edge = "top"
                self._host._out_of_frame = True

        if self._host._reacq_confirm_active:
            return self._host._occ_phase_reacq_confirm(frame)
        if self._host._occ_phase == 0:
            return self._host._occ_phase_siam(frame)
        elif 1 <= self._host._occ_phase <= self._host._cand_collection_frames:
            return self._host._occ_phase_collect(frame)
        else:
            return self._host._occ_phase_final_drm(frame)


    def begin_reacq_confirmation(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        score: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Start tentative lock-on after a successful Stage-3 verification.
    
        The tracker must stay above reacq_threshold for
        `_reacq_confirm_frames` consecutive frames before exiting occlusion.
        """
        seed_bbox = self._host._clamp_bbox_to_frame(np.array(bbox, dtype=int), frame)
        self._host.tracker.tracking_state.bbox = seed_bbox.copy()
        self._host._cand_frames = []
        self._host._occ_cam_vels = []
        self._host._occ_phase = 0
        self._host._reacq_confirm_active = True
        self._host._reacq_confirm_streak = 1 if score >= self._host.reacq_threshold else 0
        if self._host.debug:
            print(
                f"[occ frame {self._host._occ_frames}] phase=reacq_confirm_start  "
                f"score={score:.3f}  streak={self._host._reacq_confirm_streak}/"
                f"{self._host._reacq_confirm_frames}"
            )
        return self._host.held_box, score
    

    def reset_reacq_confirmation(
        self,
    ) -> None:
        self._host._reacq_confirm_active = False
        self._host._reacq_confirm_streak = 0
    

    def occ_phase_reacq_confirm(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Confirmation sub-stage after Stage-3 candidate verification.
    
        Success: require N consecutive frames with score >= reacq_threshold.
        Failure: fall back to EKF-propagated held_box and restart occlusion phases.
        """
        held_box = self._host.held_box
        assert held_box is not None
        pred_bbox, score, _ = self._host.tracker.update(frame)
        pred_bbox = np.array(pred_bbox, dtype=int)
        conf_ok = score >= self._host.reacq_threshold
        if conf_ok:
            self._host._reacq_confirm_streak += 1
            if self._host.debug:
                print(
                    f"[occ frame {self._host._occ_frames}] phase=reacq_confirm  "
                    f"score={score:.3f}  streak={self._host._reacq_confirm_streak}/"
                    f"{self._host._reacq_confirm_frames}"
                )
            if self._host._reacq_confirm_streak >= self._host._reacq_confirm_frames:
                self._host._reset_reacq_confirmation()
                self._host._set_recovered_early_occlusion_flag()
                desc = _extract_descriptor(frame, pred_bbox)
                return self._host._commit_reacquisition(frame, pred_bbox, desc, score)
            return self._host.held_box, score
    
        if self._host.debug:
            print(
                f"[occ frame {self._host._occ_frames}] phase=reacq_confirm  "
                f"score={score:.3f}  streak_broken -> restart"
            )
        self._host._reset_reacq_confirmation()
        self._host._cand_frames = []
        self._host._occ_cam_vels = []
        self._host._occ_phase = 0
        self._host.tracker.tracking_state.bbox = held_box.copy()
        return self._host.held_box, 0.0
    

    def occ_phase_siam(
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
        held_box = self._host.held_box
        assert held_box is not None
    
        rx, ry, rw, rh = self._host._get_yolo_search_roi(frame=frame)
    
        obj_w, obj_h = self._host._get_median_size()
    
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
        seed_bbox = self._host._clamp_bbox_to_frame(seed_bbox, frame)
    
        self._host.tracker.tracking_state.bbox = seed_bbox
        pred_bbox, score, _ = self._host.tracker.update(frame)
        pred_bbox = np.array(pred_bbox, dtype=int)
    
        if score >= self._host.occ_siam_reacq_threshold:
    
            if (
                not self._host._is_near_exit_edge(pred_bbox, frame, fraction=0.50)
                and self._host.recovered_early_occlusion
            ):
                if self._host.debug:
                    print(
                        f"[occ frame {self._host._occ_frames}] phase=siam  "
                        f"score={score:.3f}  REJECTED — too far from exit edge "
                        f"({self._host._exit_edge})"
                    )
                self._host.tracker.tracking_state.bbox = held_box.copy()
                self._host._cand_frames = []
                self._host._occ_cam_vels = []
                self._host._occ_phase = 1
                return self._host.held_box, score
            pred_desc = _extract_descriptor(frame, pred_bbox)

            cand_vel_phase0 = None
            if self._host._conf_history:
                last_bbox = self._host._conf_history[-1].bbox
                dx = (pred_bbox[0] + pred_bbox[2] / 2.0) - (
                    last_bbox[0] + last_bbox[2] / 2.0
                )
                dy = (pred_bbox[1] + pred_bbox[3] / 2.0) - (
                    last_bbox[1] + last_bbox[3] / 2.0
                )
                cam = self._host._cam_vel_from_H(frame)
                cand_vel_phase0 = np.array([dx - cam[0], dy - cam[1]])

            occ_match_results, occ_match_mode = self._host._occlusion_memory_match(
                frame=frame,
                candidates=[pred_bbox],
                ref_bbox=held_box,
                velocity=self._host.velocity,
                margin=self._host.occ_siam_margin,
                search_cx=self._host._search_cx,
                search_cy=self._host._search_cy,
                dist_sigma=self._host._effective_dist_sigma(frame),
                lam_iou=self._host._drm_kwargs["lam_iou"],
                lam_app=self._host._drm_kwargs["lam_app"],
                lam_mot=self._host._drm_kwargs["lam_mot"],
                lam_time=self._host._drm_kwargs["lam_time"],
                alpha=self._host._drm_kwargs["alpha"],
                gamma=self._host._drm_kwargs["gamma"],
                top_k=self._host._drm_kwargs["top_k"],
                skip_threshold=self._host._drm_kwargs["skip_threshold"],
                lam_dist=self._host._drm_kwargs["lam_dist"],
                lam_cand_dir=self._host._drm_kwargs["lam_cand_dir"],
            )

            drm_score = occ_match_results[0][1] if occ_match_results else -1.0

            lam_dir = self._host._drm_lam_cand_dir
            if lam_dir > 0 and cand_vel_phase0 is not None:
                dir_score = self._host._compute_velocity_score(cand_vel_phase0, self._host.velocity)
                drm_score += lam_dir * (2.0 * dir_score - 1.0)

            drm_ok = drm_score >= self._host.app_match_threshold
            if self._host.debug:
                print(
                    f"[occ frame {self._host._occ_frames}] phase=siam  "
                    f"score={score:.3f}  occ_match={occ_match_mode}  "
                    f"drm={drm_score:.3f}  pass={drm_ok}"
                )

            if drm_ok:
                self._host._set_recovered_early_occlusion_flag()
                return self._host._commit_reacquisition(frame, pred_bbox, pred_desc, score)

            held_box = self._host.held_box
            assert held_box is not None
            self._host.tracker.tracking_state.bbox = held_box.copy()

        self._host._cand_frames = []
        self._host._occ_cam_vels = []
        self._host._occ_phase = 1
        return self._host.held_box, score


    def occ_phase_collect(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Runs YOLO on the search ROI for this frame, extracts appearance
                    descriptors for detections (optionally capped by
                    osnet_max_candidate_batch), and appends (bbox, desc) pairs to
                    _cand_frames. Also records the camera velocity vector for this frame
                    in _occ_cam_vels so the final phase can camera-compensate each
                    per-step displacemenself._host. Updates the distractor bank with any detection
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
        cam_vel = self._host._cam_vel_from_H(frame)
        self._host._occ_cam_vels.append(cam_vel)
    
        detections = self._host._yolo_detect(frame)
        self._host._last_yolo = detections
    
        dets_for_desc = self._host._limit_osnet_candidates(detections)
        det_descs = _extract_descriptor(frame, dets_for_desc) if dets_for_desc else []
        frame_cands = [
            CandidateRecord(bbox=np.array(bbox, dtype=int), descriptor=desc.copy())
            for bbox, desc in zip(dets_for_desc, det_descs)
            if desc is not None
        ]
    
        self._host._cand_frames.append(frame_cands)
    
        if self._host._use_distractor_bank and dets_for_desc:
            held_box = self._host.held_box
            assert held_box is not None
            ious = self._host._iou_many_to_one(
                np.asarray(dets_for_desc, dtype=np.float64), held_box
            )
            new_distractors = [
                det_desc
                for det_desc, iou in zip(det_descs, ious)
                if det_desc is not None and iou >= self._host.tau_occ
            ]
            self._host._extend_distractor_descriptors(new_distractors)
    
        collection_phase_num = self._host._occ_phase
        if self._host.debug:
            print(
                f"[occ frame {self._host._occ_frames}] "
                f"phase=collect({collection_phase_num}/{self._host._cand_collection_frames})  "
                f"detections={len(detections)}  stored={len(frame_cands)}"
            )
    
        self._host._occ_phase += 1
    
        return self._host.held_box, 0.0
    

    def occlusion_memory_match(
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
        if self._host._should_use_ram_for_occlusion_recovery():
            box, score = self._host.memory.match(frame, candidates, threshold=margin)
            if box is None:
                return [], "ram"
            return [(box, float(score))], "ram"

        drm_results = self._host.memory.drm_match(
            frame=frame,
            candidates=candidates,
            ref_bbox=ref_bbox,
            velocity=velocity,
            distractor_bank=self._host._get_active_distractor_bank() if self._host._use_distractor_bank else (),
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


    def occ_phase_final_drm(
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
                       reliable velocity measuremenself._host.
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
            self._host._occ_phase = 0
            self._host._cand_frames = []
            self._host._occ_cam_vels = []
            held_box = self._host.held_box
            assert held_box is not None
            self._host.tracker.tracking_state.bbox = held_box.copy()

        last_idx = -1
        for i in range(len(self._host._cand_frames) - 1, -1, -1):
            if self._host._cand_frames[i]:
                last_idx = i
                break

        if last_idx == -1:
            if self._host.debug:
                print(
                    f"[occ frame {self._host._occ_frames}] phase=final_drm  "
                    f"no candidates in any collection frame — resetting"
                )
            _reset()
            return self._host.held_box, 0.0

        last_frame_cands = self._host._cand_frames[last_idx]
        last_cand_bboxes = [entry.bbox for entry in last_frame_cands]
        last_cand_arr = (
            np.asarray(last_cand_bboxes, dtype=np.float64)
            if last_cand_bboxes
            else np.empty((0, 4), dtype=np.float64)
        )

        cand_vels = self._host._build_candidate_velocities(last_idx)

        single_frame_mode = last_idx == 0

        fully_tracked_bboxes = [
            bbox
            for bbox, vel in zip(last_cand_bboxes, cand_vels)
            if (vel is not None or single_frame_mode)
               and self._host._is_near_exit_edge(bbox, frame, fraction=0.50)
        ]

        _n_edge_rejected = sum(
            1
            for bbox, vel in zip(last_cand_bboxes, cand_vels)
            if (vel is not None or single_frame_mode)
            and not self._host._is_near_exit_edge(bbox, frame, fraction=0.50)
        )
        if self._host.debug:
            print(
                f"[occ frame {self._host._occ_frames}] phase=final_drm  "
                f"last_cands={len(last_cand_bboxes)}  "
                f"fully_tracked={len(fully_tracked_bboxes)}  "
                f"drm_size={self._host.memory.drm_size()}  ram={len(self._host.memory)}  "
                f"ekf_unc={cast(BBoxEKF, self._host.ekf).get_uncertainty():.1f}px"
            )

        if not fully_tracked_bboxes:
            if self._host.debug:
                print(
                    f"[occ frame {self._host._occ_frames}] phase=final_drm  "
                    f"no fully-tracked candidates — resetting"
                )
            if last_cand_bboxes:
                self._host.held_box = self._host._nudge_toward_nearest(frame, last_cand_bboxes)
                ekf = self._host.ekf
                assert ekf is not None
                ekf.nudge_position(self._host.held_box)
            _reset()
            return self._host.held_box, 0.0

        dist_sigma = self._host._effective_dist_sigma(frame)

        occ_match_results, occ_match_mode = self._host._occlusion_memory_match(
            frame=frame,
            candidates=fully_tracked_bboxes,
            ref_bbox=self._host.held_box,
            velocity=self._host.velocity,
            margin=self._host._drm_kwargs["margin"],
            search_cx=self._host._search_cx,
            search_cy=self._host._search_cy,
            dist_sigma=dist_sigma,
            lam_iou=self._host._drm_kwargs["lam_iou"],
            lam_app=self._host._drm_kwargs["lam_app"],
            lam_mot=self._host._drm_kwargs["lam_mot"],
            lam_time=self._host._drm_kwargs["lam_time"],
            alpha=self._host._drm_kwargs["alpha"],
            gamma=self._host._drm_kwargs["gamma"],
            top_k=self._host._drm_kwargs["top_k"],
            skip_threshold=self._host._drm_kwargs["skip_threshold"],
            lam_dist=self._host._drm_kwargs["lam_dist"],
            lam_cand_dir=self._host._drm_kwargs["lam_cand_dir"],
        )

        if self._host.debug:
            print(
                f"[occ frame {self._host._occ_frames}] phase=final_drm  "
                f"occ_match={occ_match_mode}"
            )

        if not occ_match_results:
            if last_cand_bboxes:
                self._host.held_box = self._host._nudge_toward_nearest(frame, last_cand_bboxes)
                ekf = self._host.ekf
                assert ekf is not None
                ekf.nudge_position(self._host.held_box)
            _reset()
            return self._host.held_box, 0.0

        lam_dir = self._host._drm_lam_cand_dir
        if self._host._out_of_frame:
            lam_dir = 0.0
        elif self._host._is_long_distance(frame):
            lam_dir *= 0.5

        expected_vel = self._host.velocity

        def _find_cand_idx(
            drm_bbox,
        ):
            if last_cand_arr.size == 0:
                return None
            ious = self._host._iou_many_to_one(last_cand_arr, drm_bbox)
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
                self._host._compute_velocity_score(vel, expected_vel)
                if vel is not None
                else 0.5
            )

            augmented = drm_score + lam_dir * (2.0 * dir_score - 1.0)
            final_scored.append((drm_bbox, augmented, dir_score))

        final_scored.sort(key=lambda x: x[1], reverse=True)

        vx = float(self._host.velocity[0])
        vy = float(self._host.velocity[1])
        top_k = self._host._drm_kwargs.get("top_k", 3)

        for match_bbox, match_score, vel_score in final_scored[:top_k]:
            adjusted = match_bbox.astype(float).copy()
            adjusted[0] += vx
            adjusted[1] += vy
            adjusted = self._host._clamp_bbox_to_frame(np.array(adjusted, dtype=int), frame)

            self._host.tracker.dynamic_update = False
            verify_bbox, verify_score, _ = self._host.tracker.run_track_for_candidate(
                frame, adjusted
            )
            verify_bbox = np.array(verify_bbox, dtype=int)
            if self._host.debug:
                print(
                    f"[occ frame {self._host._occ_frames}] phase=final_drm_verify  "
                    f"drm={match_score:.3f}  vel={vel_score:.3f}  "
                    f"verify={verify_score:.3f}  "
                    f"pass={verify_score >= self._host.reacq_threshold}"
                )

            if verify_score >= self._host.reacq_threshold:
                if self._host._reacq_confirm_frames <= 1:
                    self._host._set_recovered_early_occlusion_flag()
                    desc = _extract_descriptor(frame, verify_bbox)
                    self._host._cand_frames = []
                    self._host._occ_cam_vels = []
                    return self._host._commit_reacquisition(
                        frame, verify_bbox, desc, verify_score
                    )
                return self._host._begin_reacq_confirmation(frame, verify_bbox, verify_score)

        _reset()
        return self._host.held_box, 0.0


    def commit_reacquisition(
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
                    it easy to test and audiself._host.
        Args:
            frame (any): current video frame as a numpy BGR array
            bbox (any): np.ndarray [x, y, w, h] of the verified reacquired position
            desc (any): appearance descriptor for the reacquired bbox, or None
            score (any): tracker confidence at the reacquired position
        Returns:
            ekf_bbox: np.ndarray [x, y, w, h] EKF-smoothed position after update
            score: float passed through from the caller unchanged
    """
        ekf = self._host.ekf
        assert ekf is not None
        ekf.update(bbox)
        ekf_bbox = ekf.get_bbox()
        self._host.velocity = ekf.get_velocity()
    
        self._host.in_occlusion = False
        self._host._out_of_frame = False
        self._host._exit_edge = None
        self._host._occ_frames = 0
        self._host._occ_phase = 0
        self._host._pending_candidates = []
        self._host._cand_frames = []
        self._host._occ_cam_vels = []
        self._host._reset_reacq_confirmation()
        self._host._entry_streak = 0
        self._host._distractor_mode_active = False
        self._host._distractor_mode_visual_reals = []
        self._host._distractor_mode_visual_distractors = []
        self._host._distractor_mode_roi = None
        self._host._distractor_mode_roi_size = None
        self._host._distractor_mode_stable_count = 0
        self._host._distractor_mode_ambiguous_count = 0
        self._host._distractor_mode_reentry_cooldown = 0
        self._host._distractor_mode_memory_freeze_left = 0
        self._host._distractor_mode_template_freeze_left = 0
        self._host._distractor_anchor_ekf = None
        self._host._distractor_anchor_pred_bbox = None
        self._host._distractor_mode_overlap_lock_active = False
        self._host._distractor_mode_overlap_clear_count = 0
        self._host._distractor_mode_overlap_lock_frames = 0
        self._host._distractor_anchor_uncertainty = 0.0
        self._host._clear_jump_watch_state()
        self._host._jump_reject_distractor_timer = 0
        self._host.tracker.enable_tta()
        self._host.tracker.dynamic_update = self._host.tracker.tracking_config["dynamic_update"]
    
        self._host._record_recovery(
            mode="occlusion",
            frame=frame,
            bbox=ekf_bbox,
            score=score,
            desc=desc,
        )

        if desc is not None:
            self._host.memory.try_admit(ekf_bbox, desc, self._host.held_box)
        self._host.current_bbox: NDArray = ekf_bbox.copy()
        self._host.held_box = ekf_bbox.copy()
        self._host._stable_anchor_bbox = ekf_bbox.copy()
        self._host._distractor_focus_bbox = ekf_bbox.copy()
        self._host.tracker.tracking_state.bbox = ekf_bbox.copy()
        self._host._search_cx = float(ekf_bbox[0] + ekf_bbox[2] / 2.0)
        self._host._search_cy = float(ekf_bbox[1] + ekf_bbox[3] / 2.0)
    
        self._host._commit_recovery_frame_history(frame, ekf_bbox)
        return ekf_bbox, score
    

    def cam_vel_from_h(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Projects the frame centre through self._host._last_H and returns the
                    displacement (dx, dy) as the camera velocity vector for this frame.
                    Returns zeros if _last_H is None.
    
        Called in _occ_phase_collect and _build_candidate_velocities to
                    camera-compensate candidate displacements. Without this, a candidate
                    that is stationary but appears to move because the camera panned would
                    receive an undeserved velocity score and might beat a genuinely moving
                    true targeself._host.
        Args:
            frame (any): current video frame (used only for its shape)
        Returns:
            np.ndarray of shape (2,) — (dx, dy) camera motion in pixels this frame
    """
        if self._host._last_H is None:
            return np.zeros(2)
        h_fr, w_fr = frame.shape[:2]
        cx, cy = w_fr / 2.0, h_fr / 2.0
        denom = (
            self._host._last_H[2, 0] * cx
            + self._host._last_H[2, 1] * cy
            + self._host._last_H[2, 2]
            + 1e-8
        )
        ncx = (
                  self._host._last_H[0, 0] * cx + self._host._last_H[0, 1] * cy + self._host._last_H[0, 2]
              ) / denom
        ncy = (
                  self._host._last_H[1, 0] * cx + self._host._last_H[1, 1] * cy + self._host._last_H[1, 2]
              ) / denom
        return np.array([ncx - cx, ncy - cy])
    

    def effective_dist_sigma(
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
        obj_w, obj_h = self._host._get_median_size()
        size_sigma = self._host._drm_dist_sigma_factor * float(max(obj_w, obj_h))
        ekf = self._host.ekf
        assert ekf is not None
        ekf_sigma = ekf.get_uncertainty() * 1.5
        return max(size_sigma, ekf_sigma)
    

    def build_candidate_velocities(
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
        last_frame = self._host._cand_frames[last_idx]
        if not last_frame:
            return []
    
        n_prior = last_idx
    
        results: List[Optional[np.ndarray]] = []
    
        for entry_last in last_frame:
            bbox_last = entry_last.bbox
            desc_last = entry_last.descriptor
            cx_last = float(bbox_last[0] + bbox_last[2] / 2.0)
            cy_last = float(bbox_last[1] + bbox_last[3] / 2.0)
    
            if n_prior == 0:
                results.append(None)
                continue
    
            per_frame_cx = []
            per_frame_cy = []
            all_found = True
    
            for j in range(last_idx):
                early_frame = self._host._cand_frames[j]
                if not early_frame:
                    all_found = False
                    break

                best_score = 0.35
                best_cx_e = None
                best_cy_e = None

                for entry_e in early_frame:
                    bbox_e = entry_e.bbox
                    desc_e = entry_e.descriptor
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
                if cam_idx < len(self._host._occ_cam_vels):
                    cam = self._host._occ_cam_vels[cam_idx]
                else:
                    cam = np.zeros(2)

                step_vels.append(
                    np.array([raw_dx - cam[0], raw_dy - cam[1]], dtype=float)
                )

            found_vel = np.mean(step_vels, axis=0) if step_vels else np.zeros(2)
            results.append(found_vel)

        return results
