"""Distractor-mode subsystem extracted from SiamRAM experiment tracker."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from utils.utils import _extract_descriptor, _iou

from .motion import BBoxEKF


class DistractorModeSubsystem:
    def __init__(self, host: Any) -> None:
        self._host = host

    def get_distractor_mode_roi(
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
            self._host._distractor_focus_bbox
            if self._host._distractor_focus_bbox is not None
            else self._host.current_bbox
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
    
        if self._host._distractor_mode_use_tracker_mapping:
            roi_size = self._host._distractor_mode_roi_size
            if roi_size is None:
                mapping = getattr(self._host.tracker.tracking_state, "mapping", None)
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
            self._host._distractor_mode_roi_min_side,
            int(round(base_w * self._host._distractor_mode_roi_expand)),
        )
        eh = max(
            self._host._distractor_mode_roi_min_side,
            int(round(base_h * self._host._distractor_mode_roi_expand)),
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
    

    def enter_distractor_mode(
        self,
        anchor_bbox: np.ndarray,
    ) -> None:
        self._host._distractor_mode_active = True
        self._host._materialize_distractor_prebank()
        self._host._distractor_focus_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._host._stable_anchor_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._host._distractor_mode_roi_size = None
        self._host._distractor_mode_stable_count = 0
        self._host._distractor_mode_ambiguous_count = 0
        self._host._distractor_mode_below_gate_count = 0
        self._host._distractor_mode_overlap_lock_active = False
        self._host._distractor_mode_overlap_clear_count = 0
        self._host._distractor_mode_overlap_lock_frames = 0
        self._host._distractor_anchor_pred_bbox = np.array(anchor_bbox, dtype=int).copy()
        self._host._distractor_anchor_uncertainty = 0.0
        self._host._distractor_anchor_ekf = None
        if self._host._distractor_mode_anchor_ekf_enabled:
            self._host._distractor_anchor_ekf = BBoxEKF(
                anchor_bbox,
                process_noise=self._host.ekf_process_noise,
                meas_noise=self._host.ekf_meas_noise,
            )
            src_ekf = self._host.ekf
            if src_ekf is not None:
                self._host._distractor_anchor_ekf.reseed(
                    bbox=anchor_bbox,
                    velocity=src_ekf.get_velocity(),
                )
                try:
                    self._host._distractor_anchor_ekf.P = np.array(
                        src_ekf.P, dtype=float
                    ).copy()
                except Exception:
                    pass
            self._host._distractor_anchor_uncertainty = float(
                self._host._distractor_anchor_ekf.get_uncertainty()
            )
        if self._host._distractor_mode_use_tracker_mapping:
            mapping = getattr(self._host.tracker.tracking_state, "mapping", None)
            if mapping is not None and len(mapping) >= 4:
                _mx, _my, mw, mh = map(float, mapping[:4])
                if mw > 1.0 and mh > 1.0:
                    self._host._distractor_mode_roi_size = (
                        int(round(mw)),
                        int(round(mh)),
                    )
        self._host._jump_reject_distractor_timer = max(
            self._host._jump_reject_distractor_timer,
            self._host._jump_reject_distractor_mode_frames,
        )
    

    def exit_distractor_mode(
        self,
        reason: str,
        resolved: bool = False,
    ) -> None:
        self._host._distractor_mode_active = False
        self._host._distractor_mode_visual_reals = []
        self._host._distractor_mode_visual_distractors = []
        self._host._distractor_mode_roi = None
        self._host._distractor_mode_roi_size = None
        self._host._distractor_mode_stable_count = 0
        self._host._distractor_mode_ambiguous_count = 0
        self._host._distractor_mode_below_gate_count = 0
        self._host._distractor_mode_overlap_lock_active = False
        self._host._distractor_mode_overlap_clear_count = 0
        self._host._distractor_mode_overlap_lock_frames = 0
        self._host._distractor_anchor_ekf = None
        self._host._distractor_anchor_pred_bbox = None
        self._host._distractor_anchor_uncertainty = 0.0
        self._host._jump_reject_distractor_timer = 0
        if resolved:
            self._host._distractor_mode_reentry_cooldown = (
                self._host._distractor_mode_reentry_cooldown_frames
            )
            self._host._distractor_mode_memory_freeze_left = (
                self._host._distractor_mode_post_exit_memory_freeze_frames
            )
            self._host._distractor_mode_template_freeze_left = (
                self._host._distractor_mode_post_exit_template_freeze_frames
            )
        self._host.visual_mode = "tracking"
        self._host.visual_reason = reason
        self._host.visual_details = ""
    

    def distractor_mode_update(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        effective_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        During distractor mode:
        1) run YOLO in ROI around last reliable/focus region,
        2) evaluate ROI hypotheses with OSNet against target descriptor history,
        3) pick most similar as real target and mark all other hypotheses as distractors.
        Exit only when no similar candidate exists in ROI.
        """
        (
            focus_override,
            anchor_mean,
            anchor_cov,
            uncertainty_pad_px,
        ) = self._host._advance_distractor_anchor_ekf(frame)
    
        roi = self._host._get_distractor_mode_roi(
            frame,
            focus_override=focus_override,
            uncertainty_pad_px=uncertainty_pad_px,
        )
        self._host._distractor_mode_roi = np.array(roi, dtype=int)
    
        detections = self._host._yolo_detect_in_roi(frame, roi)
        self._host._last_yolo = detections
    
        refs = self._host._target_descriptor_history()
        if not refs and self._host._distractor_focus_bbox is not None:
            focus_desc = _extract_descriptor(frame, self._host._distractor_focus_bbox)
            if focus_desc is not None:
                refs = [np.asarray(focus_desc, dtype=float)]
    
        if not detections or not refs:
            self._host._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score
    
        desired_hyps = int(self._host._distractor_mode_yolo_topk)
        if desired_hyps <= 0:
            desired_hyps = len(detections)
        if self._host._osnet_max_candidate_batch > 0:
            desired_hyps = min(desired_hyps, int(self._host._osnet_max_candidate_batch))
        n_eval = min(len(detections), max(1, desired_hyps))
        dets_for_desc = detections[:n_eval]
        det_descs = _extract_descriptor(frame, dets_for_desc) if dets_for_desc else []
    
        focus_bbox = (
            focus_override
            if focus_override is not None
            else (
                self._host._distractor_focus_bbox
                if self._host._distractor_focus_bbox is not None
                else self._host.current_bbox
            )
        )
        if focus_bbox is None:
            focus_bbox = pred_bbox
        focus_bbox_arr = np.asarray(focus_bbox, dtype=float)
        focus_cx = float(focus_bbox_arr[0] + focus_bbox_arr[2] / 2.0)
        focus_cy = float(focus_bbox_arr[1] + focus_bbox_arr[3] / 2.0)
        dist_sigma = (
            self._host._distractor_drm_dist_sigma_factor
            * max(1.0, float(max(focus_bbox_arr[2], focus_bbox_arr[3])))
        )
    
        distractor_bank = self._host._get_distractor_mode_scoring_bank()
        if distractor_bank:
            distractor_bank = distractor_bank[-self._host._distractor_drm_bank_topk:]
    
        cands = self._host._score_distractor_candidates(
            dets_for_desc=dets_for_desc,
            det_descs=det_descs,
            refs=refs,
            focus_bbox_arr=focus_bbox_arr,
            focus_cx=focus_cx,
            focus_cy=focus_cy,
            dist_sigma=dist_sigma,
            distractor_bank=distractor_bank,
            anchor_mean=anchor_mean,
            anchor_cov=anchor_cov,
        )

        max_focus_dist_frac = self._host._distractor_mode_selected_max_focus_dist_frac
        if max_focus_dist_frac > 0.0 and cands:
            focus_diag = float(
                np.hypot(focus_bbox_arr[2], focus_bbox_arr[3])
            ) + 1e-8
            max_focus_dist = max_focus_dist_frac * focus_diag
            kept: list = []
            for cand in cands:
                bb = cand[0]
                cand_cx = float(bb[0] + bb[2] / 2.0)
                cand_cy = float(bb[1] + bb[3] / 2.0)
                d = float(np.hypot(cand_cx - focus_cx, cand_cy - focus_cy))
                if d <= max_focus_dist:
                    kept.append(cand)
                elif self._host.debug:
                    print(
                        f"[distractor:far-cand-reject] frame={self._host.frame_idx} "
                        f"d={d:.1f} max={max_focus_dist:.1f}"
                    )
            cands = kept

        if not cands:
            self._host._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score
    
        max_app_sim = max(float(c[2]) for c in cands)
        if max_app_sim < self._host._distractor_mode_min_similarity:
            self._host._exit_distractor_mode("No similar objects in ROI")
            return pred_bbox, score
    
        compare_mode = self._host._distractor_compare_mode
        # Both modes rank by the same composite weighted score.
        score_idx = 3
        cands.sort(key=lambda t: t[score_idx], reverse=True)
        best_bbox, best_desc, best_sim, best_score, best_iou, best_dist, best_neg, best_maha = cands[0]
        if best_sim < self._host._distractor_mode_selected_min_similarity:
            self._host._distractor_mode_below_gate_count += 1
            hold_frames = self._host._distractor_mode_selected_below_gate_hold_frames
            if (
                hold_frames > 0
                and self._host._distractor_mode_below_gate_count <= hold_frames
            ):
                # Bounded "below-gate" motion hold. The top-ranked candidate's
                # appearance dipped under the accept gate (motion blur / pose
                # change / partial occlusion). Instead of abandoning the target
                # and handing control back to the base tracker — which is often
                # sitting on the distractor, and which also unfreezes memory and
                # risks template poisoning — ride the EKF/anchor motion
                # prediction for a few frames and wait for appearance to recover.
                hold_source = (
                    focus_override
                    if focus_override is not None
                    else (
                        self._host._distractor_focus_bbox
                        if self._host._distractor_focus_bbox is not None
                        else pred_bbox
                    )
                )
                hold_bbox = self._host._clamp_bbox_to_frame(
                    np.array(hold_source, dtype=int).copy(), frame
                )
                self._host._distractor_mode_visual_reals = [hold_bbox.copy()]
                self._host._distractor_mode_visual_distractors = [
                    np.array(bb, dtype=int).copy() for bb, *_ in cands
                ]
                self._host._distractor_mode_stable_count = 0
                self._host._distractor_focus_bbox = hold_bbox.copy()
                self._host.tracker.tracking_state.bbox = hold_bbox.copy()
                self._host.visual_mode = "distractor"
                self._host.visual_reason = "Distractor mode: below-gate motion hold"
                self._host.visual_details = (
                    f"best_app={best_sim:.2f} < "
                    f"sel_min={self._host._distractor_mode_selected_min_similarity:.2f}  "
                    f"hold={self._host._distractor_mode_below_gate_count}/{hold_frames}  "
                    f"cands={len(cands)}  roi={roi[2]}x{roi[3]}"
                )
                self._host._jump_reject_distractor_timer = max(
                    self._host._jump_reject_distractor_timer,
                    self._host._jump_reject_distractor_mode_frames,
                )
                adjusted_score = max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
                adjusted_score = self._host._apply_distractor_mode_penalty(
                    frame, hold_bbox, adjusted_score
                )
                return hold_bbox.copy(), float(adjusted_score)

            # Held long enough without the appearance recovering — the target is
            # presumed lost. Exit distractor mode and, when configured, force the
            # occlusion-recovery handoff this frame instead of reverting to the
            # base tracker (which may be confidently locked onto the distractor
            # and would otherwise never trip the score-based loss gate).
            self._host._exit_distractor_mode(
                "Best ROI candidate similarity below distractor-mode gate"
            )
            if (
                self._host._distractor_mode_selected_below_gate_force_occlusion
                and self._host.enter_occlusion_on_loss
            ):
                # Drive the standard occlusion-on-loss entry: saturate the entry
                # streak and report a sub-threshold score so the caller enters
                # occlusion recovery immediately. Still routes through the normal
                # entry path, so the camera-motion guard applies (matching
                # jump_reject_force_occlusion semantics).
                self._host._pending_distractor_occlusion = True
                self._host._entry_streak = max(
                    self._host._entry_streak, self._host._entry_patience
                )
                forced_score = min(float(score), float(effective_threshold) - 1e-6)
                return pred_bbox, float(forced_score)
            return pred_bbox, score

        # Winner cleared the appearance gate this frame — reset the hold counter.
        self._host._distractor_mode_below_gate_count = 0

        distractor_entries = cands[1:]
        max_overlap_iou = 0.0
        if distractor_entries:
            max_overlap_iou = max(
                float(_iou(best_bbox, bb)) for bb, *_ in distractor_entries
            )
    
        lock_result = self._host._maybe_engage_overlap_motion_lock(
            frame=frame,
            focus_bbox_arr=focus_bbox_arr,
            best_sim=best_sim,
            max_overlap_iou=max_overlap_iou,
            distractor_entries=distractor_entries,
            cands=cands,
            roi=roi,
            score=score,
        )
        if lock_result is not None:
            return lock_result
    
        top_margin = float("inf")
        if len(cands) > 1:
            top_margin = float(cands[0][score_idx] - cands[1][score_idx])
    
        hold_result = self._host._maybe_apply_ambiguity_hold(
            frame=frame,
            pred_bbox=pred_bbox,
            score=score,
            best_sim=best_sim,
            focus_override=focus_override,
            cands=cands,
            top_margin=top_margin,
            roi=roi,
        )
        if hold_result is not None:
            return hold_result
    
        same_iou = float(_iou(focus_bbox_arr, best_bbox))
        if same_iou >= self._host._distractor_mode_exit_same_iou:
            self._host._distractor_mode_stable_count += 1
        else:
            self._host._distractor_mode_stable_count = 1
    
        self._host._distractor_mode_visual_reals = [best_bbox.copy()]
        self._host._distractor_mode_visual_distractors = [
            np.array(bb, dtype=int).copy() for bb, *_ in distractor_entries
        ]
        if self._host._distractor_bank_maxlen > 0:
            self._host._extend_distractor_descriptors([ddesc for _bb, ddesc, *_ in distractor_entries])
    
        self._host._distractor_focus_bbox = best_bbox.copy()
        self._host._stable_anchor_bbox = best_bbox.copy()
        self._host.tracker.tracking_state.bbox = best_bbox.copy()
        self._host._record_recovery(
            mode="distractor",
            frame=frame,
            bbox=best_bbox,
            score=float(np.clip(best_sim, 0.0, 1.0)),
            sim=float(best_sim),
            iou=float(best_iou),
        )
        if self._host._distractor_mode_anchor_ekf_enabled and self._host._distractor_anchor_ekf is not None:
            try:
                self._host._distractor_anchor_ekf.update(best_bbox)
                self._host._distractor_anchor_pred_bbox = best_bbox.copy()
                self._host._distractor_anchor_uncertainty = float(
                    self._host._distractor_anchor_ekf.get_uncertainty()
                )
            except Exception:
                pass
        self._host.visual_mode = "distractor"
        if compare_mode == "ram":
            self._host.visual_reason = "Distractor mode: RAM appearance ROI ranking"
        else:
            self._host.visual_reason = "Distractor mode: DRM-style ROI re-ranking"
        maha_txt = "n/a" if not np.isfinite(best_maha) else f"{best_maha:.2f}"
        self._host.visual_details = (
            f"cmp={compare_mode}  app={best_sim:.2f}  score={best_score:.2f}  iou={best_iou:.2f}  "
            f"dist_pen={best_dist:.2f}  neg={best_neg:.2f}  maha={maha_txt}  "
            f"margin={top_margin:.3f}  same_iou={same_iou:.2f}  "
            f"stable={self._host._distractor_mode_stable_count}/{self._host._distractor_mode_exit_stable_frames}  "
            f"cands={len(cands)}  distractors={len(distractor_entries)}  roi={roi[2]}x{roi[3]}"
        )
        self._host._jump_reject_distractor_timer = max(
            self._host._jump_reject_distractor_timer,
            self._host._jump_reject_distractor_mode_frames,
        )
    
        if (
            self._host._distractor_mode_exit_reinit_enabled
            and self._host._distractor_mode_stable_count >= self._host._distractor_mode_exit_stable_frames
        ):
            reinit_bbox = self._host._clamp_bbox_to_frame(best_bbox.copy(), frame)
            try:
                reinit_bbox = self._host._reinit_dynamic_template_only(
                    frame=frame,
                    bbox=reinit_bbox.copy(),
                    score_hint=float(np.clip(best_sim, 0.0, 1.0)),
                )
                self._host._record_recovery(
                    mode="distractor",
                    frame=frame,
                    bbox=reinit_bbox,
                    score=float(np.clip(best_sim, 0.0, 1.0)),
                    sim=float(best_sim),
                    iou=float(best_iou),
                )
                self._host.current_bbox = reinit_bbox.copy()
                self._host.held_box = reinit_bbox.copy()
                self._host._stable_anchor_bbox = reinit_bbox.copy()
                self._host._distractor_focus_bbox = reinit_bbox.copy()
                self._host._exit_distractor_mode(
                    f"Distractor resolved; dynamic template reinit after {self._host._distractor_mode_stable_count} stable frames",
                    resolved=True,
                )
                return reinit_bbox.copy(), max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
            except Exception as exc:
                if self._host.debug:
                    print(f"[distractor-exit:reinit-fail] frame={self._host.frame_idx} err={exc}")

        if (
            self._host._distractor_mode_update_memory
            and best_desc is not None
            and self._host.current_bbox is not None
        ):
            self._host.memory.try_admit(best_bbox, best_desc, self._host.current_bbox)

        adjusted_score = max(float(score), float(np.clip(best_sim, 0.0, 1.0)))
        adjusted_score = self._host._apply_distractor_mode_penalty(frame, best_bbox, adjusted_score)
        return best_bbox.copy(), float(adjusted_score)


    def apply_distractor_mode_penalty(
        self,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> float:
        if not self._host._distractor_mode_active and self._host._jump_reject_distractor_timer <= 0:
            return float(score)
    
        self._host.visual_mode = "distractor"
        self._host.visual_reason = "Distractor suppression active"
        self._host.visual_details = ""
    
        adjusted = float(score)
        top_sim: Optional[float] = None
        app_penalty = 0.0
        dist_penalty = 0.0
        dist_norm: Optional[float] = None
    
        bank = self._host._get_active_distractor_bank()
        if (
            self._host._jump_reject_distractor_penalty_enabled
            and bank
            and self._host._jump_reject_distractor_penalty_weight > 0.0
        ):
            desc = _extract_descriptor(frame, pred_bbox)
            if desc is not None:
                recent = bank[-self._host._jump_reject_distractor_penalty_bank_topk:]
                recent_valid = [d for d in recent if d is not None]
                if recent_valid:
                    bank_arr = np.asarray(recent_valid, dtype=np.float64)
                    desc_arr = np.asarray(desc, dtype=np.float64)
                    dots = bank_arr @ desc_arr
                    bank_norms = np.linalg.norm(bank_arr, axis=1)
                    desc_norm = float(np.linalg.norm(desc_arr))
                    sims = dots / (bank_norms * desc_norm + 1e-8)
                    top_sim = float(np.max(sims))
                    sim_floor = self._host._jump_reject_distractor_penalty_sim_floor
                    frac = max(0.0, (top_sim - sim_floor) / (1.0 - sim_floor + 1e-8))
                    app_penalty = self._host._jump_reject_distractor_penalty_weight * frac
                    adjusted = max(0.0, adjusted - app_penalty)

        if (
            self._host._distractor_focus_dist_penalty_enabled
            and self._host._distractor_focus_bbox is not None
            and self._host._distractor_focus_dist_penalty_weight > 0.0
        ):
            dist_norm = self._host._normalized_center_distance(
                self._host._distractor_focus_bbox, pred_bbox
            )
            soft_r = self._host._distractor_focus_dist_soft_radius
            hard_r = self._host._distractor_focus_dist_hard_radius
            if dist_norm > soft_r:
                frac = float(np.clip((dist_norm - soft_r) / (hard_r - soft_r + 1e-8), 0.0, 1.0))
                dist_penalty = self._host._distractor_focus_dist_penalty_weight * frac
                adjusted = max(0.0, adjusted - dist_penalty)

        sim_txt = "n/a" if top_sim is None else f"{top_sim:.2f}"
        dist_txt = "n/a" if dist_norm is None else f"{dist_norm:.2f}"
        self._host.visual_details = (
            f"sim={sim_txt}  app_pen={app_penalty:.2f}  "
            f"dist={dist_txt}  dist_pen={dist_penalty:.2f}  "
            f"frames_left={self._host._jump_reject_distractor_timer}"
        )

        if self._host._distractor_mode_active:
            self._host._jump_reject_distractor_timer = max(
                1, self._host._jump_reject_distractor_timer
            )
        else:
            self._host._jump_reject_distractor_timer = max(0, self._host._jump_reject_distractor_timer - 1)
        if not self._host._distractor_mode_active and self._host._jump_reject_distractor_timer == 0:
            self._host._distractor_focus_bbox = (
                self._host.current_bbox.copy() if self._host.current_bbox is not None else None
            )
        return adjusted
