"""Camera-motion subsystem extracted from SiamRAM experiment tracker."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .botsort import CameraMotionEstimator as BotSortMotionEstimator


class CameraMotionSubsystem:
    def __init__(self, host: Any) -> None:
        self._host = host

    def estimate_homography(
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
        scale = self._host._flow_scale
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
        if self._host.disable_camera_motion:
            return None, False, gray
    
        if self._host._homography_mode == "botsort":
            if self._host._botsort_motion_estimator is None:
                self._host._botsort_motion_estimator = BotSortMotionEstimator(
                    first_frame=frame,
                    config=self._host._botsort_motion_cfg,
                )
                return None, False, gray
    
            ref_bbox = self._host.held_box if self._host.held_box is not None else self._host.current_bbox
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
    
            step_res = self._host._botsort_motion_estimator.step(
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

            if step_res.used_fallback and not self._host._botsort_motion_use_fallback_transform:
                return None, False, gray

            reliable = bool(step_res.confidence.valid and not step_res.used_fallback)
            if self._host.debug and (not step_res.confidence.valid):
                print(
                    f"[homography:botsort] frame={self._host.frame_idx} "
                    f"valid=0 reason={step_res.confidence.reason} "
                    f"fallback={int(step_res.used_fallback)} "
                    f"runtime_ms={step_res.runtime_ms:.1f}"
                )
            return H, reliable, gray

        if self._host.prev_gray is None:
            return None, False, gray

        if self._host.prev_gray.shape != gray.shape:
            prev_gray_scaled = cv2.resize(
                self._host.prev_gray,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            prev_gray_scaled = self._host.prev_gray

        if self._host._homography_mode == "accurate":
            H, reliable = self._host._estimate_homography_accurate(
                prev_gray_scaled=prev_gray_scaled,
                gray=gray,
                scale=scale,
            )
            if H is None:
                H, reliable = self._host._estimate_homography_classic(
                    prev_gray_scaled=prev_gray_scaled,
                    gray=gray,
                    scale=scale,
                )
            return H, reliable, gray

        H, reliable = self._host._estimate_homography_classic(
            prev_gray_scaled=prev_gray_scaled,
            gray=gray,
            scale=scale,
        )
        return H, reliable, gray


    def gmc_motion_is_valid(
        self,
        transform: Optional[np.ndarray],
        frame: np.ndarray,
    ) -> Tuple[bool, str]:
        if transform is None:
            return False, "no_h"
        if self._host._gmc_prior_require_reliable_h and not bool(self._host._last_H_reliable):
            return False, "unreliable_h"

        stats = self._host._gmc_motion_stats(transform, frame)
        if stats is None:
            return False, "invalid_h"

        dx, dy, scale, rotation_deg, max_corner_disp = stats
        h_fr, w_fr = frame.shape[:2]

        max_translation = self._host._gmc_prior_max_translation_frac * max(w_fr, h_fr)
        if abs(dx) > max_translation or abs(dy) > max_translation:
            return False, "large_translation"
        if scale < self._host._gmc_prior_min_scale or scale > self._host._gmc_prior_max_scale:
            return False, "bad_scale"
        if abs(rotation_deg) > self._host._gmc_prior_max_rotation_deg:
            return False, "large_rotation"

        max_corner_allowed = (
            self._host._gmc_prior_max_corner_displacement_frac * float(np.hypot(w_fr, h_fr))
        )
        if max_corner_disp > max_corner_allowed:
            return False, "large_corner_warp"

        return True, "ok"


    def apply_gmc_search_prior(
        self,
        frame: np.ndarray,
    ) -> None:
        if self._host._gmc_prior_skip_in_distractor_mode and self._host._distractor_mode_active:
            return
        if self._host.current_bbox is None:
            return
        if self._host._last_H is None:
            return
        if self._host._gmc_prior_require_reliable_h and not self._host._last_H_reliable:
            return
    
        valid, reason = self._host._gmc_motion_is_valid(self._host._last_H, frame)
        if not valid:
            if self._host.debug and reason not in {"no_h", "unreliable_h"}:
                print(f"[gmc-prior] frame={self._host.frame_idx} skip reason={reason}")
            return
    
        prior_bbox = self._host._warp_bbox_with_homography(
            np.asarray(self._host.current_bbox, dtype=float),
            self._host._last_H,
            frame,
        )
        if prior_bbox is None:
            if self._host.debug:
                print(f"[gmc-prior] frame={self._host.frame_idx} skip reason=warp_failed")
            return
    
        used = self._host._set_tracker_search_prior(prior_bbox)
        if self._host.debug and used:
            print(
                f"[gmc-prior] frame={self._host.frame_idx} used=1 "
                f"prior={prior_bbox.tolist()}"
            )
    

    def is_heavy_camera_motion(
        self,
        frame: np.ndarray,
        bbox_hint: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """
        Returns (is_heavy, max_disp_px) used by camera-motion guards.
        """
        if self._host._heavy_motion_cache_frame_idx == self._host.frame_idx:
            weighted_disp = self._host._heavy_motion_cache_weighted_disp
            inst_disp = self._host._heavy_motion_cache_inst_disp
        else:
            weighted_disp = self._host._camera_motion_weighted_disp(frame=None)
            inst_disp = self._host._h_translation_magnitude(self._host._last_H, frame)
            self._host._heavy_motion_cache_frame_idx = self._host.frame_idx
            self._host._heavy_motion_cache_weighted_disp = float(weighted_disp)
            self._host._heavy_motion_cache_inst_disp = float(inst_disp)
        max_disp = max(weighted_disp, inst_disp)
    
        ref_diag = self._host._motion_gate_reference_diag(frame, bbox_hint=bbox_hint)
        weighted_norm = weighted_disp / ref_diag
        inst_norm = inst_disp / ref_diag
        max_norm = max(weighted_norm, inst_norm)
        # Publish the bbox-relative camera-motion magnitude for the adaptive
        # template-rate controller (template_rate_auto) and debug overlays.
        self._host._last_cam_motion_norm = float(max_norm)

        has_px_gate = self._host._camera_motion_heavy_disp_threshold > 0.0
        has_norm_gate = self._host._camera_motion_heavy_norm_threshold > 0.0
        if not has_px_gate and not has_norm_gate:
            self._host._last_heavy_cam_motion = False
            self._host._last_heavy_cam_disp = float(max_disp)
            return False, max_disp

        heavy_px = (
            has_px_gate and max_disp >= self._host._camera_motion_heavy_disp_threshold
        )
        heavy_norm = (
            has_norm_gate and max_norm >= self._host._camera_motion_heavy_norm_threshold
        )
        heavy = bool(heavy_px or heavy_norm)
        self._host._last_heavy_cam_motion = heavy
        self._host._last_heavy_cam_disp = float(max_disp)
        return heavy, max_disp
