"""
Spike/jump watcher subsystem extracted from the tracker god-object.

The watcher holds jump-switch detection and multi-frame settle logic while
delegating host-specific side effects (memory admit, distractor-mode entry,
visual text, and thresholding) to the tracker instance passed at runtime.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from utils.utils import _cos_sim, _extract_descriptor, _iou


class SpikeWatcher:
    """Collaborator handling spike-triggered jump rejection logic."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def clear_watch_state(self) -> None:
        t = self._host
        t._jump_watch_active = False
        t._jump_watch_anchor_bbox = None
        t._jump_watch_prev_bbox = None
        t._jump_watch_frames = 0
        t._jump_watch_stable_count = 0
        t._jump_watch_baseline_norm = 0.0
        t._jump_watch_last_speed_norm = 0.0
        t._jump_watch_last_ratio = 1.0
        t._jump_watch_last_sim = None

    def start_watch(
        self,
        *,
        prev_bbox: np.ndarray,
        first_switched_bbox: np.ndarray,
        spike_speed_norm: float,
        spike_ratio: float,
        baseline_norm: float,
        sim: Optional[float],
    ) -> None:
        t = self._host
        t._jump_watch_active = True
        t._jump_watch_anchor_bbox = np.array(prev_bbox, dtype=int).copy()
        t._jump_watch_prev_bbox = np.array(first_switched_bbox, dtype=int).copy()
        t._jump_watch_frames = 0
        t._jump_watch_stable_count = 0
        t._jump_watch_baseline_norm = float(baseline_norm)
        t._jump_watch_last_speed_norm = float(spike_speed_norm)
        t._jump_watch_last_ratio = float(spike_ratio)
        t._jump_watch_last_sim = sim
        t._jump_watch_last_step_norm = float(spike_speed_norm)
        if t.debug:
            sim_txt = "n/a" if sim is None else f"{sim:.3f}"
            print(
                f"[spike-watch:start] frame={t.frame_idx}  "
                f"speed_norm={spike_speed_norm:.3f}  ratio={spike_ratio:.2f}  "
                f"baseline={baseline_norm:.3f}  app_sim={sim_txt}"
            )

    def evaluate_hard_jump_candidate(
        self,
        *,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
    ) -> Tuple[bool, Optional[np.ndarray], float, float, float, Optional[float]]:
        """
        Spike-only distractor switch trigger.

        Returns:
            trigger, candidate_descriptor, speed_norm, baseline_norm, speed_ratio, sim
        """
        t = self._host
        t._spike_debug_speed_norm = float("nan")
        t._spike_debug_baseline_norm = float("nan")
        t._spike_debug_ratio = float("nan")
        t._spike_debug_trigger = False

        if not t._spike_reject_enabled:
            return False, None, 0.0, 0.0, 1.0, None
        if t.current_bbox is None:
            return False, None, 0.0, 0.0, 1.0, None
        if t.frame_idx < t._spike_reject_min_frames:
            return False, None, 0.0, 0.0, 1.0, None
        if score < t._spike_reject_min_score:
            return False, None, 0.0, 0.0, 1.0, None

        prev_bbox = np.asarray(t.current_bbox, dtype=float)
        cand_bbox = np.asarray(pred_bbox, dtype=float)
        h_curr = t._last_H if bool(t._last_H_reliable) else None
        speed_norm = t._camera_compensated_step_norm(
            prev_bbox=prev_bbox,
            curr_bbox=cand_bbox,
            H=h_curr,
        )
        t._spike_debug_speed_norm = float(speed_norm)

        if len(t._spike_step_norms) < t._spike_reject_min_history:
            return False, None, speed_norm, 0.0, 1.0, None

        baseline_norm = float(np.median(np.asarray(t._spike_step_norms, dtype=float)))
        ratio = speed_norm / (baseline_norm + 1e-6)
        t._spike_debug_baseline_norm = float(baseline_norm)
        t._spike_debug_ratio = float(ratio)

        if speed_norm < t._spike_reject_abs_norm_min:
            return False, None, speed_norm, baseline_norm, ratio, None
        if ratio < t._spike_reject_ratio:
            return False, None, speed_norm, baseline_norm, ratio, None

        cand_desc = _extract_descriptor(frame, pred_bbox)
        if t._spike_reject_use_appearance:
            ref_desc = t.memory.best_descriptor()
            if ref_desc is not None and cand_desc is not None:
                sim = float(_cos_sim(ref_desc, cand_desc))
                if sim > t._spike_reject_max_sim:
                    return False, cand_desc, speed_norm, baseline_norm, ratio, sim
                t._spike_debug_trigger = True
                return True, cand_desc, speed_norm, baseline_norm, ratio, sim

        t._spike_debug_trigger = True
        return True, cand_desc, speed_norm, baseline_norm, ratio, None

    def select_pre_spike_anchor_bbox(self) -> Optional[np.ndarray]:
        """
        Choose a robust pre-spike anchor box.

        Priority:
        1) Explicit stable anchor tracked during normal stable motion.
        2) Box right before the largest recent normalized step jump in history.
        3) Current bbox fallback.
        """
        t = self._host
        anchor = None
        if t._stable_anchor_bbox is not None:
            anchor = np.array(t._stable_anchor_bbox, dtype=int).copy()

        hist = list(t._conf_history)
        if not hist:
            if anchor is not None:
                return anchor
            if t.current_bbox is not None:
                return np.array(t.current_bbox, dtype=int).copy()
            return None

        boxes = [
            np.asarray(entry.bbox, dtype=float)
            for entry in hist[-t._spike_anchor_history_window:]
        ]
        if t.current_bbox is not None:
            boxes.append(np.asarray(t.current_bbox, dtype=float))

        if len(boxes) >= 2:
            step_norms = [
                t._normalized_center_distance(boxes[i], boxes[i + 1])
                for i in range(len(boxes) - 1)
            ]
            if step_norms:
                j = int(np.argmax(step_norms))
                max_step = float(step_norms[j])
                if max_step >= t._spike_anchor_trigger_norm:
                    hist_anchor = np.array(np.round(boxes[j]), dtype=int)
                    anchor = hist_anchor

        if anchor is None and t.current_bbox is not None:
            anchor = np.array(t.current_bbox, dtype=int).copy()
        return anchor

    def apply_hard_jump_rejection(
        self,
        *,
        frame: np.ndarray,
        pred_bbox: np.ndarray,
        score: float,
        effective_threshold: float,
    ) -> Tuple[np.ndarray, float]:
        """Multi-frame jump handling for distractor swaps."""
        t = self._host
        if t._distractor_mode_reentry_cooldown > 0:
            t._distractor_mode_reentry_cooldown = max(
                0, t._distractor_mode_reentry_cooldown - 1
            )
            if t._jump_watch_active:
                self.clear_watch_state()
            score = t._apply_distractor_mode_penalty(frame, pred_bbox, score)
            return pred_bbox, float(score)

        heavy_cam_motion, heavy_cam_disp = t._is_heavy_camera_motion(
            frame, bbox_hint=pred_bbox
        )
        if (
            t._block_distractor_mode_on_camera_motion
            and heavy_cam_motion
        ):
            if t._jump_watch_active:
                self.clear_watch_state()
            if t.debug:
                print(
                    f"[distractor guard] frame={t.frame_idx} blocked by camera motion "
                    f"disp={heavy_cam_disp:.2f} thr={t._camera_motion_heavy_disp_threshold:.2f}"
                )
            score = t._apply_distractor_mode_penalty(frame, pred_bbox, score)
            return pred_bbox, float(score)

        spike_trigger, cand_desc, speed_norm, baseline_norm, speed_ratio, sim = (
            self.evaluate_hard_jump_candidate(
                frame=frame,
                pred_bbox=pred_bbox,
                score=score,
            )
        )
        if (
            spike_trigger
            and not t._jump_watch_active
        ):
            anchor_bbox = self.select_pre_spike_anchor_bbox()
            if anchor_bbox is None:
                anchor_bbox = (
                    np.array(t.current_bbox, dtype=int).copy()
                    if t.current_bbox is not None
                    else None
                )
            if anchor_bbox is None:
                score = t._apply_distractor_mode_penalty(frame, pred_bbox, score)
                return pred_bbox, float(score)
            self.start_watch(
                prev_bbox=anchor_bbox,
                first_switched_bbox=pred_bbox,
                spike_speed_norm=speed_norm,
                spike_ratio=speed_ratio,
                baseline_norm=baseline_norm,
                sim=sim,
            )

        if t._jump_watch_active:
            if spike_trigger:
                t._jump_watch_last_speed_norm = float(speed_norm)
                t._jump_watch_last_ratio = float(speed_ratio)
                t._jump_watch_last_sim = sim

            t._jump_watch_frames += 1

            prev_watch = t._jump_watch_prev_bbox
            if prev_watch is not None:
                step_speed_norm = t._camera_compensated_step_norm(
                    prev_bbox=np.asarray(prev_watch, dtype=float),
                    curr_bbox=np.asarray(pred_bbox, dtype=float),
                    H=t._last_H,
                )
            else:
                step_speed_norm = 0.0

            t._jump_watch_prev_bbox = pred_bbox.copy()
            settle_thresh = max(
                t._spike_reject_settle_abs_norm_max,
                t._jump_watch_baseline_norm * t._spike_reject_settle_ratio,
                t._jump_watch_last_speed_norm * t._spike_reject_settle_from_spike_frac,
            )
            t._jump_watch_last_step_norm = float(step_speed_norm)
            if step_speed_norm <= settle_thresh:
                t._jump_watch_stable_count += 1
            else:
                t._jump_watch_stable_count = 0

            anchor = t._jump_watch_anchor_bbox
            if anchor is not None and _iou(pred_bbox, anchor) >= 0.50:
                if t.debug:
                    print(
                        f"[spike-watch:cancel] frame={t.frame_idx}  "
                        "candidate returned near anchor"
                    )
                self.clear_watch_state()
                score = t._apply_distractor_mode_penalty(frame, pred_bbox, score)
                return pred_bbox, score

            stable_ready = (
                t._jump_watch_stable_count >= t._spike_reject_settle_frames
            )
            timed_out = t._jump_watch_frames >= t._spike_reject_watch_max_frames

            should_commit = False
            if stable_ready and anchor is not None:
                should_commit = True
            elif (
                timed_out
                and anchor is not None
                and t._jump_watch_stable_count > 0
            ):
                should_commit = True
                if t.debug:
                    print(
                        f"[spike-watch:fallback-commit] frame={t.frame_idx}  "
                        f"watch_frames={t._jump_watch_frames}  "
                        f"stable_count={t._jump_watch_stable_count}"
                    )

            if should_commit and anchor is not None:
                switched_desc = cand_desc
                if switched_desc is None:
                    switched_desc = _extract_descriptor(frame, pred_bbox)
                if switched_desc is not None and t._distractor_bank_maxlen > 0:
                    t._add_distractor_descriptor(switched_desc.copy())

                t.visual_mode = "distractor"
                t.visual_reason = "Distractor switch detected; snapped to anchor"
                t.visual_details = (
                    f"spike_norm={t._jump_watch_last_speed_norm:.2f}  "
                    f"ratio={t._jump_watch_last_ratio:.2f}  "
                    f"stable_frames={t._jump_watch_stable_count}  "
                    f"settle<={settle_thresh:.2f}"
                )

                if t.debug:
                    print(
                        f"[spike-watch:commit] frame={t.frame_idx}  "
                        f"watch_frames={t._jump_watch_frames}  "
                        f"stable_count={t._jump_watch_stable_count}  "
                        f"step_norm={step_speed_norm:.3f}  settle_thr={settle_thresh:.3f}"
                    )

                snap_bbox = np.array(anchor, dtype=int).copy()
                t.tracker.tracking_state.bbox = snap_bbox.copy()
                t._enter_distractor_mode(snap_bbox)
                t._distractor_mode_visual_reals = [snap_bbox.copy()]
                t._distractor_mode_visual_distractors = (
                    [np.array(pred_bbox, dtype=int).copy()]
                )
                self.clear_watch_state()

                if t._jump_reject_force_occlusion:
                    score = min(float(score), float(effective_threshold) - 1e-6)
                return snap_bbox, float(score)

            if timed_out:
                if t.debug:
                    print(
                        f"[spike-watch:timeout] frame={t.frame_idx}  "
                        f"watch_frames={t._jump_watch_frames}"
                    )
                self.clear_watch_state()

        score = t._apply_distractor_mode_penalty(frame, pred_bbox, score)
        return pred_bbox, float(score)

