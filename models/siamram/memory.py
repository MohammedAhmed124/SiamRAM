"""
Appearance memory and Dynamic Reference Memory (DRM) for SiamRAM.

This module manages the storage and matching of target appearance
descriptors. It maintains a short-term buffer for recent high-confidence
frames and a long-term DRM bank for reliable re-acquisition.
"""

from collections import deque
from typing import List, Optional, Tuple, cast

import numpy as np

from utils.utils import (
    _cos_sim,
    _extract_descriptor,
    _iou,
    _xcorr_sim_2d_many_to_one,
)


class AppearanceMemory:
    """
    Manages short-term and long-term target appearance descriptors.

    AppearanceMemory handles the admission of new descriptors based on IoU
    and area constraints, and implements the DRM scoring logic for
    candidate verification during occlusion recovery.
    """

    def __init__(
        self,
        capacity: int = 20,
        tau_iou: float = 0.40,
        tau_area: float = 0.25,
        drm_capacity: int = 10,
        tau_sim: float = 0.85,
        window_W: int = 5,
        mmin: int = 3,
        distractor_bank_maxlen: int = 50,
    ):
        """
        Initialise the AppearanceMemory.

        Args:
            capacity (int): Max size of the short-term buffer.
            tau_iou (float): IoU threshold for buffer admission.
            tau_area (float): Area ratio threshold for buffer admission.
            drm_capacity (int): Max size of the long-term DRM bank.
            tau_sim (float): Cosine similarity threshold for DRM promotion.
            window_W (int): Temporal window for DRM promotion logic.
            mmin (int): Minimum agreements required for DRM promotion.
        """
        self.capacity = capacity
        self.tau_iou = tau_iou
        self.tau_area = tau_area
        self._buf: deque = deque(maxlen=capacity)
        self.drm_capacity = drm_capacity
        self.tau_sim = tau_sim
        self.window_W = window_W
        self.mmin = mmin
        self._drm: deque = deque(maxlen=drm_capacity)
        self._distractor_bank: deque = deque(maxlen=max(0, distractor_bank_maxlen))
        self._t: int = 0

    def reset(
        self,
    ):
        """
        Clear all buffers and reset the temporal counter.
        """
        self._buf.clear()
        self._drm.clear()
        self._distractor_bank.clear()
        self._t = 0

    def add_distractor(
        self,
        desc: np.ndarray,
    ) -> None:
        """
        Add one distractor descriptor to the negative bank.
        """
        if desc is None or self._distractor_bank.maxlen == 0:
            return
        self._distractor_bank.append(np.asarray(desc, dtype=float).copy())

    def extend_distractor_bank(
        self,
        descs: List[np.ndarray],
    ) -> None:
        """
        Add multiple distractor descriptors to the negative bank.
        """
        if self._distractor_bank.maxlen == 0 or not descs:
            return
        for desc in descs:
            if desc is None:
                continue
            self._distractor_bank.append(np.asarray(desc, dtype=float).copy())

    def get_distractor_bank(
        self,
    ) -> List[np.ndarray]:
        """
        Return a copy-friendly list view of distractor descriptors.
        """
        return list(self._distractor_bank)

    @staticmethod
    def _cos_sim_many_to_one(
        lhs: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized similarity between N descriptors (lhs) and one descriptor (rhs).

        Shape-dispatch matches utils.utils._cos_sim:
            - lhs (N, D), rhs (D,)       → batched cosine dot product.
            - lhs (N, C, H, W), rhs (C, H, W) → batched 2D cross-correlation peak
              (siamese xcorr backend). Falls through to the cosine path otherwise.
        """
        a = np.asarray(lhs)
        if a.size == 0:
            return np.empty((0,), dtype=np.float64)
        b = np.asarray(rhs)

        if a.ndim == 4 and b.ndim == 3:
            return _xcorr_sim_2d_many_to_one(a, b)

        a64 = a.astype(np.float64, copy=False)
        b64 = b.astype(np.float64, copy=False)
        dots = a64 @ b64
        a_norm = np.linalg.norm(a64, axis=1)
        b_norm = float(np.linalg.norm(b64))
        return dots / (a_norm * b_norm + 1e-8)

    def try_admit(
        self,
        bbox,
        desc: np.ndarray,
        prev_bbox,
    ) -> bool:
        """
        Attempt to admit a new descriptor to the short-term buffer.

        Args:
            bbox (NDArray): Current bounding box.
            desc (np.ndarray): Extracted appearance descriptor.
            prev_bbox (NDArray): Bounding box from the previous frame.

        Returns:
            bool: True if the descriptor was admitted.
        """
        iou_ok = _iou(bbox, prev_bbox) >= self.tau_iou
        if self._buf:
            areas = np.fromiter(
                (float(e[0][2] * e[0][3]) for e in self._buf),
                dtype=np.float64,
                count=len(self._buf),
            )
            med = float(np.median(areas))
            area_ok = abs(bbox[2] * bbox[3] - med) / (med + 1e-6) <= self.tau_area
        else:
            area_ok = True
        if iou_ok and area_ok:
            self._buf.append((np.array(bbox, dtype=int), desc.copy()))
            self._t += 1
            self._try_promote_to_drm(bbox, desc)
            return True
        return False

    def _try_promote_to_drm(
        self,
        bbox,
        new_desc: np.ndarray,
    ) -> None:
        recent = list(self._buf)[-self.window_W:]
        if len(recent) < self.mmin:
            return
        agreements = sum(
            1 for (_, d) in recent if _cos_sim(new_desc, d) >= self.tau_sim
        )
        if agreements >= self.mmin:
            self._drm.append(
                (
                    np.array(bbox, dtype=int),
                    new_desc.copy(),
                    self._t,
                )
            )

    def add_drm_anchor(
        self,
        bbox,
        desc: np.ndarray,
    ) -> bool:
        """
        Write one frame into the DRM bank as a distractor-context anchor.

        This sits alongside the existing agreement-based DRM write path in
        ``_try_promote_to_drm`` (it does not replace it) and is used by the
        introspection-based DRM update (arXiv:2411.17576, Sec. 3.2.2): when the
        tracker's own response map reveals a competing distractor during reliable
        tracking, the current frame is recorded as an anchor so re-acquisition can
        discriminate the target from that distractor.

        Args:
            bbox (NDArray): Current target bounding box.
            desc (np.ndarray): Extracted appearance descriptor.

        Returns:
            bool: True if the anchor was appended.
        """
        if desc is None:
            return False
        self._drm.append(
            (
                np.array(bbox, dtype=int),
                np.asarray(desc).copy(),
                self._t,
            )
        )
        return True

    def best_descriptor(
        self,
    ) -> Optional[np.ndarray]:
        """
        Return the most recent descriptor in the short-term buffer.

        Returns:
            Optional[np.ndarray]: The latest descriptor or None.
        """
        return self._buf[-1][1] if self._buf else None

    def match(
        self,
        frame: np.ndarray,
        candidates: List[np.ndarray],
        threshold: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Match a set of candidates against the latest descriptor.

        Args:
            frame (np.ndarray): Current RGB frame.
            candidates (List[np.ndarray]): List of candidate bounding boxes.
            threshold (float): Similarity threshold for acceptance.

        Returns:
            Tuple[Optional[np.ndarray], float]: (best_box, best_score).
        """
        ref = self.best_descriptor()
        if ref is None or not candidates:
            return None, -1.0
        cand_descs = cast(list[Optional[np.ndarray]], _extract_descriptor(frame, candidates))
        valid_idx = [i for i, desc in enumerate(cand_descs) if desc is not None]
        if not valid_idx:
            return None, -1.0

        valid_descs = np.asarray(
            [cand_descs[i] for i in valid_idx],
            dtype=np.float64,
        )
        sims = self._cos_sim_many_to_one(valid_descs, np.asarray(ref, dtype=np.float64))
        best_local_idx = int(np.argmax(sims))
        best_score = float(sims[best_local_idx])
        best_box = candidates[valid_idx[best_local_idx]]
        if best_score >= threshold:
            return np.array(best_box, dtype=int), best_score
        return None, best_score

    def drm_match(
        self,
        frame: np.ndarray,
        candidates: List[np.ndarray],
        ref_bbox: np.ndarray,
        velocity: np.ndarray,
        distractor_bank,
        lam_iou: float = 0.40,
        lam_app: float = 0.30,
        lam_mot: float = 0.20,
        lam_time: float = 0.10,
        alpha: float = 0.05,
        gamma: float = 0.25,
        margin: float = 0.35,
        top_k: int = 3,
        skip_threshold: float = 0.80,
        search_cx: Optional[float] = None,
        search_cy: Optional[float] = None,
        dist_sigma: Optional[float] = None,
        lam_dist: float = 0.15,
        lam_cand_dir: float = 0.15,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Perform complex DRM matching for re-acquisition.

        This method computes a multi-component score for each candidate,
        incorporating IoU, appearance, motion consistency, and temporal decay.

        Args:
            frame (np.ndarray): Current frame.
            candidates (List[np.ndarray]): YOLO detection candidates.
            ref_bbox (np.ndarray): Reference box for motion calculation.
            velocity (np.ndarray): Expected target velocity.
            distractor_bank: Bank of negative appearance descriptors.
            lam_iou, lam_app, lam_mot, lam_time (float): Component weights.
            alpha (float): Temporal decay rate.
            gamma (float): Distractor penalty weight.
            margin (float): Acceptance margin.
            top_k (int): Number of top results to return.
            skip_threshold (float): Score to skip further verification.
            search_cx, search_cy (float): Search region center.
            dist_sigma (float): Sigma for spatial distance penalty.
            lam_dist (float): Weight for distance penalty.
            lam_cand_dir (float): Weight for direction consistency.

        Returns:
            List[Tuple[np.ndarray, float]]: Scored candidates.
        """
        if not self._drm:
            box, score = self.match(frame, candidates, threshold=margin)
            if box is not None:
                return [(box, score)]
            return []

        if not candidates:
            return []

        ref_cx = ref_bbox[0] + ref_bbox[2] / 2.0
        ref_cy = ref_bbox[1] + ref_bbox[3] / 2.0
        vel_norm = float(np.linalg.norm(velocity)) + 1e-8

        scored: List[Tuple[np.ndarray, float]] = []
        cand_descs = cast(list[Optional[np.ndarray]], _extract_descriptor(frame, candidates))
        drm_entries = list(self._drm)

        distractor_arr = (
            np.asarray(list(distractor_bank), dtype=np.float64)
            if gamma > 0.0 and distractor_bank
            else None
        )

        precomp = []
        for dk_bbox, dk_desc, rho_k in drm_entries:
            dk_cx = dk_bbox[0] + dk_bbox[2] / 2.0
            dk_cy = dk_bbox[1] + dk_bbox[3] / 2.0
            motion_vec = np.array([dk_cx - ref_cx, dk_cy - ref_cy])
            mot_norm = float(np.linalg.norm(motion_vec)) + 1e-8
            pi_t = float(np.dot(velocity, motion_vec) / (vel_norm * mot_norm))
            pi_t = max(0.0, pi_t)
            age = max(0, self._t - rho_k)
            s_mot_time = lam_mot * pi_t + lam_time * float(np.exp(-alpha * age))
            if distractor_arr is not None:
                pen = float(np.max(self._cos_sim_many_to_one(distractor_arr, dk_desc)))
                s_mot_time -= gamma * pen
            precomp.append((dk_bbox, dk_desc, s_mot_time))

        cand_arr = np.asarray(candidates, dtype=np.float64)
        cand_centers = np.column_stack(
            (
                cand_arr[:, 0] + cand_arr[:, 2] / 2.0,
                cand_arr[:, 1] + cand_arr[:, 3] / 2.0,
            )
        )

        for cand_idx, (cand_bbox, cand_desc) in enumerate(zip(candidates, cand_descs)):
            if cand_desc is None:
                continue

            anchor_scores = []
            for dk_bbox, dk_desc, s_mot_time in precomp:
                s_iou = lam_iou * _iou(dk_bbox, cand_bbox)
                s_app = lam_app * _cos_sim(dk_desc, cand_desc)
                raw = s_iou + s_app + s_mot_time
                anchor_scores.append(raw)

            cand_score = max(anchor_scores) if anchor_scores else -np.inf

            if (
                search_cx is not None
                and search_cy is not None
                and dist_sigma is not None
                and dist_sigma > 0
            ):
                cand_cx = float(cand_centers[cand_idx, 0])
                cand_cy = float(cand_centers[cand_idx, 1])
                d = np.hypot(cand_cx - search_cx, cand_cy - search_cy)
                cand_score -= lam_dist * (1.0 - np.exp(-0.5 * (d / dist_sigma) ** 2))

            if lam_cand_dir > 0 and vel_norm > 1e-3:
                cand_cx = float(cand_centers[cand_idx, 0])
                cand_cy = float(cand_centers[cand_idx, 1])
                cand_vec = np.array([cand_cx - ref_cx, cand_cy - ref_cy])
                cand_d = float(np.linalg.norm(cand_vec)) + 1e-8
                cos_dir = float(np.dot(velocity, cand_vec) / (vel_norm * cand_d))
                cand_score += lam_cand_dir * cos_dir

            if cand_score > margin:
                scored.append((np.array(cand_bbox, dtype=int), float(cand_score)))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored[0][1] >= skip_threshold:
            return [scored[0]]

        return scored[:top_k]

    def score_target_against_drm(
        self,
        target_bbox,
        target_desc: Optional[np.ndarray],
        ref_bbox,
        velocity,
        distractor_bank=(),
        lam_iou: float = 0.40,
        lam_app: float = 0.30,
        lam_mot: float = 0.20,
        lam_time: float = 0.10,
        alpha: float = 0.05,
        gamma: float = 0.30,
    ) -> Optional[float]:
        """
        Score the *current genuine target* as if it were a re-acquisition
        candidate, using the same composite formula as ``drm_match`` so the
        result is on the same scale as ``margin``.

        This exists purely to feed the adaptive ``drm_margin="auto"`` estimator
        (see ``auto_margin.AutoDrmMargin``); ``drm_match`` is left untouched.
        Mirrors the per-candidate scoring of ``drm_match`` (max over DRM anchors
        of IoU + appearance + motion/time, minus the distractor penalty); the
        spatial-distance penalty and direction term are skipped because for the
        live target the search centre coincides with the target (penalty ~0) and
        the candidate/ref displacement is degenerate.

        Returns the composite score, or ``None`` when there are no DRM anchors
        yet (in which case the matcher would fall back to the pure-cosine path).
        """
        if target_desc is None or not self._drm:
            return None

        ref_cx = ref_bbox[0] + ref_bbox[2] / 2.0
        ref_cy = ref_bbox[1] + ref_bbox[3] / 2.0
        vel = np.asarray(velocity, dtype=np.float64)
        vel_norm = float(np.linalg.norm(vel)) + 1e-8

        distractor_arr = (
            np.asarray(list(distractor_bank), dtype=np.float64)
            if gamma > 0.0 and distractor_bank
            else None
        )
        cand_desc = np.asarray(target_desc, dtype=np.float64)

        best = -np.inf
        for dk_bbox, dk_desc, rho_k in list(self._drm):
            dk_cx = dk_bbox[0] + dk_bbox[2] / 2.0
            dk_cy = dk_bbox[1] + dk_bbox[3] / 2.0
            motion_vec = np.array([dk_cx - ref_cx, dk_cy - ref_cy])
            mot_norm = float(np.linalg.norm(motion_vec)) + 1e-8
            pi_t = max(0.0, float(np.dot(vel, motion_vec) / (vel_norm * mot_norm)))
            age = max(0, self._t - rho_k)
            s_mot_time = lam_mot * pi_t + lam_time * float(np.exp(-alpha * age))
            if distractor_arr is not None:
                pen = float(np.max(self._cos_sim_many_to_one(distractor_arr, dk_desc)))
                s_mot_time -= gamma * pen
            s_iou = lam_iou * _iou(dk_bbox, target_bbox)
            s_app = lam_app * _cos_sim(dk_desc, cand_desc)
            raw = s_iou + s_app + s_mot_time
            if raw > best:
                best = raw

        if best == -np.inf:
            return None
        return float(best)

    def drm_size(
        self,
    ) -> int:
        """
        Return the current number of entries in the DRM bank.

        Returns:
            int: DRM bank size.
        """
        return len(self._drm)

    def __len__(
        self,
    ):
        return len(self._buf)
