from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from utils.utils import _cos_sim, _extract_descriptor, _iou


class AppearanceMemory:
    def __init__(
        self,
        capacity:     int   = 20,
        tau_iou:      float = 0.40,
        tau_area:     float = 0.25,
        drm_capacity: int   = 10,
        tau_sim:      float = 0.85,
        window_W:     int   = 5,
        mmin:         int   = 3,
    ):
        self.capacity = capacity
        self.tau_iou  = tau_iou
        self.tau_area = tau_area
        self._buf: deque = deque(maxlen=capacity)
        self.drm_capacity = drm_capacity
        self.tau_sim      = tau_sim
        self.window_W     = window_W
        self.mmin         = mmin
        self._drm: deque  = deque(maxlen=drm_capacity)
        self._t:   int    = 0

    def reset(self):
        self._buf.clear()
        self._drm.clear()
        self._t = 0

    def try_admit(self, bbox, desc: np.ndarray, prev_bbox) -> bool:
        iou_ok = _iou(bbox, prev_bbox) >= self.tau_iou
        if self._buf:
            med     = float(np.median([e[0][2] * e[0][3] for e in self._buf]))
            area_ok = abs(bbox[2] * bbox[3] - med) / (med + 1e-6) <= self.tau_area
        else:
            area_ok = True
        if iou_ok and area_ok:
            self._buf.append((np.array(bbox, dtype=int), desc.copy()))
            self._t += 1
            self._try_promote_to_drm(bbox, desc)
            return True
        return False

    def _try_promote_to_drm(self, bbox, new_desc: np.ndarray) -> None:
        recent = list(self._buf)[-self.window_W:]
        if len(recent) < self.mmin:
            return
        agreements = sum(
            1 for (_, d) in recent
            if _cos_sim(new_desc, d) >= self.tau_sim
        )
        if agreements >= self.mmin:
            self._drm.append((
                np.array(bbox, dtype=int),
                new_desc.copy(),
                self._t,
            ))

    def best_descriptor(self) -> Optional[np.ndarray]:
        return self._buf[-1][1] if self._buf else None

    def match(self, frame: np.ndarray, candidates: List[np.ndarray],
              threshold: float) -> Tuple[Optional[np.ndarray], float]:
        ref = self.best_descriptor()
        if ref is None or not candidates:
            return None, -1.0
        best_box, best_score = None, -1.0
        for bbox in candidates:
            desc = _extract_descriptor(frame, bbox)
            if desc is None:
                continue
            s = _cos_sim(ref, desc)
            if s > best_score:
                best_score, best_box = s, bbox
        if best_score >= threshold:
            return np.array(best_box, dtype=int), best_score
        return None, best_score

    def drm_match(
        self,
        frame:            np.ndarray,
        candidates:       List[np.ndarray],
        ref_bbox:         np.ndarray,
        velocity:         np.ndarray,
        distractor_bank,
        lam_iou:          float = 0.40,
        lam_app:          float = 0.30,
        lam_mot:          float = 0.20,
        lam_time:         float = 0.10,
        alpha:            float = 0.05,
        gamma:            float = 0.25,
        margin:           float = 0.35,
        top_k:            int   = 3,
        skip_threshold:   float = 0.80,
        search_cx:        Optional[float] = None,
        search_cy:        Optional[float] = None,
        dist_sigma:       Optional[float] = None,
        lam_dist:         float           = 0.15,
        lam_cand_dir:     float           = 0.15,
    ) -> List[Tuple[np.ndarray, float]]:
        if not self._drm:
            box, score = self.match(frame, candidates, threshold=margin)
            if box is not None:
                return [(box, score)]
            return []

        if not candidates:
            return []

        ref_cx   = ref_bbox[0] + ref_bbox[2] / 2.0
        ref_cy   = ref_bbox[1] + ref_bbox[3] / 2.0
        vel_norm = float(np.linalg.norm(velocity)) + 1e-8

        scored: List[Tuple[np.ndarray, float]] = []

        for cand_bbox in candidates:
            cand_desc = _extract_descriptor(frame, cand_bbox)
            if cand_desc is None:
                continue

            anchor_scores = []
            for (dk_bbox, dk_desc, rho_k) in self._drm:
                s_iou = lam_iou * _iou(dk_bbox, cand_bbox)
                s_app = lam_app * _cos_sim(dk_desc, cand_desc)

                dk_cx      = dk_bbox[0] + dk_bbox[2] / 2.0
                dk_cy      = dk_bbox[1] + dk_bbox[3] / 2.0
                motion_vec = np.array([dk_cx - ref_cx, dk_cy - ref_cy])
                mot_norm   = float(np.linalg.norm(motion_vec)) + 1e-8
                pi_t       = float(np.dot(velocity, motion_vec) /
                                   (vel_norm * mot_norm))
                pi_t       = max(0.0, pi_t)
                s_mot      = lam_mot * pi_t

                age    = max(0, self._t - rho_k)
                s_time = lam_time * float(np.exp(-alpha * age))

                raw = s_iou + s_app + s_mot + s_time

                if distractor_bank:
                    pen = max(_cos_sim(dk_desc, nu) for nu in distractor_bank)
                    raw -= gamma * pen

                anchor_scores.append(raw)

            cand_score = max(anchor_scores) if anchor_scores else -np.inf

            if (search_cx is not None and search_cy is not None
                    and dist_sigma is not None and dist_sigma > 0):
                cand_cx    = cand_bbox[0] + cand_bbox[2] / 2.0
                cand_cy    = cand_bbox[1] + cand_bbox[3] / 2.0
                d          = np.hypot(cand_cx - search_cx, cand_cy - search_cy)
                cand_score -= lam_dist * (1.0 - np.exp(-0.5 * (d / dist_sigma) ** 2))

            if lam_cand_dir > 0 and vel_norm > 1e-3:
                cand_cx  = cand_bbox[0] + cand_bbox[2] / 2.0
                cand_cy  = cand_bbox[1] + cand_bbox[3] / 2.0
                cand_vec = np.array([cand_cx - ref_cx, cand_cy - ref_cy])
                cand_d   = float(np.linalg.norm(cand_vec)) + 1e-8
                cos_dir  = float(np.dot(velocity, cand_vec) / (vel_norm * cand_d))
                cand_score += lam_cand_dir * cos_dir

            if cand_score > margin:
                scored.append((np.array(cand_bbox, dtype=int), float(cand_score)))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored[0][1] >= skip_threshold:
            return [scored[0]]

        return scored[:top_k]

    def drm_size(self) -> int:
        return len(self._drm)

    def __len__(self):
        return len(self._buf)

