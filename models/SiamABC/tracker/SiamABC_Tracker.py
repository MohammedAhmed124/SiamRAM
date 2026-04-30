"""
SiamABC Tracker implementation.

This module provides the main tracker class for the SiamABC model,
integrating feature extraction, correlation, and test-time adaptation
into a cohesive tracking pipeline.
"""
from collections import deque
from typing import Dict, Tuple, Union

import numpy as np
import torch
from numpy._typing import NDArray

from utils.box_coder import SiamABCBoxCoder, TrackerDecodeResult
from utils.utils import clamp_bbox, extend_bbox, get_extended_crop

from ..model import constants
from ..model.adaptive_batch_norm import AdaptiveBatchNorm
from .base_tracker import Tracker


class SiamABCTracker(Tracker):
    def __init__(self, model, cuda_id=0, **tracking_config):
        super().__init__(model, cuda_id, **tracking_config)

        self._norm_lambda_tta: float = next(
            (m._norm_lambda for m in self.net.modules() if isinstance(m, AdaptiveBatchNorm)),
            0.1,
        )
        self._tta_lam: torch.Tensor = torch.zeros(1, device=f"cuda:{cuda_id}")

    def get_box_coder(self, tracking_config, cuda_id: str | int = 0):
        """
        Return the appropriate box coder for decoding model outputs.

        Args:
            tracking_config (Dict): Configuration dictionary for tracking.
            cuda_id (Union[str, int]): GPU device ID.

        Returns:
            SiamABCBoxCoder: An initialised box coder.
        """
        return SiamABCBoxCoder(tracking_config)

    def initialize(self, image: NDArray, rect: NDArray, **kwargs) -> None:
        """
        args:
            img(np.ndarray): RGB image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        self.smooth_pred = self.tracking_config['smooth']
        self._prev_bbox = rect.copy()
        self.N = self.tracking_config["N"]
        self.dynamic_update = self.tracking_config["dynamic_update"]
        self.similarity_score = self.tracking_config["similarity_score"]
        self.dynamic_update_threshold = self.tracking_config["dynamic_update_threshold"]

        rect = clamp_bbox(rect, image.shape)
        self.tracking_state.bbox = rect
        self._init_rect = rect.copy()

        self.tracking_state.pred_score = 1.0
        self.prev_good_bbox = rect
        self.tracking_state.paths = deque([], maxlen=70)

        self.tracking_state.mean_color = np.mean(image, axis=(0, 1))

        self._template_features, _ = self.get_template_features(image, rect)
        self.dynamic_template_features = self._template_features.clone()
        search_feats, search_bbox_in_crop, _, search_crop_raw = self.get_search_features(image, rect)
        self.dynamic_search_features = search_feats

        self.warmup_frames = self.tracking_config.get("warmup_frames", 0)
        warmup_window = self.tracking_config.get("warmup_window_size", self.memory_window_size)
        init_window = warmup_window if self.warmup_frames > 0 else self.memory_window_size

        # Seed entry has all 5 elements so select_representatives never crashes
        # on the first selection interval.
        self.all_memory_imgs = deque(
            [[search_crop_raw, search_bbox_in_crop, rect,
              self.dynamic_template_features, search_feats]],
            maxlen=init_window,
        )
        self.classification_scores = deque([0.5], maxlen=init_window)
        self.running_dynamic_image = search_crop_raw
        self.running_dynamic_bbox = rect

        self.offset = self.tracking_config["search_context"]
        self.cls_threshold = 0.9

        self.idx = 0
        self.lost_idx = 0
        self.slection_method = 'mean'

        self.update_lambda = 0.1
        self.running_confidence = 0.5

        self._best_idx = 0
        self._best_score = 0.5
        self._is_full = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_template_features(self, image, rect):
        """Returns (features, raw_template_crop_numpy)."""
        context = extend_bbox(rect,
                              offset=self.tracking_config["template_bbox_offset"],
                              image_width=image.shape[1], image_height=image.shape[0])
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            context=context,
            crop_size=self.tracking_config["template_size"],
            padding_value=self.tracking_state.mean_color,
        )
        img = self._preprocess_image(template_crop, self._template_transform)
        return self.net.get_features(img), template_crop

    @torch.no_grad()
    def get_search_features(self, image, bbox):
        """Returns (features, search_bbox_in_crop, search_context, raw_search_crop_numpy)."""
        context = extend_bbox(
            bbox,
            offset=self.tracking_config["search_context"],
            image_width=image.shape[1],
            image_height=image.shape[0]
        )
        search_crop, search_bbox, search_context = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=self.tracking_config["instance_size"],
            padding_value=self.tracking_state.mean_color,
            context=context
        )
        search_tensor = self._preprocess_image(search_crop, self._search_transform)
        return self.net.get_features(search_tensor), search_bbox, search_context, search_crop

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def check_validity(self, bbox_window, bbox):
        return (
            bbox[0] >= bbox_window[0]
            and bbox[1] >= bbox_window[1]
            and bbox[2] <= bbox_window[2]
            and bbox[3] <= bbox_window[3]
        )

    def _update_best_index(self, pred_score: float, evicting: bool) -> None:
        """O(1) amortised best-index maintenance."""
        if evicting:
            if self._best_idx == 0:
                scores = np.array(self.classification_scores, dtype=np.float16)
                self._best_idx = int(np.argmax(scores))
                self._best_score = float(scores[self._best_idx])
            else:
                self._best_idx -= 1
                if pred_score > self._best_score:
                    self._best_score = pred_score
                    self._best_idx = len(self.classification_scores) - 1
        else:
            if pred_score > self._best_score:
                self._best_score = pred_score
                self._best_idx = len(self.classification_scores) - 1

    def _maybe_store_frame(
        self,
        search_crop_raw: np.ndarray,
        search_bbox_in_crop: np.ndarray,
        pred_bbox: np.ndarray,
        pred_score: float,
        template_feats: torch.Tensor,
        search_feats: torch.Tensor,
    ) -> None:
        """
        Store a memory entry with pre-computed features.

        template_feats and search_feats are computed from the full-res frame
        at admission time — identical quality to Version 2, but paid only on
        admitted frames instead of on every select_representatives call.
        """
        evicting = len(self.classification_scores) == self.memory_window_size
        self.all_memory_imgs.append(
            [search_crop_raw, search_bbox_in_crop, pred_bbox, template_feats, search_feats]
        )
        self.classification_scores.append(pred_score)
        self._update_best_index(pred_score, evicting)

    def select_representatives(self) -> None:
        """
        Update the dynamic template using the best frame in the memory window.
        """
        if not self.classification_scores:
            return
        if self._best_score < self.dynamic_update_threshold:
            return

        _, _, best_bbox, template_feats, search_feats = self.all_memory_imgs[self._best_idx]

        # Features were computed from the full-res frame at storage time —
        # no re-derivation, no transform mismatch, no quality loss.
        self.dynamic_template_features = template_feats
        self.dynamic_search_features = search_feats
        self.running_dynamic_bbox = best_bbox.copy()

        if hasattr(self.net, "invalidate_template_cache"):
            self.net.invalidate_template_cache()

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------

    def update(self, search: NDArray, *kw):
        pred_bbox, pred_score, sim_score, search_crop_raw, search_bbox_in_crop, search_features = self.run_track(search)

        self.tracking_state.bbox = pred_bbox
        self.tracking_state.pred_score = pred_score
        self.tracking_state.paths.append(pred_bbox)

        if self.dynamic_update:
            score_ok = (
                pred_score > self.running_confidence
                or pred_score >= self.dynamic_update_threshold
            )
            if score_ok:
                iou_ok = (
                    self._compute_iou(pred_bbox, self._prev_bbox)
                    >= self.tracking_config.get("iou_threshold", 0.3)
                )
                if iou_ok:
                    # get_template_features is only called on admitted frames.
                    # search_features is free — already computed in run_track.
                    template_feats, _ = self.get_template_features(search, pred_bbox)
                    self._maybe_store_frame(
                        search_crop_raw, search_bbox_in_crop, pred_bbox, pred_score,
                        template_feats, search_features,
                    )

            self._prev_bbox = pred_bbox.copy()

            if self.warmup_frames > 0 and self.idx == self.warmup_frames - 1:
                full_imgs   = deque(self.all_memory_imgs,       maxlen=self.memory_window_size)
                full_scores = deque(self.classification_scores, maxlen=self.memory_window_size)
                self.all_memory_imgs       = full_imgs
                self.classification_scores = full_scores
                scores = np.array(self.classification_scores, dtype=np.float16)
                self._best_idx   = int(np.argmax(scores))
                self._best_score = float(scores[self._best_idx])

            self.idx += 1

            if self.idx % self.N == 0:
                self.select_representatives()

            self.running_confidence = (
                self.update_lambda * pred_score
                + (1 - self.update_lambda) * self.running_confidence
            )
            self.running_confidence = min(self.running_confidence, self.running_confidence_floor_value)

        return pred_bbox, pred_score, sim_score

    def run_track(self, search):
        """
        Returns (pred_bbox, pred_score, sim_score, search_crop_raw,
                 search_bbox_in_crop, search_features).

        search_crop_raw, search_bbox_in_crop, and search_features are free
        byproducts of get_search_features — returning them costs nothing and
        lets update() store them without a second forward pass.
        """
        search_features, search_bbox_in_crop, search_context, search_crop_raw = self.get_search_features(
            search, self.tracking_state.bbox
        )
        self.tracking_state.mapping = search_context
        self.tracking_state.prev_size = search_bbox_in_crop[2:]
        pred_bbox, pred_score, sim_score, _ = self.track(
            search_features, self.dynamic_search_features, self.dynamic_template_features
        )
        pred_bbox = self._rescale_bbox(pred_bbox, self.tracking_state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, search.shape)
        return pred_bbox, pred_score, sim_score, search_crop_raw, search_bbox_in_crop, search_features

    @torch.no_grad()
    def track(self, search_features, dynamic_search_features, dynamic_template_features):
        track_result = self.net.track(
            search_features=search_features,
            dynamic_search_features=dynamic_search_features,
            template_features=self._template_features,
            dynamic_template_features=dynamic_template_features,
            lam=self._tta_lam,
        )
        return self._postprocess(track_result=track_result)

    def _postprocess(
        self,
        track_result: Dict[str, torch.Tensor]
    ) -> Tuple[NDArray, float, Union[int, float, bool], torch.Tensor]:
        cls_score = track_result[constants.TARGET_CLASSIFICATION_KEY].float().sigmoid()
        regression_map = track_result[constants.TARGET_REGRESSION_LABEL_KEY].detach().float()
        classification_map, penalty, pred_location = self._confidence_postprocess(
            cls_score=cls_score,
            regression_map=regression_map
        )

        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=classification_map,
            regression_map=track_result[constants.TARGET_REGRESSION_LABEL_KEY],
            use_sigmoid=False,
            pred_location=pred_location,
        )

        cls_score = cls_score.squeeze().cpu().numpy()
        pred_bbox = self._postprocess_bbox(decoded_info=decoded_info, cls_score=cls_score, penalty=penalty)
        r_max, c_max = decoded_info.pred_coords[0]
        sim_score_raw = track_result[constants.TRACKER_TARGET_SEARCH_SIM_SCORE]
        sim_score = sim_score_raw.item() if sim_score_raw is not None else 0.0
        return pred_bbox, cls_score[r_max, c_max].item(), sim_score, track_result[constants.TRACKER_ATTENTION_MAP]

    def run_track_for_candidate(self, search: np.ndarray, candidate_bbox: np.ndarray):
        """
        Evaluate a specific candidate bounding box using the tracker.

        This method temporarily sets the tracker state to the candidate's
        location to perform a tracking update, then restores the original state.

        Args:
            search (np.ndarray): RGB image of the search region.
            candidate_bbox (np.ndarray): Candidate box to evaluate [x, y, w, h].

        Returns:
            Tuple[NDArray, float, float]: (pred_bbox, pred_score, sim_score).
        """
        cand_x, cand_y, cand_w, cand_h = [float(v) for v in candidate_bbox]
        h_fr, w_fr = search.shape[:2]
        context_ratio = self.tracking_config["search_context"]

        pad_w = cand_w * context_ratio
        pad_h = cand_h * context_ratio

        px1 = max(0, int(cand_x - pad_w))
        py1 = max(0, int(cand_y - pad_h))
        px2 = min(w_fr, int(cand_x + cand_w + pad_w))
        py2 = min(h_fr, int(cand_y + cand_h + pad_h))

        padded_context = np.array([px1, py1, px2 - px1, py2 - py1], dtype=int)

        saved_bbox = self.tracking_state.bbox
        try:
            self.tracking_state.bbox = candidate_bbox.copy()
            self.tracking_state.mapping = padded_context.copy()
            pred_bbox, pred_score, sim_score, _, _, _ = self.run_track(search)
            return pred_bbox, pred_score, sim_score
        finally:
            self.tracking_state.bbox = saved_bbox

    # ------------------------------------------------------------------
    # TTA
    # ------------------------------------------------------------------

    def set_tta(self, enabled: bool) -> None:
        val = self._norm_lambda_tta if enabled else 0.0
        self._tta_lam.fill_(val)

    def enable_tta(self) -> None:
        """
        Enable Test-Time Adaptation.
        """
        self.set_tta(True)

    def disable_tta(self) -> None:
        """
        Disable Test-Time Adaptation.
        """
        self.set_tta(False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        ix1 = max(boxA[0], boxB[0])
        iy1 = max(boxA[1], boxB[1])
        ix2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        iy2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        return inter / union if union > 0 else 0.0
