"""
SiamABC Tracker implementation.

This module provides the main tracker class for the SiamABC model,
integrating feature extraction, correlation, and test-time adaptation
into a cohesive tracking pipeline.
"""
from collections import deque
from statistics import mean
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
    """
    State-of-the-art Single Object Tracker based on SiamABCNet.

    This tracker implements a dual-template strategy with polarized attention
    and test-time adaptation (TTA) via Adaptive Batch Normalisation. It
    maintains a memory window of reliable target appearances to update the
    dynamic template branch.
    """
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
        self._init_rect = rect.copy()  # ← add this

        self.tracking_state.pred_score = 1.0
        self.prev_good_bbox = rect
        self.tracking_state.paths.clear()

        self.tracking_state.mean_color = np.mean(image, axis=(0, 1))

        self._template_features = self.get_template_features(image, rect)
        self.dynamic_template_features = self._template_features.clone()
        self.dynamic_search_features, _, _ = self.get_search_features(image, rect)

        self.all_memory_imgs = deque([[image, rect]], maxlen=self.memory_window_size)
        self.classification_scores = deque([0.5], maxlen=self.memory_window_size)

        self.running_dynamic_image = image
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
        self._is_full = False  # tracks whether deque has started evicti

    @torch.no_grad()
    def get_template_features(self, image, rect):
        """
        Extract features from a template crop.

        Args:
            image (NDArray): Full RGB image.
            rect (NDArray): Target bounding box [x, y, w, h].

        Returns:
            torch.Tensor: Encoded template features.
        """
        context = extend_bbox(rect,
                              offset=self.tracking_config["template_bbox_offset"],
                              image_width=image.shape[1], image_height=image.shape[0])
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            context=context,
            crop_size=self.tracking_config["template_size"],
        )

        img = self._preprocess_image(template_crop, self._template_transform)

        return self.net.get_features(img)

    @torch.no_grad()
    def get_search_features(self, image, bbox):
        """
        Extract features from a search region crop.

        Args:
            image (NDArray): Full RGB image.
            bbox (NDArray): Center bounding box for the search region.

        Returns:
            Tuple[torch.Tensor, NDArray, NDArray]:
                (features, search_bbox, search_context).
        """

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
        search_crop = self._preprocess_image(search_crop, self._search_transform)
        return self.net.get_features(search_crop), search_bbox, search_context

    def check_validity(self, bbox_window, bbox):
        """
        Check if a bounding box is within the specified window.

        Args:
            bbox_window (NDArray): Window boundaries [x1, y1, x2, y2].
            bbox (NDArray): Bounding box to check [x, y, w, h].

        Returns:
            bool: True if the box is within the window.
        """
        return bbox[0] >= bbox_window[0] and bbox[1] >= bbox_window[1] and bbox[2] <= bbox_window[2] and bbox[3] <= \
            bbox_window[3]

    def select_representatives_from_all(self):
        """
        Select a representative frame from the memory window to update templates.

        This method uses a mean-based selection strategy to find a frame with
        above-average classification confidence.
        """
        for num in range(len(self.classification_scores) - 1, -1, -1):
            if self.classification_scores[num] > mean(self.classification_scores):
                self.dynamic_template_features = self.get_template_features(self.all_memory_imgs[num][0],
                                                                            self.all_memory_imgs[num][1])
                self.dynamic_search_features, _, _ = self.get_search_features(self.all_memory_imgs[num][0],
                                                                              self.all_memory_imgs[num][1])
                return

    def _update_best_index(self, pred_score: float, evicting: bool) -> None:
        """
        O(1) amortized best-index maintenance.
        Only O(window_size) on the rare event the best element is evicted.
        Must be called AFTER appending to both deques.
        """
        if evicting:
            if self._best_idx == 0:
                # Best was just dropped — rescan (new element already in deque)
                scores = np.array(self.classification_scores, dtype=np.float16)
                self._best_idx = int(np.argmax(scores))
                self._best_score = float(scores[self._best_idx])
            else:
                # Best survived — shift index left to account for eviction
                self._best_idx -= 1
                # Still check if new element beats it
                if pred_score > self._best_score:
                    self._best_score = pred_score
                    self._best_idx = len(self.classification_scores) - 1
        else:
            # No eviction — just check new element
            if pred_score > self._best_score:
                self._best_score = pred_score
                self._best_idx = len(self.classification_scores) - 1

    def _maybe_store_frame(self, search: np.ndarray, pred_bbox: np.ndarray, pred_score: float) -> None:
        """Append to memory and keep best-index consistent. Single responsibility."""
        evicting = len(self.classification_scores) == self.memory_window_size
        self.all_memory_imgs.append([search, pred_bbox])
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

        best_img, best_bbox = self.all_memory_imgs[self._best_idx]
        self.dynamic_template_features = self.get_template_features(best_img, best_bbox)
        self.dynamic_search_features, _, _ = self.get_search_features(best_img, best_bbox)
        self.running_dynamic_image = best_img.copy()
        self.running_dynamic_bbox = best_bbox.copy()

        if hasattr(self.net, "invalidate_template_cache"):
            self.net.invalidate_template_cache()

    def update(self, search: NDArray, *kw):
        """
        Process a new frame and update the tracker state.

        Args:
            search (NDArray): RGB image of the new frame.
            *kw: Additional keyword arguments.

        Returns:
            Tuple[NDArray, float, float]: (pred_bbox, pred_score, sim_score).
        """
        pred_bbox, pred_score, sim_score = self.run_track(search)

        # Always update state
        self.tracking_state.bbox = pred_bbox
        self.tracking_state.pred_score = pred_score
        self.tracking_state.paths.append(pred_bbox)

        if self.dynamic_update:
            score_ok = (
                pred_score > self.running_confidence or
                pred_score >= self.dynamic_update_threshold
            )
            if score_ok:
                iou_ok = (
                    self._compute_iou(pred_bbox, self._prev_bbox)
                    >= self.tracking_config.get("iou_threshold", 0.3)
                )
                if iou_ok:
                    self._maybe_store_frame(search, pred_bbox, pred_score)

            self._prev_bbox = pred_bbox.copy()
            self.idx += 1

            if self.idx % self.N == 0:
                self.select_representatives()

            # Single EMA update, then clamp — fixes double-update bug
            self.running_confidence = (
                self.update_lambda * pred_score
                + (1 - self.update_lambda) * self.running_confidence
            )
            self.running_confidence = min(self.running_confidence, self.running_confidence_floor_value)

        return pred_bbox, pred_score, sim_score

    def run_track(self, search):
        """
        Execute tracking on the given search image.

        Args:
            search (NDArray): RGB image of the search region.

        Returns:
            Tuple[NDArray, float, float]: (pred_bbox, pred_score, sim_score).
        """
        search_features, search_bbox, search_context = self.get_search_features(search, self.tracking_state.bbox)
        self.tracking_state.mapping = search_context
        self.tracking_state.prev_size = search_bbox[2:]
        pred_bbox, pred_score, sim_score, _ = self.track(search_features, self.dynamic_search_features,
                                                         self.dynamic_template_features)
        pred_bbox = self._rescale_bbox(pred_bbox, self.tracking_state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, search.shape)
        return pred_bbox, pred_score, sim_score

    @torch.no_grad()
    def track(self, search_features, dynamic_search_features, dynamic_template_features):
        """
        Perform a model forward pass using pre-extracted features.

        Args:
            search_features (torch.Tensor): Features from the current search region.
            dynamic_search_features (torch.Tensor): Features from the dynamic search region.
            dynamic_template_features (torch.Tensor): Features from the dynamic template.

        Returns:
            Tuple[NDArray, float, float, torch.Tensor]: Post-processed results.
        """
        track_result = self.net.track(
            search_features=search_features,
            dynamic_search_features=dynamic_search_features,
            template_features=self._template_features,
            dynamic_template_features=dynamic_template_features
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

        # Use the same context ratio as normal tracking so the feature extractor
        # sees the same relative amount of context it was trained with.
        context_ratio = self.tracking_config["search_context"]  # e.g. 2.0

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
            # search_context stays UNCHANGED — already the right ratio
            return self.run_track(search)
        finally:
            self.tracking_state.bbox = saved_bbox

    def set_tta(self, enabled: bool) -> None:
        """
        Enable or disable Test-Time Adaptation (Adaptive Batch Normalisation).

        Args:
            enabled (bool): Whether to enable TTA.
        """
        for module in self.net.modules():
            if isinstance(module, (AdaptiveBatchNorm)):
                if enabled:
                    module.enable_tta()
                else:
                    module.disable_tta()

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

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        ix1 = max(boxA[0], boxB[0])
        iy1 = max(boxA[1], boxB[1])
        ix2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        iy2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        return inter / union if union > 0 else 0.0
