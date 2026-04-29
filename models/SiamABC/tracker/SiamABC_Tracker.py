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
    def __init__(self, model, cuda_id=0, **tracking_config):
        super().__init__(model, cuda_id, **tracking_config)

        self._norm_lambda_tta: float = next(
            (m._norm_lambda for m in self.net.modules() if isinstance(m, AdaptiveBatchNorm)),
            0.1,
        )
        self._tta_lam: torch.Tensor = torch.zeros(1, device=f"cuda:{cuda_id}")

    def get_box_coder(self, tracking_config, cuda_id: str | int = 0):
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

        # Compute mean_color once here so we can pass it everywhere — including
        # get_template_features — and avoid re-computing np.mean over the full
        # frame inside get_extended_crop (which was the previous behaviour when
        # padding_value was not supplied to that call).
        self.tracking_state.mean_color = np.mean(image, axis=(0, 1))

        self._template_features, _ = self.get_template_features(image, rect)
        self.dynamic_template_features = self._template_features.clone()
        search_feats, search_bbox_in_crop, _, search_crop_raw = self.get_search_features(image, rect)
        self.dynamic_search_features = search_feats

        self.warmup_frames = self.tracking_config.get("warmup_frames", 0)
        warmup_window = self.tracking_config.get("warmup_window_size", self.memory_window_size)
        init_window = warmup_window if self.warmup_frames > 0 else self.memory_window_size

        # Memory stores [search_crop_256, search_bbox_in_crop, frame_bbox].
        # search_bbox_in_crop lets select_representatives re-derive a template crop
        # from the small 256x256 image — no full-res frame ever stored or re-read.
        self.all_memory_imgs = deque(
            [[search_crop_raw, search_bbox_in_crop, rect]],
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
            # FIX: pass pre-computed mean_color so get_extended_crop never falls
            # back to np.mean(image, axis=(0,1)) — an O(resolution) operation on
            # the full raw frame.  mean_color is always set before this call in
            # initialize(), so tracking_state.mean_color is always valid here.
            padding_value=self.tracking_state.mean_color,
        )
        img = self._preprocess_image(template_crop, self._template_transform)
        return self.net.get_features(img), template_crop

    @torch.no_grad()
    def get_search_features(self, image, bbox):
        """
        Returns (features, search_bbox_in_crop, search_context, raw_search_crop_numpy).

        search_bbox_in_crop is the tracked object bbox scaled into 256x256 crop space.
        Storing it alongside the crop allows select_representatives to re-derive a
        128x128 template crop from the small image only — never the full-res frame.
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
        search_tensor = self._preprocess_image(search_crop, self._search_transform)
        return self.net.get_features(search_tensor), search_bbox, search_context, search_crop

    @torch.no_grad()
    def _get_features_from_crop(self, crop: np.ndarray):
        """Run net.get_features on an already-resized small crop (128 or 256 px)."""
        tensor = self._preprocess_image(crop, None)
        return self.net.get_features(tensor)

    def _extract_template_from_search_crop(
        self, search_crop: np.ndarray, bbox_in_crop: np.ndarray
    ) -> np.ndarray:
        """
        Re-derive a 128x128 template crop from the stored 256x256 search crop.

        Source area: 256x256 = 65 K px.
        Full-res 4K source area: ~8 M px.
        This is ~120x cheaper than cropping from the original frame, and happens
        only every N=10 frames in select_representatives — not on the per-frame path.
        """
        context = extend_bbox(
            bbox_in_crop,
            offset=self.tracking_config["template_bbox_offset"],
            image_width=search_crop.shape[1],
            image_height=search_crop.shape[0],
        )
        template_crop, _, _ = get_extended_crop(
            image=search_crop,
            bbox=bbox_in_crop,
            context=context,
            crop_size=self.tracking_config["template_size"],  # 128
        )
        return template_crop

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
    ) -> None:
        """
        Zero extra computation on the per-frame hot path.

        Both arguments are already computed during run_track — this is purely
        appending references that already exist. No crop, no resize, no copy.

        Memory: 256x256x3 uint8 ≈ 196 KB per slot.
        window_size=20 → ~4 MB total vs ~500 MB in the original.
        """
        evicting = len(self.classification_scores) == self.memory_window_size
        self.all_memory_imgs.append([search_crop_raw, search_bbox_in_crop, pred_bbox])
        self.classification_scores.append(pred_score)
        self._update_best_index(pred_score, evicting)

    def select_representatives(self) -> None:
        if not self.classification_scores:
            return
        if self._best_score < self.dynamic_update_threshold:
            return

        best_search_crop, best_bbox_in_crop, best_bbox = self.all_memory_imgs[self._best_idx]

        # Cheap crop from 256x256 source — ~120x less work than cropping a 4K frame.
        # This runs every N=10 frames; with the previous bug it paid full-res crop cost here.
        best_template_crop = self._extract_template_from_search_crop(best_search_crop, best_bbox_in_crop)

        self.dynamic_template_features = self._get_features_from_crop(best_template_crop)
        self.dynamic_search_features = self._get_features_from_crop(best_search_crop)

        self.running_dynamic_image = best_search_crop   # 256x256 ref, not a 25 MB memcpy
        self.running_dynamic_bbox = best_bbox.copy()

        if hasattr(self.net, "invalidate_template_cache"):
            self.net.invalidate_template_cache()

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------

    def update(self, search: NDArray, *kw):
        pred_bbox, pred_score, sim_score, search_crop_raw, search_bbox_in_crop = self.run_track(search)

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
                    # Both values already computed in run_track — storing them is free.
                    self._maybe_store_frame(search_crop_raw, search_bbox_in_crop, pred_bbox, pred_score)

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
        Returns (pred_bbox, pred_score, sim_score, search_crop_raw, search_bbox_in_crop).

        search_crop_raw and search_bbox_in_crop are free byproducts of get_search_features.
        Returning them here means update() can call _maybe_store_frame at zero extra cost.
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
        return pred_bbox, pred_score, sim_score, search_crop_raw, search_bbox_in_crop

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
            pred_bbox, pred_score, sim_score, _, _ = self.run_track(search)
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
        self.set_tta(True)

    def disable_tta(self) -> None:
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