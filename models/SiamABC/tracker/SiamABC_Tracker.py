from collections import deque
from typing import Dict, Tuple, Union

import numpy as np
import torch
from numpy._typing import NDArray

from utils.box_coder import SiamABCBoxCoder, TrackerDecodeResult
from utils.utils import clamp_bbox, extend_bbox, get_extended_crop
from .base_tracker import Tracker
from ..model import constants
from ..model.adaptive_batch_norm import AdaptiveBatchNorm


class SiamABCTracker(Tracker):
    """
    Siamese Actor-Critic based Tracker.

    This tracker uses dynamic templates and memory windows to maintain tracking robustly.
    It manages the tracking state, handles dynamic updates, and coordinates with
    a box coder to decode network outputs into bounding boxes.
    """

    def __init__(
        self,
        model,
        cuda_id=0,
        **tracking_config,
    ):
        """
        Initialize the SiamABCTracker.

        Args:
            model (torch.nn.Module): The underlying SiamABC neural network model.
            cuda_id (int, optional): The ID of the CUDA device to use. Defaults to 0.
            **tracking_config: Dictionary containing arbitrary tracking configurations.
        """
        super().__init__(model, cuda_id, **tracking_config)

        self._norm_lambda_tta: float = next(
            (
                m._norm_lambda
                for m in self.net.modules()
                if isinstance(m, AdaptiveBatchNorm)
            ),
            0.1,
        )

        self._tta_lam: torch.Tensor = torch.zeros(1, device=f"cuda:{cuda_id}")

    def get_box_coder(
        self,
        tracking_config,
        cuda_id: str | int = 0,
    ):
        """
        Create and return the bounding box coder for decoding network outputs.

        Args:
            tracking_config (dict): The configuration dictionary for the tracker.
            cuda_id (str | int, optional): The CUDA device ID. Defaults to 0.

        Returns:
            SiamABCBoxCoder: An instance of the box coder initialized with the configuration.
        """
        return SiamABCBoxCoder(tracking_config)

    def initialize(
        self,
        image: NDArray,
        rect: NDArray,
        **kwargs,
    ) -> None:
        """
        Initialize the tracker state with the first frame and bounding box.

        This sets up the initial template and search features by calling
        `get_template_features()` and `get_search_features()`. It also resets
        history buffers and tracking states.

        Args:
            image (NDArray): The initial RGB image frame.
            rect (NDArray): The initial bounding box as [x, y, width, height],
                where x and y are 0-based.
            **kwargs: Additional arguments.

        Returns:
            None
        """

        self.smooth_pred = self.tracking_config["smooth"]
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
        self.slection_method = "mean"

        self.update_lambda = 0.1
        self.running_confidence = 0.5

        self._best_idx = 0
        self._best_score = 0.5
        self._is_full = False

    @torch.no_grad()
    def get_template_features(
        self,
        image,
        rect,
    ):
        """
        Extract template features for a specific bounding box in an image.

        Args:
            image (NDArray): The source image array.
            rect (NDArray): The bounding box [x, y, w, h] indicating the template region.

        Returns:
            torch.Tensor: The extracted features for the template crop.
        """
        context = extend_bbox(
            rect,
            offset=self.tracking_config["template_bbox_offset"],
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            context=context,
            crop_size=self.tracking_config["template_size"],
        )

        img = self._preprocess_image(template_crop, self._template_transform)

        return self.net.get_features(img)

    @torch.no_grad()
    def get_search_features(
        self,
        image,
        bbox,
    ):
        """
        Extract search features and compute search context mapping for a bounding box.

        Args:
            image (NDArray): The search image array.
            bbox (NDArray): The bounding box [x, y, w, h] to extract features around.

        Returns:
            Tuple[torch.Tensor, NDArray, NDArray]:
                - Extracted features for the search crop.
                - The scaled bounding box within the crop.
                - The padded search context used for mapping coordinates back.
        """

        context = extend_bbox(
            bbox,
            offset=self.tracking_config["search_context"],
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

        search_crop, search_bbox, search_context = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=self.tracking_config["instance_size"],
            padding_value=self.tracking_state.mean_color,
            context=context,
        )
        search_crop = self._preprocess_image(search_crop, self._search_transform)
        return self.net.get_features(search_crop), search_bbox, search_context

    def check_validity(
        self,
        bbox_window,
        bbox,
    ):
        """
        Check if a bounding box is fully contained within another window bounding box.

        Args:
            bbox_window (Sequence[float]): The reference bounding box [x, y, right, bottom].
            bbox (Sequence[float]): The bounding box to check [x, y, right, bottom].

        Returns:
            bool: True if `bbox` is inside `bbox_window`, False otherwise.
        """
        return (
            bbox[0] >= bbox_window[0]
            and bbox[1] >= bbox_window[1]
            and bbox[2] <= bbox_window[2]
            and bbox[3] <= bbox_window[3]
        )

    def _update_best_index(
        self,
        pred_score: float,
        evicting: bool,
    ) -> None:
        """
        Update the index of the best score in the classification memory window.

        Maintains O(1) amortized tracking of the highest score index.
        Called by `_maybe_store_frame()`.

        Args:
            pred_score (float): The newly added classification score.
            evicting (bool): Whether the memory deque has reached capacity and is evicting the oldest element.

        Returns:
            None
        """
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
        search: np.ndarray,
        pred_bbox: np.ndarray,
        pred_score: float,
    ) -> None:
        """
        Optionally store the current frame in memory if conditions are met.

        This appends the frame to `all_memory_imgs`, the score to `classification_scores`,
        and manages the best index using `_update_best_index()`.

        Args:
            search (np.ndarray): The current image frame.
            pred_bbox (np.ndarray): The predicted bounding box.
            pred_score (float): The prediction confidence score.

        Returns:
            None
        """
        evicting = len(self.classification_scores) == self.memory_window_size
        self.all_memory_imgs.append([search, pred_bbox])
        self.classification_scores.append(pred_score)
        self._update_best_index(pred_score, evicting)

    def select_representatives(
        self,
    ) -> None:
        """
        Select the best memory frame and update dynamic tracking features.

        Calls `get_template_features()` and `get_search_features()` on the best
        stored frame to update dynamic representative features if its score exceeds
        `dynamic_update_threshold`.

        Returns:
            None
        """
        if not self.classification_scores:
            return
        if self._best_score < self.dynamic_update_threshold:
            return
        best_img, best_bbox = self.all_memory_imgs[self._best_idx]
        self.dynamic_template_features = self.get_template_features(best_img, best_bbox)
        self.dynamic_search_features, _, _ = self.get_search_features(
            best_img, best_bbox
        )
        self.running_dynamic_image = best_img.copy()
        self.running_dynamic_bbox = best_bbox.copy()

        if hasattr(self.net, "invalidate_template_cache"):
            self.net.invalidate_template_cache()

    def update(
        self,
        search: NDArray,
        *kw,
    ):
        """
        Update the tracker using the current frame, generating the next bounding box.

        This invokes `run_track()` to perform the tracking step, evaluates conditions
        like IoU with `_compute_iou()` and stores confident frames using
        `_maybe_store_frame()`. Finally, it occasionally updates dynamic features
        via `select_representatives()`.

        Args:
            search (NDArray): The current image frame array.
            *kw: Additional unused positional arguments.

        Returns:
            Tuple[NDArray, float, float]:
                - The predicted bounding box.
                - The predicted classification score.
                - The similarity score.
        """
        pred_bbox, pred_score, sim_score = self.run_track(search)

        self.tracking_state.bbox = pred_bbox
        self.tracking_state.pred_score = pred_score
        self.tracking_state.paths.append(pred_bbox)
        if self.dynamic_update:
            score_ok = (
                pred_score > self.running_confidence
                or pred_score >= self.dynamic_update_threshold
            )
            if score_ok:
                iou_ok = self._compute_iou(
                    pred_bbox, self._prev_bbox
                ) >= self.tracking_config.get("iou_threshold", 0.3)
                if iou_ok:
                    self._maybe_store_frame(search, pred_bbox, pred_score)

            self._prev_bbox = pred_bbox.copy()

            if self.warmup_frames > 0 and self.idx == self.warmup_frames - 1:
                full_imgs = deque(self.all_memory_imgs, maxlen=self.memory_window_size)
                full_scores = deque(
                    self.classification_scores, maxlen=self.memory_window_size
                )
                self.all_memory_imgs = full_imgs
                self.classification_scores = full_scores
                scores = np.array(self.classification_scores, dtype=np.float16)
                self._best_idx = int(np.argmax(scores))
                self._best_score = float(scores[self._best_idx])

            self.idx += 1

            if self.idx % self.N == 0:
                self.select_representatives()

            self.running_confidence = (
                self.update_lambda * pred_score
                + (1 - self.update_lambda) * self.running_confidence
            )
            self.running_confidence = min(
                self.running_confidence, self.running_confidence_floor_value
            )

        return pred_bbox, pred_score, sim_score

    def run_track(
        self,
        search,
    ):
        """
        Run the fundamental tracking sequence for a search frame.

        Extracts search features using `get_search_features()`, forwards them to `track()`,
        and rescales the output bounding box back to image coordinates.

        Args:
            search (NDArray): The search image frame array.

        Returns:
            Tuple[NDArray, float, float]:
                - The predicted bounding box in image coordinates.
                - The classification score.
                - The similarity score.
        """
        search_features, search_bbox, search_context = self.get_search_features(
            search, self.tracking_state.bbox
        )
        self.tracking_state.mapping = search_context
        self.tracking_state.prev_size = search_bbox[2:]
        pred_bbox, pred_score, sim_score, _ = self.track(
            search_features,
            self.dynamic_search_features,
            self.dynamic_template_features,
        )
        pred_bbox = self._rescale_bbox(pred_bbox, self.tracking_state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, search.shape)
        return pred_bbox, pred_score, sim_score

    @torch.no_grad()
    def track(
        self,
        search_features,
        dynamic_search_features,
        dynamic_template_features,
    ):
        """
        Execute the core neural network tracking operation and postprocess results.

        Passes features to `self.net.track()` and hands the raw output to `_postprocess()`.

        Args:
            search_features (torch.Tensor): Current frame search features.
            dynamic_search_features (torch.Tensor): Best historical search features.
            dynamic_template_features (torch.Tensor): Best historical template features.

        Returns:
            Tuple[NDArray, float, float, torch.Tensor]: Outputs derived from `_postprocess()`.
        """
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
        track_result: Dict[str, torch.Tensor],
    ) -> Tuple[NDArray, float, Union[int, float, bool], torch.Tensor]:
        """
        Post-process raw tracking network output into actionable bounding boxes and scores.

        Uses the box coder to decode classification and regression maps, applying
        penalties as necessary.

        Args:
            track_result (Dict[str, torch.Tensor]): The raw output tensor dictionary from the tracking network.

        Returns:
            Tuple[NDArray, float, Union[int, float, bool], torch.Tensor]:
                - The predicted bounding box within the crop.
                - The maximum classification score.
                - The similarity score.
                - The attention map.
        """
        cls_score = track_result[constants.TARGET_CLASSIFICATION_KEY].float().sigmoid()
        regression_map = (
            track_result[constants.TARGET_REGRESSION_LABEL_KEY].detach().float()
        )
        classification_map, penalty, pred_location = self._confidence_postprocess(
            cls_score=cls_score, regression_map=regression_map
        )

        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=classification_map,
            regression_map=track_result[constants.TARGET_REGRESSION_LABEL_KEY],
            use_sigmoid=False,
            pred_location=pred_location,
        )

        cls_score = cls_score.squeeze().cpu().numpy()
        pred_bbox = self._postprocess_bbox(
            decoded_info=decoded_info, cls_score=cls_score, penalty=penalty
        )
        r_max, c_max = decoded_info.pred_coords[0]
        sim_score_raw = track_result[constants.TRACKER_TARGET_SEARCH_SIM_SCORE]
        sim_score = sim_score_raw.item() if sim_score_raw is not None else 0.0
        return (
            pred_bbox,
            cls_score[r_max, c_max].item(),
            sim_score,
            track_result[constants.TRACKER_ATTENTION_MAP],
        )

    def run_track_for_candidate(
        self,
        search: np.ndarray,
        candidate_bbox: np.ndarray,
    ):
        """
        Temporarily set tracking state to a candidate bbox and run tracking.

        Evaluates a hypothesis candidate by injecting its bounding box into
        tracking state and invoking `run_track()`. Restores state afterward.

        Args:
            search (np.ndarray): The search image frame array.
            candidate_bbox (np.ndarray): The hypothesized bounding box [x, y, w, h].

        Returns:
            Tuple[NDArray, float, float]: The predicted bounding box, score, and
            similarity score as returned by `run_track()`.
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

            return self.run_track(search)
        finally:
            self.tracking_state.bbox = saved_bbox

    def set_tta(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable Test-Time Augmentation (TTA).

        Updates the `_tta_lam` tensor parameter.

        Args:
            enabled (bool): Whether TTA should be enabled.

        Returns:
            None
        """
        val = self._norm_lambda_tta if enabled else 0.0
        self._tta_lam.fill_(val)

    def enable_tta(
        self,
    ) -> None:
        """
        Enable Test-Time Augmentation by calling `set_tta(True)`.

        Returns:
            None
        """
        self.set_tta(True)

    def disable_tta(
        self,
    ) -> None:
        """
        Disable Test-Time Augmentation by calling `set_tta(False)`.

        Returns:
            None
        """
        self.set_tta(False)

    @staticmethod
    def _compute_iou(
        boxA,
        boxB,
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            boxA (Sequence[float]): First bounding box [x, y, w, h].
            boxB (Sequence[float]): Second bounding box [x, y, w, h].

        Returns:
            float: The IoU ratio ranging from 0.0 to 1.0.
        """
        ix1 = max(boxA[0], boxB[0])
        iy1 = max(boxA[1], boxB[1])
        ix2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        iy2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        return inter / union if union > 0 else 0.0
