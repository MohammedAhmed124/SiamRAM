from collections import deque
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from numpy._typing import NDArray

from utils.box_coder import SiamABCBoxCoder, TrackerDecodeResult
from utils.utils import clamp_bbox, extend_bbox, get_extended_crop

from ..model import constants
from ..model.adaptive_batch_norm import AdaptiveBatchNorm
from .base_tracker import Tracker


# Process-wide latch: the DRM-introspection signal shape/sample is logged at most
# once per process (and only when research telemetry is enabled) so a run visibly
# confirms the secondary-peak signal is live rather than stuck at its inert 1.0
# default. See the DRM block in track().
_DRM_INTROSPECTION_SHAPE_LOGGED = False


class SiamABCTracker(Tracker):
    """
    The main tracker class for SiamRAM — built on top of the SiamABC core.

    This is the heart of the tracking system described in the SiamRAM paper.
    During normal operation (Section 2 of the paper), this tracker runs every
    frame: it extracts features from both the search region and the template,
    fuses them through an attention mechanism, and predicts where the target
    object is in the current frame.

    What makes this tracker smarter than a basic Siamese tracker is that it
    doesn't just rely on the very first frame's template forever. Instead, it
    maintains a rolling memory window of the most recent high-quality frames
    (Section 2.2 of the paper) and continuously updates what it calls the
    "dynamic template" — a fresh, high-confidence snapshot of the target that
    replaces the static first-frame crop whenever the tracker is confident enough.

    The tracker also supports Test-Time Augmentation (TTA) via Adaptive Batch
    Normalization (Section 2.3), which helps it handle gradual illumination
    changes and sensor noise without any retraining.

    In the broader SiamRAM pipeline, this class is the engine that the occlusion
    recovery and reacquisition phases (Section 4) call back into once they've
    found a candidate location for the lost target.

    Attributes:
        _norm_lambda_tta (float): The lambda value used to modulate Adaptive
            Batch Norm behavior during TTA. Pulled directly from the model's
            AdaptiveBatchNorm layers at init time.
        _tta_lam (torch.Tensor): A scalar tensor that acts as the live TTA
            switch. Setting it to `_norm_lambda_tta` turns TTA on; zeroing it
            turns TTA off. This tensor is passed directly into the network's
            forward pass so the change takes effect immediately.
    """

    def __init__(
        self,
        model,
        cuda_id=0,
        **tracking_config,
    ):
        """
        Set up the tracker and wire in the TTA (Test-Time Augmentation) machinery.

        This constructor does two things beyond the standard base-class setup:

        1. It scans the model's layers for an AdaptiveBatchNorm module and grabs
           its `_norm_lambda` value. This value controls how aggressively the
           batch-norm statistics adapt at test time (see Section 2.3 of the paper).
           If no AdaptiveBatchNorm layer is found, it safely defaults to 0.1.

        2. It initialises `_tta_lam` as a zero tensor on the target GPU. A value
           of zero means TTA is OFF by default — the tracker runs in standard mode
           until `enable_tta()` is explicitly called.

        Args:
            model (torch.nn.Module): The fully loaded SiamABC neural network. This
                should already be moved to the correct device before being passed in.
            cuda_id (int, optional): Which GPU to run on. Defaults to 0. Used to
                place the `_tta_lam` tensor on the right device from the start.
            **tracking_config: A flat dictionary of all tracker hyperparameters —
                things like memory window size, score thresholds, IoU gates, search
                context ratios, etc. These come from a config file and are passed
                through to the base class and then used throughout tracking.
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
        Build and return the bounding box decoder that the tracker relies on.

        In the SiamABC architecture, the network doesn't output a bounding box
        directly. Instead, it produces two raw maps: a classification map (which
        spatial locations look like the target?) and a regression map (by how much
        should each candidate location's default box be offset to fit the target?).

        The SiamABCBoxCoder is the component that takes those raw maps and converts
        them into a final [x, y, w, h] bounding box in image coordinates. It applies
        the cosine penalty window (to suppress false positives near the search region
        boundary) and handles all the anchor-free decoding math.

        This method is called once during base-class initialisation, so the returned
        coder is available as `self.box_coder` throughout the tracker's lifetime.

        Args:
            tracking_config (dict): The full tracking config dictionary. The box coder
                reads things like grid stride, penalty weights, and score normalization
                settings from here.
            cuda_id (str | int, optional): The GPU index. Passed to the coder so it
                can place intermediate tensors on the right device. Defaults to 0.

        Returns:
            SiamABCBoxCoder: A fully configured box decoder instance, ready to
                process network outputs frame by frame.
        """
        return SiamABCBoxCoder(tracking_config)

    def initialize(
        self,
        image: NDArray,
        rect: NDArray,
        **kwargs,
    ) -> None:
        """
        Cold-start the tracker on the very first frame of a sequence.

        This is called exactly once per video sequence, before any calls to
        `update()`. Everything that the tracker needs to know about the target's
        initial appearance is set up here.

        Concretely, this method:

        - Reads all the relevant thresholds and flags from the config (memory
          window size N, optional warm_up_n cadence, dynamic update on/off,
          similarity score mode, etc.)
        - Clamps the given bounding box so it can't spill outside the image frame.
        - Extracts the static template features from the initial crop — these
          features are kept frozen for the entire sequence and represent the
          "ground truth" appearance of the target at t=0.
        - Also builds the initial dynamic template and dynamic search features,
          which start as copies of the static ones but get updated as tracking
          proceeds (see Section 2.2 of the paper).
        - Initialises the memory deque (`all_memory_imgs`) with the first frame
          and its ground-truth box, so the memory window starts populated.
        - Resets all counters, confidence trackers, and score history so the
          tracker is in a clean state.

        After this call, the tracker is fully ready to receive frames via `update()`.

        Args:
            image (NDArray): The RGB image array for the very first frame (H x W x 3,
                uint8). This is the frame where the target is known with certainty.
            rect (NDArray): The initial ground-truth bounding box as [x, y, width, height]
                in 0-based pixel coordinates. This is provided by the benchmark or the
                human annotator and is assumed to be accurate.
            **kwargs: Reserved for future use. Not currently read.

        Returns:
            None
        """

        self.smooth_pred = self.tracking_config["smooth"]
        self._prev_bbox = rect.copy()
        self.N = max(1, int(self.tracking_config["N"]))
        raw_warm_up_n = self.tracking_config.get("warm_up_n", self.N)
        self.warm_up_n = max(1, int(raw_warm_up_n))
        raw_warmup_window_size = self.tracking_config.get(
            "warmup_window_size", self.memory_window_size
        )
        self.warmup_window_size = max(1, int(raw_warmup_window_size))
        self.dynamic_update = self.tracking_config["dynamic_update"]
        self.similarity_score = bool(self.tracking_config.get("similarity_score", False))
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
        self.use_iou_score = bool(
            self.tracking_config.get(
                "use_iou_score",
                self.tracking_config.get("use_quality_score", False),
            )
        )
        self.iou_power = float(
            self.tracking_config.get(
                "iou_power",
                self.tracking_config.get("quality_power", 1.0),
            )
        )
        self.use_iou_gate = bool(self.tracking_config.get("use_iou_gate", False))
        self.iou_gate_threshold = float(
            self.tracking_config.get("iou_gate_threshold", 0.35)
        )
        self.use_iou_gate_for_score = bool(
            self.tracking_config.get("use_iou_gate_for_score", False)
        )
        self.iou_gate_fail_closed = bool(
            self.tracking_config.get("iou_gate_fail_closed", False)
        )
        self.iou_gate_warmup_frames = int(
            self.tracking_config.get("iou_gate_warmup_frames", 20)
        )
        self.iou_gate_ema_beta = float(
            self.tracking_config.get("iou_gate_ema_beta", 0.9)
        )
        self.iou_gate_use_ema = bool(
            self.tracking_config.get("iou_gate_use_ema", True)
        )
        self.latest_iou_score = 1.0
        self.latest_iou_score_ema = 1.0
        self._last_iou_gate_pass = True

        # Introspection-based DRM update (arXiv:2411.17576). The SiamRAM tracker
        # sets ``_drm_introspection_enabled = True`` on this instance only when the
        # feature is enabled; otherwise the response-map secondary-peak signal is
        # never computed (legacy path untouched) and these inert defaults remain.
        self._drm_introspection_enabled = False
        # Hypothesis-divergence ratio (primary/union bbox-area) and the primary /
        # secondary response-map peak strengths from the most recent frame.
        self.latest_secondary_peak_ratio = 1.0
        self.latest_primary_peak_score = 0.0
        self.latest_secondary_peak_score = 0.0
        # Secondary-peak (distractor) descriptor write-back into the NEGATIVE DRM
        # bank. The SiamRAM tracker flips ``_drm_write_distractor_bank = True`` only
        # when the distractor-bank sub-feature is enabled; otherwise the frame-space
        # secondary-peak bbox is never computed and this inert default remains.
        self.latest_secondary_peak_bbox = None
        self._drm_write_distractor_bank = False

        # Research telemetry (arXiv features). The SiamRAM tracker flips
        # ``_research_telemetry_enabled = True`` and stashes a ``_tlm_inner`` dict
        # here only when telemetry is enabled; otherwise these inert defaults mean
        # every telemetry site (DRM secondary-peak counts) is skipped and
        # behaviour is byte-for-byte unchanged.
        self._research_telemetry_enabled = False
        self._tlm_inner = None

        raw_floor = float(
            self.tracking_config.get("running_confidence_floor_value", 0.6)
        )
        if raw_floor > 1.0:
            raw_floor /= 100.0
        self.running_confidence_floor_value = float(np.clip(raw_floor, 0.0, 1.0))

        self._best_idx = 0
        self._best_score = 0.5
        self._is_full = False

    def _in_warmup_phase(self) -> bool:
        """True while update() is still inside the warm-up frame budget."""
        return self.warmup_frames > 0 and self.idx < self.warmup_frames

    def _current_template_update_period(self) -> int:
        """
        Return the current select_representatives() cadence.

        During the initial warm-up region, we can use a dedicated cadence
        (`warm_up_n`) to refresh dynamic templates faster/slower than the
        steady-state cadence (`N`). Outside warm-up we always use `N`.
        """
        if self._in_warmup_phase():
            return self.warm_up_n
        return self.N

    def _current_template_selection_window(self) -> int:
        """
        Return how many recent memory entries may be considered for template refresh.

        During warm-up, selection is restricted to the latest
        `warmup_window_size` entries to encourage fast adaptation to recent
        appearance. After warm-up, the full memory window is eligible.
        """
        if self._in_warmup_phase():
            return max(1, min(int(self.warmup_window_size), int(self.memory_window_size)))
        return max(1, int(self.memory_window_size))

    @torch.no_grad()
    def get_template_features(
        self,
        image,
        rect,
    ):
        """
        Crop out the target region from an image and run it through the feature extractor.

        The SiamABC architecture (Section 2.1) always has a "template branch" and
        a "search branch". This method handles the template side: given an image and
        a bounding box, it adds some padding around the box (controlled by
        `template_bbox_offset` in the config), crops out that padded region, resizes
        it to the standard template size, and then passes it through the shared
        ResNet backbone + neck to get a feature tensor.

        This is used in two contexts:
        1. At initialisation — to extract the static first-frame template features
           that remain frozen for the whole sequence.
        2. During dynamic template updates (`select_representatives()`) — to refresh
           the dynamic template features from the best high-confidence frame stored
           in memory.

        Running under `@torch.no_grad()` because we never backprop through tracking.

        Args:
            image (NDArray): The full source frame (H x W x 3). Can be any frame
                from the sequence — first frame, or a stored memory frame.
            rect (NDArray): The bounding box [x, y, w, h] indicating where the
                target is in `image`. Used as the centre of the template crop.

        Returns:
            torch.Tensor: The feature map extracted from the template crop. Shape
                depends on the backbone/neck configuration, but this tensor is
                what gets passed into the attention fusion module during tracking.
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
        Extract features from the search region centred around a predicted bounding box.

        While the template tells the network "what the target looks like", the search
        region tells it "where to look in the current frame". This method crops out a
        larger region centred on the current bounding box estimate (using the
        `search_context` ratio from config to decide how much padding to add), resizes
        it to the standard instance size, and extracts features from it.

        Critically, this method also returns the spatial mapping information needed to
        translate predicted coordinates inside the crop back to full image coordinates.
        That mapping (the `search_context` array) is saved into `tracking_state.mapping`
        during `run_track()` and later used by `_rescale_bbox()`.

        Used in three places:
        1. At initialisation — to set the initial dynamic search features.
        2. Every frame in `run_track()` — to process the live search region.
        3. During `select_representatives()` — to refresh dynamic search features
           from the best memory frame.

        Running under `@torch.no_grad()` because we never backprop through tracking.

        Args:
            image (NDArray): The full current frame (H x W x 3).
            bbox (NDArray): The current bounding box estimate [x, y, w, h] — used as
                the centre of the search crop. During normal tracking this is the
                most recent prediction.

        Returns:
            Tuple[torch.Tensor, NDArray, NDArray]:
                - The feature tensor extracted from the search crop.
                - The scaled bounding box of the target within the crop (useful for
                  computing relative offsets during decoding).
                - The padded search context box in full image coordinates — this is
                  the "mapping" needed to invert the crop transform and recover
                  predicted coordinates in the original image space.
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
        Check whether a candidate bounding box is fully enclosed within a reference window.

        This is a spatial containment test used to validate that a predicted or
        candidate bounding box doesn't stray outside a defined region of interest.
        In the context of SiamRAM's recovery pipeline (Section 4), this kind of
        check guards against accepting a "recovered" box that is actually sitting
        outside the expected search window — which would be a sign of a false match
        rather than a genuine reacquisition.

        Both boxes are expected in [x, y, right, bottom] (i.e. corner format, not
        [x, y, w, h]), so the caller is responsible for converting if needed.

        Args:
            bbox_window (Sequence[float]): The outer boundary box expressed as
                [x_left, y_top, x_right, y_bottom]. Everything must stay inside this.
            bbox (Sequence[float]): The box to validate, also as
                [x_left, y_top, x_right, y_bottom].

        Returns:
            bool: True if `bbox` is completely inside `bbox_window` (all four edges
                are within bounds). False if any edge pokes outside.
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
        Keep track of which memory slot currently holds the highest-scoring frame.

        The memory window (`all_memory_imgs`) is a fixed-size deque. As new frames
        are pushed in, old ones fall off the left end. The tracker needs to always
        know which slot holds the best frame — because that's the one it will use
        to refresh the dynamic template on the configured cadence
        (warm-up `warm_up_n`, then steady-state `N`; see `select_representatives()`
        and Section 2.2 of the paper).

        Recomputing `argmax` over the full deque every frame would be wasteful.
        This method maintains `_best_idx` and `_best_score` incrementally:

        - If we're NOT evicting (the window still has space), we just check if the
          new score beats the current best and update accordingly.

        - If we ARE evicting (oldest frame is being dropped), we have to be more
          careful. If the frame being evicted was the best one (index 0), we do a
          full re-scan — there's no way around it when the champion leaves. If it
          wasn't the best frame, we simply shift the best index left by one (since
          every existing entry moves one slot to the left) and check the new entry.

        This gives O(1) amortized cost for the common case, with O(M) only when
        the best frame falls off the window — which is relatively infrequent.

        Called exclusively by `_maybe_store_frame()` immediately after appending
        the new entry to the deque.

        Args:
            pred_score (float): The classification score of the frame that was
                just appended to the memory window.
            evicting (bool): True if the deque was already at full capacity before
                this append, meaning the oldest element was simultaneously dropped.

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
        Attempt to add the current frame to the rolling memory window.

        This is the write path for the short-term memory buffer described in
        Section 2.2 of the paper. The memory window M (default size 20) stores
        the most recent high-quality (image, bounding-box) pairs so that the
        dynamic template can be periodically refreshed from the best one.

        Note: the filtering decision (whether this frame is good enough to store)
        is NOT made here — that happens in `update()` before this method is called.
        By the time we reach `_maybe_store_frame()`, the decision to store has
        already been made. This method's job is purely mechanical: append the data
        and update the best-index bookkeeping via `_update_best_index()`.

        The deque's `maxlen` ensures automatic eviction of the oldest frame when
        the window is full — we just need to tell `_update_best_index()` whether
        an eviction is happening so it can maintain the best-index correctly.

        Args:
            search (np.ndarray): The full current frame image (H x W x 3). Stored
                alongside its bounding box so we can re-extract features from it
                later during `select_representatives()`.
            pred_bbox (np.ndarray): The predicted bounding box [x, y, w, h] for
                the target in this frame. Stored as the crop centre for future
                feature extraction.
            pred_score (float): The network's classification confidence for this
                frame. Stored in `classification_scores` and used by
                `_update_best_index()` to track which frame is the best.

        Returns:
            None
        """
        evicting = len(self.classification_scores) == self.memory_window_size
        self.all_memory_imgs.append([search, pred_bbox])
        self.classification_scores.append(pred_score)
        self._update_best_index(pred_score, evicting)

    def select_representatives(
        self,
        window_size: Optional[int] = None,
    ) -> None:
        """
        Refresh dynamic features from the newest memory frame above update threshold.

        This is the core of the online template update mechanism described in
        Section 2.2 of the paper. Instead of always tracking against the first-frame
        template (which can become stale as the target changes appearance), we
        periodically pick the newest stored frame whose confidence clears the
        update threshold and refresh dynamic template/search features from it.

        This method is called on the cadence chosen in `update()`:
        `warm_up_n` during warm-up, then `N` afterward (default 10). When called, it:

        1. Checks that there's actually something in memory, then scans from newest
           to oldest for the first score >= `dynamic_update_threshold` (τu = 0.87
           in the paper). If `window_size` is set, the scan is restricted to the
           latest `window_size` entries. If no stored frame clears the threshold,
           we keep the previous dynamic template.

        2. Retrieves that newest qualifying (image, bbox) pair from
           `all_memory_imgs`.

        3. Re-extracts both `dynamic_template_features` and `dynamic_search_features`
           from that frame by calling `get_template_features()` and
           `get_search_features()` respectively.

        4. Saves the raw image and box as `running_dynamic_image` / `running_dynamic_bbox`
           so other parts of the system can reference the current dynamic frame.

        5. Invalidates any cached template computation in the network (if the model
           supports it), so the next forward pass uses the freshly updated features.

        Returns:
            None
        """
        if not self.classification_scores:
            return

        total = len(self.classification_scores)
        if window_size is None:
            first_idx = 0
        else:
            window = max(1, int(window_size))
            first_idx = max(0, total - window)

        latest_idx = None
        for idx in range(total - 1, first_idx - 1, -1):
            if float(self.classification_scores[idx]) >= self.dynamic_update_threshold:
                latest_idx = idx
                break
        if latest_idx is None:
            return

        best_img, best_bbox = self.all_memory_imgs[latest_idx]

        self.dynamic_template_features = self.get_template_features(
            best_img,
            best_bbox,
        )
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
        Process a new frame and return the tracker's best estimate of the target location.

        This is the main entry point called once per frame during normal tracking
        (Section 2 of the paper). It ties together the full tracking loop:
        feature extraction → network forward → box decoding → memory update →
        dynamic template refresh.

        Here's what happens step by step:

        1. `run_track()` is called to get the predicted bounding box and scores for
           this frame. This is the actual neural network inference step.

        2. The tracking state (current bbox, score, path history) is updated.

        3. If `dynamic_update` is enabled, the frame goes through a two-gate filter
           before being admitted to memory:
           - Score gate: the score must beat the running exponential average of
             recent scores, OR clear the hard update threshold. This prevents
             storing frames when the tracker is confused.
           - IoU gate: the new box must overlap sufficiently with the previous
             frame's box (default τ_iou=0.3). This prevents storing frames where
             the box jumped suddenly, which often indicates a distractor switch.

        4. After the warmup period (if configured), the memory deque is sealed at
           full capacity and the best-index is computed fresh over the whole window.

        5. On each configured cadence tick (`warm_up_n` during warm-up, otherwise `N`),
           `select_representatives()` is called to refresh the dynamic template
           from the newest stored frame above `dynamic_update_threshold`
           (Section 2.2).

        6. The running confidence EMA is updated and clamped at a floor value, so
           the threshold adapts to the sequence difficulty without collapsing.

        Args:
            search (NDArray): The current full frame image (H x W x 3, uint8).
                This is the frame to track the target in.
            *kw: Unused. Exists for API compatibility with the base class interface.

        Returns:
            Tuple[NDArray, float, float]:
                - pred_bbox (NDArray): The predicted bounding box [x, y, w, h] in
                  image pixel coordinates.
                - pred_score (float): The classification confidence score from the
                  network's output map at the predicted location. Higher is more
                  confident. Values below the occlusion threshold (0.55) trigger
                  the SiamRAM recovery pipeline.
                - sim_score (float): A similarity score between the current search
                  features and the stored dynamic features, providing an auxiliary
                  signal about how much the target's appearance has drifted.
        """
        pred_bbox, pred_score, sim_score = self.run_track(search)

        self.tracking_state.bbox = pred_bbox
        self.tracking_state.pred_score = pred_score
        self.tracking_state.paths.append(pred_bbox)
        if self.dynamic_update:
            gate_iou = (
                self.latest_iou_score_ema
                if self.iou_gate_use_ema
                else self.latest_iou_score
            )
            gate_ready = self.idx >= self.iou_gate_warmup_frames
            iou_head_ok = (
                not self.use_iou_gate
                or not gate_ready
                or gate_iou >= self.iou_gate_threshold
            )
            score_ok = (
                pred_score > self.running_confidence
                or pred_score >= self.dynamic_update_threshold
            )
            if score_ok and iou_head_ok:
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

            period = self._current_template_update_period()
            if self.idx % period == 0:
                window = self._current_template_selection_window()
                self.select_representatives(window_size=window)

            self.running_confidence = (
                self.update_lambda * pred_score
                + (1 - self.update_lambda) * self.running_confidence
            )
            self.running_confidence = max(
                self.running_confidence, self.running_confidence_floor_value
            )

        return pred_bbox, pred_score, sim_score

    def run_track(
        self,
        search,
    ):
        """
        Run one complete tracking inference step for the current frame.

        This method sits between `update()` (which handles memory and state) and
        `track()` (which calls the raw neural network). Its job is the geometric
        pipeline around the network call:

        1. Call `get_search_features()` to extract features from a region centred
           on the most recent bounding box estimate. This also returns the `search_context`
           — the mapping from crop coordinates back to full image coordinates.

        2. Save the context mapping into `tracking_state` so `_rescale_bbox()` can
           later invert the crop transform.

        3. Call `track()` to run the full network forward pass and get a predicted
           box in crop-relative coordinates.

        4. Call `_rescale_bbox()` to convert the predicted box back into full
           image pixel coordinates using the saved mapping.

        5. Call `clamp_bbox()` to ensure the final box doesn't go outside the image
           boundaries (handles edge cases near frame borders).

        This method is also the one called by `run_track_for_candidate()` when the
        recovery pipeline (Section 4.3) wants to test a hypothesised target location.

        Args:
            search (NDArray): The current full frame image (H x W x 3).

        Returns:
            Tuple[NDArray, float, float]:
                - pred_bbox (NDArray): The predicted bounding box [x, y, w, h] in
                  full image pixel coordinates, guaranteed to be within image bounds.
                - pred_score (float): The peak classification score at the predicted
                  location, used to decide whether tracking is reliable.
                - sim_score (float): The similarity score between the current search
                  and the stored dynamic context features.
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
        Execute the SiamABC neural network forward pass and postprocess its output.

        This is the lowest-level tracking method that actually touches the neural
        network (`self.net`). The SiamABC architecture (Section 2.1 of the paper)
        takes four feature inputs in a single forward pass:

        - Static template features (always `self._template_features` from frame 0,
          held frozen — passed from `self` not as an argument here).
        - Dynamic template features (the freshest high-confidence template, updated
          on the configured cadence by `select_representatives()`).
        - Current search features (extracted from the live frame by `run_track()`).
        - Dynamic search features (from the best historical frame, same source as
          dynamic template features).

        These four are fused inside the network via the Fast Parallel Polarized
        Self-Attention module and the BoxTower head, producing a classification map
        and regression map. This method then hands those raw outputs to
        `_postprocess()` which decodes them into an actual bounding box.

        Running under `@torch.no_grad()` — tracking is purely inference, no gradients.

        Args:
            search_features (torch.Tensor): Features extracted from the current
                frame's search crop — represents "where to look right now".
            dynamic_search_features (torch.Tensor): Features from the best recent
                high-confidence search crop — the tracker's updated reference for
                the target's current environment context.
            dynamic_template_features (torch.Tensor): Features from the best recent
                high-confidence template crop — the tracker's updated visual model
                of what the target looks like.

        Returns:
            Tuple[NDArray, float, float, torch.Tensor]: The four outputs from
                `_postprocess()`:
                - Predicted bounding box in crop coordinates (before rescaling).
                - Peak classification confidence score.
                - Similarity score.
                - Raw attention map tensor (for debugging / visualisation).
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
        Convert raw network output tensors into a bounding box, score, and similarity.

        The SiamABC network produces raw logits and regression offsets — it doesn't
        directly output a box. This method is the bridge between the network's
        tensor world and the tracker's geometric world. Here's what it does:

        1. Pulls out the raw classification logits and applies sigmoid to get
           per-location confidence scores in [0, 1].

        2. Pulls out the raw regression map (location offsets from anchor points).

        3. Calls `_confidence_postprocess()` (inherited from the base class) which
           applies the cosine penalty window — this suppresses spuriously high
           responses near the edges of the search crop, which is a common failure
           mode in Siamese trackers.

        4. Calls `self.box_coder.decode()` (the SiamABCBoxCoder) to anchor-free
           decode the penalised classification map + regression map into a candidate
           bounding box in crop-relative coordinates.

        5. Identifies the peak score location (r_max, c_max) from the decoded result
           and extracts the actual confidence value at that point.

        6. Reads out the similarity score from the network's output dictionary.

        Args:
            track_result (Dict[str, torch.Tensor]): The raw dictionary returned by
                `self.net.track()`. Expected keys include:
                - `TARGET_CLASSIFICATION_KEY`: Raw classification logit map.
                - `TARGET_REGRESSION_LABEL_KEY`: Raw regression offset map.
                - `TRACKER_TARGET_SEARCH_SIM_SCORE`: Scalar similarity score.
                - `TRACKER_ATTENTION_MAP`: Attention weights map (for visualisation).

        Returns:
            Tuple[NDArray, float, Union[int, float, bool], torch.Tensor]:
                - pred_bbox (NDArray): The decoded bounding box [x, y, w, h] in
                  crop-relative pixel coordinates. Will be rescaled to image
                  coordinates by the caller.
                - cls_score (float): The classification confidence at the peak
                  location — this is the score used for memory gating and occlusion
                  detection.
                - sim_score (float): The feature similarity score. Returns 0.0 if
                  the network didn't produce one.
                - attention_map (torch.Tensor): The raw attention map tensor, useful
                  for debugging or visualisation. Not used in the main tracking loop.
        """
        cls_score = track_result[constants.TARGET_CLASSIFICATION_KEY].float().sigmoid()
        iou_map = track_result.get(constants.TARGET_IOU_KEY)
        iou_score_map = iou_map.float().sigmoid() if iou_map is not None else None
        if self.use_iou_score and iou_score_map is not None:
            cls_score = cls_score * iou_score_map.pow(self.iou_power)
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
        peak_iou_score: Optional[float] = None
        if iou_score_map is not None:
            iou_np = iou_score_map.squeeze().detach().cpu().numpy()
            peak_iou_score = float(iou_np[r_max, c_max])

        if peak_iou_score is None:
            self.latest_iou_score = 0.0 if self.iou_gate_fail_closed else 1.0
        else:
            self.latest_iou_score = peak_iou_score
        beta = float(np.clip(self.iou_gate_ema_beta, 0.0, 0.999))
        self.latest_iou_score_ema = (
            beta * self.latest_iou_score_ema
            + (1.0 - beta) * self.latest_iou_score
        )

        # Introspection-based DRM update (arXiv:2411.17576). Only when explicitly
        # enabled on this instance: read the second-strongest local peak in the
        # response map (the analogue of SAM2's "alternative mask") and stash the
        # hypothesis-divergence signal for the SiamRAM tracker to consume. When
        # disabled this block is skipped entirely (no extra compute, legacy path
        # byte-for-byte unchanged).
        if getattr(self, "_drm_introspection_enabled", False):
            from models.siamram.drm_introspection import secondary_peak_divergence

            target_half = None
            if getattr(self, "_drm_scale_aware_divergence", False):
                try:
                    gh, gw = int(cls_score.shape[0]), int(cls_score.shape[1])
                    inst = float(self.tracking_config["instance_size"])
                    if gh > 0 and gw > 0 and inst > 0.0:
                        cell_y = inst / gh
                        cell_x = inst / gw
                        hy = 0.5 * float(pred_bbox[3]) / cell_y
                        hx = 0.5 * float(pred_bbox[2]) / cell_x
                        up_y = max(1.0, gh / 2.0 - 1.0)
                        up_x = max(1.0, gw / 2.0 - 1.0)
                        hy = min(max(hy, 1.0), up_y)
                        hx = min(max(hx, 1.0), up_x)
                        target_half = (hy, hx)
                except Exception:
                    target_half = None
            ratio, prim_score, sec_score, sec_rc = secondary_peak_divergence(
                cls_score, (int(r_max), int(c_max)), target_half_cells=target_half
            )
            self.latest_secondary_peak_ratio = ratio
            self.latest_primary_peak_score = prim_score
            self.latest_secondary_peak_score = sec_score

            # Distractor-bank write-back (sub-feature of DRM introspection). When
            # enabled, map the secondary-peak grid location into a frame-space bbox
            # so the outer SiamRAM tracker can record it in the NEGATIVE distractor
            # bank. Gated so it costs nothing when the sub-feature is off.
            self.latest_secondary_peak_bbox = None
            if getattr(self, "_drm_write_distractor_bank", False) and sec_rc is not None:
                try:
                    gh, gw = int(cls_score.shape[0]), int(cls_score.shape[1])
                    inst = float(self.tracking_config["instance_size"])
                    dx = (int(sec_rc[1]) - int(c_max)) * (inst / gw)
                    dy = (int(sec_rc[0]) - int(r_max)) * (inst / gh)
                    sec_crop = np.array(
                        [float(pred_bbox[0]) + dx, float(pred_bbox[1]) + dy,
                         float(pred_bbox[2]), float(pred_bbox[3])], dtype=float)
                    self.latest_secondary_peak_bbox = self._rescale_bbox(
                        sec_crop, self.tracking_state.mapping)
                except Exception:
                    self.latest_secondary_peak_bbox = None

            # One-time-per-process diagnostic so the next run visibly confirms the
            # secondary-peak signal is now real (non-1.0) rather than stuck at the
            # inert default. Guarded to at-most-once and only when telemetry is on;
            # never emitted when telemetry is disabled (zero overhead off-path).
            global _DRM_INTROSPECTION_SHAPE_LOGGED
            if (
                not _DRM_INTROSPECTION_SHAPE_LOGGED
                and getattr(self, "_research_telemetry_enabled", False)
            ):
                _DRM_INTROSPECTION_SHAPE_LOGGED = True
                try:
                    from utils.console import siamram_log

                    siamram_log(
                        "[research telemetry] DRM introspection signal live: "
                        f"cls_score.shape={tuple(cls_score.shape)} "
                        f"(ratio={ratio:.4f}, primary={prim_score:.4f}, "
                        f"secondary={sec_score:.4f})"
                    )
                except Exception:
                    pass

            # Research telemetry (inert unless the outer SiamRAM tracker enabled
            # it and stashed _tlm_inner here). Count frames where a genuine
            # competing secondary peak existed, and where the divergence ratio
            # fell below the DRM-introspection theta_anc default (0.7).
            if getattr(self, "_research_telemetry_enabled", False):
                tlm = getattr(self, "_tlm_inner", None)
                if tlm is not None:
                    if sec_score > 0.0:
                        tlm["drm_secondary_peak_frames"] += 1
                    if ratio < 0.7:
                        tlm["drm_divergence_below_theta"] += 1

        peak_cls_score = float(cls_score[r_max, c_max].item())
        gate_iou = (
            self.latest_iou_score_ema
            if self.iou_gate_use_ema
            else self.latest_iou_score
        )
        gate_ready = self.idx >= self.iou_gate_warmup_frames
        if (
            self.use_iou_gate
            and self.use_iou_gate_for_score
            and gate_ready
            and gate_iou < self.iou_gate_threshold
        ):
            # Keep a confidence penalty instead of hard-zeroing the score.
            # Hard-zero caused frequent false occlusion entries with noisy IoU maps.
            peak_cls_score *= 0.25
            self._last_iou_gate_pass = False
        else:
            self._last_iou_gate_pass = True

        sim_score_raw = track_result[constants.TRACKER_TARGET_SEARCH_SIM_SCORE]
        sim_score = sim_score_raw.item() if sim_score_raw is not None else 0.0
        return (
            pred_bbox,
            peak_cls_score,
            sim_score,
            track_result[constants.TRACKER_ATTENTION_MAP],
        )

    def run_track_for_candidate(
        self,
        search: np.ndarray,
        candidate_bbox: np.ndarray,
    ):
        """
        Test whether a hypothesised bounding box is a plausible target location.

        This method is used by the Multi-Phase Reacquisition Pipeline (Section 4.3
        of the paper) during Phase 3 — Tracker Verification. When the recovery
        pipeline has found a promising candidate (e.g. a YOLO detection that scored
        well against the DAM memory), it calls this method to verify the candidate
        by re-running the Siamese tracker as if the target were at that location.

        The trick here is "hypothesis injection": rather than using the tracker's
        current bounding box estimate (which may be wrong or stale during occlusion),
        we temporarily override `tracking_state.bbox` with the candidate box,
        compute the appropriate search context around it, run `run_track()`, and
        then restore the original state. This lets us evaluate any location in the
        image without permanently committing to it.

        The caller (Phase 3) accepts the candidate if the returned score is ≥ 0.70,
        at which point the tracker is reinitialised at the candidate location and
        normal tracking resumes.

        Args:
            search (np.ndarray): The current full frame image (H x W x 3). The same
                frame that the recovery pipeline is operating on.
            candidate_bbox (np.ndarray): The hypothesis bounding box [x, y, w, h] to
                evaluate. This comes from Phase 2 scoring — the top-ranked DAM
                candidate after appearance and velocity filtering.

        Returns:
            Tuple[NDArray, float, float]:
                - pred_bbox (NDArray): The tracker's predicted box when seeded at
                  the candidate location. A well-matched candidate will produce a
                  box close to the input candidate.
                - pred_score (float): The classification confidence at the prediction.
                  This is the key verification score — ≥ 0.70 means confirmed.
                - sim_score (float): The similarity score from the network forward pass.
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
        Turn Test-Time Augmentation (TTA) on or off for all subsequent frames.

        As described in Section 2.3 of the paper, Adaptive Batch Normalization
        allows certain encoder layers to incrementally update their running mean
        and variance statistics based on each incoming search crop. This is a
        lightweight form of test-time adaptation that helps the tracker handle
        gradual shifts in illumination, white balance, or sensor noise — without
        any retraining.

        The `_tta_lam` tensor acts as the live gate for this mechanism. When it's
        nonzero (equal to `_norm_lambda_tta`), the AdaptiveBatchNorm layers blend
        their stored statistics with the current crop's statistics. When it's zero,
        they behave like standard frozen BatchNorm.

        Using `fill_()` modifies the tensor in-place, which means any reference
        to `_tta_lam` elsewhere (e.g. inside the network's forward graph) sees
        the update immediately — no need to pass a new tensor.

        Args:
            enabled (bool): True to activate TTA (sets lam to `_norm_lambda_tta`),
                False to deactivate it (sets lam to 0.0 — standard frozen BN).

        Returns:
            None
        """
        val = self._norm_lambda_tta if enabled else 0.0
        self._tta_lam.fill_(val)

    def enable_tta(
        self,
    ) -> None:
        """
        Activate Test-Time Augmentation for all subsequent network forward passes.

        A convenience wrapper around `set_tta(True)`. Call this before processing
        a sequence where you expect significant appearance variation — e.g. outdoor
        scenes with changing sunlight, cameras with auto-exposure, or sequences
        with motion blur and sensor noise.

        Internally, this sets `_tta_lam` to the AdaptiveBatchNorm lambda value
        found in the model, causing the encoder's normalization layers to adapt
        their statistics incrementally at each step (Section 2.3 of the paper).

        Returns:
            None
        """
        self.set_tta(True)

    def disable_tta(
        self,
    ) -> None:
        """
        Deactivate Test-Time Augmentation, reverting to standard frozen BatchNorm.

        A convenience wrapper around `set_tta(False)`. Call this when you want
        strict reproducibility or when the sequence has stable, predictable
        illumination where adaptation would add noise rather than help.

        Internally, this zeros out `_tta_lam`, which makes the AdaptiveBatchNorm
        layers behave identically to regular frozen BatchNorm — no statistics
        are updated during inference.

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
        Compute the Intersection over Union (IoU) between two bounding boxes.

        IoU is the standard overlap metric in object tracking and detection. It
        measures what fraction of the combined area of two boxes is actually
        shared between them. A value of 1.0 means perfect overlap; 0.0 means
        no overlap at all.

        In SiamRAM this is used in two critical places:

        1. In `update()` as the second gate before storing a frame to memory:
           IoU(pred_bbox, prev_bbox) ≥ τ_iou (default 0.3). This prevents the
           memory from being polluted by frames where the predicted box jumped
           suddenly — a sign of a distractor switch or momentary confusion.
           (See Equation 4 in the paper, where τ_iou = 0.6 for the full DAM
           admission criteria.)

        2. In the DAM module's RAM admission logic (Section 3.2), where the same
           kind of geometric coherence check is applied to short-term memory entries.

        The implementation handles the degenerate case where union is zero
        (e.g. one box has zero area) by returning 0.0 rather than dividing by zero.

        Args:
            boxA (Sequence[float]): First bounding box in [x, y, width, height]
                format. x and y are the top-left corner in pixel coordinates.
            boxB (Sequence[float]): Second bounding box, also [x, y, width, height].

        Returns:
            float: IoU value in the range [0.0, 1.0]. Returns 0.0 if the union
                area is zero (e.g. either box has zero dimensions).
        """
        ix1 = max(boxA[0], boxB[0])
        iy1 = max(boxA[1], boxB[1])
        ix2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        iy2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        return inter / union if union > 0 else 0.0
