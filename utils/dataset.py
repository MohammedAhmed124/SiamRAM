"""
Multi-dataset tracking dataset supporting UAV123, UAVTrack112, and DTB70.

Dataset-specific differences handled here
──────────────────────────────────────────
UAV123      : frames/{frame_idx:06d}.jpg, index starts at 1
UAVTrack112 : frames/{frame_idx:06d}.jpg, index starts at 1  (same as UAV123)
DTB70       : img/{frame_idx:04d}.jpg,    index starts at 1

Frame naming is auto-detected from existing files on first access, so new
datasets that follow any common convention will work without code changes.

Negative sample structure (guaranteed)
───────────────────────────────────────
  template         – positive target crop (template-style) from frame t_idx
  dynamic_template – positive target crop (template-style) from frame dyn_idx
  dynamic_search   – positive target crop (search-style)  from frame dyn_idx
                     dyn_idx ∈ [t_idx, s_idx]; SAME frame for both dynamic crops
  search           – negative crop: the context window is GUARANTEED not to
                     overlap the ground-truth target bbox, verified explicitly
                     via _bboxes_overlap() after every candidate is proposed.
                     No target pixel can appear in the search crop.

Why the old shift-magnitude heuristic was insufficient
───────────────────────────────────────────────────────
With search_context = 2.0, extend_bbox expands the context window by 2× the
object width/height on every side.  The full context window therefore spans
5× the object in each dimension (2 left + 1 object + 2 right).  The context
left/right edges are at anchor_center ± 2.5 × w.  For a purely horizontal
shift of magnitude s, the nearest context edge is at distance s – 2.5w from
the target center, so NO overlap requires s > (1 + search_context) × w = 3w.
The old code shifted by only 1–2 × w, which is provably insufficient.

The fix uses a minimum shift magnitude of (search_context + 1.1) × dimension
as a sampling bias AND always validates with the explicit geometry check.
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.box_coder import SiamABCBoxCoder
from utils.utils import (
    clamp_bbox,
    extend_bbox,
    get_extended_crop,
)

_FRAME_PATTERNS = [
    "{:06d}.jpg",
    "img/{:04d}.jpg",
    "{:04d}.jpg",
    "img/{:06d}.jpg",
    "{:06d}.png",
    "img/{:04d}.png",
]


def _detect_frame_pattern(seq_path: str, probe_indices: List[int]) -> Optional[str]:
    """
    Try each pattern in _FRAME_PATTERNS against a handful of probe frame
    indices.  Return the first pattern where ALL probes resolve to existing
    files, or None if nothing matches.
    """
    for pattern in _FRAME_PATTERNS:
        if all(
            os.path.exists(os.path.join(seq_path, pattern.format(idx)))
            for idx in probe_indices
        ):
            return pattern
    return None


def _regression_weight_label(
    bbox,
    image_size: int = 255,
    map_size: int = 25,
    r_pos: int = 2,
    r_neg: int = 0,
) -> torch.Tensor:
    bbox_c_x = bbox[0] + bbox[2] / 2.0
    bbox_c_y = bbox[1] + bbox[3] / 2.0

    sz_x = np.floor(float(bbox_c_x / image_size * map_size))
    sz_y = np.floor(float(bbox_c_y / image_size * map_size))

    x, y = np.meshgrid(
        np.arange(0, map_size) - sz_x,
        np.arange(0, map_size) - sz_y,
    )
    dist_to_center = np.abs(x) + np.abs(y)
    label = np.where(
        dist_to_center <= r_pos,
        np.ones_like(y),
        np.where(dist_to_center < r_neg, 0.5 * np.ones_like(y), np.zeros_like(y)),
    )
    return torch.from_numpy(label.astype(np.float32))


class TrackingSequence:
    """
    Dataset-agnostic sequence wrapper.

    The dataframe row must supply:
        seq_path    – directory containing frame images
        annot_path  – path to groundtruth annotation file
        start_idx   – first frame index (inclusive)
        end_idx     – last  frame index (inclusive)
        n_frames    – total number of frames
        class       – object class label (string)
        dataset     – source dataset name, e.g. "UAV123" / "DTB70" / "UAVTrack112"

    Optional row column:
        frame_pattern – printf-style pattern relative to seq_path, e.g.
                        "img/{:04d}.jpg".  If absent or empty the pattern is
                        auto-detected from existing files.
    """

    def __init__(self, row: Dict):
        """
        Initialise a TrackingSequence from a dataframe row.

        Args:
            row (Dict): Dictionary containing 'seq_path', 'annot_path', etc.
        """
        self.seq_path = row["seq_path"]
        self.annot_path = row["annot_path"]
        self.start_idx = int(row["start_idx"])
        self.end_idx = int(row["end_idx"])
        self.n_frames = int(row["n_frames"])
        self.cls = row["class"]
        self.dataset = row["dataset"]

        self._frame_pattern: Optional[str] = row.get("frame_pattern") or None
        self._bboxes: Optional[List[np.ndarray]] = None

    def _resolve_frame_pattern(self) -> str:
        """
        Auto-detect the frame-naming pattern by probing a few known indices.
        Raises RuntimeError if none of the known patterns match.
        """
        mid = (self.start_idx + self.end_idx) // 2
        probes = list({self.start_idx, mid, self.end_idx})

        pattern = _detect_frame_pattern(self.seq_path, probes)
        if pattern is None:
            pattern = self._infer_pattern_from_directory()
        if pattern is None:
            raise RuntimeError(
                f"Cannot detect frame naming pattern for sequence at '{self.seq_path}'. "
                f"Add a 'frame_pattern' column to your dataframe or extend _FRAME_PATTERNS."
            )
        return pattern

    def _infer_pattern_from_directory(self) -> Optional[str]:
        """
        Fallback: list the sequence directory (or its 'img/' subdirectory) and
        guess the zero-padding width and extension from the first image found.
        """
        for subdir in ("", "img"):
            dirpath = os.path.join(self.seq_path, subdir) if subdir else self.seq_path
            if not os.path.isdir(dirpath):
                continue
            candidates = sorted(
                f for f in os.listdir(dirpath)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not candidates:
                continue
            name, ext = os.path.splitext(candidates[0])
            width = len(name)
            prefix = (subdir + "/") if subdir else ""
            return f"{prefix}{{:{width:02d}d}}{ext}"
        return None

    def frame_path(self, frame_idx: int) -> str:
        """
        Return the absolute path to the image at the given frame index.

        Args:
            frame_idx (int): Frame index.

        Returns:
            str: Absolute file path.
        """
        if self._frame_pattern is None:
            self._frame_pattern = self._resolve_frame_pattern()
        return os.path.join(self.seq_path, self._frame_pattern.format(frame_idx))

    def _load_bboxes(self) -> None:
        if self._bboxes is not None:
            return
        bboxes: List[np.ndarray] = []
        with open(self.annot_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                vals = line.replace("\t", ",").split(",")
                try:
                    x, y, w, h = (float(v) for v in vals[:4])
                    bboxes.append(np.array([x, y, w, h], dtype=np.float32))
                except (ValueError, IndexError):
                    print(f"[warn] failed to parse bbox: {self.annot_path!r} | {line!r}")
                    bboxes.append(np.full(4, np.nan, dtype=np.float32))
        self._bboxes = bboxes

    def get_bbox(self, frame_idx: int) -> np.ndarray:
        """
        Return the ground-truth bounding box for the given frame index.

        Args:
            frame_idx (int): Frame index.

        Returns:
            np.ndarray: Bounding box [x, y, w, h].
        """
        self._load_bboxes()
        assert self._bboxes is not None
        local = frame_idx - self.start_idx
        if local < 0 or local >= len(self._bboxes):
            return np.full(4, np.nan, dtype=np.float32)
        return self._bboxes[local].copy()

    @staticmethod
    def is_valid_bbox(bbox: np.ndarray) -> bool:
        """
        Check if a bounding box is valid (not NaN, positive dimensions).

        Args:
            bbox (np.ndarray): Bounding box to check.

        Returns:
            bool: True if valid.
        """
        return (
            not np.any(np.isnan(bbox))
            and bbox[2] > 5
            and bbox[3] > 5
            and bbox[0] >= 0
            and bbox[1] >= 0
        )

    def sample_valid_frame(
        self,
        rng: random.Random,
        max_tries: int = 50,
        lower_end: Optional[int] = None,
        upper_end: Optional[int] = None,
    ) -> Optional[int]:
        """
        Randomly sample a frame index with a valid bounding box.

        Args:
            rng (random.Random): Random number generator.
            max_tries (int): Maximum number of random attempts.
            lower_end (Optional[int]): Lower bound for sampling.
            upper_end (Optional[int]): Upper bound for sampling.

        Returns:
            Optional[int]: Valid frame index or None.
        """
        lower_end = max(lower_end, self.start_idx) if lower_end is not None else self.start_idx
        upper_end = min(upper_end, self.end_idx) if upper_end is not None else self.end_idx
        self._load_bboxes()
        indices = list(range(lower_end, upper_end + 1))
        rng.shuffle(indices)
        for idx in indices[:max_tries]:
            bbox = self.get_bbox(idx)
            if self.is_valid_bbox(bbox) and os.path.exists(self.frame_path(idx)):
                return idx
        return None

    def sample_search_template_idx_pair(
        self,
        rng: random.Random,
        max_frame_gap: int,
        lower_end: Optional[int] = None,
        upper_end: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int], Optional[np.ndarray]]:
        """
        Sample a pair of (template, search) frame indices.

        Args:
            rng (random.Random): Random number generator.
            max_frame_gap (int): Maximum temporal distance between frames.
            lower_end, upper_end (Optional[int]): Search range bounds.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[np.ndarray]]:
                (template_idx, search_idx, search_bbox).
        """
        lower_end = max(lower_end, self.start_idx) if lower_end is not None else self.start_idx
        upper_end = min(upper_end, self.end_idx) if upper_end is not None else self.end_idx

        failure = (None, None, None)

        t_idx = self.sample_valid_frame(rng, lower_end=lower_end, upper_end=upper_end)
        if t_idx is None:
            return failure

        lo = max(lower_end, t_idx)
        hi = min(upper_end, t_idx + max_frame_gap)
        candidates = [i for i in range(lo, hi + 1) if (max_frame_gap == 0 or i != t_idx)]
        if not candidates:
            return failure

        s_idx = rng.choice(candidates)
        s_bbox = self.get_bbox(s_idx)
        if not self.is_valid_bbox(s_bbox):
            return failure
        if not os.path.exists(self.frame_path(s_idx)):
            return failure

        return t_idx, s_idx, s_bbox


class UAVTrackingDataset(Dataset):
    """
    Unified tracking dataset for UAV123, UAVTrack112, and DTB70.

    The dataframe must have columns described in TrackingSequence.__init__.
    Mixed-dataset dataframes (rows from different sources) are fully supported.
    """

    TEMPLATE_CONTEXT = 0.5
    SEARCH_CONTEXT = 2.0

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tracking_config: Dict[str, Any],
        num_samples: int = 10_000,
        neg_ratio: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialise the UAVTrackingDataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing sequence info.
            tracking_config (Dict[str, Any]): Tracker hyper-parameters.
            num_samples (int): Total number of samples per epoch.
            neg_ratio (float): Probability of sampling a negative pair.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        self.template_size = tracking_config["template_size"]
        self.instance_size = tracking_config["instance_size"]
        self.score_size = tracking_config["score_size"]
        self.max_frame_gap = tracking_config.get("max_frame_gap", 50)
        self.gaussian_sigma = tracking_config.get("gaussian_sigma", 0.125)
        self.template_bbox_offset = tracking_config["template_bbox_offset"]
        self.search_context = tracking_config["search_context"]
        self.search_image_shift = tracking_config["search_image_shift"]
        self.search_image_scale = tracking_config["search_image_scale"]
        self.template_image_shift = tracking_config["template_image_shift"]
        self.template_image_scale = tracking_config["template_image_scale"]
        self.cuda_id = 0

        self._template_transform = self._get_default_transform(tracking_config["template_size"])
        self._search_transform = self._get_default_transform(tracking_config["instance_size"])
        self._dynamic_search_transform = self._get_default_transform(tracking_config["instance_size"])

        self.neg_ratio = neg_ratio
        self.num_samples = num_samples
        self.rng = random.Random(seed)

        self.box_coder = SiamABCBoxCoder(tracking_config)

        self.sequences: List[TrackingSequence] = [
            TrackingSequence(row.to_dict())
            for _, row in dataframe.reset_index(drop=True).iterrows()
        ]

        if "dataset" in dataframe.columns:
            breakdown = dataframe["dataset"].value_counts().to_dict()
        else:
            breakdown = {"unknown": len(self.sequences)}

        print(
            f"[Dataset] {len(self.sequences)} sequences | "
            f"sources={breakdown} | "
            f"num_samples={num_samples} | neg_ratio={neg_ratio:.0%} | "
            f"template_size={self.template_size} | instance_size={self.instance_size} | "
            f"score_size={self.score_size}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        for attempt in range(5):
            try:
                if self.rng.random() >= self.neg_ratio:
                    return self._positive_sample()
                else:
                    return self._negative_sample()
            except (ValueError, RuntimeError):
                if attempt == 4:
                    return self._dummy_negative()
        return self._dummy_negative()

    def _positive_sample(self) -> Dict[str, torch.Tensor]:
        for _ in range(10):
            seq = self.rng.choice(self.sequences)

            t_crop, t_idx, template_bbox, s_crop, s_idx, bbox_in_c, ok = \
                self.sample_search_template_pair(self.rng, seq, max_frame_gap=self.max_frame_gap)
            if not ok:
                continue

            dt_crop, _, _, ds_crop, _, _, ok_dyn = self.sample_search_template_pair(
                self.rng, seq,
                lower_end=t_idx, upper_end=s_idx,
                max_frame_gap=0,
            )
            if not ok_dyn:
                dt_crop = t_crop
                ds_crop = s_crop

            if not self.is_valid_bbox(bbox_in_c):
                continue
            x, y, w, h = bbox_in_c
            if (x + w) > self.instance_size or (y + h) > self.instance_size:
                continue

            encoded = self.box_coder.encode(torch.from_numpy(bbox_in_c).reshape(1, 4))
            cls_label = encoded.classification_label[0].numpy()
            bbox_label = encoded.regression_map[0].numpy()

            reg_weight = _regression_weight_label(
                bbox_in_c, self.instance_size, self.score_size
            ).numpy()

            return self._pack(
                t_crop, dt_crop, s_crop, ds_crop,
                cls_label, bbox_label, reg_weight,
                is_positive=True, path=seq.frame_path(s_idx),
            )

        return self._dummy_negative()

    def sample_search_template_pair(
        self,
        rng: random.Random,
        seq: TrackingSequence,
        lower_end=None,
        upper_end=None,
        max_frame_gap=None,
    ):
        """
        Sample a (template_crop, search_crop) pair from a sequence.

        Args:
            rng (random.Random): Random number generator.
            seq (TrackingSequence): The sequence to sample from.
            lower_end, upper_end (Optional[int]): Range bounds.
            max_frame_gap (Optional[int]): Max temporal distance.

        Returns:
            Tuple: (t_crop, t_idx, t_bbox, s_crop, s_idx, s_bbox, success).
        """
        failure = (None, None, None, None, None, None, False)
        frame_gap = self.max_frame_gap if max_frame_gap is None else max_frame_gap

        t_idx, s_idx, s_bbox = seq.sample_search_template_idx_pair(
            rng, max_frame_gap=frame_gap,
            lower_end=lower_end, upper_end=upper_end,
        )
        if t_idx is None or s_idx is None or s_bbox is None:
            return failure

        t_bbox = seq.get_bbox(t_idx)
        t_img = self._load_image(seq.frame_path(t_idx))
        s_img = self._load_image(seq.frame_path(s_idx))

        t_crop, template_bbox = self.get_template_crop(t_img, t_bbox)
        s_crop, bbox_in_c, _ = self.get_search_crop(s_img, s_bbox)

        return t_crop, t_idx, template_bbox, s_crop, s_idx, bbox_in_c, True

    def get_template_crop(self, image: np.ndarray, rect: np.ndarray):
        """
        Extract and preprocess a template crop from an image.

        Args:
            image (np.ndarray): Source RGB image.
            rect (np.ndarray): Target bounding box [x, y, w, h].

        Returns:
            Tuple[torch.Tensor, np.ndarray]: (preprocessed_crop, crop_bbox).
        """
        shifted = self.apply_shift_scale(
            rect, image.shape,
            shift_factor=self.template_image_shift,
            scale_factor=self.template_image_scale,
        )
        if not self.is_valid_bbox(shifted):
            shifted = rect

        context = extend_bbox(
            shifted, offset=self.template_bbox_offset,
            image_width=image.shape[1], image_height=image.shape[0],
        )
        crop, template_bbox, _ = get_extended_crop(
            image=image, bbox=rect, context=context,
            crop_size=self.template_size,
        )
        img = self._preprocess_image(crop, self._template_transform)
        return img, template_bbox

    def get_search_crop(self, image: np.ndarray, bbox: np.ndarray):
        """
        Extract and preprocess a search crop from an image.

        Args:
            image (np.ndarray): Source RGB image.
            bbox (np.ndarray): Target bounding box [x, y, w, h].

        Returns:
            Tuple[torch.Tensor, np.ndarray, np.ndarray]:
                (preprocessed_crop, search_bbox, context_rect).
        """
        shifted = self.apply_shift_scale(
            bbox, image.shape,
            shift_factor=self.search_image_shift,
            scale_factor=self.search_image_scale,
        )
        bbox_for_crop = clamp_bbox(shifted, shape=image.shape[:2])
        context = extend_bbox(
            bbox_for_crop, offset=self._get_random_context(),
            image_width=image.shape[1], image_height=image.shape[0],
        )
        crop, search_bbox, ctx = get_extended_crop(
            image=image, bbox=bbox,
            crop_size=self.instance_size, context=context,
            padding_value=np.mean(image, axis=(0, 1)),
        )
        img = self._preprocess_image(crop, self._search_transform)
        return img, search_bbox, ctx

    def _negative_sample(self) -> Dict[str, torch.Tensor]:
        """
        Build a negative training sample with the following strict structure:

          template         – positive target crop (template-style) from frame t_idx
          dynamic_template – positive target crop (template-style) from frame dyn_idx
          dynamic_search   – positive target crop (search-style)   from frame dyn_idx
                             dyn_idx ∈ [t_idx, s_idx] — SAME frame for both dynamic crops
          search           – negative crop: context window is GUARANTEED not to
                             overlap the actual target bbox in its source frame.

        The "guaranteed" part is enforced by _bboxes_overlap(), which is called
        after every candidate crop window is proposed.  No target pixel can appear
        in the search crop.
        """
        for _ in range(15):
            seq = self.rng.choice(self.sequences)

            t_idx, s_idx, s_bbox = seq.sample_search_template_idx_pair(
                self.rng, max_frame_gap=self.max_frame_gap
            )
            if t_idx is None:
                continue

            t_bbox = seq.get_bbox(t_idx)
            if not seq.is_valid_bbox(t_bbox):
                continue

            t_img = self._load_image(seq.frame_path(t_idx))

            t_crop, _ = self.get_template_crop(t_img, t_bbox)

            dyn_idx = seq.sample_valid_frame(
                self.rng, lower_end=t_idx, upper_end=s_idx
            )
            if dyn_idx is None:

                dyn_idx = t_idx
                dyn_img = t_img
                dyn_bbox = t_bbox.copy()
            else:
                dyn_img = self._load_image(seq.frame_path(dyn_idx))
                dyn_bbox = seq.get_bbox(dyn_idx)

            if not seq.is_valid_bbox(dyn_bbox):
                continue

            dynamic_template, _ = self.get_template_crop(dyn_img, dyn_bbox)

            dynamic_search, _, _ = self.get_search_crop(dyn_img, dyn_bbox)

            if self.rng.random() < 0.75:
                s_crop = self._build_shifted_negative_search(seq, s_idx, s_bbox)
            else:
                s_crop = self._build_distractor_negative_search(seq, s_idx, s_bbox)

            if s_crop is None:
                continue

            S = self.score_size
            cls_label = np.zeros((1, S, S), dtype=np.float32)
            bbox_label = np.zeros((4, S, S), dtype=np.float32)
            reg_weight = np.zeros((S, S), dtype=np.float32)

            return self._pack(
                t_crop, dynamic_template, s_crop, dynamic_search,
                cls_label, bbox_label, reg_weight,
                is_positive=False,
            )

        return self._dummy_negative()

    @staticmethod
    def _bboxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
        """
        Return True if two [x, y, w, h] bounding boxes have any pixel overlap.

        Handles negative coordinates correctly (context boxes can extend off-screen).
        This is the definitive safety check used to guarantee that the target bbox
        does not appear inside any negative search crop.

        Geometry: two axis-aligned rectangles overlap iff their projections onto
        BOTH axes overlap simultaneously.
          x-overlap: a.x1 < b.x2  AND  a.x2 > b.x1
          y-overlap: a.y1 < b.y2  AND  a.y2 > b.y1
        """
        ax1, ay1 = float(a[0]), float(a[1])
        ax2, ay2 = ax1 + float(a[2]), ay1 + float(a[3])
        bx1, by1 = float(b[0]), float(b[1])
        bx2, by2 = bx1 + float(b[2]), by1 + float(b[3])
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def _build_shifted_negative_search(
        self,
        seq: TrackingSequence,
        frame_idx: int,
        target_bbox: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """
        Build a negative search crop from `frame_idx` by shifting the crop
        anchor far away from `target_bbox`.

        Strategy
        --------
        1. Clamp `target_bbox` to image boundaries → `target_clamped`.
        2. Pick a random cardinal direction and a shift magnitude of at least
           (search_context + 1.1) × object_dimension.  This magnitude is the
           theoretical minimum needed to clear the context window past the target
           edge in the chosen direction, plus a 10% margin.
        3. After clamping the shifted anchor to the image, compute the context
           window and call `_bboxes_overlap(context, target_clamped)`.
        4. If overlap → reject and retry.  This is the definitive correctness
           guarantee — the magnitude is just a sampling bias.

        Returns None after 50 failed attempts.
        """
        if not os.path.exists(seq.frame_path(frame_idx)):
            return None

        img = self._load_image(seq.frame_path(frame_idx))
        H, W = img.shape[:2]
        mean_color = np.mean(img, axis=(0, 1))

        target_clamped = clamp_bbox(target_bbox.astype(np.float32), img.shape[:2])
        if not self.is_valid_bbox(target_clamped):
            return None

        x, y, w, h = target_clamped.astype(float)

        min_shift_x = (1.0 + self.search_context + 0.1) * w
        min_shift_y = (1.0 + self.search_context + 0.1) * h

        for _ in range(50):
            direction = self.rng.choice(('left', 'right', 'up', 'down'))

            if direction == 'right':
                shift_x = self.rng.uniform(min_shift_x, min_shift_x + 2.0 * w)
                shift_y = self.rng.uniform(-0.5, 0.5) * h
            elif direction == 'left':
                shift_x = -self.rng.uniform(min_shift_x, min_shift_x + 2.0 * w)
                shift_y = self.rng.uniform(-0.5, 0.5) * h
            elif direction == 'down':
                shift_x = self.rng.uniform(-0.5, 0.5) * w
                shift_y = self.rng.uniform(min_shift_y, min_shift_y + 2.0 * h)
            else:
                shift_x = self.rng.uniform(-0.5, 0.5) * w
                shift_y = -self.rng.uniform(min_shift_y, min_shift_y + 2.0 * h)

            anchor = clamp_bbox(
                np.array([x + shift_x, y + shift_y, w, h], dtype=np.float32),
                img.shape[:2],
            )
            if not self.is_valid_bbox(anchor):
                continue

            context = extend_bbox(
                anchor, offset=self.search_context,
                image_width=W, image_height=H,
            )

            if self._bboxes_overlap(context, target_clamped):
                continue

            raw_crop, _, _ = get_extended_crop(
                image=img, bbox=anchor, context=context,
                crop_size=self.instance_size,
                padding_value=mean_color,
            )
            return self._preprocess_image(raw_crop, self._search_transform)

        return None

    def _build_distractor_negative_search(
        self,
        seq: TrackingSequence,
        frame_idx: int,
        target_bbox: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """
        Build a negative search crop from `frame_idx` by randomly sampling
        a background anchor position whose context window does NOT overlap
        with `target_bbox`.

        The anchor's size matches the target so the crop has similar scale
        characteristics (hard negative).  All positions are tested with
        `_bboxes_overlap` before accepting — this is the definitive guarantee.

        Returns None after 50 failed attempts.
        """
        if not os.path.exists(seq.frame_path(frame_idx)):
            return None

        img = self._load_image(seq.frame_path(frame_idx))
        H, W = img.shape[:2]
        mean_color = np.mean(img, axis=(0, 1))

        target_clamped = clamp_bbox(target_bbox.astype(np.float32), img.shape[:2])
        if not self.is_valid_bbox(target_clamped):
            return None

        _, _, w, h = target_clamped.astype(float)

        for _ in range(50):

            cx_bg = self.rng.uniform(0.0, float(W))
            cy_bg = self.rng.uniform(0.0, float(H))

            anchor = clamp_bbox(
                np.array([cx_bg - w / 2.0, cy_bg - h / 2.0, w, h], dtype=np.float32),
                img.shape[:2],
            )
            if not self.is_valid_bbox(anchor):
                continue

            context = extend_bbox(
                anchor, offset=self.search_context,
                image_width=W, image_height=H,
            )

            if self._bboxes_overlap(context, target_clamped):
                continue

            raw_crop, _, _ = get_extended_crop(
                image=img, bbox=anchor, context=context,
                crop_size=self.instance_size,
                padding_value=mean_color,
            )
            return self._preprocess_image(raw_crop, self._search_transform)

        return None

    def _get_template_crop(
        self, img: np.ndarray, bbox: np.ndarray
    ) -> Optional[torch.Tensor]:
        bbox = clamp_bbox(bbox, img.shape)
        if not self.is_valid_bbox(bbox):
            return None
        context = extend_bbox(
            bbox, offset=self.template_bbox_offset,
            image_width=img.shape[1], image_height=img.shape[0],
        )
        crop, _, _ = get_extended_crop(
            image=img, bbox=bbox, context=context,
            crop_size=self.template_size,
        )
        return self._preprocess_image(crop, self._template_transform)

    def _get_search_crop(
        self, img: np.ndarray, bbox: np.ndarray
    ) -> Optional[torch.Tensor]:
        bbox = clamp_bbox(bbox, img.shape)
        if not self.is_valid_bbox(bbox):
            return None
        context = extend_bbox(
            bbox, offset=self.search_context,
            image_width=img.shape[1], image_height=img.shape[0],
        )
        crop, _, _ = get_extended_crop(
            image=img, bbox=bbox, context=context,
            crop_size=self.instance_size,
            padding_value=np.mean(img, axis=(0, 1)),
        )
        return self._preprocess_image(crop, self._search_transform)

    def _pack(
        self,
        template,
        dynamic_template,
        search,
        dynamic_search,
        cls_label: np.ndarray,
        bbox_label: np.ndarray,
        reg_weight: np.ndarray,
        is_positive: bool,
        path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        return {
            "template": self._to_tensor(template),
            "dynamic_template": self._to_tensor(dynamic_template),
            "search": self._to_tensor(search),
            "dynamic_search": self._to_tensor(dynamic_search),
            "cls_label": torch.from_numpy(cls_label),
            "bbox_label": torch.from_numpy(bbox_label),
            "reg_weight": torch.from_numpy(reg_weight),
            "is_positive": torch.tensor(is_positive, dtype=torch.bool),
            "path": path if path is not None else "None",
        }

    def _dummy_negative(self) -> Dict[str, torch.Tensor]:
        S, T, i = self.score_size, self.template_size, self.instance_size
        return {
            "template": torch.zeros(3, T, T),
            "dynamic_template": torch.zeros(3, T, T),
            "search": torch.zeros(3, i, i),
            "dynamic_search": torch.zeros(3, i, i),
            "cls_label": torch.zeros(1, S, S),
            "bbox_label": torch.zeros(4, S, S),
            "reg_weight": torch.zeros(S, S),
            "is_positive": torch.tensor(False, dtype=torch.bool),
            "path": "None",
        }

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(x).float()

    def _preprocess_image(self, image: np.ndarray, transform: Callable) -> torch.Tensor:
        img = transform(image[:, :, :3])
        if image.shape[2] > 3:
            img = np.concatenate([img, image[:, :, 3:]], axis=2)
        return self._array_to_batch(img).float()

    @staticmethod
    def _array_to_batch(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.transpose(x, (2, 0, 1)))

    def _get_default_transform(self, img_size: int) -> Callable:
        pipeline = albu.Compose([
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return lambda a: pipeline(image=a)["image"]

    def apply_shift_scale(
        self, bbox: np.ndarray, img_shape, shift_factor=0.0, scale_factor=0.0
    ) -> np.ndarray:
        """
        Apply random translation and scaling to a bounding box.

        Args:
            bbox (np.ndarray): Original bounding box.
            img_shape (Tuple[int, int]): Image dimensions (H, W).
            shift_factor (float): Max translation ratio.
            scale_factor (float): Max scaling ratio.

        Returns:
            np.ndarray: Shifted and scaled bounding box.
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        shift_x = np.random.uniform(-shift_factor, shift_factor) * w
        shift_y = np.random.uniform(-shift_factor, shift_factor) * h
        scale = np.random.uniform(1.0 - scale_factor, 1.0 + scale_factor)
        new_w, new_h = w * scale, h * scale
        new_bbox = np.array(
            [cx + shift_x - new_w / 2, cy + shift_y - new_h / 2, new_w, new_h],
            dtype=np.float32,
        )
        return clamp_bbox(new_bbox, shape=img_shape[:2])

    def _get_random_context(self) -> float:
        return np.random.uniform(self.search_context * 0.8, self.search_context * 1.2)

    @staticmethod
    def is_valid_bbox(bbox: np.ndarray) -> bool:
        """
        Check if a bounding box is valid.

        Args:
            bbox (np.ndarray): Bounding box [x, y, w, h].

        Returns:
            bool: True if valid.
        """
        return (
            not np.any(np.isnan(bbox))
            and bbox[2] > 5
            and bbox[3] > 5
            and bbox[0] >= 0
            and bbox[1] >= 0
        )


UAV123TrackingDataset = UAVTrackingDataset
