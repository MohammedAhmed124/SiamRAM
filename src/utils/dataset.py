"""
UAV tracking dataset that feeds SiamABC during training.
 
Supports UAV123, UAVTrack112, and DTB70 out of the box. Mixed dataframes
(rows from multiple datasets) work fine — each sequence figures out its own
frame-naming convention automatically.
 
──────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS: THE CONFIDENCE COLLAPSE PROBLEM
──────────────────────────────────────────────────────────────────────────────
 
SiamABC (and Siamese trackers in general) learn by comparing a template crop
of the target to a search-region crop and predicting whether the target is
present and where.  The head outputs a classification score (confidence) and
a regression map (bounding box).
 
The problem: if you only ever train on POSITIVE pairs — template and search
both containing the real target — the head never learns to say "no".  During
inference it sees backgrounds, distractors, and completely empty regions, but
it was never penalised for being confident on those.  The result is that the
confidence score is always high regardless of what's actually in the search
crop.  The tracker can't distinguish "I found the target" from "I'm looking
at a patch of sky".  This kills any re-detection or confidence-gating logic
you put downstream.
 
The fix is to inject NEGATIVE pairs during training: samples where the search
crop is explicitly guaranteed to NOT contain the target.  The head then learns
to output low confidence when the target isn't there.  That's the entire
reason _negative_sample(), _build_shifted_negative_search(), and
_build_distractor_negative_search() exist in this file.
 
──────────────────────────────────────────────────────────────────────────────
WHY "GUARANTEED" MATTERS
──────────────────────────────────────────────────────────────────────────────
 
Early versions of negative sampling just shifted the crop anchor by "a lot"
and hoped for the best.  That doesn't work reliably.  With search_context=2.0
the context window is 5× the object wide (2 left + 1 object + 2 right), so
you need to shift by more than (1 + search_context) × object_width to clear
it — at least 3× the object width horizontally.  The old code shifted by only
1–2× and would frequently still overlap the target.
 
The current code uses _bboxes_overlap() as a hard geometric gate after EVERY
candidate crop is proposed.  No crop is accepted until it provably has zero
pixel overlap with the target bounding box.  The minimum-shift heuristic is
just a sampling bias to make the search faster; the overlap check is the
actual guarantee.
 

 
You don't need to hard-code these.  The pattern is auto-detected from files
on disk the first time a frame is accessed.  New datasets work automatically
as long as they follow any common numbering convention.
 
──────────────────────────────────────────────────────────────────────────────
SAMPLE STRUCTURE (both positive and negative)
──────────────────────────────────────────────────────────────────────────────
 
Every sample contains four crops:
 
  template         – target crop (template-sized) from frame t_idx
  dynamic_template – target crop (template-sized) from an intermediate frame
  dynamic_search   – target crop (search-sized)   from that SAME intermediate frame
  search           – either the real target region (positive) or a crop that
                     is 100% confirmed to not contain the target (negative)
"""
 
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
 
import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
 
from utils.box_coder import SiamABCBoxCoder
from utils.utils import (
    clamp_bbox,
    extend_bbox,
    get_extended_crop,
)
 
_FRAME_PATTERNS = [
    "{:08d}.jpg",
    "img/{:08d}.jpg",
    "{:06d}.jpg",
    "img/{:04d}.jpg",
    "{:04d}.jpg",
    "img/{:06d}.jpg",
    "{:06d}.png",
    "img/{:04d}.png",
]
 
 
def _detect_frame_pattern(
    seq_path: str,
    probe_indices: List[int],
) -> Optional[str]:
    """
    Figure out how a dataset names its frame images by testing a handful
    of known frame indices against every pattern we know about.
 
    We probe the first frame, last frame, and middle frame — if all three
    exist under the same pattern, that's almost certainly the right one.
    Returns None if nothing matches, which triggers the directory-listing
    fallback in TrackingSequence._infer_pattern_from_directory().
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
    """
    Build the per-cell regression weight map for a given bounding box.
 
    Cells near the target center get weight 1.0, cells far away get 0.0.
    The head uses this to focus regression loss on cells that are actually
    close to the target — there's no point penalising the regression output
    at a cell that's three object-widths away from the target center.
 
    The "label" is essentially a Manhattan-distance threshold: cells within
    r_pos steps of the target's cell on the score map are positive, cells
    between r_pos and r_neg are neutral (0.5), everything else is background.
    """
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


def _iou_label_from_regression_map(
    regression_map: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Build an IoU-proxy target map from FCOS distances.

    regression_map is expected in [4, H, W] channel order:
      0: left, 1: top, 2: right, 3: bottom
    """
    left = regression_map[0]
    top = regression_map[1]
    right = regression_map[2]
    bottom = regression_map[3]

    valid = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)
    lr = np.minimum(left, right) / (np.maximum(left, right) + eps)
    tb = np.minimum(top, bottom) / (np.maximum(top, bottom) + eps)
    centerness = np.sqrt(np.clip(lr * tb, 0.0, 1.0))
    iou_proxy = np.where(valid, centerness, 0.0).astype(np.float32)
    return iou_proxy[None, ...]


class TrackingSequence:
    """
    A single annotated video sequence, dataset-agnostic.
 
    This wraps all the per-sequence bookkeeping — reading annotations,
    finding frame images, sampling valid frames — so the main dataset
    class doesn't have to think about it.
 
    The row dict you pass in must contain:
        seq_path    – folder where frame images live
        annot_path  – path to the groundtruth .txt file (x,y,w,h per line)
        start_idx   – first frame index (inclusive, 1-based for UAV datasets)
        end_idx     – last frame index (inclusive)
        n_frames    – total frame count
        class       – what kind of object is being tracked
        dataset     – which dataset this came from (UAV123 / DTB70 / etc.)
 
    Optionally:
        frame_pattern – e.g. "img/{:04d}.jpg". If you leave this out, we
                        auto-detect it from the files on disk.
    """
 
    def __init__(
        self,
        row: Dict,
    ):
        self.seq_path = row["seq_path"]
        self.annot_path = row["annot_path"]
        self.start_idx = int(row["start_idx"])
        self.end_idx = int(row["end_idx"])
        self.n_frames = int(row["n_frames"])
        cls_value = row.get("yolo_target_class", row.get("class"))
        if pd.isna(cls_value):
            cls_value = row.get("class")
        self.cls = cls_value
        self.dataset = row["dataset"]
 
        self._frame_pattern: Optional[str] = row.get("frame_pattern") or None
        self._bboxes: Optional[List[np.ndarray]] = None
 
    def _resolve_frame_pattern(
        self,
    ) -> str:
        """
        Try to figure out the frame naming pattern by probing a few frames
        from this sequence.  Falls back to scanning the directory if probing
        fails.  Raises RuntimeError if nothing works — at that point you need
        to either add the pattern to _FRAME_PATTERNS or put a 'frame_pattern'
        column in your dataframe.
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
 
    def _infer_pattern_from_directory(
        self,
    ) -> Optional[str]:
        """
        Last-resort frame pattern detection: list the folder, look at the
        first image filename, and guess the zero-padding width and extension
        from it.  Checks both the sequence root and an 'img/' subfolder since
        some datasets (like DTB70) put frames there.
        """
        for subdir in ("", "img"):
            dirpath = os.path.join(self.seq_path, subdir) if subdir else self.seq_path
            if not os.path.isdir(dirpath):
                continue
            candidates = sorted(
                f
                for f in os.listdir(dirpath)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not candidates:
                continue
            name, ext = os.path.splitext(candidates[0])
            width = len(name)
            prefix = (subdir + "/") if subdir else ""
            return f"{prefix}{{:{width:02d}d}}{ext}"
        return None
 
    def frame_path(
        self,
        frame_idx: int,
    ) -> str:
        """
        Get the full path to a specific frame image.
        Auto-detects the naming pattern on first call, caches it after that.
        """
        if self._frame_pattern is None:
            self._frame_pattern = self._resolve_frame_pattern()
        return os.path.join(self.seq_path, self._frame_pattern.format(frame_idx))
 
    def _load_bboxes(
        self,
    ) -> None:
        """
        Read the annotation file into memory.  We load it all at once and
        cache it — annotation files are small and we'll typically access many
        frames from the same sequence during training.
 
        Handles both comma-separated and tab-separated files.  Lines that
        can't be parsed (partial occlusion flags, comment lines, etc.) become
        NaN bounding boxes, which is_valid_bbox() will reject safely.
        """
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
                    bboxes.append(np.full(4, np.nan, dtype=np.float32))
        self._bboxes = bboxes
 
    def get_bbox(
        self,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Return the ground-truth bounding box for a given frame.
        Returns a NaN array if the index is out of range — callers should
        always check is_valid_bbox() before using the result.
        """
        self._load_bboxes()
        assert self._bboxes is not None
        local = frame_idx - self.start_idx
        if local < 0 or local >= len(self._bboxes):
            return np.full(4, np.nan, dtype=np.float32)
        return self._bboxes[local].copy()
 
    @staticmethod
    def is_valid_bbox(
        bbox: np.ndarray,
    ) -> bool:
        """
        Check whether a bounding box is actually usable for training.
 
        We reject boxes that are NaN (annotation parse failures), have
        dimensions of 5 pixels or less (too small to crop meaningfully),
        or have negative coordinates (off-screen boxes from some annotation
        styles).  These edge cases come up more than you'd expect in real
        UAV datasets.
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
        Randomly pick a frame index that has a valid bounding box and an
        image file that actually exists on disk.
 
        We shuffle the candidate indices instead of sampling with replacement
        so we don't waste tries on the same bad frames repeatedly.  Returns
        None if we exhausted max_tries without finding anything usable —
        callers should handle this gracefully (skip the sequence, try another).
        """
        lower_end = (
            max(lower_end, self.start_idx) if lower_end is not None else self.start_idx
        )
        upper_end = (
            min(upper_end, self.end_idx) if upper_end is not None else self.end_idx
        )
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
        Pick a (template frame, search frame) pair from this sequence.
 
        The template frame is sampled first.  The search frame is then
        picked from the window [template_frame, template_frame + max_frame_gap].
        This enforces a temporal ordering — the template always comes before
        (or at) the search frame — which mirrors how a real tracker works.
 
        When max_frame_gap=0 is passed, we allow t_idx == s_idx, which is
        used for the dynamic template/search pair where we want both crops
        from the exact same moment in time.
 
        Returns (None, None, None) if no valid pair could be found, which
        happens with short sequences or sequences with many invalid frames.
        """
        lower_end = (
            max(lower_end, self.start_idx) if lower_end is not None else self.start_idx
        )
        upper_end = (
            min(upper_end, self.end_idx) if upper_end is not None else self.end_idx
        )
 
        failure = (None, None, None)
 
        t_idx = self.sample_valid_frame(rng, lower_end=lower_end, upper_end=upper_end)
        if t_idx is None:
            return failure
 
        lo = max(lower_end, t_idx)
        hi = min(upper_end, t_idx + max_frame_gap)
        candidates = [
            i for i in range(lo, hi + 1) if (max_frame_gap == 0 or i != t_idx)
        ]
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
    Training dataset for SiamABC on UAV footage.
 
    ──────────────────────────────────────────────────────────────────────────
    THE CORE PROBLEM THIS DATASET SOLVES
    ──────────────────────────────────────────────────────────────────────────
 
    SiamABC's classification head learns to compare a template (what we're
    tracking) against a search region (where we think it might be).  If
    training only ever shows it positive pairs — search regions that DO
    contain the target — the head never learns what "not there" looks like.
 
    In practice this means the confidence score stays high everywhere.
    The tracker can't tell the difference between "I found the target" and
    "I'm looking at a blank wall".  This kills re-detection, makes the
    tracker drift to distractors, and makes confidence-based early stopping
    useless.
 
    The solution: neg_ratio controls what fraction of training samples are
    NEGATIVE — search crops explicitly confirmed to contain zero target pixels.
    The head learns to output low confidence on those, which means high
    confidence becomes a meaningful signal again.
 
    ──────────────────────────────────────────────────────────────────────────
    TWO KINDS OF NEGATIVES
    ──────────────────────────────────────────────────────────────────────────
 
    75% of negatives use the "shifted" strategy: take the same frame as the
    search crop but shift the anchor far away from the target.  This teaches
    the head to reject background from the same sequence as the target.
 
    25% use the "distractor" strategy: randomly sample a background patch
    anywhere in the image.  This provides more variety and catches cases where
    the shifted strategy keeps hitting similar textures.
 
    Both strategies use _bboxes_overlap() as a hard geometric gate — no crop
    is accepted until it's provably non-overlapping with the target.
 
    ──────────────────────────────────────────────────────────────────────────
    SAMPLE STRUCTURE
    ──────────────────────────────────────────────────────────────────────────
 
    Every sample (positive or negative) returns the same four crops:
 
        template         – what the target looked like at time t
        dynamic_template – what the target looked like at some time between t and s
        dynamic_search   – same intermediate frame, search-crop style
        search           – positive: target at time s / negative: guaranteed non-target
 
    The dynamic pair exists so the model can learn from target appearance
    changes over time, not just the initial template vs. final search.
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
        Set up the dataset.
 
        dataframe      – one row per sequence, see TrackingSequence for required columns
        tracking_config – dict of tracker hyperparams (sizes, context factors, etc.)
        num_samples    – how many samples to report per epoch (sampling is random so
                         this is just a schedule length, not a data limit)
        neg_ratio      – fraction of samples that will be negative pairs.
                         0.5 means the head sees as many "not there" crops as "there"
                         crops, which works well in practice.
        seed           – fixed seed so runs are reproducible
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
 
        self._template_transform = self._get_default_transform(
            tracking_config["template_size"]
        )
        self._search_transform = self._get_default_transform(
            tracking_config["instance_size"]
        )
        self._dynamic_search_transform = self._get_default_transform(
            tracking_config["instance_size"]
        )

        self.occlusion_aug_prob = float(
            np.clip(tracking_config.get("occlusion_aug_prob", 0.35), 0.0, 1.0)
        )
        self.occlusion_min_area = float(
            tracking_config.get("occlusion_min_area", 0.04)
        )
        self.occlusion_max_area = float(
            tracking_config.get("occlusion_max_area", 0.20)
        )
        self.occlusion_max_patches = int(
            max(1, tracking_config.get("occlusion_max_patches", 2))
        )
        self.search_blur_prob = float(
            np.clip(tracking_config.get("search_blur_prob", 0.15), 0.0, 1.0)
        )
        self.search_jpeg_prob = float(
            np.clip(tracking_config.get("search_jpeg_prob", 0.15), 0.0, 1.0)
        )

        self.neg_cross_seq_prob = float(
            np.clip(tracking_config.get("neg_cross_seq_prob", 0.45), 0.0, 1.0)
        )
        self.neg_same_class_prob = float(
            np.clip(tracking_config.get("neg_same_class_prob", 0.70), 0.0, 1.0)
        )
        self.neg_shifted_prob = float(
            np.clip(tracking_config.get("neg_shifted_prob", 0.40), 0.0, 1.0)
        )
 
        self.neg_ratio = neg_ratio
        self.num_samples = num_samples
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self._rng_worker_id: Optional[int] = None
        self._rng_worker_seed: Optional[int] = None
 
        self.box_coder = SiamABCBoxCoder(tracking_config)
 
        self.sequences: List[TrackingSequence] = [
            TrackingSequence(row.to_dict())
            for _, row in dataframe.reset_index(drop=True).iterrows()
        ]
        self.class_to_indices: Dict[str, List[int]] = {}
        for idx, seq in enumerate(self.sequences):
            cls_key = self._sequence_class_key(seq.cls)
            if cls_key:
                self.class_to_indices.setdefault(cls_key, []).append(idx)
 
        if "dataset" in dataframe.columns:
            breakdown = dataframe["dataset"].value_counts().to_dict()
        else:
            breakdown = {"unknown": len(self.sequences)}
 
        print(
            f"[Dataset] {len(self.sequences)} sequences | "
            f"sources={breakdown} | "
            f"num_samples={num_samples} | neg_ratio={neg_ratio:.0%} | "
            f"cross_seq_neg={self.neg_cross_seq_prob:.0%} | "
            f"same_class_neg={self.neg_same_class_prob:.0%} | "
            f"occlusion_aug={self.occlusion_aug_prob:.0%} | "
            f"template_size={self.template_size} | instance_size={self.instance_size} | "
            f"score_size={self.score_size}"
        )
 
    def __len__(
        self,
    ) -> int:
        return self.num_samples
 
    def __getitem__(
        self,
        _idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Return one training sample.
 
        The index is ignored — sampling is random each call because we want
        each epoch to see a different random mix.  If sampling fails after 5
        attempts (bad sequence, all frames corrupt, etc.) we fall back to a
        zero-filled dummy negative so the DataLoader never stalls.
        """
        self._ensure_worker_rng()
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

    def _ensure_worker_rng(
        self,
    ) -> None:
        """
        Ensure each DataLoader worker has an independent RNG stream.

        PyTorch forks dataset objects across workers. If we keep a single RNG that
        was seeded in __init__, workers will generate identical sample sequences.
        We derive worker-local seeds from PyTorch's worker seed so every worker
        follows a distinct, deterministic random stream.
        """
        worker = get_worker_info()
        if worker is None:
            worker_id = -1
            worker_seed = self.seed
        else:
            worker_id = int(worker.id)
            worker_seed = int(worker.seed) & 0xFFFFFFFF

        if (
            self._rng_worker_id == worker_id
            and self._rng_worker_seed == worker_seed
        ):
            return

        self.rng = random.Random(worker_seed)
        self.np_rng = np.random.default_rng(worker_seed)
        self._rng_worker_id = worker_id
        self._rng_worker_seed = worker_seed
 
    def _positive_sample(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a positive training sample — one where the search crop DOES
        contain the target and the labels point to its real location.
 
        We also sample a "dynamic" pair from an intermediate frame between
        the template and search frames.  This gives the model a view of how
        the target appearance evolves over time, which helps with sequences
        that have scale changes, rotation, or deformation.
 
        Falls back to _dummy_negative() if 10 consecutive sequence attempts
        all fail (e.g. the chosen sequences have mostly invalid annotations).
        """
        for _ in range(10):
            seq = self.rng.choice(self.sequences)
 
            t_crop, t_idx, template_bbox, s_crop, s_idx, bbox_in_c, ok = (
                self.sample_search_template_pair(
                    self.rng, seq, max_frame_gap=self.max_frame_gap
                )
            )
            if not ok:
                continue
 
            dt_crop, _, _, ds_crop, _, _, ok_dyn = self.sample_search_template_pair(
                self.rng,
                seq,
                lower_end=t_idx,
                upper_end=s_idx,
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
            iou_label = _iou_label_from_regression_map(bbox_label)

            reg_weight = _regression_weight_label(
                bbox_in_c, self.instance_size, self.score_size
            ).numpy()
 
            return self._pack(
                t_crop,
                dt_crop,
                s_crop,
                ds_crop,
                cls_label,
                bbox_label,
                iou_label,
                reg_weight,
                is_positive=True,
                path=seq.frame_path(s_idx),
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
        Pick and crop a (template, search) pair from a sequence.
 
        This is the main building block for both positive and negative samples.
        It handles frame index sampling, image loading, and crop extraction.
        Returns a 7-tuple ending in False if anything goes wrong — callers
        check the last element before using the rest.
        """
        failure = (None, None, None, None, None, None, False)
        frame_gap = self.max_frame_gap if max_frame_gap is None else max_frame_gap
 
        t_idx, s_idx, s_bbox = seq.sample_search_template_idx_pair(
            rng,
            max_frame_gap=frame_gap,
            lower_end=lower_end,
            upper_end=upper_end,
        )
        if t_idx is None or s_idx is None or s_bbox is None:
            return failure
 
        t_bbox = seq.get_bbox(t_idx)
        t_img = self._load_image(seq.frame_path(t_idx))
        s_img = self._load_image(seq.frame_path(s_idx))
 
        t_crop, template_bbox = self.get_template_crop(t_img, t_bbox)
        s_crop, bbox_in_c, _ = self.get_search_crop(s_img, s_bbox)
 
        return t_crop, t_idx, template_bbox, s_crop, s_idx, bbox_in_c, True
 
    def get_template_crop(
        self,
        image: np.ndarray,
        rect: np.ndarray,
    ):
        """
        Cut out and preprocess a template crop from an image.
 
        We apply a small random shift and scale to the bounding box before
        cropping.  This augments the template so the model sees the target
        from slightly different viewpoints and scales during training, which
        makes it more robust to the kind of small jitter that happens between
        frames in real tracking.
        """
        shifted = self.apply_shift_scale(
            rect,
            image.shape,
            shift_factor=self.template_image_shift,
            scale_factor=self.template_image_scale,
        )
        if not self.is_valid_bbox(shifted):
            shifted = rect
 
        context = extend_bbox(
            shifted,
            offset=self.template_bbox_offset,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            context=context,
            crop_size=self.template_size,
        )
        img = self._preprocess_image(crop, self._template_transform)
        return img, template_bbox
 
    def get_search_crop(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        apply_occlusion_aug: bool = True,
    ):
        """
        Cut out and preprocess a search-region crop from an image.
 
        Like the template crop but larger (controlled by search_context).
        We also randomise the context multiplier slightly on each call via
        _get_random_context() — this teaches the model to handle search
        regions of varying scale, which matters when the tracker's estimated
        object size drifts during inference.
        """
        shifted = self.apply_shift_scale(
            bbox,
            image.shape,
            shift_factor=self.search_image_shift,
            scale_factor=self.search_image_scale,
        )
        bbox_for_crop = clamp_bbox(shifted, shape=image.shape[:2])
        context = extend_bbox(
            bbox_for_crop,
            offset=self._get_random_context(),
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        crop, search_bbox, ctx = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=self.instance_size,
            context=context,
            padding_value=np.mean(image, axis=(0, 1)),
        )
        if apply_occlusion_aug:
            crop = self._apply_search_augmentations(crop)
        img = self._preprocess_image(crop, self._search_transform)
        return img, search_bbox, ctx
 
    def _negative_sample(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a negative training sample — one where the search crop is
        GUARANTEED to not contain the target.
 
        ──────────────────────────────────────────────────────────────────────
        WHY WE NEED THIS
        ──────────────────────────────────────────────────────────────────────
 
        SiamABC's head assigns a classification confidence score to every
        cell in its output map.  If you only train it on positive pairs (where
        the target is always present in the search region), the head learns
        to be confident everywhere — it literally never sees a case where "no
        target" is the right answer.
 
        During inference this is catastrophic.  The tracker drifts, gets
        stuck on distractors, and its confidence output is useless as a signal
        because it's always high regardless of whether the target is actually
        visible.
 
        By feeding negative samples, we give the head explicit training signal
        for the "target is not here" case.  The all-zeros classification label
        means the loss pushes confidence toward zero for these crops.  After
        training, high confidence becomes a genuinely informative signal.
 
        ──────────────────────────────────────────────────────────────────────
        STRUCTURE
        ──────────────────────────────────────────────────────────────────────
 
        template + dynamic_template + dynamic_search are all POSITIVE crops —
        we still want the model to see a real target in those positions.
        Only the `search` crop is negative.  The all-zeros cls_label tells
        the loss that there's nothing to find in the search region.
 
        75% of the time we use the "shifted" strategy (anchor pushed far from
        the target in a random cardinal direction).  25% of the time we use
        the "distractor" strategy (random background patch anywhere in the
        image).  Both strategies validate every candidate with _bboxes_overlap()
        before accepting it — no target pixel can sneak into the crop.
        """
        for _ in range(20):
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

            dyn_idx = seq.sample_valid_frame(self.rng, lower_end=t_idx, upper_end=s_idx)
            if dyn_idx is None:
                dyn_img = t_img
                dyn_bbox = t_bbox.copy()
            else:
                dyn_img = self._load_image(seq.frame_path(dyn_idx))
                dyn_bbox = seq.get_bbox(dyn_idx)

            if not seq.is_valid_bbox(dyn_bbox):
                continue

            dynamic_template, _ = self.get_template_crop(dyn_img, dyn_bbox)
            dynamic_search, _, _ = self.get_search_crop(dyn_img, dyn_bbox)

            strategy_roll = self.rng.random()
            s_crop: Optional[torch.Tensor] = None

            if strategy_roll < self.neg_cross_seq_prob:
                s_crop = self._build_cross_sequence_negative_search(anchor_seq=seq)

            if s_crop is None and strategy_roll < (self.neg_cross_seq_prob + self.neg_shifted_prob):
                s_crop = self._build_shifted_negative_search(
                    seq,
                    s_idx,
                    s_bbox,
                    hard_mode=True,
                )

            if s_crop is None:
                s_crop = self._build_distractor_negative_search(seq, s_idx, s_bbox)
            if s_crop is None:
                s_crop = self._build_shifted_negative_search(
                    seq,
                    s_idx,
                    s_bbox,
                    hard_mode=False,
                )
            if s_crop is None:
                continue

            S = self.score_size
            cls_label = np.zeros((1, S, S), dtype=np.float32)
            bbox_label = np.zeros((4, S, S), dtype=np.float32)
            iou_label = np.zeros((1, S, S), dtype=np.float32)
            reg_weight = np.zeros((S, S), dtype=np.float32)

            return self._pack(
                t_crop,
                dynamic_template,
                s_crop,
                dynamic_search,
                cls_label,
                bbox_label,
                iou_label,
                reg_weight,
                is_positive=False,
            )
 
        return self._dummy_negative()

    @staticmethod
    def _sequence_class_key(
        cls_value: Any,
    ) -> str:
        if cls_value is None:
            return ""
        key = str(cls_value).strip().lower()
        if key in {"", "nan", "none"}:
            return ""
        return key

    def _sample_negative_source_sequence(
        self,
        anchor_seq: TrackingSequence,
    ) -> Optional[TrackingSequence]:
        if len(self.sequences) < 2:
            return None

        anchor_cls = self._sequence_class_key(anchor_seq.cls)
        same_class_pool: List[TrackingSequence] = []
        if anchor_cls:
            same_indices = self.class_to_indices.get(anchor_cls, [])
            same_class_pool = [
                self.sequences[idx]
                for idx in same_indices
                if self.sequences[idx] is not anchor_seq
            ]

        if same_class_pool and self.rng.random() < self.neg_same_class_prob:
            return self.rng.choice(same_class_pool)

        other_pool = [seq for seq in self.sequences if seq is not anchor_seq]
        if not other_pool:
            return None
        return self.rng.choice(other_pool)

    def _build_cross_sequence_negative_search(
        self,
        anchor_seq: TrackingSequence,
    ) -> Optional[torch.Tensor]:
        source_seq = self._sample_negative_source_sequence(anchor_seq)
        if source_seq is None:
            return None

        frame_idx = source_seq.sample_valid_frame(self.rng)
        if frame_idx is None:
            return None

        frame_path = source_seq.frame_path(frame_idx)
        if not os.path.exists(frame_path):
            return None
        img = self._load_image(frame_path)

        bbox = source_seq.get_bbox(frame_idx)
        if not source_seq.is_valid_bbox(bbox):
            return None

        search_crop, _, _ = self.get_search_crop(
            image=img,
            bbox=bbox,
            apply_occlusion_aug=False,
        )
        return search_crop
 
    @staticmethod
    def _bboxes_overlap(
        a: np.ndarray,
        b: np.ndarray,
    ) -> bool:
        """
        Return True if two [x, y, w, h] boxes share any pixels at all.
 
        This is the definitive safety gate for negative sample construction.
        Every candidate negative crop goes through this check — if it returns
        True, we throw the candidate away and try again.
 
        The math: two axis-aligned rectangles overlap on both axes at the same
        time iff their x-intervals overlap AND their y-intervals overlap.
        Intervals [a1, a2] and [b1, b2] overlap iff a1 < b2 AND a2 > b1.
 
        Handles negative coordinates correctly, which matters because context
        windows can extend off the image boundary.
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
        hard_mode: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Build a negative search crop by moving the crop anchor far away from
        the target in one of the four cardinal directions.
 
        ──────────────────────────────────────────────────────────────────────
        WHY THIS STRATEGY EXISTS
        ──────────────────────────────────────────────────────────────────────
 
        We want the negative search crop to come from the SAME frame as the
        real search crop.  That makes the negative hard: the background texture
        and lighting are identical to what the tracker actually sees during
        inference, so the head can't cheat by learning "brighter = target" or
        similar low-level cues.
 
        ──────────────────────────────────────────────────────────────────────
        WHY THE SHIFT MAGNITUDE IS (search_context + 1.1) × dimension
        ──────────────────────────────────────────────────────────────────────
 
        With search_context=2.0, the context window extends 2× the object
        width on every side.  The full window is therefore 5× the object wide
        (2 left + 1 object + 2 right).  For a shift of magnitude s along X,
        the nearest edge of the context window is at distance s - 2.5w from
        the target center.  No overlap requires that distance to be positive:
            s > 2.5w  →  s > (1 + search_context) × w
 
        We use (search_context + 1.1) × w as the minimum — the extra 0.1
        is a margin so we don't land exactly on the edge.  The minimum is
        just a sampling bias to avoid wasting rejection-sampling iterations;
        the actual correctness guarantee comes from _bboxes_overlap() at the end.
 
        Returns None after 50 failed attempts (e.g. the object is so large
        there's nowhere on this frame to put a non-overlapping crop).
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
        max_extra_x = (0.65 if hard_mode else 2.0) * w
        max_extra_y = (0.65 if hard_mode else 2.0) * h
        ortho_jitter = 0.25 if hard_mode else 0.5
 
        for _ in range(50):
            direction = self.rng.choice(("left", "right", "up", "down"))
 
            if direction == "right":
                shift_x = self.rng.uniform(min_shift_x, min_shift_x + max_extra_x)
                shift_y = self.rng.uniform(-ortho_jitter, ortho_jitter) * h
            elif direction == "left":
                shift_x = -self.rng.uniform(min_shift_x, min_shift_x + max_extra_x)
                shift_y = self.rng.uniform(-ortho_jitter, ortho_jitter) * h
            elif direction == "down":
                shift_x = self.rng.uniform(-ortho_jitter, ortho_jitter) * w
                shift_y = self.rng.uniform(min_shift_y, min_shift_y + max_extra_y)
            else:
                shift_x = self.rng.uniform(-ortho_jitter, ortho_jitter) * w
                shift_y = -self.rng.uniform(min_shift_y, min_shift_y + max_extra_y)
 
            anchor = clamp_bbox(
                np.array([x + shift_x, y + shift_y, w, h], dtype=np.float32),
                img.shape[:2],
            )
            if not self.is_valid_bbox(anchor):
                continue
 
            context = extend_bbox(
                anchor,
                offset=self.search_context,
                image_width=W,
                image_height=H,
            )
 
            if self._bboxes_overlap(context, target_clamped):
                continue
 
            raw_crop, _, _ = get_extended_crop(
                image=img,
                bbox=anchor,
                context=context,
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
        Build a negative search crop by randomly placing a same-size anchor
        anywhere in the image and keeping it only if it doesn't overlap the target.
 
        ──────────────────────────────────────────────────────────────────────
        WHY THIS STRATEGY EXISTS (AND COMPLEMENTS THE SHIFTED ONE)
        ──────────────────────────────────────────────────────────────────────
 
        The shifted strategy always places the anchor in a specific cardinal
        direction from the target.  Over many training steps the head could
        theoretically learn "if the crop is to the right of the target it's
        negative" — a spatial bias we don't want.
 
        Random placement avoids that.  It also samples background regions that
        the shifted strategy might never reach (e.g. the far corner diagonally
        opposite the target), giving the head more variety in what "no target"
        looks like.
 
        We keep the anchor size the same as the target to produce hard negatives:
        same scale, similar context window size, just a different location.
 
        Like the shifted strategy, _bboxes_overlap() is the definitive check.
        We never accept a crop until it provably contains zero target pixels.
 
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
                anchor,
                offset=self.search_context,
                image_width=W,
                image_height=H,
            )
 
            if self._bboxes_overlap(context, target_clamped):
                continue
 
            raw_crop, _, _ = get_extended_crop(
                image=img,
                bbox=anchor,
                context=context,
                crop_size=self.instance_size,
                padding_value=mean_color,
            )
            return self._preprocess_image(raw_crop, self._search_transform)
 
        return None

    def _apply_search_augmentations(
        self,
        crop: np.ndarray,
    ) -> np.ndarray:
        """
        Apply robustification transforms to positive search crops.
        """
        if crop.ndim != 3 or crop.shape[2] < 3:
            return crop

        out = crop.copy()

        if self.np_rng.random() < self.occlusion_aug_prob:
            out = self._apply_random_occluder(out)

        if self.np_rng.random() < self.search_blur_prob:
            k = int(self.rng.choice((3, 5)))
            out = cv2.GaussianBlur(out, (k, k), 0)

        if self.np_rng.random() < self.search_jpeg_prob:
            quality = int(self.np_rng.integers(30, 86))
            out = self._jpeg_compress(out, quality=quality)

        return out

    def _apply_random_occluder(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if h < 8 or w < 8:
            return image

        out = image.copy()
        min_area = max(1e-6, min(self.occlusion_min_area, self.occlusion_max_area))
        max_area = max(min_area, self.occlusion_max_area)
        patch_count = int(self.np_rng.integers(1, self.occlusion_max_patches + 1))
        mean_rgb = np.mean(out[:, :, :3], axis=(0, 1))

        for _ in range(patch_count):
            area = self.np_rng.uniform(min_area, max_area) * float(h * w)
            aspect = self.np_rng.uniform(0.5, 2.0)
            occ_w = int(np.sqrt(area * aspect))
            occ_h = int(np.sqrt(area / max(aspect, 1e-6)))
            occ_w = int(np.clip(occ_w, 4, w))
            occ_h = int(np.clip(occ_h, 4, h))
            max_x = max(1, w - occ_w + 1)
            max_y = max(1, h - occ_h + 1)
            x0 = int(self.np_rng.integers(0, max_x))
            y0 = int(self.np_rng.integers(0, max_y))
            x1 = x0 + occ_w
            y1 = y0 + occ_h

            if self.np_rng.random() < 0.55:
                color = np.clip(
                    mean_rgb + self.np_rng.normal(0.0, 35.0, size=3),
                    0,
                    255,
                ).astype(np.uint8)
                out[y0:y1, x0:x1, :3] = color
            else:
                patch = out[y0:y1, x0:x1, :3]
                out[y0:y1, x0:x1, :3] = cv2.GaussianBlur(patch, (5, 5), 0)

        return out

    @staticmethod
    def _jpeg_compress(
        image: np.ndarray,
        quality: int = 70,
    ) -> np.ndarray:
        quality = int(np.clip(quality, 5, 100))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            return image
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
 
    def _get_template_crop(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """
        Convenience helper that clamps the bbox, builds a context window,
        and returns a preprocessed template crop — or None if the bbox
        is invalid after clamping.
        """
        bbox = clamp_bbox(bbox, img.shape)
        if not self.is_valid_bbox(bbox):
            return None
        context = extend_bbox(
            bbox,
            offset=self.template_bbox_offset,
            image_width=img.shape[1],
            image_height=img.shape[0],
        )
        crop, _, _ = get_extended_crop(
            image=img,
            bbox=bbox,
            context=context,
            crop_size=self.template_size,
        )
        return self._preprocess_image(crop, self._template_transform)
 
    def _get_search_crop(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """
        Convenience helper that clamps the bbox, builds a search-size context
        window, and returns a preprocessed search crop — or None if invalid.
        """
        bbox = clamp_bbox(bbox, img.shape)
        if not self.is_valid_bbox(bbox):
            return None
        context = extend_bbox(
            bbox,
            offset=self.search_context,
            image_width=img.shape[1],
            image_height=img.shape[0],
        )
        crop, _, _ = get_extended_crop(
            image=img,
            bbox=bbox,
            context=context,
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
        iou_label: np.ndarray,
        reg_weight: np.ndarray,
        is_positive: bool,
        path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble all the pieces of a training sample into a single dict
        that the DataLoader can batch and the trainer can consume directly.
        """
        return {
            "template": self._to_tensor(template),
            "dynamic_template": self._to_tensor(dynamic_template),
            "search": self._to_tensor(search),
            "dynamic_search": self._to_tensor(dynamic_search),
            "cls_label": torch.from_numpy(cls_label),
            "bbox_label": torch.from_numpy(bbox_label),
            "iou_label": torch.from_numpy(iou_label),
            "reg_weight": torch.from_numpy(reg_weight),
            "is_positive": torch.tensor(is_positive, dtype=torch.bool),
            "path": path if path is not None else "None",
        }
 
    def _dummy_negative(
        self,
    ) -> Dict[str, torch.Tensor]:
        """
        Return a zero-filled sample when everything else has failed.
 
        This exists purely to keep the DataLoader alive.  A run that
        produces many dummy negatives is a sign that something is wrong
        with the data paths or annotations — check the [warn] logs.
        """
        S, T, i = self.score_size, self.template_size, self.instance_size
        return {
            "template": torch.zeros(3, T, T),
            "dynamic_template": torch.zeros(3, T, T),
            "search": torch.zeros(3, i, i),
            "dynamic_search": torch.zeros(3, i, i),
            "cls_label": torch.zeros(1, S, S),
            "bbox_label": torch.zeros(4, S, S),
            "iou_label": torch.zeros(1, S, S),
            "reg_weight": torch.zeros(S, S),
            "is_positive": torch.tensor(False, dtype=torch.bool),
            "path": "None",
        }
 
    @staticmethod
    def _load_image(
        path: str,
    ) -> np.ndarray:
        """
        Load an image from disk as RGB.  Raises FileNotFoundError immediately
        if the file is missing — we prefer a clear error over a silent NaN.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    @staticmethod
    def _to_tensor(
        x,
    ) -> torch.Tensor:
        """Convert a numpy array or existing tensor to float32."""
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(x).float()
 
    def _preprocess_image(
        self,
        image: np.ndarray,
        transform: Callable,
    ) -> torch.Tensor:
        """
        Normalise an image and convert it to a (C, H, W) float tensor.
 
        We normalise with ImageNet statistics (mean/std) because the backbone
        was pretrained on ImageNet.  The channel transpose from (H,W,C) to
        (C,H,W) is what PyTorch's conv layers expect.
        """
        img = transform(image[:, :, :3])
        if image.shape[2] > 3:
            img = np.concatenate([img, image[:, :, 3:]], axis=2)
        return self._array_to_batch(img).float()
 
    @staticmethod
    def _array_to_batch(
        x: np.ndarray,
    ) -> torch.Tensor:
        """Transpose (H, W, C) → (C, H, W) for PyTorch."""
        return torch.from_numpy(np.transpose(x, (2, 0, 1)))
 
    def _get_default_transform(
        self,
        img_size: int,
    ) -> Callable:
        """
        Build the standard normalisation pipeline.
 
        Just ImageNet mean/std normalisation via albumentations.
        We keep this as a callable so it's easy to swap in heavier augmentation
        pipelines (colour jitter, blur, etc.) later without changing the
        calling code.
        """
        pipeline = albu.Compose(
            [
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return lambda a: pipeline(image=a)["image"]
 
    def apply_shift_scale(
        self,
        bbox: np.ndarray,
        img_shape,
        shift_factor=0.0,
        scale_factor=0.0,
    ) -> np.ndarray:
        """
        Randomly jitter a bounding box position and size.
 
        shift_factor controls how far the center can move (as a fraction of
        the box dimension).  scale_factor controls how much the size can change.
        Both default to 0 so the method is safe to call even when no augmentation
        is wanted.
 
        The result is clamped to the image boundaries so we never try to crop
        outside the frame.
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        shift_x = self.np_rng.uniform(-shift_factor, shift_factor) * w
        shift_y = self.np_rng.uniform(-shift_factor, shift_factor) * h
        scale = self.np_rng.uniform(1.0 - scale_factor, 1.0 + scale_factor)
        new_w, new_h = w * scale, h * scale
        new_bbox = np.array(
            [cx + shift_x - new_w / 2, cy + shift_y - new_h / 2, new_w, new_h],
            dtype=np.float32,
        )
        return clamp_bbox(new_bbox, shape=img_shape[:2])
 
    def _get_random_context(
        self,
    ) -> float:
        """
        Return a slightly randomised context multiplier around search_context.
 
        Varying the context scale by ±20% during training means the model
        learns to handle search regions that are a bit tighter or looser than
        the nominal size.  This matters because the tracker's size estimate
        drifts during long sequences, so the search region it gets during
        inference won't always be exactly the right scale.
        """
        return self.np_rng.uniform(self.search_context * 0.8, self.search_context * 1.2)
 
    @staticmethod
    def is_valid_bbox(
        bbox: np.ndarray,
    ) -> bool:
        """
        Check whether a bounding box is safe to use.
        Same criteria as TrackingSequence.is_valid_bbox — no NaN, non-negative
        coordinates, and at least 5px in each dimension.
        """
        return (
            not np.any(np.isnan(bbox))
            and bbox[2] > 5
            and bbox[3] > 5
            and bbox[0] >= 0
            and bbox[1] >= 0
        )
 
 
UAV123TrackingDataset = UAVTrackingDataset
