"""
Bounding box encoding and decoding for Siamese trackers.

This module is the bridge between the raw bounding boxes that come from ground-truth
annotations (or from the tracker's own previous predictions) and the spatial map
representations that the SiamABC neural network actually learns from and outputs.

In the SiamABC tracking system (Section 2.1 of the SiamRAM paper), the convolutional
BoxTower head produces two output maps over a spatial grid:
  - A regression map (shape [B, 4, H, W]) encoding how far each grid cell is from
    the four sides of the target box (left, top, right, bottom distances).
  - A classification map (shape [B, 1, H, W]) scoring how likely each grid cell is
    to lie inside the target box.

This module provides:
  - BoxCoder (abstract base): the shared interface and coordinate grid setup.
  - SiamABCBoxCoder: the concrete encoder/decoder for SiamABC.
  - get_box_coder: a factory function to pick the right coder by tracker name.
  - gauss_1d / gauss_2d / gaussian_label_function: utilities for generating soft
    Gaussian classification labels used during training.
"""

import math
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .utils import make_grid

TrackerEncodeResult = namedtuple(
    "TrackerEncodeResult", ["regression_map", "classification_label"]
)
TrackerDecodeResult = namedtuple("TrackerDecodeResult", ["bbox", "pred_coords"])


class BoxCoder(ABC):
    """
    Abstract base class that defines the contract for all bounding-box coders in
    the SiamABC / SiamRAM tracking framework.

    Why does this class exist?
    --------------------------
    The SiamABC tracker (Section 2.1) operates in two directions:

      1. ENCODE (training time): We have a ground-truth box [x, y, w, h] in image
         coordinates and we need to convert it into the spatial maps (regression and
         classification) that the network is trained to predict.

      2. DECODE (inference time): The network outputs regression and classification
         maps, and we need to convert those back into a single bounding box in image
         coordinates that we can draw on the frame and feed into the next tracking step.

    Both directions rely on a fixed spatial coordinate grid — a lookup table that maps
    each cell (row, col) of the score map to a real pixel position in the search-region
    image. That grid is pre-computed once here in __init__ via make_grid() and reused
    across every frame, which is why it lives in the base class.

    Subclasses (e.g. SiamABCBoxCoder) implement encode() and decode() according to the
    specific box parameterisation used by their tracker head.
    """

    def __init__(
        self,
        tracker_config: Dict[str, Any],
    ) -> None:
        """
        Set up the spatial coordinate grid shared by all encode/decode operations.

        What is the grid?
        -----------------
        The BoxTower head in SiamABC outputs predictions over an H×W grid (typically
        16×16). Each cell (i, j) in that grid corresponds to a specific pixel location
        (grid_x[i,j], grid_y[i,j]) in the 256×256 search-region image. The grid is
        determined by the stride (how many pixels each grid step represents) and the
        overall image size.

        This grid is critical for both encoding (converting pixel boxes into per-cell
        distance targets) and decoding (converting per-cell predictions back into pixel
        coordinates).

        Args:
            tracker_config (Dict[str, Any]):
                A configuration dictionary that must contain:
                  - "score_size" (int): The spatial resolution of the output map, e.g. 16
                    means the head produces a 16×16 grid of predictions.
                  - "total_stride" (int): The effective stride of the network backbone,
                    e.g. 16, meaning each grid cell corresponds to a 16×16 pixel patch
                    in the input image.
                  - "instance_size" (int): The width/height of the search-region crop fed
                    into the network, e.g. 256 pixels.
        """
        super().__init__()
        self.tracker_config = tracker_config
        self.grid_x, self.grid_y = make_grid(
            tracker_config["score_size"],
            tracker_config["total_stride"],
            tracker_config["instance_size"],
        )

    def to_device(
        self,
        device: Union[str, int],
    ) -> "BoxCoder":
        """
        Move the internal coordinate grids to a specific compute device (CPU or GPU).

        Why is this needed?
        -------------------
        During training and inference, the regression map and classification map tensors
        live on whatever device the GPU is running on (e.g. 'cuda:0'). The encode() and
        decode() methods perform element-wise arithmetic between those tensors and the
        grid arrays (grid_x, grid_y). For that arithmetic to work without copying data
        back and forth between devices, the grids must be on the same device as the
        model outputs.

        This method is typically called once, right after the coder is instantiated and
        the model is moved to GPU.

        Args:
            device (Union[str, int]):
                The target device identifier. Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'.

        Returns:
            BoxCoder:
                Returns self so calls can be chained, e.g.:
                    coder = SiamABCBoxCoder(config).to_device('cuda:0')
        """
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        return self

    @abstractmethod
    def encode(
        self,
        bboxes: np.array,
    ) -> TrackerEncodeResult:
        """
        Convert ground-truth bounding boxes into the spatial maps used for training.

        Why do we need this?
        --------------------
        The SiamABC BoxTower head does not directly predict [x, y, w, h] at a single
        point. Instead, it predicts a dense map: for every cell in the 16×16 output grid,
        it outputs how far that cell is from each of the four sides of the target box
        (left, top, right, bottom — the FCOS-style parameterisation). The binary
        classification map then labels which cells actually fall inside the box.

        This encode() method creates those two training targets from a batch of raw boxes.

        Args:
            bboxes (np.array):
                Ground-truth boxes in [x, y, w, h] format (top-left corner + size),
                one row per sample in the batch.
                Shape: [batch_size, 4]

        Returns:
            TrackerEncodeResult — a named tuple with:
              - regression_map (np.array, shape [batch, 4, 16, 16]):
                    For every grid cell, the distances to the left, top, right, and
                    bottom edges of the ground-truth box, in pixels.
              - classification_label (np.array, shape [batch, 1, 16, 16]):
                    A binary mask that is 1.0 for every grid cell that lies strictly
                    inside the ground-truth box, and 0.0 everywhere else.
        """
        pass

    @abstractmethod
    def decode(
        self,
        regression_map: torch.Tensor,
        classification_map: torch.Tensor,
        use_sigmoid: bool = True,
    ) -> TrackerDecodeResult:
        """
        Convert the network's raw output maps into a single bounding box in image
        coordinates.

        Why do we need this?
        --------------------
        At inference time (every frame of the video), SiamABC produces two maps:
          - regression_map: per-cell predicted distances to box edges.
          - classification_map: per-cell confidence that the target is there.

        To actually track the object, we need to collapse these maps into one box
        [x, y, w, h] that can be drawn on the frame, stored in the RAM buffer,
        and fed back into the next frame's search crop. This decode() method does
        that collapse by finding the highest-confidence cell and reading off the
        corresponding box.

        Args:
            regression_map (torch.Tensor):
                The raw regression output from the BoxTower head.
                Shape: [batch, 4, H, W] — 4 channels for (left, top, right, bottom).
            classification_map (torch.Tensor):
                The raw classification output from the BoxTower head.
                Shape: [batch, 1, H, W] — one confidence score per grid cell.
            use_sigmoid (bool):
                If True, applies a sigmoid activation to classification_map before
                finding the peak. Set to True when passing raw logits from the network.
                Set to False when classification_label is already a 0/1 map (e.g.
                during encode verification).

        Returns:
            TrackerDecodeResult — a named tuple with:
              - bbox (torch.Tensor, shape [1, 4]):
                    The predicted box in [x, y, w, h] image-pixel coordinates.
              - pred_coords (list of (int, int)):
                    The (row, col) index of the grid cell with the highest classification
                    score — useful for debugging and visualisation.
        """
        pass


class SiamABCBoxCoder(BoxCoder):
    """
    Concrete bounding-box coder for the SiamABC tracker.

    How does it work?
    -----------------
    SiamABC adopts the FCOS-style box parameterisation: instead of predicting an
    (x, y, w, h) tuple at a single anchor point, every cell (i, j) in the output
    grid predicts four distances:
        left   = grid_x[i,j] - box_left
        top    = grid_y[i,j] - box_top
        right  = box_right  - grid_x[i,j]
        bottom = box_bottom - grid_y[i,j]

    A cell is considered a positive (inside the box) when all four distances are > 0,
    meaning the cell literally lies within the ground-truth rectangle. The network is
    trained to regress those distances and to classify positive vs. negative cells.

    At decode time, the cell with the highest classification score is picked, and its
    four predicted distances are combined with the grid coordinates to recover [x, y,
    w, h] in image space.

    This coder is used throughout the SiamRAM pipeline:
      - During training, encode() builds the supervision signal for the BoxTower loss.
      - During normal tracking (Section 2.1), decode() turns each forward pass into a
        box estimate that is stored in the RAM buffer (Section 3.2) and optionally used
        to update the dynamic template (Section 2.2).
      - During reacquisition (Section 4.3, Phase 3), decode() is used to verify whether
        a YOLO candidate, after re-initialising SiamABC, produces a confident track.
    """

    def __init__(
        self,
        tracker_config: Dict[str, Any],
    ) -> None:
        """
        Initialise the SiamABCBoxCoder.

        Delegates entirely to the parent BoxCoder.__init__ to build the coordinate grid.
        No additional state is needed beyond what the base class provides.

        Args:
            tracker_config (Dict[str, Any]):
                Must contain 'score_size', 'total_stride', and 'instance_size'.
                See BoxCoder.__init__ for full details.
        """
        super().__init__(tracker_config=tracker_config)

    @torch.no_grad()
    def encode(
        self,
        bboxes: torch.Tensor,
    ) -> TrackerEncodeResult:
        """
        Convert a batch of ground-truth boxes into regression maps and classification
        labels for training the SiamABC BoxTower head.

        Step-by-step logic:
        -------------------
        1. Each box [x, y, w, h] is broadcast across the full 16×16 grid.
        2. For every cell, compute the four signed distances to each box edge:
               left   = grid_x - box_x
               top    = grid_y - box_y
               right  = (box_x + box_w) - grid_x
               bottom = (box_y + box_h) - grid_y
           These four channels form the regression target.
        3. A cell is labelled positive (1.0) if and only if ALL four distances are
           strictly positive — i.e. the cell lies entirely inside the box. Cells on the
           boundary or outside are labelled 0.0.

        The @torch.no_grad() decorator ensures this runs without building a computation
        graph, since it's pure label generation, not part of the learnable forward pass.

        Args:
            bboxes (torch.Tensor):
                Batch of ground-truth boxes in [x, y, w, h] format (top-left origin).
                Shape: [batch_size, 4]
                Example: tensor([[54, 60, 84, 90]]) — one box, x=54, y=60, w=84, h=90.

        Returns:
            TrackerEncodeResult — a named tuple containing:
              - regression_map (torch.Tensor, shape [batch, 4, 16, 16], dtype float32):
                    Channel 0: distance from each grid cell to the LEFT edge of the box.
                    Channel 1: distance from each grid cell to the TOP edge of the box.
                    Channel 2: distance from each grid cell to the RIGHT edge of the box.
                    Channel 3: distance from each grid cell to the BOTTOM edge of the box.
                    Negative values indicate the cell is outside the box on that side.
              - classification_label (torch.Tensor, shape [batch, 1, 16, 16], dtype float32):
                    1.0 where the cell is strictly inside the ground-truth box, else 0.0.
        """
        bboxes = bboxes.unsqueeze(-1).unsqueeze(-1)
        left = self.grid_x - bboxes[:, 0]
        top = self.grid_y - bboxes[:, 1]
        right = bboxes[:, 0] + bboxes[:, 2] - self.grid_x
        bottom = bboxes[:, 1] + bboxes[:, 3] - self.grid_y
        regression_map = torch.stack((left, top, right, bottom), dim=1).float()
        regression_map_min, _ = torch.min(regression_map, dim=1)
        classification_label = (regression_map_min.unsqueeze(1) > 0).float()
        return TrackerEncodeResult(
            regression_map=regression_map, classification_label=classification_label
        )

    @torch.no_grad()
    def decode(
        self,
        regression_map,
        classification_map,
        use_sigmoid=True,
        pred_location=None,
    ):
        """
        Decode the SiamABC network output maps into a single bounding box in image
        pixel coordinates.

        This is called every frame during normal tracking (Section 2.1) and again
        during Phase-3 tracker verification in the reacquisition pipeline (Section 4.3).
        The result is the [x, y, w, h] box that gets stored in the RAM buffer,
        evaluated for the DRM promotion check, and fed back as the search region
        centre for the next frame.

        Step-by-step logic:
        -------------------
        1. Optionally apply sigmoid to turn raw classification logits into [0,1] scores.
        2. If pred_location is not pre-computed, reconstruct absolute pixel coordinates
           for every cell's four edges using the stored grid:
               x1[i,j] = grid_x[i,j] - regression_map[:, 0, i, j]   (left edge)
               y1[i,j] = grid_y[i,j] - regression_map[:, 1, i, j]   (top edge)
               x2[i,j] = grid_x[i,j] + regression_map[:, 2, i, j]   (right edge)
               y2[i,j] = grid_y[i,j] + regression_map[:, 3, i, j]   (bottom edge)
        3. Find the grid cell (r_max, c_max) with the highest classification score
           using argmax on the flattened map.
        4. Read the four edge coordinates at that cell and convert to [x, y, w, h].

        The @torch.no_grad() decorator keeps inference memory-efficient.

        Args:
            regression_map (torch.Tensor):
                Raw regression output from the BoxTower head.
                Shape: [batch, 4, H, W]. Channels are (left, top, right, bottom) distances.
            classification_map (torch.Tensor):
                Raw classification output from the BoxTower head.
                Shape: [batch, 1, H, W]. One logit (or probability) per grid cell.
            use_sigmoid (bool):
                Whether to apply sigmoid before finding the peak. Default True.
                Pass False only when classification_map already contains probabilities
                (e.g. during encode-decode round-trip tests with binary labels).
            pred_location (Optional[torch.Tensor]):
                If you have already computed the full [x1, y1, x2, y2] absolute location
                tensor (shape [batch, 4, H, W]), pass it here to skip recomputation.
                This is an optional performance optimisation for callers that pre-compute
                locations once and call decode() repeatedly (e.g. beam-search decoding).
                If None (the default), locations are computed internally.

        Returns:
            TrackerDecodeResult — a named tuple containing:
              - bbox (torch.Tensor, shape [1, 4]):
                    The predicted bounding box in [x, y, w, h] format, in search-region
                    pixel coordinates (i.e. relative to the 256×256 crop, not the full frame).
              - pred_coords (list of (int, int)):
                    A one-element list containing the (row, col) index of the winning
                    grid cell. Useful for debugging and overlay visualisation.
        """
        if use_sigmoid:
            classification_map = classification_map.float().sigmoid()
        cls_map = classification_map[0, 0]

        if pred_location is None:
            pred_location = torch.stack(
                [
                    self.grid_x - regression_map[:, 0],
                    self.grid_y - regression_map[:, 1],
                    self.grid_x + regression_map[:, 2],
                    self.grid_y + regression_map[:, 3],
                ],
                dim=1,
            )
        loc = pred_location[0]

        H, W = cls_map.shape
        flat_idx = int(torch.argmax(cls_map).item())
        r_max, c_max = divmod(flat_idx, W)

        x1, y1, x2, y2 = loc[:, r_max, c_max]
        bbox = torch.stack([x1, y1, x2 - x1, y2 - y1])

        return TrackerDecodeResult(bbox=bbox.unsqueeze(0), pred_coords=[(r_max, c_max)])


def get_box_coder(
    tracker_config: Dict[str, Any],
    tracker_name: str = "SiamABC",
) -> Optional[BoxCoder]:
    """
    Factory function that returns the correct BoxCoder instance for a given tracker.

    Why does this exist?
    --------------------
    The SiamRAM system is designed to support multiple tracker backends. Rather than
    scattering if/else logic throughout the training and inference code, this single
    factory function centralises tracker-to-coder mapping. Callers just pass the
    tracker name and get back a fully initialised coder, without needing to know which
    concrete class to import or instantiate.

    Currently supported trackers:
      - "SiamABC": Returns a SiamABCBoxCoder configured with the given tracker_config.

    Args:
        tracker_config (Dict[str, Any]):
            Configuration dictionary forwarded to the BoxCoder constructor.
            Must contain 'score_size', 'total_stride', and 'instance_size'.
            Example: {"score_size": 16, "total_stride": 16, "instance_size": 256}

        tracker_name (str):
            Name of the tracker whose coder to instantiate. Default is "SiamABC".
            If an unrecognised name is given, returns None — the caller should handle
            this gracefully (e.g. raise a descriptive error or fall back to a default).

    Returns:
        Optional[BoxCoder]:
            A fully initialised BoxCoder subclass ready for encode() and decode() calls,
            or None if the tracker_name is not recognised.
    """
    if tracker_name == "SiamABC":
        return SiamABCBoxCoder(tracker_config=tracker_config)
    return None


def gauss_1d(
    sz,
    sigma,
    center,
    end_pad=0,
    density=False,
) -> torch.Tensor:
    """
    Compute a 1D Gaussian distribution evaluated at integer positions along an axis.

    Why do we need this?
    --------------------
    Hard binary classification labels (1 inside box, 0 outside) create a sharp, brittle
    training signal. In practice, SiamABC can benefit from soft Gaussian labels that
    smoothly peak at the box centre and fall off toward the edges. This encourages the
    network to produce smooth, peaked score maps rather than flat plateaus, making
    the argmax-based decode() step more reliable.

    gauss_1d() is the building block for gauss_2d() and gaussian_label_function(). It
    evaluates a zero-mean Gaussian (shifted by `center`) at every integer position from
    -(sz-1)/2 to (sz+1)/2, optionally padded by end_pad extra positions on the right.

    Mathematical form:
        gauss[i] = exp( -1/(2*sigma^2) * (k_i - center)^2 )
    where k_i are the sample positions.

    Args:
        sz (int):
            The number of positions to evaluate along the axis (e.g. 16 for a 16-cell
            score map dimension).
        sigma (float):
            Standard deviation of the Gaussian in grid-cell units. Larger sigma → wider,
            softer label; smaller sigma → sharper, more localised label.
        center (torch.Tensor):
            The centre position(s) of the Gaussian in grid-cell coordinates, relative
            to the middle of the axis. Shape: [batch_size] — one centre per sample.
        end_pad (int):
            Number of extra positions appended to the right of the evaluation range.
            Used internally by gauss_2d() for even-kernel alignment. Default is 0.
        density (bool):
            If True, normalises the output so the values integrate (approximately) to 1
            over the evaluated range, making it a proper probability density. If False
            (default), the peak is 1.0 and the tails decay toward 0.

    Returns:
        torch.Tensor:
            Shape: [batch_size, sz + end_pad]. Each row is the Gaussian curve for one
            sample in the batch, evaluated at the sz + end_pad integer positions.
    """
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(
    sz,
    sigma,
    center,
    end_pad=(0, 0),
    density=False,
) -> torch.Tensor:
    """
    Compute a 2D Gaussian label map by taking the outer product of two 1D Gaussians.

    Why do we need this?
    --------------------
    The SiamABC classification head outputs a 2D spatial map (H×W). A soft 2D Gaussian
    label centred on the target provides a smooth training target that is more robust
    than a hard binary mask: cells near the centre receive high label values, cells far
    from it receive low values, and the transition is gradual.

    This function factorises the 2D Gaussian into two independent 1D Gaussians (one per
    axis) and computes their outer product — a standard trick that is both efficient
    and exact for axis-aligned Gaussians.

    Args:
        sz (Tuple[torch.Tensor, torch.Tensor]):
            A pair of tensors (or ints) giving the height and width of the output map,
            e.g. (torch.tensor(16), torch.tensor(16)) for a 16×16 score map.
        sigma (Union[float, Tuple[float, float]]):
            Standard deviation of the Gaussian. Can be a single float (same sigma for
            both axes) or a (sigma_h, sigma_w) pair for an anisotropic Gaussian.
        center (torch.Tensor):
            Centre coordinates in grid-cell space. Shape: [batch_size, 2], where
            column 0 is the horizontal offset and column 1 is the vertical offset.
        end_pad (Tuple[int, int]):
            Extra padding on the right/bottom of each axis, passed through to gauss_1d().
            Typically (0, 0) unless even-kernel alignment is needed.
        density (bool):
            If True, normalise the output to unit volume (probability density). If False
            (default), the peak is 1.0.

    Returns:
        torch.Tensor:
            Shape: [batch_size, sz[0]+end_pad[0], sz[1]+end_pad[1]].
            The 2D Gaussian label map for each sample in the batch.
    """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(
        center.shape[0], 1, -1
    ) * gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(
        center.shape[0], -1, 1
    )


def gaussian_label_function(
    target_bb,
    sigma_factor=0.1,
    kernel_sz=1,
    feat_sz=16,
    image_sz=256,
    end_pad_if_even=True,
    density=False,
    uni_bias=0,
) -> torch.Tensor:
    """
    Build a 2D Gaussian classification label map centred on the target bounding box.

    Why do we need this?
    --------------------
    During training, the SiamABC classification head needs a spatially soft label: rather
    than a hard 0/1 mask (which SiamABCBoxCoder.encode() can also produce), this function
    generates a smooth Gaussian that peaks at the box centre and decays outward. This is
    the approach adopted from DiMP / PrDiMP [see SiamRAM references], and it encourages
    the network to produce peaked, unambiguous classification maps — which in turn makes
    the argmax-based decode() step in SiamABCBoxCoder more reliable, even in cluttered
    search regions or in the presence of distractors (Section 2.1).

    An optional uniform bias (uni_bias) mixes in a small flat floor, preventing the label
    from being exactly zero outside the target and providing a softer gradient signal.

    Coordinate mapping:
    -------------------
    The function converts the box centre from image-pixel space to feature-map space:
        1. Compute target_center in pixels from the [x1,y1,x2,y2] corners.
        2. Normalise to [-0.5, 0.5] relative to the image centre.
        3. Scale by feat_sz to get the centre in grid-cell units.
        4. Adjust by a half-pixel offset for even-sized kernels.

    Args:
        target_bb (torch.Tensor):
            Ground-truth boxes in [x1, y1, x2, y2] format (pixel coordinates, top-left
            and bottom-right corners). Shape: [batch_size, 4].
        sigma_factor (float):
            Controls the width of the Gaussian relative to the feature map size. The
            effective sigma is: sigma = sigma_factor * sqrt(feat_sz_h * feat_sz_w).
            Default 0.1 gives a moderately tight label. Larger values spread the label
            wider; smaller values concentrate it near the box centre.
        kernel_sz (Union[int, Tuple[int, int]]):
            Kernel size used only to compute the even-padding offset. Typically 1 (no
            effect). Set to an even integer when the upstream conv kernel needs alignment.
        feat_sz (Union[int, Tuple[int, int]]):
            Spatial size of the output label map. Matches the score map resolution of
            the BoxTower head, typically 16 (giving a 16×16 label). Can be (H, W) for
            non-square maps.
        image_sz (Union[int, Tuple[int, int]]):
            The pixel size of the search-region image crop, typically 256. Used to
            normalise target_center before mapping to grid-cell coordinates. Can be
            (H, W) for non-square crops.
        end_pad_if_even (bool):
            If True, adds one extra cell of padding on each axis whose kernel_sz is even,
            ensuring correct alignment for even-sized convolution kernels. Default True.
        density (bool):
            If True, normalise the Gaussian to unit volume. Default False (peak = 1.0).
        uni_bias (float):
            A small constant (e.g. 0.01) mixed into the label to avoid exactly-zero
            targets outside the Gaussian. Default 0 (no floor).

    Returns:
        torch.Tensor:
            Shape: [batch_size, feat_sz_h (+ pad), feat_sz_w (+ pad)].
            The soft Gaussian classification label, ready to be used as the target for
            the classification head's BCE or focal loss.

    Example:
        >>> boxes = torch.tensor([[80., 90., 170., 180.]])  # [x1,y1,x2,y2]
        >>> labels = gaussian_label_function(boxes, sigma_factor=0.1, feat_sz=16, image_sz=256)
        >>> labels.shape
        torch.Size([1, 16, 16])
    """

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = (target_bb[:, 0:2] + target_bb[:, 2:4]) * 0.5
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * torch.Tensor(
        [(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2]
    )

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label


if __name__ == "__main__":
    tracker_config = {"score_size": 16, "total_stride": 16, "instance_size": 256}

    box_coder = SiamABCBoxCoder(tracker_config=tracker_config)
    bboxes = torch.tensor([[54, 60, 84, 90]])
    encoded = box_coder.encode(bboxes)
    print()