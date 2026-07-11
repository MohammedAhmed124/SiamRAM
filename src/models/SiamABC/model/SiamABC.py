"""
SiamABCNet — Siamese Attention-Based Correlation Network for Single Object Tracking.

This module implements SiamABCNet, a dual-branch Siamese network designed for
real-time single object tracking. The architecture pairs a shared feature encoder
with a Polarized Self-Attention mechanism that fuses static template features with
a continuously updated dynamic template, allowing the tracker to adapt to appearance
changes over time without retraining.

At a high level, the pipeline works as follows:
    1. Four image crops are passed in — a static template, a dynamic (updated) template,
       a static search region, and a dynamic search region.
    2. Each crop is independently encoded by a shared-weight backbone.
    3. Template and search features are pairwise concatenated and refined through
       a Polarized Self-Attention block, producing mixed attention maps that capture
       both static appearance and recent motion context.
    4. A Box Tower head consumes the attention-refined features to regress a bounding
       box and predict a classification score for the tracked target.

Architecture variants:
    - 'tiny' (Tiny) — MobileNet-style lightweight encoder (``Encoder``).
    - 'S' (Small)   — ResNet-based encoder (``EncoderResNet``) for higher accuracy.

Compatibility note:
    Several ``collections`` aliases are patched at import time to restore deprecated
    names (e.g., ``collections.Mapping``) that were removed in Python 3.10. This is
    required for compatibility with older third-party libraries that have not yet
    migrated to ``collections.abc``.

Modified from the FEAR tracker by PinataFarms:
    - Repository : https://github.com/PinataFarms/FEARTracker
    - Original file: model_training/model/fear_net.py
    - Author: FEAR (https://github.com/PinataFarms/FEARTracker/blob/main/model_training/model/fear_net.py)
"""

import collections.abc
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

from . import constants
from .blocks import (AdjustLayer, BoxTower, Encoder, EncoderResNet,
                     FastParallelPolarizedSelfAttention)


def _patch_collections_compat_aliases() -> None:
    """
    Restore deprecated `collections` aliases expected by legacy dependencies.
    """
    collections_any = cast(Any, collections)
    alias_names = (
        "Mapping",
        "MutableMapping",
        "Iterable",
        "Iterator",
        "Sequence",
        "MutableSequence",
        "Set",
        "MutableSet",
        "Callable",
        "Hashable",
        "Sized",
        "Container",
        "ValuesView",
        "KeysView",
        "ItemsView",
    )
    for alias in alias_names:
        setattr(collections_any, alias, getattr(collections.abc, alias))


_patch_collections_compat_aliases()


class SiamABCNet(nn.Module):
    """
    Siamese Attention-Based Correlation Network (SiamABCNet).

    SiamABCNet is a single object tracker that extends the classic Siamese
    framework with a dual-template strategy and polarized self-attention.
    Rather than comparing the target against a single frozen template, the model
    maintains two parallel representations — a static template captured at
    initialisation and a dynamic template that is updated during inference to
    reflect recent appearance changes. These two views are fused via attention
    before being cross-correlated with the corresponding search-region features.

    The model exposes two execution paths:
        - ``forward``  — used during training; accepts a packed tuple of four
          tensors and returns a dictionary of losses / auxiliary outputs.
        - ``track``    — used during inference; accepts pre-extracted features
          to avoid redundant computation when the template features are cached
          across frames.

    Args:
        simsiam_dim (int):
            Output dimensionality of the SimSiam projection head.
            Retained for API compatibility; currently unused in the forward pass.
            Defaults to ``2048``.
        simsiam_pred_dim (int):
            Hidden dimensionality of the SimSiam prediction MLP.
            Retained for API compatibility; currently unused in the forward pass.
            Defaults to ``512``.
        pretrained (bool):
            Whether to initialise the backbone with ImageNet-pretrained weights.
            Set to ``False`` when training from scratch or when weights are
            loaded separately. Defaults to ``True``.
        adjust_channels (int):
            Number of output channels for the ``AdjustLayer`` neck. This value
            controls the width of all subsequent attention and regression heads.
            Defaults to ``256``.
        towernum (int):
            Number of convolutional blocks stacked inside the ``BoxTower`` head.
            Higher values increase capacity at the cost of latency.
            Defaults to ``2``.
        max_layer (int):
            Index controlling how many backbone stages are used for feature
            extraction. Accepted values are ``3`` (extracts up to ``layer2``)
            and ``4`` (extracts up to ``layer1``). Only relevant for model
            size ``'tiny'``. Defaults to ``4``.
        conv_block (str):
            Convolution block type passed to ``BoxTower``. Use ``"regular"`` for
            standard convolutions or another supported variant for depthwise-
            separable alternatives. Defaults to ``"regular"``.
        model_size (str):
            Selects the backbone family:
            - ``'tiny'`` — lightweight MobileNet-style encoder (SiamABC-Tiny).
            - ``'S'`` — heavier ResNet-based encoder (default / "Small").
            Defaults to ``'S'``.
        build_simsiam_heads (bool):
            Flag indicating whether SimSiam projection/prediction heads should
            be constructed. Retained for API compatibility; the heads are not
            actively used in the current forward pass. Defaults to ``True``.
        **kwargs:
            Additional keyword arguments accepted for forward compatibility.

    Attributes:
        encoder (nn.Sequential):
            Shared backbone network used to extract spatial features from every
            input crop. Weights are shared across all four branches.
        neck (AdjustLayer):
            Channel-projection layer that maps raw backbone features to the
            uniform ``adjust_channels`` width expected by downstream modules.
        polarized_self_attention (FastParallelPolarizedSelfAttention):
            Attention module that operates on the concatenation of two feature
            maps (e.g., static + dynamic template) and produces a refined,
            context-aware representation.
        attention_neck (AdjustLayer):
            Second channel-projection layer that reduces the doubled channel
            count (``2 × adjust_channels``) produced by concatenation back to
            ``adjust_channels`` after the attention pass.
        connect_model (BoxTower):
            Regression and classification head that cross-correlates the
            attention-refined template with the search region to predict the
            target bounding box and confidence score.
        similarity (nn.CosineSimilarity):
            Cosine similarity module. Available for optional SimSiam-style
            self-supervised auxiliary losses.

    Raises:
        AssertionError:
            Raised during construction if ``max_layer`` is not one of the
            supported values (``3`` or ``4``).
        Exception:
            Raised during construction if ``model_size`` is not ``'S'`` or
            ``'tiny'``.

    Example:
        >>> model = SiamABCNet(model_size='S', pretrained=False).cuda()
        >>> template        = torch.randn(1, 3, 64,  64).cuda()   # static template crop
        >>> dynamic_template= torch.randn(1, 3, 64,  64).cuda()   # updated template crop
        >>> search          = torch.randn(1, 3, 128, 128).cuda()  # static search region
        >>> dynamic_search  = torch.randn(1, 3, 128, 128).cuda()  # dynamic search region
        >>> outputs = model((template, dynamic_template, search, dynamic_search))
        >>> print(outputs.keys())
    """

    def __init__(
        self,
        simsiam_dim: int = 2048,
        simsiam_pred_dim: int = 512,
        pretrained: bool = True,
        adjust_channels: int = 256,
        towernum: int = 2,
        max_layer: int = 4,
        conv_block: str = "regular",
        model_size="S",
        build_simsiam_heads=True,
        build_iou_head: bool = False,
        inference_mode: bool = False,
        norm_lambda: float = 0.1,
        **kwargs,
    ):
        legacy_build_quality = kwargs.pop("build_quality_head", None)
        if legacy_build_quality is not None and not build_iou_head:
            build_iou_head = bool(legacy_build_quality)
        kwargs.pop("build_embedding_head", None)
        kwargs.pop("embedding_dim", None)

        max_layer2name = {3: "layer2", 4: "layer1"}
        self.build_simsiam_heads = build_simsiam_heads
        self.build_iou_head = build_iou_head
        assert max_layer in max_layer2name

        super().__init__()
        if model_size == "tiny":
            self.max_layer = max_layer
            base_encoder = Encoder(pretrained)
            self.encoder = nn.Sequential(*base_encoder.stages[: self.max_layer])
            adjust_in_channels = base_encoder.encoder_channels[
                max_layer2name[max_layer]
            ]
        elif model_size == "S":
            base_encoder = EncoderResNet(pretrained=pretrained)
            adjust_in_channels = base_encoder.last_layer_channels
            self.encoder = nn.Sequential(*base_encoder.layers)

        else:
            raise Exception("Not Implemented")

        self.neck = AdjustLayer(
            in_channels=adjust_in_channels, out_channels=adjust_channels
        )

        self.polarized_self_attention = FastParallelPolarizedSelfAttention(
            adjust_channels + adjust_channels, 2
        )

        self.attention_neck = AdjustLayer(
            adjust_channels + adjust_channels, adjust_channels
        )

        self.connect_model = BoxTower(
            inchannels=adjust_channels,
            outchannels=adjust_channels,
            towernum=towernum,
            conv_block=conv_block,
            inference_mode=inference_mode,
            norm_lambda=norm_lambda,
            model_size=model_size,
        )
        iou_hidden = max(32, adjust_channels // 2)
        self.iou_head = (
            nn.Sequential(
                nn.Conv2d(adjust_channels, iou_hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(iou_hidden, 1, kernel_size=1),
            )
            if self.build_iou_head
            else None
        )

        self.similarity = nn.CosineSimilarity(dim=1)

    def feature_extractor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass a single image crop through the shared backbone encoder.

        This is the lowest-level extraction step and is intentionally kept
        separate so that callers can cache its output and avoid redundant
        computation across frames (see ``track``).
        It is called by `get_features()`.

        Args:
            x (torch.Tensor):
                Input image crop of shape ``(B, 3, H, W)`` normalised to the
                range expected by the backbone (typically ImageNet statistics).

        Returns:
            torch.Tensor:
                Spatial feature map of shape ``(B, C_enc, H', W')``, where
                ``C_enc`` is the backbone's output channel count and ``H'``,
                ``W'`` are spatially downsampled relative to the input.
        """
        x = self.encoder(x)
        return x

    def get_features(
        self,
        crop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract and project features for a given image crop.

        This function passes the crop through the `feature_extractor()` backbone
        and then applies the `neck` layer to project the channel dimensionality
        to `adjust_channels`. It is called by `forward()` for each input crop.

        Args:
            crop (torch.Tensor):
                Input image crop of shape ``(B, 3, H, W)``.

        Returns:
            torch.Tensor:
                Projected feature map of shape ``(B, adjust_channels, H', W')``.
        """
        features = self.feature_extractor(crop)
        features = self.neck(features)
        return features

    def connector(
        self,
        template_mixed_attention: torch.Tensor,
        search_mixed_attention: torch.Tensor,
        search: torch.Tensor,
        lam: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Connect the attention-refined features to the regression head.

        Passes the mixed attention maps and original search features into
        the `BoxTower` (`connect_model`) to compute the bounding box regression
        and target classification scores. Called by both `forward()` and `track()`.

        Args:
            template_mixed_attention (torch.Tensor): The attention-refined template features.
            search_mixed_attention (torch.Tensor): The attention-refined search features.
            search (torch.Tensor): The original raw search features.
            lam (torch.Tensor): The lambda parameter for TTA (Test-Time Augmentation).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The predicted bounding box regression map.
                - The classification score map.
        """
        bbox_pred, cls_pred, _, _ = self.connect_model(
            search_org=search,
            search=search_mixed_attention,
            kernel=template_mixed_attention,
            lam=lam,
        )
        return bbox_pred, cls_pred

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        lam: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Execute the forward pass for training.

        Takes the four image crops, processes each through `get_features()`,
        combines them in the polarized self-attention module, and then calls
        `connector()` to produce tracking outputs.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                Tuple containing the static template, dynamic template,
                static search region, and dynamic search region respectively.
            lam (Optional[torch.Tensor], optional):
                Lambda value for TTA. Defaults to None, in which case a
                zero tensor is used.

        Returns:
            Dict[str, torch.Tensor | List[torch.Tensor] | None]:
                A dictionary containing:
                - Target regression map
                - Target classification map
                - SimSiam auxiliary outputs (None by default)
                - Search to template similarity score (None by default)
                - The attention map generated for the search region
        """
        template, dynamic_template, search, dynamic_search = x

        if lam is None:
            lam = torch.zeros(1, device=template.device)

        template_features = self.get_features(template)
        dynamic_template_features = self.get_features(dynamic_template)

        template_combined_features = torch.concat(
            [template_features, dynamic_template_features], dim=1
        )
        template_attention = self.polarized_self_attention(template_combined_features)
        template_mixed_attention = self.attention_neck(template_attention)

        search_features = self.get_features(search)
        dynamic_search_features = self.get_features(dynamic_search)

        search_combined_features = torch.concat(
            [dynamic_search_features, search_features], dim=1
        )
        search_attention = self.polarized_self_attention(search_combined_features)
        search_mixed_attention = self.attention_neck(search_attention)

        bbox_pred, cls_pred = self.connector(
            template_mixed_attention=template_mixed_attention,
            search_mixed_attention=search_mixed_attention,
            search=search_features,
            lam=lam,
        )
        iou_pred = (
            self.iou_head(search_mixed_attention)
            if self.iou_head is not None
            else None
        )

        simsiam_out_search = None
        simsiam_out_dynamic = None

        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred,
            constants.TARGET_CLASSIFICATION_KEY: cls_pred,
            constants.TARGET_IOU_KEY: iou_pred,
            constants.SIMSIAM_SEARCH_OUT_KEY: simsiam_out_search,
            constants.SIMSIAM_DYNAMIC_OUT_KEY: simsiam_out_dynamic,
            constants.TRACKER_TARGET_SEARCH_SIM_SCORE: None,
            constants.TRACKER_ATTENTION_MAP: search_mixed_attention,
        }

    def track(
        self,
        search_features: torch.Tensor,
        dynamic_search_features: torch.Tensor,
        template_features: torch.Tensor,
        dynamic_template_features: torch.Tensor,
        lam: torch.Tensor,
    ) -> Dict[str, torch.Tensor | None]:
        """
        Execute tracking inference given pre-computed features.

        This bypasses the `get_features()` extraction step for performance,
        taking raw backbone features directly. It passes the concatenated features
        into the polarized self attention, then calls `connector()` to obtain the
        final predictions.

        Args:
            search_features (torch.Tensor): Encoded current search region.
            dynamic_search_features (torch.Tensor): Encoded historical search region.
            template_features (torch.Tensor): Encoded static template.
            dynamic_template_features (torch.Tensor): Encoded dynamic template.
            lam (torch.Tensor): The lambda parameter for test-time augmentation.

        Returns:
            Dict[str, torch.Tensor | None]:
                A dictionary containing the regression map, classification map,
                attention map, and a placeholder for similarity score.
        """

        template_combined_features = torch.concat(
            [template_features, dynamic_template_features], dim=1
        )
        template_attention = self.polarized_self_attention(template_combined_features)
        template_mixed_attention = self.attention_neck(template_attention)

        search_combined_features = torch.concat(
            [dynamic_search_features, search_features], dim=1
        )
        search_attention = self.polarized_self_attention(search_combined_features)
        search_mixed_attention = self.attention_neck(search_attention)

        bbox_pred, cls_pred = self.connector(
            template_mixed_attention=template_mixed_attention,
            search_mixed_attention=search_mixed_attention,
            search=search_features,
            lam=lam,
        )
        iou_pred = (
            self.iou_head(search_mixed_attention)
            if self.iou_head is not None
            else None
        )
        return {
            constants.TARGET_REGRESSION_LABEL_KEY: bbox_pred,
            constants.TARGET_CLASSIFICATION_KEY: cls_pred,
            constants.TARGET_IOU_KEY: iou_pred,
            constants.TRACKER_TARGET_SEARCH_SIM_SCORE: None,
            constants.TRACKER_ATTENTION_MAP: search_mixed_attention,
        }


if __name__ == "__main__":
    model = SiamABCNet(gaussian_map=True).cuda()
    search = torch.randn((2, 3, 128, 128)).cuda()
    dynamic = torch.randn((2, 3, 128, 128)).cuda()
    template = torch.randn((2, 3, 64, 64)).cuda()
