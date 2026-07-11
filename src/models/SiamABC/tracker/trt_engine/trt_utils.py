import copy

import torch
import torch.nn as nn

from ...model.SiamABC import SiamABCNet



class _FeatureExtractorModule(nn.Module):
    """encoder + neck  (deep-copied, dtype-isolated)."""

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(net.encoder)
        self.neck = copy.deepcopy(net.neck)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.neck(self.encoder(x))


class _EncoderModule(nn.Module):
    """Deep-copied encoder used by the hybrid FP16-encoder backbone mode."""

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(net.encoder)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x)


class _NeckModule(nn.Module):
    """Deep-copied neck whose dtype is independent from the source model."""

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.neck = copy.deepcopy(net.neck)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.neck(x)


class _AttentionNeck(nn.Module):
    """
    polarized_self_attention + attention_neck for ONE branch.

    Called twice per track step (template branch, search branch).
    We compile a single dynamic-shape engine and call it for both —
    the weights are shared so this is correct, and one engine guarantees
    the same kernel family for both spatial sizes.

    Input : (B, 2*C, h, w)   — concatenated feature pair
    Output: (B, C,   h, w)   — mixed attention features

    FIX 4: weights are deep-copied (previously held live references to the
    original model, which TRT tracing could silently mutate).
    """

    def __init__(
        self,
        model: SiamABCNet,
    ) -> None:
        super().__init__()

        self.attn = copy.deepcopy(model.polarized_self_attention)
        self.neck = copy.deepcopy(model.attention_neck)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.neck(self.attn(x))


def _cast_module(
    module: nn.Module,
    dtype: torch.dtype,
) -> nn.Module:
    """Cast all params/buffers of *module* to *dtype* in-place and return it."""
    return module.half() if dtype == torch.float16 else module.float()
