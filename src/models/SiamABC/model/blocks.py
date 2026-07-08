"""

Modified from
Main Author of this file: FEAR
Repo: https://github.com/PinataFarms/FEARTracker/tree/main
File: https://github.com/PinataFarms/FEARTracker/blob/main/model_training/model/blocks.py

"""

from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from utils.console import quiet_external_logs

from .adaptive_batch_norm import AdaptiveBatchNorm

with quiet_external_logs():
    from mobile_cv.model_zoo.models.fbnet_v2 import fbnet



def _make_norm(
    num_features: int,
    inference_mode: bool,
    norm_lambda: float = 0.1,
) -> nn.Module:
    """Return AdaptiveBatchNorm when building for inference, plain BatchNorm2d otherwise."""
    if inference_mode:
        return AdaptiveBatchNorm(num_features, norm_lambda=norm_lambda)
    return nn.BatchNorm2d(num_features)


class AdaptiveSequential(nn.Sequential):
    """``nn.Sequential`` that forwards the ``lam`` tensor to any
    ``AdaptiveBatchNorm`` children and calls every other layer normally.

    Using a subclass (rather than a plain loop) means the module tree is
    identical to what ``torch.compile`` / TensorRT see — there is no
    Python-level branching on the *value* of ``lam``, only on its *type*
    at graph-build time (``isinstance`` is resolved at compile time).
    """

    def forward(
        self,
        x: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, AdaptiveBatchNorm):
                x = module(x, lam)
            else:
                x = module(x)
        return x


def conv_2d(
    inp,
    oup,
    kernel_size=3,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    norm=True,
    act=True,
):
    """
    Utility function to create a sequential 2D convolution block.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        groups (int): Number of blocked connections from input to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        norm (bool): If True, adds a BatchNorm2d layer.
        act (bool): If True, adds a SiLU activation layer.

    Returns:
        nn.Sequential: A sequential container of the constructed layers.
    """
    conv = nn.Sequential()
    conv.add_module(
        "conv",
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups),
    )
    if norm:
        conv.add_module("BatchNorm2d", nn.BatchNorm2d(oup))
    if act:
        conv.add_module("Activation", nn.SiLU())
    return conv


class FastParallelPolarizedSelfAttention(nn.Module):
    """
    Polarized Self-Attention (PSA) module for dual-template feature fusion.

    PSA separately models channel-wise and spatial-wise attention maps, then
    combines them via summation. This mechanism is designed to capture high-
    resolution spatial context and long-range channel dependencies efficiently.
    """

    def __init__(
        self,
        channel=256,
        squeeze=2,
    ):
        """
        Initialise the FastParallelPolarizedSelfAttention module.

        Args:
            channel (int): Number of input feature channels.
            squeeze (int): Channel reduction factor for the internal projections.
        """
        super().__init__()

        self.squeeze = squeeze
        self.wv = nn.Conv2d(channel, channel // self.squeeze, kernel_size=(1, 1))

        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // self.squeeze, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)

        self.sigmoid = nn.Sigmoid()
        self.sp_wq = nn.Conv2d(channel, channel // self.squeeze, kernel_size=(1, 1))

        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        x1,
    ):
        """
        Apply polarized self-attention to the input feature map.

        Args:
            x1 (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attended feature tensor of the same shape as input.
        """
        b, c, h, w = x1.size()

        wv = self.wv(x1)
        wv = wv.reshape(b, c // self.squeeze, -1)

        channel_wq = self.ch_wq(x1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = (
            torch.sum(wv * channel_wq.permute(0, 2, 1), dim=2)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        channel_weight = (
            self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )

        spatial_wq = self.sp_wq(x1)
        spatial_wq = self.agp(spatial_wq)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // self.squeeze)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.sum(spatial_wq.permute(0, 2, 1) * wv, dim=1).unsqueeze(1)
        spatial_weight = spatial_wz.reshape(b, 1, h, w)

        out = (self.sigmoid(channel_weight) + self.sigmoid(spatial_weight)) * x1

        return out


class EncoderResNet(nn.Module):
    """
    ResNet-50 based feature encoder for medium-sized SiamABC variants.

    This encoder uses a pretrained ResNet-50 backbone, truncated after the
    third stage (layer3) to produce high-resolution feature maps suitable
    for object tracking.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        """
        Initialise the EncoderResNet.

        Args:
            pretrained (bool): Whether to load ImageNet-pretrained weights.
        """
        super().__init__()
        self.pretrained = pretrained
        self.last_layer_channels = 1024
        self.model = self._load_model()
        self.layers = self._get_layers()

    def _load_model(
        self,
    ) -> Any:
        model = resnet50(pretrained=self.pretrained)
        return model

    def _get_layers(
        self,
    ) -> List[Any]:
        layers = [
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
        ]
        return layers

    def forward(
        self,
        x: Any,
    ) -> List[Any]:
        """
        Extract features from the ResNet backbone.

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            List[torch.Tensor]: A list of feature maps from each backbone stage.
        """
        encoder_maps = []
        for layer in self.layers:
            x = layer(x)
            encoder_maps.append(x)
        return encoder_maps


class Encoder(nn.Module):
    """
    Lightweight FBNet-based feature encoder for small SiamABC variants.

    Optimised for real-time inference on mobile or edge devices, this encoder
    uses an FBNet architecture to extract hierarchical features.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        """
        Initialise the Encoder.

        Args:
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        self.encoder_channels = {
            "layer0": 352,
            "layer1": 112,
            "layer2": 32,
            "layer3": 24,
            "layer4": 16,
        }

    def _load_model(
        self,
    ) -> Any:
        model_name = "fbnet_c"
        model = fbnet(model_name, pretrained=self.pretrained)
        return model

    def _get_stages(
        self,
    ) -> List[Any]:
        stages = [
            self.model.backbone.stages[:2],
            self.model.backbone.stages[2:5],
            self.model.backbone.stages[5:9],
            self.model.backbone.stages[9:18],
            self.model.backbone.stages[18:23],
        ]
        return stages

    def forward(
        self,
        x: Any,
    ) -> List[Any]:
        """
        Extract features from the FBNet backbone.

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            List[torch.Tensor]: A list of feature maps from each FBNet stage.
        """
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class ConvBlock(nn.Module):
    """
    Depthwise-Separable Convolution block.

    This block implements a standard depthwise convolution followed by a
    pointwise convolution, significantly reducing the parameter count and
    computational cost compared to standard convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        """
        Initialise the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the depthwise convolution kernel.
            stride (int): Stride of the depthwise convolution.
            padding (int): Padding for the depthwise convolution.
            dilation (int): Dilation for the depthwise convolution.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass the input through the depthwise and pointwise layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with out_channels.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdjustLayer(nn.Module):
    """
    Feature projection and alignment neck.

    Used to map backbone features of varying channel widths to a uniform
    target dimensionality, ensuring compatibility with the attention and
    regression heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        crop_rate: int = 4,
    ):
        """
        Initialise the AdjustLayer.

        Args:
            in_channels (int): Number of channels from the backbone.
            out_channels (int): Desired output channel count.
            crop_rate (int): Spatial cropping rate (unused in current implementation).
        """
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.size_threshold = 20
        self.crop_rate = crop_rate

    def forward(
        self,
        x,
    ):
        """
        Pass features through the adjustment projection.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Projected feature map.
        """
        x_ori = self.downsample(x)
        adjust = x_ori
        return adjust


class BoxTower(nn.Module):
    """
    Box Tower for FCOS regression
    """

    def __init__(
        self,
        towernum: int = 4,
        conv_block: str = "regular",
        inchannels: int = 512,
        outchannels: int = 256,
        gaussian_map=False,
        inference_mode: bool = False,
        norm_lambda: float = 0.1,
        model_size: str = "M",
    ):
        """
        Initialise the BoxTower regression and classification head.

        Args:
            towernum (int): Depth of the classification branch.
            conv_block (str): Convolution type ('regular' or depthwise-separable).
            inchannels (int): Input channel count.
            outchannels (int): Hidden and output channel count for the branches.
            gaussian_map (bool): Reserved for auxiliary heatmap prediction.
            inference_mode (bool): Whether to build the block for inference.
            norm_lambda (float): Normalization momentum factor.
            model_size (str): Preset model size ('S' or 'M').
        """
        super().__init__()
        tower = []
        cls_tower = []

        self.cls_encode = EncodeBackbone(
            in_channels=inchannels, out_channels=outchannels, conv_block=conv_block
        )
        self.reg_encode = EncodeBackbone(
            in_channels=inchannels, out_channels=outchannels, conv_block=conv_block
        )

        if model_size == "S":
            self.cls_dw = CorrelationConcat(
                num_channels=outchannels,
                inference_mode=inference_mode,
                norm_lambda=norm_lambda,
            )
            self.reg_dw = CorrelationConcat(
                num_channels=outchannels,
                inference_mode=inference_mode,
                norm_lambda=norm_lambda,
            )
        else:
            self.cls_dw = CorrelationConcatAtt(
                num_channels=outchannels,
                inference_mode=inference_mode,
                norm_lambda=norm_lambda,
            )
            self.reg_dw = CorrelationConcatAtt(
                num_channels=outchannels,
                inference_mode=inference_mode,
                norm_lambda=norm_lambda,
            )

        for i in range(4):
            tower.append(
                ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
            )
            tower.append(_make_norm(outchannels, inference_mode, norm_lambda))
            tower.append(nn.ReLU())

        for i in range(towernum):
            cls_tower.append(
                ConvBlock(outchannels, outchannels, kernel_size=3, stride=1, padding=1)
            )
            cls_tower.append(_make_norm(outchannels, inference_mode, norm_lambda))
            cls_tower.append(nn.ReLU())

        self.add_module("bbox_tower", AdaptiveSequential(*tower))
        self.add_module("cls_tower", AdaptiveSequential(*cls_tower))

        self.bbox_pred = ConvBlock(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = ConvBlock(outchannels, 1, kernel_size=3, stride=1, padding=1)

        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(
        self,
        search_org,
        search,
        kernel,
        lam: torch.Tensor,
    ):

        cls_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        cls_x = self.cls_encode(search)

        reg_z = kernel.reshape(kernel.size(0), kernel.size(1), -1)
        reg_x = self.reg_encode(search)

        cls_dw = self.cls_dw(cls_z, cls_x, search_org, lam)
        reg_dw = self.reg_dw(reg_z, reg_x, search_org, lam)

        x_reg = self.bbox_tower(reg_dw, lam)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)

        c = self.cls_tower(cls_dw, lam)
        cls = 0.1 * self.cls_pred(c)

        return x, cls, cls_dw, x_reg


class EncodeBackbone(nn.Module):
    """
    Encode backbone feature
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_block: str = "regular",
    ):
        """
        Initialise the EncodeBackbone module.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
            conv_block (str): Convolution type.
        """
        super().__init__()
        self.matrix11_s = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x,
    ):
        """
        Encode the backbone features.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Refined feature map.
        """
        return self.matrix11_s(x)


class CorrelationConcat(nn.Module):
    """
    Correlation module
    """

    def __init__(
        self,
        num_channels: int,
        num_corr_channels: int = 64,
        inference_mode: bool = False,
        norm_lambda: float = 0.1,
    ):
        """
        Initialise the CorrelationConcat module.

        Args:
            num_channels (int): Output channel count.
            num_corr_channels (int): Number of correlation map channels.
            inference_mode (bool): Whether to build the block for inference.
            norm_lambda (float): Normalization momentum factor.
        """
        super().__init__()

        in_size = num_channels + num_corr_channels
        self.enc = AdaptiveSequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            _make_norm(num_channels, inference_mode, norm_lambda),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        z,
        x,
        d,
        lam: torch.Tensor = None,
    ):
        """
        Perform correlation via matrix multiplication and concatenation.

        Args:
            z (torch.Tensor): Template feature map (reshaped).
            x (torch.Tensor): Search region feature map.
            d (torch.Tensor): Original search features for residual connection.
            lam (torch.Tensor, optional): Lambda value for test-time adaptation.

        Returns:
            torch.Tensor: Combined correlation features.
        """
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1)
        if lam is None:
            lam = torch.zeros(1, device=x.device)
        s = self.enc(s, lam)
        return s


class CorrelationConcatAtt(nn.Module):
    """
    Correlation module
    """

    def __init__(
        self,
        num_channels: int,
        num_corr_channels: int = 64,
        inference_mode: bool = False,
        norm_lambda: float = 0.1,
    ):
        super().__init__()

        in_size = num_channels + num_corr_channels
        self.enc = AdaptiveSequential(
            ConvBlock(in_size, num_channels, kernel_size=3, padding=1),
            _make_norm(num_channels, inference_mode, norm_lambda),
            nn.ReLU(inplace=True),
        )

        self.att = FastParallelPolarizedSelfAttention(num_channels, 1)

    def forward(
        self,
        z,
        x,
        d,
        lam: torch.Tensor,
    ):
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([s, d], dim=1)
        s = self.enc(s, lam)
        s = self.att(s)
        return s
