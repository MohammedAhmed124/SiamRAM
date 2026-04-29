"""
https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""

import torch
from torch import nn


class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive Batch Normalisation (ABN) layer for Test-Time Adaptation (TTA).

    ABN extends standard BatchNorm2d by allowing the blending of running
    statistics (calculated during training) with batch statistics (calculated
    during inference). This allows the model to adapt to distribution shifts
    in the search region or template during tracking.

    The blending is controlled by a 'lambda' parameter:
        μ_eff = λ * μ_batch + (1 - λ) * μ_running
        σ_eff = λ * σ_batch + (1 - λ) * σ_running

    When λ = 0, it behaves like standard inference BN.
    When λ > 0, it incorporates local batch statistics.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 contineous=False, norm_lambda=0.1):
        """
        Initialise the AdaptiveBatchNorm layer.

        Args:
            num_features (int): Number of channels in the input tensor.
            eps (float): Small constant for numerical stability.
            momentum (float): Momentum factor for running stats updates.
            affine (bool): Whether to use learnable scale and shift parameters.
            track_running_stats (bool): Whether to track running mean and variance.
            contineous (bool): Reserved for future use in continuous adaptation.
            norm_lambda (float): Default blending factor for Test-Time Adaptation.
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.contineous = contineous
        self._norm_lambda = norm_lambda

        self.register_buffer('_lam', torch.tensor(norm_lambda, dtype=torch.float32))

    def enable_tta(self):
        """
        Enable Test-Time Adaptation by setting the blending factor to norm_lambda.
        """
        self._lam.fill_(self._norm_lambda)

    def disable_tta(self):
        """
        Disable Test-Time Adaptation by setting the blending factor to 0.0.
        """
        self._lam.fill_(0.0)

    @property
    def tta_enabled(self) -> bool:
        """
        Return whether Test-Time Adaptation is currently enabled.
        """
        return self._lam.item() > 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the AdaptiveBatchNorm layer.

        In training mode, this behaves as standard Batch Normalisation.
        In evaluation mode, it blends the running statistics with the current
        batch statistics if TTA is enabled.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalised output tensor.
        """
        self._check_input_dim(x)

        # Training path — standard BN, untouched
        if self.training:
            return torch.nn.functional.batch_norm(
                x,
                self.running_mean, self.running_var,
                self.weight, self.bias,
                True,
                self.momentum or (1.0 / float(self.num_batches_tracked + 1)),
                self.eps,
            )

        # ── Single eval path — no Python branch on tta_enabled ───────────
        # When _lam = 0.0  → mean = running_mean, var = running_var  (TTA off)
        # When _lam > 0.0  → blended stats                           (TTA on)
        # The compiler sees ONE graph for both modes.
        batch_mean = x.mean([0, 2, 3])
        batch_var = x.var([0, 2, 3], unbiased=False)

        mean = self._lam * batch_mean + (1.0 - self._lam) * self.running_mean
        var = self._lam * batch_var + (1.0 - self._lam) * self.running_var

        x = (x - mean[None, :, None, None]) \
            / torch.sqrt(var[None, :, None, None] + self.eps)

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


def replace_layers_adaptive_bn(model, norm_lambda, continuous):
    """
    Recursively replace BatchNorm2d layers in a model with AdaptiveBatchNorm.

    Args:
        model (nn.Module): The model or submodule to process.
        norm_lambda (float): Blending factor for the new ABN layers.
        continuous (bool): Whether to enable continuous adaptation (reserved).
    """
    for name, module in model.named_children():

        if len(list(module.children())) > 0:
            replace_layers_adaptive_bn(module, norm_lambda, continuous)

        if isinstance(module, nn.BatchNorm2d):
            mybatch_norm = AdaptiveBatchNorm(module.num_features, norm_lambda=norm_lambda)
            if isinstance(model, (nn.Sequential, nn.ModuleList)) and name.isdigit():
                model[int(name)] = mybatch_norm
            else:
                setattr(model, name, mybatch_norm)
