"""
https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""

import torch
from torch import nn


class AdaptiveBatchNorm(nn.BatchNorm2d):
    """BatchNorm2d whose TTA blending weight is supplied at forward-time.

    ``lam`` is the actual blending coefficient passed in by the caller:
      - ``lam = 0.0``           → pure running statistics  (TTA off)
      - ``lam = norm_lambda``   → blended batch + running  (TTA on)

    Keeping ``lam`` as a tensor *input* (rather than a stored buffer) means
    there are no side-effecting mutations between forward calls, so the module
    is fully compatible with ``torch.compile`` / TensorRT.

    During *training* the standard batch-normalisation update is used and
    ``lam`` is ignored, so training behaviour is unchanged.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 contineous=False, norm_lambda=0.1):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.contineous = contineous
        # Stored so the tracker can recover the configured strength when
        # enabling TTA; not used inside forward().
        self._norm_lambda = norm_lambda

    def forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   input feature map  [B, C, H, W]
            lam: scalar tensor — blending weight for batch statistics.
                 0.0 → TTA off (use running stats only).
                 norm_lambda → TTA on (blend batch and running stats).
        """
        self._check_input_dim(x)

        # Training path — standard BN; lam is intentionally ignored.
        if self.training:
            return torch.nn.functional.batch_norm(
                x,
                self.running_mean, self.running_var,
                self.weight, self.bias,
                True,
                self.momentum or (1.0 / float(self.num_batches_tracked + 1)),
                self.eps,
            )

        # ── Single eval path — branch-free, TRT-safe ──────────────────────
        # lam = 0.0 → mean/var collapse to running_mean/var  (TTA off)
        # lam > 0.0 → blended statistics                     (TTA on)
        # The compiler sees ONE static graph regardless of the lam value.
        batch_mean = x.mean([0, 2, 3])
        batch_var = x.var([0, 2, 3], unbiased=False)

        mean = lam * batch_mean + (1.0 - lam) * self.running_mean
        var  = lam * batch_var  + (1.0 - lam) * self.running_var

        x = (x - mean[None, :, None, None]) \
            / torch.sqrt(var[None, :, None, None] + self.eps)

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


def replace_layers_adaptive_bn(model, norm_lambda, continuous):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers_adaptive_bn(module, norm_lambda, continuous)

        if isinstance(module, nn.BatchNorm2d) and not isinstance(module, AdaptiveBatchNorm):
            new_bn = AdaptiveBatchNorm(
                module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                norm_lambda=norm_lambda,
            )
            # ← copy all trained state
            new_bn.load_state_dict(module.state_dict())
            new_bn.to(next(module.parameters(), module.running_mean).device)

            if isinstance(model, (nn.Sequential, nn.ModuleList)) and name.isdigit():
                model[int(name)] = new_bn
            else:
                setattr(model, name, new_bn)