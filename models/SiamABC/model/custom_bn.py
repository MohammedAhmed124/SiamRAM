"""
https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""

import torch
from torch import nn


class AdaptiveBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 contineous=False, norm_lambda=0.1):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.contineous  = contineous
        self._norm_lambda = norm_lambda


        self.register_buffer('_lam', torch.tensor(norm_lambda, dtype=torch.float32))

    def enable_tta(self):
        self._lam.fill_(self._norm_lambda)

    def disable_tta(self):
        self._lam.fill_(0.0)
    @property
    def tta_enabled(self) -> bool:
        return self._lam.item() > 0.0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        batch_var  = x.var([0, 2, 3], unbiased=False)

        mean = self._lam * batch_mean + (1.0 - self._lam) * self.running_mean
        var  = self._lam * batch_var  + (1.0 - self._lam) * self.running_var

        x = (x - mean[None, :, None, None]) \
            / torch.sqrt(var[None, :, None, None] + self.eps)

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


def replace_layers_adaptive_bn(model, norm_lambda, contineous ):
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            replace_layers_adaptive_bn(module, norm_lambda, contineous)
            
        if isinstance(module, nn.BatchNorm2d):
            mybatch_norm = AdaptiveBatchNorm(module.num_features, norm_lambda=norm_lambda)
            try:
                n = int(n)
                model[n] = mybatch_norm
            except:
                setattr(model, n, mybatch_norm)

