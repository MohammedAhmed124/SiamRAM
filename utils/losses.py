"""
Loss functions for fine-tuning the SiamABC classification and regression heads.

Why these two losses?
─────────────────────
cls  → Focal Loss (binary)
    The classification map is extremely sparse: for a 16×16 score map there are
    256 locations and only a handful are 'positive' (near the target centre).
    Standard BCE would collapse to predicting 0 everywhere to minimise loss.
    Focal Loss solves this by down-weighting easy negatives (already predicted
    correctly with high confidence) so the model focuses on hard examples.
    This is especially important when training on cross-sequence NEGATIVES
    (all-zero labels) because the model must learn "nothing here" without
    ignoring the gradients entirely.

bbox → GIoU Loss (masked to positive locations only)
    Standard L1 / smooth-L1 has a scale problem: absolute pixel errors are
    meaningless across objects of wildly different sizes.  IoU is scale-
    invariant.  GIoU additionally penalises non-overlapping predictions by
    measuring the gap between the prediction and the enclosing box.
    We mask the regression loss to locations where cls_label > threshold
    so negative samples (all-zero cls) contribute zero regression loss —
    exactly what we want.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from external.SiamABC.core.models import SiamABC

from external.SiamABC.core.models.loss import BoxLoss



class TrackingHeadLoss(nn.Module):
    def __init__(self, cls_weight=1.0, bbox_weight=1.0, 
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cls_weight  = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce  = nn.BCEWithLogitsLoss(reduction='none')  
        self.bbox = BoxLoss()

    def _cls_loss(self, pred, label):
        pred  = pred.view(-1)
        label = label.view(-1).float()

        pos_idx = label.eq(1).nonzero().squeeze()
        neg_idx = label.eq(0).nonzero().squeeze()

        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            return pred.sum() * 0.0

        def focal_loss_at(idx):
            p    = torch.sigmoid(pred[idx])
            lbl  = label[idx]
            bce  = self.bce(pred[idx], lbl)

            p_t     = p * lbl + (1 - p) * (1 - lbl) # high when model is confident and correect, low when confident and wrong
            alpha_t = self.focal_alpha * lbl + (1 - self.focal_alpha) * (1 - lbl)
            focal_w = alpha_t * (1 - p_t) ** self.focal_gamma

            return (focal_w * bce).mean()

        if pos_idx.numel() == 0:
            return focal_loss_at(neg_idx)
        if neg_idx.numel() == 0:
            return focal_loss_at(pos_idx)

        return 0.5 * focal_loss_at(pos_idx) + 0.5 * focal_loss_at(neg_idx)

    def _bbox_loss(self, pred, target, reg_weight):
        p = pred.permute(0,2,3,1).reshape(-1, 4)
        t = target.permute(0,2,3,1).reshape(-1, 4)
        w = reg_weight.reshape(-1)
        pos = w.gt(0).nonzero().squeeze(1)
        if pos.numel() == 0:
            return pred.sum() * 0.0
        return self.bbox(p[pos], t[pos])

    def forward(self, cls_pred, bbox_pred, cls_label, bbox_label, reg_weight):
        cls_loss  = self._cls_loss(cls_pred, cls_label)
        bbox_loss = self._bbox_loss(bbox_pred, bbox_label, reg_weight)
        total = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        return {
            "total":     total,
            "cls_loss":  cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
        }