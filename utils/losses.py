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
from typing import Optional

import torch
import torch.nn as nn

from utils.utils import calc_iou


class BoxLoss(nn.Module):
    """
    BBOX Loss: optimizes IoU of bounding boxes
    Original implentation:
    losses = -torch.log(calc_iou(reg_target=target, pred=pred)) was computationally unstable
    those was replaced with: 1 - IoU
    """

    def __init__(self) -> None:
        """
        Initialise the BoxLoss.
        """
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the IoU loss.

        Args:
            pred (torch.Tensor): Predicted bounding boxes.
            target (torch.Tensor): Ground-truth bounding boxes.
            weight (Optional[torch.Tensor]): Mask for positive locations.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        losses = 1 - calc_iou(target, pred)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()


class TrackingHeadLoss(nn.Module):
    """
    Combined loss for SiamABC classification and regression heads.

    Computes Focal Loss for the classification map and IoU Loss (masked by
    positive labels) for the regression map.
    """

    def __init__(self, cls_weight=1.0, bbox_weight=1.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        Initialise the TrackingHeadLoss.

        Args:
            cls_weight (float): Weight for the classification loss.
            bbox_weight (float): Weight for the regression loss.
            focal_alpha (float): Alpha parameter for Focal Loss.
            focal_gamma (float): Gamma parameter for Focal Loss.
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox = BoxLoss()

    def _cls_loss(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1).float()

        pos_idx = label.eq(1).nonzero().squeeze()
        neg_idx = label.eq(0).nonzero().squeeze()

        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            return pred.sum() * 0.0

        def focal_loss_at(idx):
            p = torch.sigmoid(pred[idx])
            lbl = label[idx]
            bce = self.bce(pred[idx], lbl)

            p_t = p * lbl + (1 - p) * (
                1 - lbl)
            alpha_t = self.focal_alpha * lbl + (1 - self.focal_alpha) * (1 - lbl)
            focal_w = alpha_t * (1 - p_t) ** self.focal_gamma

            return (focal_w * bce).mean()

        if pos_idx.numel() == 0:
            return focal_loss_at(neg_idx)
        if neg_idx.numel() == 0:
            return focal_loss_at(pos_idx)

        return 0.5 * focal_loss_at(pos_idx) + 0.5 * focal_loss_at(neg_idx)

    def _bbox_loss(self, pred, target, reg_weight):
        p = pred.permute(0, 2, 3, 1).reshape(-1, 4)
        t = target.permute(0, 2, 3, 1).reshape(-1, 4)
        w = reg_weight.reshape(-1)
        pos = w.gt(0).nonzero().squeeze(1)
        if pos.numel() == 0:
            return pred.sum() * 0.0
        return self.bbox(p[pos], t[pos])

    def forward(self, cls_pred, bbox_pred, cls_label, bbox_label, reg_weight):
        """
        Compute the total combined loss for a batch.

        Args:
            cls_pred (torch.Tensor): Predicted classification logit map.
            bbox_pred (torch.Tensor): Predicted regression map.
            cls_label (torch.Tensor): Ground-truth classification map.
            bbox_label (torch.Tensor): Ground-truth regression map.
            reg_weight (torch.Tensor): Mask for valid regression samples.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'total', 'cls_loss', and 'bbox_loss'.
        """
        cls_loss = self._cls_loss(cls_pred, cls_label)
        bbox_loss = self._bbox_loss(bbox_pred, bbox_label, reg_weight)
        total = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        return {
            "total": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
        }
