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
    Implements the geometric Bounding Box loss based on Intersection-over-Union (IoU).

    ### Why we need this
    Standard coordinate regression (like L1) fails because a 10-pixel error on a 
    tiny drone is a disaster, while a 10-pixel error on a close-up car is negligible. 
    IoU is scale-invariant, making it much more robust for UAV tracking where object 
    sizes vary wildly. We use $1 - IoU$ to transform a similarity measure 
    into a minimizable loss function.

    ### Inputs
    *   **pred**: (Tensor) The predicted bounding boxes from the Box Tower.
    *   **target**: (Tensor) The ground-truth boxes.
    *   **weight**: (Optional Tensor) A spatial mask to prioritize specific regions.

    ### Returns
    *   **Tensor**: A scalar representing the average geometric error.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialise the BoxLoss.
        """
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
    The unified loss objective for the SiamABC core tracker.

    ### Why we need this
    SiamABC produces two outputs simultaneously: a classification map $\hat{c}$ 
    and a regression map $\hat{\Delta}$. This class combines Focal Loss 
    (for finding the object) and BoxLoss (for sizing it) into a single gradient 
    that updates the Box Tower heads[cite: 1]. It ensures the model stays 
    discriminative even when hard negatives are present in the search region.
    """

    def __init__(
        self,
        cls_weight=1.0,
        bbox_weight=1.0,
        iou_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        """
        Configure the loss hyperparameters.

        Args:
            cls_weight (float): Importance of the classification branch.
            bbox_weight (float): Importance of the box refinement branch.
            focal_alpha (float): Balancing factor for rare positive samples.
            focal_gamma (float): The "focusing" parameter that down-weights easy examples.
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox = BoxLoss()

    def _cls_loss(
        self,
        pred,
        label,
    ):
        """
        Internal implementation of Focal Loss for the sparse classification map.

        ### Why we need this
        In a typical search crop, 99% of the pixels are background. A standard loss 
        would make the model "lazy"—it would just predict 'background' for everything 
        to get a low error. Focal Loss forces the model to focus on the 1% of pixels 
        that actually belong to the target.

        ### Inputs
        *   **pred**: Raw logits from the classification head.
        *   **label**: Binary ground-truth map (1 for target center, 0 for background).

        ### Returns
        *   **Tensor**: The weighted classification penalty.
        """
        pred = pred.view(-1)
        label = label.view(-1).float()

        pos_idx = label.eq(1).nonzero().squeeze()
        neg_idx = label.eq(0).nonzero().squeeze()

        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            return pred.sum() * 0.0

        def focal_loss_at(
            idx,
        ):
            p = torch.sigmoid(pred[idx])
            lbl = label[idx]
            bce = self.bce(pred[idx], lbl)

            p_t = p * lbl + (1 - p) * (1 - lbl)
            alpha_t = self.focal_alpha * lbl + (1 - self.focal_alpha) * (1 - lbl)
            focal_w = alpha_t * (1 - p_t) ** self.focal_gamma

            return (focal_w * bce).mean()

        if pos_idx.numel() == 0:
            return focal_loss_at(neg_idx)
        if neg_idx.numel() == 0:
            return focal_loss_at(pos_idx)

        return 0.5 * focal_loss_at(pos_idx) + 0.5 * focal_loss_at(neg_idx)

    def _bbox_loss(
        self,
        pred,
        target,
        reg_weight,
    ):
        """
        Computes regression loss only for the cells that actually contain the target.

        ### Why we need this
        It makes no sense to calculate bounding box error for a pixel that is 
        background. We use `reg_weight` as a mask to ensure that only "positive" 
        locations contribute to the box refinement gradients.

        ### Inputs
        *   **pred**: Predicted regression offsets.
        *   **target**: Ground-truth offsets.
        *   **reg_weight**: A mask where > 0 indicates a positive target location.

        ### Returns
        *   **Tensor**: Scaled regression loss.
        """
        p = pred.permute(0, 2, 3, 1).reshape(-1, 4)
        t = target.permute(0, 2, 3, 1).reshape(-1, 4)
        w = reg_weight.reshape(-1)
        pos = w.gt(0).nonzero().squeeze(1)
        if pos.numel() == 0:
            return pred.sum() * 0.0
        return self.bbox(p[pos], t[pos])

    def _iou_loss(
        self,
        iou_pred: Optional[torch.Tensor],
        iou_label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute IoU-map loss (IoU-proxy supervision).

        We balance positive and negative cells equally to avoid the huge
        background majority dominating gradients.
        """
        if iou_pred is None or iou_label is None:
            if iou_pred is not None:
                return iou_pred.sum() * 0.0
            if iou_label is not None:
                return iou_label.sum() * 0.0
            return torch.zeros((), dtype=torch.float32)

        pred = iou_pred.view(-1)
        label = iou_label.view(-1).float()
        bce = self.bce(pred, label)

        pos = label.gt(0)
        neg = label.eq(0)

        if pos.any() and neg.any():
            return 0.5 * bce[pos].mean() + 0.5 * bce[neg].mean()
        if pos.any():
            return bce[pos].mean()
        if neg.any():
            return bce[neg].mean()
        return iou_pred.sum() * 0.0

    def forward(
        self,
        cls_pred,
        bbox_pred,
        cls_label,
        bbox_label,
        reg_weight,
        iou_pred: Optional[torch.Tensor] = None,
        iou_label: Optional[torch.Tensor] = None,
    ):
        """
        Executes the full forward pass for the multi-task loss objective.

        ### Inputs
        *   **cls_pred**: Classification logits from the network.
        *   **bbox_pred**: Regression map from the network.
        *   **cls_label**: Target classification grid.
        *   **bbox_label**: Target regression grid.
        *   **reg_weight**: Mask for valid regression samples.

        ### Returns
        *   **Dict**: A collection containing the 'total' loss for backprop, 
            plus individual detached losses for logging/monitoring.
        """
        cls_loss = self._cls_loss(cls_pred, cls_label)
        bbox_loss = self._bbox_loss(bbox_pred, bbox_label, reg_weight)
        if iou_pred is None and iou_label is None:
            iou_loss = cls_pred.sum() * 0.0
        else:
            iou_loss = self._iou_loss(iou_pred, iou_label)
        total = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.iou_weight * iou_loss
        )
        return {
            "total": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "iou_loss": iou_loss.detach(),
        }
