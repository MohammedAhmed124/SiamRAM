"""
Training loop for fine-tuning the SiamABC BoxTower (cls + bbox head).

Usage
─────
    from your_model_file import SiamABCNet
    from train import train_model

    model = SiamABCNet(pretrained=True, model_size='S', max_layer=3)
    # load your pre-trained weights here if needed:
    # model.load_state_dict(torch.load('siamambc_pretrained.pth'))

    model = train_model(
        model        = model,
        csv_path     = 'UAV123.csv',
        data_root    = '/path/to/UAV123',   # root that contains data_seq/ and anno/
        num_epochs   = 10,
        batch_size   = 32,
        score_size   = 16,                # MUST match model's output stride
    )
"""

import time
from collections import defaultdict
from typing import Dict, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.SiamABC.model.constants as constants

from .losses import TrackingHeadLoss


class OptimizerParamGroup(TypedDict):
    """
    A helper structure for PyTorch's parameter group logic.

    ### Why we need this
    In SiamRAM, different modules (like the Attention Neck vs. the Box Tower) require 
    different learning rates to stay stable[cite: 1]. This dictionary format allows us 
    to pass these specific configurations to the AdamW optimizer in an organized way.
    """

    params: list[nn.Parameter]
    lr: float
    name: str


def freeze_backbone_only(
    model,
    verbose: bool = True,
) -> None:
    """
    Lock the ResNet encoder weights and unlock the attention and head modules.

    ### Why we need this exactly
    The SiamABC backbone is already pre-trained on massive datasets to recognize 
    general shapes. We don't want to "break" that knowledge by over-training it 
    on a smaller UAV dataset. Instead, we freeze the backbone and only train the 
    'Polarized Self-Attention' and 'Box Tower' modules[cite: 1]. This forces the 
    model to focus exclusively on learning how to better distinguish targets from 
    distractors and how to draw more accurate boxes[cite: 1].

    ### Inputs
    *   **model**: The full SiamABC architecture.
    *   **verbose**: If True, prints a summary of trainable vs. frozen parameters.
    """

    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable_modules = [
        model.neck,
        model.polarized_self_attention,
        model.attention_neck,
        model.connect_model,
        model.similarity_avgpool,
    ]
    if model.build_simsiam_heads:
        trainable_modules += [model.avgpool, model.classifier, model.predictor]

    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    if verbose:
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"Frozen    : {frozen:,}  ({100 * frozen / total:.1f}%)")
        print(f"Trainable : {trainable:,}  ({100 * trainable / total:.1f}%)")
        print(f"Total     : {total:,}")
        print("\nTrainable modules:")
        for m in trainable_modules:
            name = type(m).__name__
            params = sum(p.numel() for p in m.parameters())
            print(f"  {name:<40} {params:>10,} params")


def get_trainable_optimizer(
    model,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """
    Configures a multi-group AdamW optimizer with custom learning rates.

    ### Why we need this exactly
    The 'Neck' and 'Attention' modules are the bridge between the frozen backbone 
    and the trainable heads. We give them a lower learning rate (0.5x) so they 
    don't change too violently and destroy the feature distribution[cite: 1]. 
    The 'Box Head' gets the full learning rate because it needs to aggressively 
    learn the specific geometry of the new target objects[cite: 1].

    ### Inputs
    *   **model**: The SiamABC network.
    *   **lr**: The base learning rate.
    *   **weight_decay**: Regularization factor to prevent over-fitting.

    ### Returns
    *   **optimizer**: A PyTorch AdamW optimizer ready for training.
    """
    param_groups: list[OptimizerParamGroup] = [
        {
            "params": list(model.neck.parameters())
                      + list(model.polarized_self_attention.parameters())
                      + list(model.attention_neck.parameters()),
            "lr": lr * 0.5,
            "name": "neck_attention",
        },
        {
            "params": list(model.connect_model.parameters()),
            "lr": lr,
            "name": "box_head",
        },
    ]

    if model.build_simsiam_heads:
        param_groups.append(
            {
                "params": list(model.classifier.parameters())
                          + list(model.predictor.parameters()),
                "lr": lr,
                "name": "simsiam_heads",
            }
        )

    for group in param_groups:
        group["params"] = [p for p in group["params"] if p.requires_grad]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def set_reg_bn_train(
    model,
):
    """
    Forces the Batch Normalization layers in the Regression (BBox) branch into training mode.

    ### Why we need this exactly
    SiamABC relies on 'Adaptive Batch Calibration'[cite: 1]. Even if we are in an 
    evaluation phase, we sometimes want these specific layers to keep updating 
    their statistics so they can adapt to new visual environments or sensor noise 
    on the fly[cite: 1].
    """
    for module in [
        model.connect_model.reg_encode,
        model.connect_model.reg_dw,
        model.connect_model.bbox_tower,
        model.connect_model.bbox_pred,
    ]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


def set_bn_eval(
    model,
):
    """
    Locks all Batch Normalization layers to use their stored running statistics.

    ### Why we need this exactly
    When validating the model, we want it to behave exactly as it would in 
    the field. By setting BN to eval mode, we stop it from shifting its 
    internal normalization based on the current validation batch[cite: 1].
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def set_cls_bn_train(
    model,
):
    """
    Forces the Batch Normalization layers in the Classification branch into training mode.

    ### Why we need this exactly
    Similar to the regression branch, the classification head needs to adapt 
    its internal statistics to the current 'Search Region' to ensure the 
    confidence scores remain sharp and accurate[cite: 1].
    """
    for module in [
        model.connect_model.cls_encode,
        model.connect_model.cls_dw,
        model.connect_model.cls_tower,
        model.connect_model.cls_pred,
    ]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TrackingHeadLoss,
    device: torch.device,
    epoch: int,
    log_every: int = 20,
) -> Dict[str, float]:
    """
    Executes a single pass over the entire training dataset.

    ### Why we need this exactly
    This is the engine of the fine-tuning process. It processes the 
    four-input stream (Static/Dynamic Template and Search Regions) 
    through the ResNet and Attention blocks to produce the final 
    tracking predictions[cite: 1]. It also applies 'Gradient Norm Clipping' 
    to prevent the 'Box Tower' weights from exploding if it encounters 
    a particularly difficult training sample[cite: 1].

    ### Inputs
    *   **model**: The SiamABC network.
    *   **loader**: The training data provider.
    *   **optimizer**: The AdamW optimizer.
    *   **criterion**: The loss function (Focal + GIoU).
    *   **device**: The GPU/CPU to run on.

    ### Returns
    *   **Dict**: Average losses (Total, Classification, and BBox) for the epoch.
    """
    model.train()

    running = defaultdict(float)
    t_start = time.time()
    n_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        template = batch["template"].to(device)
        dynamic_template = batch["dynamic_template"].to(device)
        search = batch["search"].to(device)
        dynamic_search = batch["dynamic_search"].to(device)
        cls_label = batch["cls_label"].to(device)
        bbox_label = batch["bbox_label"].to(device)

        # SiamABC Forward Pass: Processes 4 input crops simultaneously[cite: 1]
        out = model((template, dynamic_template, search, dynamic_search))
        cls_pred = out[constants.TARGET_CLASSIFICATION_KEY]
        bbox_pred = out[constants.TARGET_REGRESSION_LABEL_KEY]

        reg_weight = batch["reg_weight"].to(device)
        losses = criterion(cls_pred, bbox_pred, cls_label, bbox_label, reg_weight)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()

        # Stabilize training by limiting the magnitude of gradient updates
        nn.utils.clip_grad_norm_(
            [p for p in model.connect_model.parameters() if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()

        for k, v in losses.items():
            running[k] += v.item()

        if step % log_every == 0 or step == n_steps:
            elapsed = time.time() - t_start
            avg = {k: v / step for k, v in running.items()}
            n_pos = batch["is_positive"].sum().item()
            print(
                f"  step {step:>4}/{n_steps} | "
                f"loss={avg['total']:.4f}  "
                f"cls={avg['cls_loss']:.4f}  "
                f"bbox={avg['bbox_loss']:.4f}  "
                f"pos={n_pos}/{len(batch['is_positive'])}  "
                f"({elapsed:.0f}s)"
            )

    return {k: v / n_steps for k, v in running.items()}


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TrackingHeadLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the model on a separate validation set without updating weights.

    ### Why we need this exactly
    We need to check if our fine-tuning is actually helping or if the 
    model is just memorizing the training data (over-fitting). Using 
    `@torch.no_grad()` is vital here because it tells PyTorch not to 
    waste memory or time tracking gradients, making validation much faster[cite: 1].

    ### Returns
    *   **Dict**: Average validation losses for monitoring performance.
    """
    model.eval()
    running = defaultdict(float)
    n_steps = len(loader)

    for batch in loader:
        template = batch["template"].to(device)
        dynamic_template = batch["dynamic_template"].to(device)
        search = batch["search"].to(device)
        dynamic_search = batch["dynamic_search"].to(device)
        cls_label = batch["cls_label"].to(device)
        bbox_label = batch["bbox_label"].to(device)
        reg_weight = batch["reg_weight"].to(device)

        out = model((template, dynamic_template, search, dynamic_search))
        cls_pred = out[constants.TARGET_CLASSIFICATION_KEY]
        bbox_pred = out[constants.TARGET_REGRESSION_LABEL_KEY]

        losses = criterion(cls_pred, bbox_pred, cls_label, bbox_label, reg_weight)
        for k, v in losses.items():
            running[k] += v.item()

    return {k: v / n_steps for k, v in running.items()}

