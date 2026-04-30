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
        score_size   = 16,                  # MUST match model's output stride
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
    Dictionary representing a parameter group for a PyTorch optimizer.
    """

    params: list[nn.Parameter]
    lr: float
    name: str


def freeze_backbone_only(
    model,
    verbose: bool = True,
) -> None:
    """
    Freeze the encoder (backbone) and leave everything after it trainable:
        neck, polarized_self_attention, attention_neck,
        connect_model, classifier, predictor, similarity_avgpool
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
    Build an optimizer that only touches the unfrozen parameters.
    Optionally use different LRs per group (neck vs heads).
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
    Set BatchNorm layers in the regression branch to training mode.
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
    Set all BatchNorm layers in the model to evaluation mode.
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def set_cls_bn_train(
    model,
):
    """
    Set BatchNorm layers in the classification branch to training mode.
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
    Run one training epoch.

    Returns:
        dict with average 'total', 'cls_loss', 'bbox_loss' over the epoch.
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

        out = model((template, dynamic_template, search, dynamic_search))
        cls_pred = out[constants.TARGET_CLASSIFICATION_KEY]
        bbox_pred = out[constants.TARGET_REGRESSION_LABEL_KEY]

        reg_weight = batch["reg_weight"].to(device)
        losses = criterion(cls_pred, bbox_pred, cls_label, bbox_label, reg_weight)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()

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
    Run one validation epoch (no gradients).

    Returns:
        dict with average 'total', 'cls_loss', 'bbox_loss' over the loader.
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
