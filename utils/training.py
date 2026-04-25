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
from typing import Dict

import external.SiamABC.core.constants as constants
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import TrackingHeadLoss

# ──────────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────────

def freeze_backbone_only(model, verbose: bool = True) -> None:
    """
    Freeze the encoder (backbone) and leave everything after it trainable:
        neck, polarized_self_attention, attention_neck,
        connect_model, classifier, predictor, similarity_avgpool
    """
    # ── Freeze encoder ────────────────────────────────────────────────────────
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ── Unfreeze everything else ──────────────────────────────────────────────
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


def get_trainable_optimizer(model, lr: float = 1e-4,
                            weight_decay: float = 1e-4):
    """
    Build an optimizer that only touches the unfrozen parameters.
    Optionally use different LRs per group (neck vs heads).
    """
    param_groups = [
        # Neck + attention: lower LR (closer to backbone, more sensitive)
        {
            "params": list(model.neck.parameters()) +
                      list(model.polarized_self_attention.parameters()) +
                      list(model.attention_neck.parameters()),
            "lr": lr * 0.5,
            "name": "neck_attention",
        },
        # Box head: normal LR
        {
            "params": model.connect_model.parameters(),
            "lr": lr,
            "name": "box_head",
        },
    ]

    if model.build_simsiam_heads:
        param_groups.append({
            "params": list(model.classifier.parameters()) +
                      list(model.predictor.parameters()),
            "lr": lr,
            "name": "simsiam_heads",
        })

    # Safety net: make sure nothing frozen sneaks in
    for group in param_groups:
        group["params"] = [p for p in group["params"] if p.requires_grad]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def set_reg_bn_train(model):
    for module in [
        model.connect_model.reg_encode,
        model.connect_model.reg_dw,
        model.connect_model.bbox_tower,
        model.connect_model.bbox_pred,
    ]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


# After model.train(), also freeze ALL BN outside cls branch:
def set_bn_eval(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def set_cls_bn_train(model):
    for module in [
        model.connect_model.cls_encode,
        model.connect_model.cls_dw,
        model.connect_model.cls_tower,
        model.connect_model.cls_pred,
    ]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    # set_bn_eval(model)
    # set_cls_bn_train(model)  # Only unfreeze cls BN if we're fine-tuning the cls branch; otherwise keep all BN frozen to preserve pre-trained stats.
    # set_reg_bn_train(model)
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

        # ── forward ──────────────────────────────────────────────────────────
        # model.forward() returns a dict; we only use cls + bbox keys.
        # SimSiam outputs are computed inside forward() but we don't add their
        # loss here — gradients only flow through our criterion.
        out = model((template, dynamic_template, search, dynamic_search))
        cls_pred = out[constants.TARGET_CLASSIFICATION_KEY]  # (B, 1, S, S)
        bbox_pred = out[constants.TARGET_REGRESSION_LABEL_KEY]  # (B, 4, S, S)

        reg_weight = batch["reg_weight"].to(device)
        losses = criterion(cls_pred, bbox_pred, cls_label, bbox_label, reg_weight)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        # Clip gradients to prevent exploding updates in early fine-tuning
        nn.utils.clip_grad_norm_(
            [p for p in model.connect_model.parameters() if p.requires_grad],
            max_norm=5.0
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
