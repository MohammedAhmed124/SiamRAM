"""
SiamABC Fine-Tuning Script
Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --csv_path /path/to/data.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, csv_path: str | None) -> dict:
    """Apply CLI overrides on top of the loaded YAML config."""
    if csv_path:
        cfg["data"]["csv_path"] = csv_path
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the single combined CSV and produce train / val DataFrames."""
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    print(f"Loaded {len(df)} sequences from {data_cfg['csv_path']}")

    val_ratio = cfg["dataset"]["val_split_ratio"]
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=data_cfg["seed"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Split → train: {len(train_df)}  val: {len(val_df)}")
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    from utils.dataset import UAV123TrackingDataset

    tracking_config = cfg["tracking"]
    ds_cfg = cfg["dataset"]
    dl_cfg = cfg["dataloader"]
    seed = cfg["data"]["seed"]

    train_dataset = UAV123TrackingDataset(
        dataframe=train_df,
        tracking_config=tracking_config,
        num_samples=ds_cfg["train_num_samples"],
        neg_ratio=ds_cfg["train_neg_ratio"],
        seed=seed,
    )

    val_dataset = UAV123TrackingDataset(
        dataframe=val_df,
        tracking_config=tracking_config,
        num_samples=ds_cfg["val_num_samples"],
        neg_ratio=ds_cfg["val_neg_ratio"],
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["train_batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
    )

    print(
        f"DataLoaders ready — "
        f"train batches: {len(train_loader)}  val batches: {len(val_loader)}"
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device):
    from models.SiamABC.tracker.tracker_setup import load_model
    from utils.hydra import load_hydra_config_from_path

    model_cfg = cfg["model"]
    tracker_cfg = cfg["tracker"]

    hydra_config = load_hydra_config_from_path(
        config_path=model_cfg["simabc_config_path"],
        config_name=model_cfg["simabc_config_name"],
    )

    hydra_config["model"]["model_size"] = model_cfg["model_size"]
    hydra_config["model"]["build_simsiam_heads"] = model_cfg["build_simsiam_heads"]
    hydra_config["tracker"]["dynamic_update"] = tracker_cfg["dynamic_update"]
    hydra_config["tracker"]["window_influence"] = tracker_cfg["window_influence"]
    hydra_config["tracker"]["N"] = tracker_cfg["N"]

    model = instantiate(hydra_config["model"])
    model = load_model(model, model_cfg["weights_path"], strict=False)
    model = model.to(device)

    print(f"Model loaded from {model_cfg['weights_path']}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, cfg: dict):
    train_cfg = cfg["training"]

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.connect_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": train_cfg["backbone_lr"]},
            {"params": head_params, "lr": train_cfg["head_lr"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=train_cfg["warmup_epochs"],
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"] - train_cfg["warmup_epochs"],
        eta_min=train_cfg["eta_min"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[train_cfg["warmup_epochs"]],
    )

    return optimizer, scheduler


def build_criterion(cfg: dict, device: torch.device):
    from utils.losses import TrackingHeadLoss

    loss_cfg = cfg["loss"]
    return TrackingHeadLoss(
        cls_weight=loss_cfg["cls_weight"],
        bbox_weight=loss_cfg["bbox_weight"],
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    from utils.training import _train_one_epoch, freeze_backbone_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──
    train_df, val_df = load_and_split(cfg)
    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg)

    # ── model ──
    model = build_model(cfg, device)
    freeze_backbone_only(model)

    # ── optimisation ──
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    criterion = build_criterion(cfg, device)

    # ── checkpointing ──
    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = cfg["training"]
    num_epochs = train_cfg["num_epochs"]
    start_epoch = train_cfg["start_epoch"]

    # ── loop ──
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n── Epoch {epoch}/{num_epochs} ──")
        model.train()

        train_losses = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            log_every=train_cfg["log_every"],
        )

        scheduler.step()

        print(
            f"  train loss={train_losses['total']:.4f}  "
            f"cls={train_losses['cls_loss']:.4f}  "
            f"bbox={train_losses['bbox_loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        ckpt_path = os.path.join(save_dir, f"head_epoch_{epoch:03d}.pth")
        state = {"module." + k: v for k, v in model.state_dict().items()}
        torch.save(state, ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SiamABC fine-tuning script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Override data.csv_path from config (path to the combined sequences CSV)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_config(cfg, args.csv_path)
    train(cfg)
