import argparse
import os
import sys

import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path: str) -> dict:
    """
    Reads the YAML config file from disk and returns it as a plain Python dict.
    Every section of the config (data, model, training, etc.) becomes a nested key in that dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, csv_path: str | None) -> dict:
    """
    Lets you swap out the CSV path at launch time without editing the config file.
    If you pass --csv_path on the command line, this function writes it into the config
    so the rest of the code never needs to know the override came from the CLI.
    If no override is given, the config is returned unchanged.
    """
    if csv_path:
        cfg["data"]["csv_path"] = csv_path
    return cfg


def load_and_split(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the single combined CSV that contains all tracking sequences,
    then performs a reproducible random split into training and validation sets.
    The split ratio and random seed both come from the config so results are consistent
    across runs as long as the config doesn't change.
    Returns two DataFrames: one for training and one for validation.
    """
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


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[DataLoader, DataLoader]:
    """
    Wraps the train and validation DataFrames in UAV123TrackingDataset instances,
    which handle all the crop-and-augment logic for siamese tracking pairs.
    Then wraps those datasets in standard PyTorch DataLoaders.
    The training loader shuffles its data every epoch; the validation loader does not,
    since order doesn't matter when you're just measuring performance.
    Returns the train DataLoader and the val DataLoader.
    """
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


def build_model(cfg: dict, device: torch.device):
    """
    Constructs the SiamABCNet model directly from the model section of the config.
    The config uses Hydra's _target_ convention, so instantiate() reads that key,
    imports the right class, and passes every other key in the section as a constructor argument.
    weights_path is pulled out before instantiation because it is not a constructor argument —
    it tells us where to load pretrained weights from after the model is built.
    The model is moved to the requested device before being returned.
    """
    from models.SiamABC.tracker.tracker_setup import load_model

    model_cfg = dict(cfg["model"])
    weights_path = model_cfg.pop("weights_path", None)

    model = instantiate(model_cfg)

    if weights_path:
        model = load_model(model, weights_path, strict=False)
        print(f"Weights loaded from {weights_path}")

    model = model.to(device)
    return model


def build_optimizer_and_scheduler(model, cfg: dict):
    """
    Sets up AdamW with two separate learning-rate groups: one for the backbone encoder
    (which we want to nudge gently because it's already well-trained) and one for the
    tracking head (which we want to adapt more aggressively).
    The learning-rate schedule starts with a short linear warm-up phase to avoid
    destabilising the pretrained weights in the first few epochs, then transitions
    into a cosine annealing decay for the rest of training.
    Returns the optimizer and the combined scheduler.
    """
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
    """
    Builds the combined tracking loss, which adds a focal classification loss
    (for deciding whether a location contains the target) and a bounding-box
    regression loss (for refining its position). The weights and focal parameters
    all come from the loss section of the config.
    """
    from utils.losses import TrackingHeadLoss

    loss_cfg = cfg["loss"]
    return TrackingHeadLoss(
        cls_weight=loss_cfg["cls_weight"],
        bbox_weight=loss_cfg["bbox_weight"],
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
    ).to(device)


def train(cfg: dict) -> None:
    """
    The main training loop. Pulls everything together: data, model, optimizer, scheduler,
    and loss function. Runs for the configured number of epochs, printing a loss summary
    after each one and saving a checkpoint to disk so you can resume or evaluate at any point.
    Backbone layers are frozen at the start so only the tracking head gets updated,
    which is a standard fine-tuning strategy that keeps the pretrained representations intact.
    """
    from utils.training import _train_one_epoch, freeze_backbone_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df = load_and_split(cfg)
    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg)

    model = build_model(cfg, device)
    freeze_backbone_only(model)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    criterion = build_criterion(cfg, device)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = cfg["training"]
    num_epochs = train_cfg["num_epochs"]
    start_epoch = train_cfg["start_epoch"]

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


def parse_args() -> argparse.Namespace:
    """
    Defines the two command-line arguments the script accepts.
    --config points to the YAML file that controls everything.
    --csv_path is an optional shortcut to override the dataset path without
    touching the config file, which is handy when running quick experiments.
    """
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