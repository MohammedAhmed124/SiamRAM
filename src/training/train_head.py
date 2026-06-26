"""
SiamABC Classification Head Fine-Tuning
========================================

Background & Motivation
------------------------
When we received the pretrained SiamABC checkpoints, we discovered a critical flaw in
the classification head: it had essentially zero discrimination power. Regardless of
the input, the head would almost always saturate to a score of 1 — confidently labelling
every candidate region as a positive match, even in clearly negative situations where the
target is absent or the search region contains nothing but background.

This makes the out-of-the-box model unreliable for real-world UAV tracking scenarios,
where the tracker must robustly decide *whether* the target is present before committing
to *where* it is. A head that always fires positive collapses that two-stage reasoning
into a blind "always track something", causing the tracker to latch onto irrelevant
background clutter whenever the true target drifts out of frame.

Purpose of This Script
-----------------------
This script fine-tunes *only* the classification head (and the lightweight connection
module) of SiamABC on the UAV123 and other datasets, while keeping the backbone encoder frozen.
The goal is to teach the head to correctly distinguish between:

  - Positive pairs  → the search crop genuinely contains the target object
  - Negative pairs  → the search crop is background-only, or from a different sequence

We use a focal loss on the classification branch (to handle the natural class imbalance
between positives and the flood of easy negatives) combined with a bounding-box
regression loss on positive samples to preserve localisation quality throughout training.

By the end of fine-tuning the head should produce well-calibrated scores: high confidence
when the target is present, and near-zero confidence when it is not — restoring the
discriminative behaviour that was missing from the original checkpoints.

Usage example:
-----
    python train_head.py --config config.yaml
    python train_head.py --config config.yaml --csv_path /data/uav123_sequences.csv
"""

import argparse
import os
import random
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


def _seed_worker(
    worker_id: int,
) -> None:
    """
    Seed each DataLoader worker deterministically.

    This prevents duplicated random streams across workers for python's random
    module and NumPy augmentations.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _clamp_ratio(
    value: float,
    name: str,
) -> float:
    """
    Clamp a ratio into [0, 1] and log when config values are invalid.
    """
    v = float(value)
    if 0.0 <= v <= 1.0:
        return v
    clamped = min(1.0, max(0.0, v))
    print(
        f"[warn] {name}={v} is outside [0,1]; using {clamped} instead."
    )
    return clamped


def _repair_path_if_needed(
    path_value: str,
    repo_root: str,
) -> str:
    """
    Remap stale absolute paths (e.g. /app/data/...) to local repo paths.
    """
    path_str = str(path_value)
    if os.path.exists(path_str):
        return path_str

    norm = path_str.replace("\\", "/")
    candidates: list[str] = []

    if "/app/" in norm:
        candidates.append(os.path.join(repo_root, norm.split("/app/", 1)[1]))
    if "/data_imgs/" in norm:
        tail = norm.split("/data_imgs/", 1)[1]
        candidates.append(os.path.join(repo_root, "data", tail))
    if "/data/" in norm:
        tail = norm.split("/data/", 1)[1]
        candidates.append(os.path.join(repo_root, "data", tail))
    if norm.startswith("data_imgs/"):
        tail = norm[len("data_imgs/"):]
        candidates.append(os.path.join(repo_root, "data", tail))
    if norm.startswith("data/"):
        candidates.append(os.path.join(repo_root, norm))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path_str


def _repair_dataframe_paths(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Repair CSV path columns when rows were generated in a different mount root.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fixed = df.copy()
    for col in ("seq_path", "annot_path"):
        if col not in fixed.columns:
            continue
        before = fixed[col].astype(str)
        after = before.map(lambda p: _repair_path_if_needed(p, repo_root))
        fixed[col] = after
        changed = int((before != after).sum())
        missing = int((~after.map(os.path.exists)).sum())
        if changed > 0:
            print(f"[PathRepair] {col}: remapped {changed} rows")
        print(f"[PathRepair] {col}: missing after repair = {missing}")
    return fixed


def load_config(path: str) -> dict:
    """
    Reads the YAML config file from disk and returns it as a plain Python dict.
    Every section of the config (data, model, training, etc.) becomes a nested key in that dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _format_lr_groups(
    optimizer: torch.optim.Optimizer,
) -> str:
    """
    Build a compact string with per-parameter-group learning rates.
    """
    parts: list[str] = []
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group{idx}"))
        lr = float(group["lr"])
        parts.append(f"{name}={lr:.2e}")
    return " | ".join(parts)


def _weighted_terms(
    losses: dict,
    criterion,
) -> tuple[float, float, float]:
    """
    Return weighted cls/bbox/iou loss components for clearer logging.
    """
    cls_term = float(criterion.cls_weight) * float(losses["cls_loss"])
    bbox_term = float(criterion.bbox_weight) * float(losses["bbox_loss"])
    iou_term = float(criterion.iou_weight) * float(losses.get("iou_loss", 0.0))
    return cls_term, bbox_term, iou_term


def _validate_train_mode(
    train_mode: str,
) -> str:
    """
    Validate and normalize supported train modes.
    """
    mode = str(train_mode).strip()
    if mode not in {"cls_and_bbox_head", "cls_head_only", "heads_only"}:
        raise ValueError(
            f"Unsupported train mode {train_mode!r}. "
            "Use 'cls_and_bbox_head', 'heads_only', or 'cls_head_only'."
        )
    return mode


def _resolve_train_mode_for_epoch(
    training_cfg: dict,
    epoch: int,
    start_epoch: int,
) -> str:
    """
    Resolve train mode for a specific epoch.

    Supported scheduling formats:
      1) mode_schedule:
         - train_mode: cls_and_bbox_head
           epochs: 5
         - train_mode: cls_head_only
           epochs: 5

      2) stage1/stage2 convenience keys:
         stage1_epochs: 5
         stage1_mode: cls_and_bbox_head
         stage2_mode: cls_head_only
    """
    schedule = training_cfg.get("mode_schedule")
    if schedule:
        if not isinstance(schedule, list):
            raise ValueError("training.mode_schedule must be a list of stage dicts.")
        cursor = int(start_epoch)
        for idx, stage in enumerate(schedule, start=1):
            if not isinstance(stage, dict):
                raise ValueError(
                    f"training.mode_schedule[{idx - 1}] must be a dict."
                )
            mode = _validate_train_mode(stage.get("train_mode", ""))
            span = int(stage.get("epochs", 0))
            if span <= 0:
                raise ValueError(
                    f"training.mode_schedule[{idx - 1}].epochs must be > 0."
                )
            stage_end = cursor + span - 1
            if epoch <= stage_end:
                return mode
            cursor = stage_end + 1
        return _validate_train_mode(schedule[-1]["train_mode"])

    stage1_epochs = int(training_cfg.get("stage1_epochs", 0))
    if stage1_epochs > 0:
        stage1_mode = _validate_train_mode(
            training_cfg.get("stage1_mode", "cls_and_bbox_head")
        )
        stage2_mode = _validate_train_mode(
            training_cfg.get("stage2_mode", "cls_head_only")
        )
        return stage1_mode if epoch <= stage1_epochs else stage2_mode

    return _validate_train_mode(training_cfg.get("train_mode", "cls_and_bbox_head"))


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
    csv_path = data_cfg["csv_path"]
    partial_csv_path = f"{csv_path}.partial"
    if not os.path.exists(csv_path) and os.path.exists(partial_csv_path):
        print(
            f"[warn] {csv_path} not found. Falling back to partial index: {partial_csv_path}"
        )
        csv_path = partial_csv_path

    df = pd.read_csv(csv_path)
    df = _repair_dataframe_paths(df)
    if "ann_len" in df.columns:
        df = df[df["ann_len"] > 1].copy()
    print(f"Loaded {len(df)} sequences from {csv_path}")

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
    seed = int(cfg["data"]["seed"])
    train_neg_ratio = _clamp_ratio(ds_cfg["train_neg_ratio"], "dataset.train_neg_ratio")
    val_neg_ratio = _clamp_ratio(ds_cfg["val_neg_ratio"], "dataset.val_neg_ratio")
    num_workers = int(dl_cfg["num_workers"])

    if num_workers == 0:
        print("[DataLoader] num_workers=0 (single-process loading).")
    else:
        print(f"[DataLoader] num_workers={num_workers} with per-worker seeding enabled.")

    train_dataset = UAV123TrackingDataset(
        dataframe=train_df,
        tracking_config=tracking_config,
        num_samples=ds_cfg["train_num_samples"],
        neg_ratio=train_neg_ratio,
        seed=seed,
    )

    val_dataset = UAV123TrackingDataset(
        dataframe=val_df,
        tracking_config=tracking_config,
        num_samples=ds_cfg["val_num_samples"],
        neg_ratio=val_neg_ratio,
        seed=seed,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["train_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
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
        load_device = "cpu" if device.type == "cpu" else None
        model = load_model(
            model,
            weights_path,
            map_location=load_device,
            strict=False,
        )
        print(f"Weights loaded from {weights_path}")

    model = model.to(device)
    return model


def configure_trainable_mode(model, train_mode: str) -> None:
    """
    Configure which parts of the tracking head are trainable.

    Supported modes:
      - cls_and_bbox_head: train SiamABC head branches (cls + bbox) and neck/attention
      - heads_only:        train only head modules (cls + bbox + iou), freeze neck/attention
      - cls_head_only:     train classification branch only (all other parts frozen)
    """
    from utils.training import freeze_backbone_only

    train_mode = _validate_train_mode(train_mode)
    for p in model.parameters():
        p.requires_grad = False

    if train_mode == "cls_and_bbox_head":
        freeze_backbone_only(model)
        print("[TrainMode] cls_and_bbox_head")
        return

    if train_mode == "heads_only":
        for p in model.connect_model.parameters():
            p.requires_grad = True
        if hasattr(model, "iou_head") and model.iou_head is not None:
            for p in model.iou_head.parameters():
                p.requires_grad = True

        allowed_prefixes = ("connect_model.", "iou_head.")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        unexpected = [n for n in trainable_names if not n.startswith(allowed_prefixes)]
        if unexpected:
            raise RuntimeError(
                "heads_only must only train connect_model/iou_head parameters, "
                f"but found unexpected trainables: {unexpected[:10]}"
            )

        trainable_count = sum(
            p.numel() for _, p in model.named_parameters() if p.requires_grad
        )
        print("[TrainMode] heads_only (connect_model + iou_head trainable)")
        print(f"[TrainMode] heads_only trainable params: {trainable_count:,}")
        return

    cls_modules = [
        model.connect_model.cls_encode,
        model.connect_model.cls_dw,
        model.connect_model.cls_tower,
        model.connect_model.cls_pred,
    ]
    for module in cls_modules:
        for p in module.parameters():
            p.requires_grad = True

    allowed_prefixes = (
        "connect_model.cls_encode.",
        "connect_model.cls_dw.",
        "connect_model.cls_tower.",
        "connect_model.cls_pred.",
    )
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    unexpected = [n for n in trainable_names if not n.startswith(allowed_prefixes)]
    if unexpected:
        raise RuntimeError(
            "cls_head_only must only train classification branch parameters, "
            f"but found unexpected trainables: {unexpected[:10]}"
        )

    trainable_count = sum(
        p.numel() for _, p in model.named_parameters() if p.requires_grad
    )
    print("[TrainMode] cls_head_only (only classification branch trainable)")
    print(f"[TrainMode] cls_head_only trainable params: {trainable_count:,}")


def build_optimizer_and_scheduler(
    model,
    cfg: dict,
    *,
    total_epochs_override: int | None = None,
    warmup_epochs_override: int | None = None,
):
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

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    backbone_params = [p for n, p in trainable if n.startswith("encoder.")]
    head_params = [p for n, p in trainable if not n.startswith("encoder.")]

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": train_cfg["backbone_lr"],
            }
        )
    if head_params:
        param_groups.append(
            {
                "name": "head",
                "params": head_params,
                "lr": train_cfg["head_lr"],
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found. Check train_mode/freeze logic.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    num_epochs = (
        int(total_epochs_override)
        if total_epochs_override is not None
        else int(train_cfg["num_epochs"])
    )
    num_epochs = max(1, num_epochs)
    warmup_epochs = (
        int(warmup_epochs_override)
        if warmup_epochs_override is not None
        else int(train_cfg["warmup_epochs"])
    )
    warmup_epochs = max(0, min(warmup_epochs, num_epochs))
    if warmup_epochs <= 0:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=train_cfg["eta_min"],
        )
    elif warmup_epochs >= num_epochs:
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, num_epochs),
        )
    else:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=train_cfg["eta_min"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return optimizer, scheduler


def build_criterion(
    cfg: dict,
    device: torch.device,
    train_mode: str | None = None,
):
    """
    Builds the combined tracking loss, which adds a focal classification loss
    (for deciding whether a location contains the target) and a bounding-box
    regression loss (for refining its position). The weights and focal parameters
    all come from the loss section of the config.
    """
    from utils.losses import TrackingHeadLoss

    loss_cfg = cfg["loss"]
    if train_mode is None:
        train_mode = cfg["training"].get("train_mode", "cls_and_bbox_head")
    train_mode = _validate_train_mode(train_mode)
    bbox_weight = float(loss_cfg["bbox_weight"])
    iou_weight = float(loss_cfg.get("iou_weight", loss_cfg.get("quality_weight", 1.0)))
    if train_mode == "cls_head_only":
        bbox_weight = 0.0
        iou_weight = 0.0
        print("[Loss] cls_head_only active → bbox_weight=0.0, iou_weight=0.0")
    criterion = TrackingHeadLoss(
        cls_weight=loss_cfg["cls_weight"],
        bbox_weight=bbox_weight,
        iou_weight=iou_weight,
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
    ).to(device)
    print(
        "[Loss] weights "
        f"cls={criterion.cls_weight} "
        f"bbox={criterion.bbox_weight} "
        f"iou={criterion.iou_weight} | "
        f"focal(alpha={criterion.focal_alpha}, gamma={criterion.focal_gamma})"
    )
    return criterion


def train(cfg: dict) -> None:
    """
    The main training loop. Pulls everything together: data, model, optimizer, scheduler,
    and loss function. Runs for the configured number of epochs, printing a loss summary
    after each one and saving a checkpoint to disk so you can resume or evaluate at any point.
    Backbone layers are frozen at the start so only the tracking head gets updated,
    which is a standard fine-tuning strategy that keeps the pretrained representations intact.
    """
    from utils.training import _train_one_epoch, _validate_one_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df = load_and_split(cfg)
    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg)

    model = build_model(cfg, device)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = cfg["training"]
    num_epochs = train_cfg["num_epochs"]
    start_epoch = train_cfg["start_epoch"]
    stage2_warmup_epochs = int(train_cfg.get("stage2_warmup_epochs", 0))
    batch_level_logs = bool(train_cfg.get("batch_level_logs", True))
    configured_log_every = int(train_cfg.get("log_every", 1))
    train_log_every = 1 if batch_level_logs else max(1, configured_log_every)

    active_mode: str | None = None
    optimizer = None
    scheduler = None
    criterion = None

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_mode = _resolve_train_mode_for_epoch(
            training_cfg=train_cfg,
            epoch=epoch,
            start_epoch=start_epoch,
        )
        if epoch_mode != active_mode:
            stage_total_epochs = num_epochs - epoch + 1
            if active_mode is None:
                warmup_override = (
                    0
                    if int(start_epoch) > 1
                    else int(train_cfg["warmup_epochs"])
                )
            else:
                warmup_override = stage2_warmup_epochs
            print(
                f"[StageSwitch] epoch={epoch} mode={epoch_mode} "
                f"(remaining_epochs={stage_total_epochs}, warmup={warmup_override})"
            )
            configure_trainable_mode(model, epoch_mode)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model,
                cfg,
                total_epochs_override=stage_total_epochs,
                warmup_epochs_override=warmup_override,
            )
            criterion = build_criterion(cfg, device, train_mode=epoch_mode)
            active_mode = epoch_mode

        assert optimizer is not None
        assert scheduler is not None
        assert criterion is not None

        print("\n" + "=" * 108)
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Mode: {active_mode}")
        print(
            f"Batch logs: every {train_log_every} step(s)"
            + (" (forced batch-level)" if batch_level_logs else "")
        )
        print("=" * 108)
        model.train()

        train_losses = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            log_every=train_log_every,
            train_mode=active_mode,
        )

        val_losses = _validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        train_cls_w, train_bbox_w, train_iou_w = _weighted_terms(train_losses, criterion)
        val_cls_w, val_bbox_w, val_iou_w = _weighted_terms(val_losses, criterion)
        lr_groups = _format_lr_groups(optimizer)
        pos_rate = 100.0 * float(train_losses.get("pos_rate", 0.0))
        epoch_seconds = float(train_losses.get("epoch_seconds", 0.0))

        print("Epoch Summary")
        print(
            f"  train total={train_losses['total']:.4f} | "
            f"raw(c/b/i)={train_losses['cls_loss']:.4f}/{train_losses['bbox_loss']:.4f}/{train_losses.get('iou_loss', 0.0):.4f} | "
            f"weighted(c/b/i)={train_cls_w:.4f}/{train_bbox_w:.4f}/{train_iou_w:.4f}"
        )
        print(
            f"  val   total={val_losses['total']:.4f} | "
            f"raw(c/b/i)={val_losses['cls_loss']:.4f}/{val_losses['bbox_loss']:.4f}/{val_losses.get('iou_loss', 0.0):.4f} | "
            f"weighted(c/b/i)={val_cls_w:.4f}/{val_bbox_w:.4f}/{val_iou_w:.4f}"
        )
        print(
            f"  train pos_rate={pos_rate:.1f}% | epoch_time={epoch_seconds:.1f}s | lr: {lr_groups}"
        )

        ckpt_path = os.path.join(save_dir, f"head_epoch_{epoch:03d}.pth")
        state = {"module." + k: v for k, v in model.state_dict().items()}
        torch.save(state, ckpt_path)
        print(f"  checkpoint: {ckpt_path}")


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
        default="config/training_config.yaml",
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
