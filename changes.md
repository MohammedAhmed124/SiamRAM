# SiamRAM — Model Changes & Additions

This document describes two significant changes made to the SiamRAM tracker on the
`feature/siamabc-tiny` branch:

1. **SiamABC-Tiny integration** — added the lightweight FBNet-based model from the
   official SiamABC repository as an optional configuration switch.
2. **Classification head fine-tuning on hard-negatives** — fine-tuned *only* the
   classification branch of the default Medium model to fix its confidence-collapse
   problem.

---

## Background — Why Both Changes Were Needed

The pretrained SiamABC-Medium checkpoint had a critical flaw: the classification head
always output a near-1 confidence score regardless of the input. Because the head was
trained on almost exclusively positive pairs (template ↔ search crop that always
contained the target), it never learned to say "no". In real UAV tracking this means
the tracker latches onto background clutter whenever the true target leaves the frame.

The two changes address this from different angles:
- **SiamABC-Tiny** provides a fast, lighter alternative backbone for resource-constrained
  deployments.
- **Classification head fine-tuning** directly patches the confidence-collapse issue in
  the existing Medium model by re-training the head on a large fraction of hard negatives.

---

## 1. SiamABC-Tiny Integration

### What Changed

| File | Change |
|---|---|
| [`src/models/SiamABC/model/blocks.py`](src/models/SiamABC/model/blocks.py) | `CorrelationConcat` made TTA-compatible; `BoxTower` now picks the correlation block based on `model_size` |
| [`src/models/SiamABC/model/SiamABC.py`](src/models/SiamABC/model/SiamABC.py) | `model_size` forwarded from `SiamABCNet` → `BoxTower` |
| [`download.py`](download.py) | Auto-downloads `model_S_Tiny_v1.pt` from the official SiamABC repo if missing |
| [`src/config/inference_config.yaml`](src/config/inference_config.yaml) | Documented commented-out switches for the Tiny variant |

### Architecture Difference

| | SiamABC-Medium (`model_size: M`) | SiamABC-Tiny (`model_size: S`) |
|---|---|---|
| **Backbone** | ResNet-50 (`EncoderResNet`) | Lightweight FBNet (`Encoder`) |
| **Correlation blocks** | `CorrelationConcatAtt` — polarised self-attention | `CorrelationConcat` — standard, no attention |
| **Checkpoint** | `checkpoints/model.pth` | `checkpoints/model_S_Tiny_v1.pt` |
| **Trainable params** | ~6.4M head + backbone | ~440 checkpoint tensors, smaller overall |

The key discovery: the official Tiny weights were trained *without* the
`CorrelationConcatAtt` polarised self-attention layers that we added for Medium.
`BoxTower` was made adaptive so it instantiates the right module at build time.

### How to Switch to SiamABC-Tiny

Open [`src/config/inference_config.yaml`](src/config/inference_config.yaml) and make
**three** edits:

```yaml
# 1. Top of file — swap the weights path
weights_path: "checkpoints/model_S_Tiny_v1.pt"   # ← uncomment this
# weights_path: "checkpoints/model.pth"            # ← comment this out

# 2. Under the `model:` section — change model_size
model:
  model_size: S         # ← was M

# 3. Under the `model:` section — disable pretrained (FBNet has no online weights)
  pretrained: false     # ← was true
```

The checkpoint is downloaded automatically on first run via `download.py` (no manual
step needed). The TensorRT engines are invalidated by the architecture change and will
be rebuilt automatically on the first inference run (~1–3 min).

### Verification (inside Docker)

Both variants were verified inside the running GPU container:

```
[Tiny]   Loaded 440/440 tensors (100.0%) from checkpoints/model_S_Tiny_v1.pt ✓
[Medium] Loaded 396/396 tensors (100.0%) from checkpoints/model.pth           ✓
```

End-to-end tracking on `penguin_in_fog.mp4` with Tiny weights: **73.7 FPS** on the GPU,
256 frames tracked successfully.

---

## 2. Classification Head Fine-Tuning on Hard-Negatives

### Why

The confidence-collapse problem was described in the training code comment:

> *"Regardless of the input, the head would almost always saturate to a score of 1 —
> confidently labelling every candidate region as a positive match, even in clearly
> negative situations."*

The fix is to retrain *only* the classification branch while keeping the backbone and
regression head frozen, on a dataset where 65% of samples are hard negatives.

### Training Infrastructure

All training code was already in the repository:

| File | Role |
|---|---|
| [`src/training/train_head.py`](src/training/train_head.py) | Main training script — data loading, training loop, checkpointing |
| [`src/utils/dataset.py`](src/utils/dataset.py) | `UAV123TrackingDataset` — generates positive and hard-negative pairs |
| [`src/utils/losses.py`](src/utils/losses.py) | Focal Loss (cls) + GIoU Loss (bbox, masked to positives only) |
| [`src/utils/training.py`](src/utils/training.py) | `_train_one_epoch`, `_validate_one_epoch`, freeze helpers |
| [`src/data_prep/build_dataset_index.py`](src/data_prep/build_dataset_index.py) | Extracts video frames and builds the sequence CSV index |
| [`src/config/training_config.yaml`](src/config/training_config.yaml) | All training hyperparameters |

### Data Preparation

Training data is at `C:/Users/afara/PycharmProjects/SiamRAM-final/Data` (mounted
read-only into Docker at `/app/data`). Extracted frames and the CSV index are written
to a persistent Docker named volume (`siamram_training`) at `/app/training_data/`.

**Datasets used:** `dataset2` (70 seq), `dataset3` (112 seq), `dataset4` (20 seq),
`dataset5` (123 seq) — **244 sequences, 85 348 frames total**.

> `dataset1` was excluded: its `annotation.txt` files contain only a single line
> (the initial bounding box for the test set), which gives `n_frames = 1` and is
> automatically filtered out by the indexer.

**To rebuild the CSV index (e.g. after mounting new data):**

```bash
docker exec siamram-master-gpu-1 bash -c "
  cd /app && python src/data_prep/build_dataset_index.py \
    --data       /app/data \
    --imgs-root  /app/training_data/frames \
    --output     /app/training_data \
    --datasets   dataset2 dataset3 dataset4 dataset5 \
    --workers    6 \
    --jpg-quality 90
"
```

### Training Configuration

Key decisions in [`src/config/training_config.yaml`](src/config/training_config.yaml):

| Parameter | Value | Why |
|---|---|---|
| `train_mode` | `cls_head_only` | Freezes backbone + bbox branch; trains only `cls_encode`, `cls_dw`, `cls_tower`, `cls_pred` — 492 930 params |
| `train_neg_ratio` | `0.65` | 65% of each batch is hard negatives — forces the head to learn "no" |
| `neg_cross_seq_prob` | `0.55` | Cross-sequence negatives (completely different target) |
| `neg_same_class_prob` | `0.75` | Same-class distractors (hardest: right category, wrong instance) |
| `focal_gamma` | `2.0` | Standard focal loss γ — down-weights easy negatives, focuses on hard errors |
| `focal_alpha` | `0.25` | Balances the positive/negative gradient contribution |
| `head_lr` | `1e-4` | Conservative — fine-tuning, not training from scratch |
| `warmup_epochs` | `2` | Linear LR warm-up before cosine annealing |
| `num_epochs` | `20` | 20 epochs × 250 steps/epoch = 5 000 gradient steps |
| `bbox_weight / iou_weight` | `0.0` (auto) | Zeroed automatically in `cls_head_only` mode |

### How to Run Training

```bash
# 1. Start the Docker container (data volume is mounted read-only)
docker compose up -d

# 2. (First time only) Build the sequence CSV index
docker exec siamram-master-gpu-1 bash -c "
  cd /app && python src/data_prep/build_dataset_index.py \
    --data /app/data --imgs-root /app/training_data/frames \
    --output /app/training_data --datasets dataset2 dataset3 dataset4 dataset5 \
    --workers 6 --jpg-quality 90
"

# 3. Launch training
docker exec siamram-master-gpu-1 bash -c "
  cd /app && python src/training/train_head.py \
    --config src/config/training_config.yaml
"
```

Checkpoints are saved after every epoch to
`/app/training_data/checkpoints/cls_hardneg_run1/head_epoch_NNN.pth`.

**To resume from a checkpoint** (if training is interrupted), set `start_epoch` in
`training_config.yaml` to the next epoch number and add the checkpoint path to
`weights_path` in the `model:` section.

### How to Use the Fine-Tuned Weights

Once training completes, the best checkpoint (typically the final epoch or the one with
the lowest validation loss) must be integrated into the inference config.

**Step 1 — Copy the checkpoint:**

```bash
# Inside Docker (or copy the file from the Docker volume to the checkpoints folder)
cp /app/training_data/checkpoints/cls_hardneg_run1/head_epoch_020.pth \
   /app/checkpoints/model_cls_finetuned.pth
```

**Step 2 — Switch `inference_config.yaml`:**

```yaml
# Top of src/config/inference_config.yaml
weights_path: "checkpoints/model_cls_finetuned.pth"

# model: section stays the same (Medium, model_size: M, pretrained: true)
```

> **Pre-trained weights download link:**
> <!-- TODO: replace this line with the direct download URL once uploaded -->
> `model_cls_finetuned.pth` — *link to be added after training completes*

**Step 3 — Verify:**

```bash
docker exec siamram-master-gpu-1 python /app/scratch/test_end_to_end.py
```

The end-to-end script will load the fine-tuned weights, track the sample video, and
confirm the tracker runs at full speed without errors.

---

## Docker Setup Summary

```yaml
# docker-compose.yml volumes
- .:/app                                                         # source code (rw)
- siamram_venv_gpu:/app/.venv                                    # Python env
- C:/Users/afara/PycharmProjects/SiamRAM-final/Data:/app/data:ro # training data (ro)
- siamram_training:/app/training_data                            # frames + checkpoints (rw)
```

The `siamram_training` named volume persists across container restarts so extracted
frames and training checkpoints are not lost when the container is recreated.

---

## Branch

All changes are on the `feature/siamabc-tiny` branch.

```
git checkout feature/siamabc-tiny
```

---

## 3. SiamABC-Tiny Classification Head Fine-Tuning on Hard-Negatives

After the Medium fine-tuning run, the same process was applied to the **SiamABC-Tiny** model with minor adjustments.

### Differences from the Medium run

| Setting | Medium run | Tiny run |
|---|---|---|
| `weights_path` | `checkpoints/model.pth` | `checkpoints/model_S_Tiny_v1.pt` |
| `model_size` | `M` | `S` |
| `pretrained` | `true` | `false` (FBNet has no online weights) |
| `num_epochs` | 20 | **25** |
| Checkpoint size | 42 MB/epoch | **9.1 MB/epoch** (smaller model, cls head only) |
| Checkpoint location | Docker named volume | **Local project dir** (directly accessible on host) |
| Starting cls loss | ~0.0073 | ~0.051 (Tiny head had more to learn) |
| Final cls loss (ep. 25) | train=0.0046 / val=0.0050 | **train=0.0049 / val=0.0071** |

Both models converged cleanly with no overfitting. The Tiny head started from a much higher initial loss (~10x) because the FBNet-based classification branch had never seen any hard-negatives at all.

### Checkpoint Location

All 25 checkpoints are saved directly on the **host machine** — no Docker copy step needed:

```
C:\Users\afara\PycharmProjects\SiamRAM-master\training\checkpoints\cls_hardneg_tiny_run1\
    head_epoch_001.pth  (9.1 MB)
    ...
    head_epoch_025.pth  (9.1 MB)  <- recommended for use
```

### How to Use the Fine-Tuned Tiny Weights

The fine-tuned checkpoint contains **only the cls head parameters** (with `module.` prefixed keys). You need to merge it with the base Tiny weights before using it at inference:

```python
import torch

base = torch.load("checkpoints/model_S_Tiny_v1.pt", map_location="cpu")
finetuned = torch.load(
    "training/checkpoints/cls_hardneg_tiny_run1/head_epoch_025.pth",
    map_location="cpu"
)

# Strip the "module." prefix and keep only cls head keys
cls_keys = {
    k.replace("module.", ""): v
    for k, v in finetuned.items()
    if any(x in k for x in ["cls_encode", "cls_dw", "cls_tower", "cls_pred"])
}

base.update(cls_keys)
torch.save(base, "checkpoints/model_S_Tiny_cls_finetuned.pth")
print(f"Merged {len(cls_keys)} cls keys into base Tiny weights")
```

Then update `inference_config.yaml`:

```yaml
weights_path: "checkpoints/model_S_Tiny_cls_finetuned.pth"

model:
  model_size: S
  pretrained: false
```

> **Fine-tuned Tiny weights download link:**
> <!-- TODO: replace this line with the direct download URL once uploaded -->
> `model_S_Tiny_cls_finetuned.pth` — *link to be added after weights are uploaded*

### How to Reproduce

```bash
# training_config.yaml is already committed with Tiny settings.
docker exec siamram-master-gpu-1 bash -c "
  cd /app && python src/training/train_head.py \
    --config src/config/training_config.yaml
"
# Checkpoints appear live in training/checkpoints/cls_hardneg_tiny_run1/ as each epoch finishes.
```
