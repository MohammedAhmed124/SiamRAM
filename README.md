# SiamRAM

<div align="center">

**Robust long-term visual object tracking with occlusion recovery and distractor-aware reacquisition**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Abstract

SiamRAM is a hybrid visual object tracker built for robust long-term tracking under occlusion. It combines a Siamese network base tracker (SiamABC) with YOLO-based re-detection, an Extended Kalman Filter (EKF) for motion-compensated ROI prediction, and a Distractor-Aware Memory (DAM/DRM) bank for candidate verification during reacquisition. When the primary tracker loses confidence, SiamRAM enters a structured recovery pipeline that expands the search region, filters YOLO detections against the appearance memory, and re-initialises the tracker only on high-confidence matches — substantially reducing false reacquisitions caused by distractors.

## Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Quick Start](#quick-start)
- [Training](#training)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [Citation](#citation)

---

## System Architecture

![System Architecture](docs/system_diagram.png)

SiamRAM is organised around three cooperating subsystems.

### 1. Normal Tracking

SiamABC runs every frame, producing a bounding-box prediction and a confidence score. Frames that exceed the confidence threshold are admitted into a short-term appearance buffer (`AppearanceMemory`) and used to keep the EKF updated.

### 2. EKF and Camera-Motion Compensation

`BBoxEKF` tracks the target centre using a constant-velocity model, with optional camera-motion compensation via a homography matrix. When tracking fails, the EKF prediction defines where in the frame to search next.

### 3. Multi-Phase Reacquisition Pipeline (DAM/DRM)

When confidence drops below `conf_threshold`, SiamRAM switches to occlusion mode and works through three steps to find the target again:

1. **ROI search** — a window centred on the EKF prediction is expanded progressively.
2. **YOLO candidates** — detections inside the ROI are scored against the appearance memory using cosine similarity, IoU, motion consistency, and temporal decay.
3. **DRM verification** — the best candidate re-initialises SiamABC, and tracking only resumes if the resulting confidence clears `reacq_threshold`.

For full design details see the [System Description](docs/system_description.pdf).

---

## Installation

Depending on whether your machine has a GPU, follow one of these guides:

- [CUDA / GPU Installation Guide](docs/install-CUDA.md) — Native Docker, VSCode Devcontainer, or local `uv` with CUDA
- [CPU-only Installation Guide](docs/install-CPU.md) — Native Docker, VSCode Devcontainer, or local `uv` without a GPU

Throughout this README, every runnable step is shown for all three environments. If you are unsure which one applies to you:

> 💡 **Which environment am I in?**
>
> | How you installed | Your environment |
> |---|---|
> | Ran `poe gpu_setup` or `poe cpu_setup` in a terminal | **Native Docker** |
> | Opened the project via *Dev Containers: Reopen in Container* in VSCode | **VSCode Devcontainer** |
> | Ran `uv sync` directly on your machine | **Local uv** |

---

## Checkpoints

The model needs two weight files to run, both stored in `checkpoints/`:

- `inference_checkpoint.pth` — the SiamABC weights.
- `yolo11n.pt` — the YOLO weights.
- `SiamABC_init_checkpoint.pth` — only needed if you plan to train from scratch.

You can download them by running:

```bash
python checkpoints/download_checkpoints.py
```

Or grab them directly from [Google Drive](https://drive.google.com/drive/folders/1BRPhnBnU9CDLU5qQPv-zQeKtqv1HMsl4?usp=drive_link) and place them in `checkpoints/`.

> 📝 **Note:** If you skip this, `run_inference.py` will download the checkpoints automatically on first run.

---

## Quick Start

`run_inference.py` expects the AIC-4 competition data layout under `data/`:

```
data/
├── metadata/
│   └── contestant_manifest.json
└── <dataset_name>/
    ├── <video_id>/
    │   └── video.mp4
    └── annotations/
        └── <video_id>.txt     # single line: x,y,w,h
```

All paths default to being relative to the repo root, so it doesn't matter where you call the script from.

---

### Native Docker

> 📝 **Note:** You only need to start the container once per session — it stays running until you stop it or restart your machine. If it isn't up yet, run `poe gpu_up` (or `poe cpu_up` for CPU-only) before your first command. You don't need to repeat this between scripts.

Run with all defaults:

```bash
poe gpu_run run_inference.py
```

Or with custom paths and settings:

```bash
poe gpu_run run_inference.py \
    --data_dir data/ \
    --manifest_path data/metadata/contestant_manifest.json \
    --weights_path checkpoints/inference_checkpoint.pth \
    --outputs_dir outputs/SiamRAM \
    --submission_csv submission.csv
```

### VSCode Devcontainer

Open a terminal in VSCode — you are already inside the container, so just run the script directly.

Run with all defaults:

```bash
python run_inference.py
```

Or with custom paths and settings:

```bash
python run_inference.py \
    --data_dir data/ \
    --manifest_path data/metadata/contestant_manifest.json \
    --weights_path checkpoints/inference_checkpoint.pth \
    --outputs_dir outputs/SiamRAM \
    --submission_csv submission.csv
```

### Local uv

Run with all defaults:

```bash
uv run run_inference.py
```

Or with custom paths and settings:

```bash
uv run run_inference.py \
    --data_dir data/ \
    --manifest_path data/metadata/contestant_manifest.json \
    --weights_path checkpoints/inference_checkpoint.pth \
    --outputs_dir outputs/SiamRAM \
    --submission_csv submission.csv
```

---

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Root directory containing video files and annotation sub-folders |
| `--manifest_path` | `data/metadata/contestant_manifest.json` | Path to the competition manifest JSON |
| `--weights_path` | `checkpoints/head_epoch_000.pth` | SiamABC checkpoint to use for inference |
| `--yaml_config_path` | `config/inference_config.yaml` | Inference config YAML |
| `--outputs_dir` | `outputs/SiamRAM` | Where per-video bounding-box predictions are written |
| `--model_size` | `M` | SiamABC model size — `S`, `M`, or `L` |
| `--lambda_tta` | `0.1` | TTA lambda for the base tracker |
| `--datasets` | all sub-dirs in `--data_dir` | Specific dataset folders to run; defaults to everything in `data/` except `metadata/` |
| `--submission_csv` | `submission.csv` | Output path for the final submission CSV |

---

## Training

Training is a two-step process: first you build a frame-level index from the raw videos, then you fine-tune the tracking head against it.

### Step 1 — Build the dataset index

This decodes the raw videos in `data/` into individual frames under `data_imgs/` and writes the CSV index files the training loader expects. You only need to do this once per dataset.

**Native Docker:**

```bash
poe gpu_run data_prep/build_dataset_index.py
```

**VSCode Devcontainer:**

```bash
python data_prep/build_dataset_index.py
```

**Local uv:**

```bash
uv run data_prep/build_dataset_index.py
```

### Step 2 — Fine-tune the tracking head

All training hyperparameters — learning rate, batch size, number of epochs, checkpoint interval — are set in `config/training_config.yaml`. Edit that file before running.

**Native Docker:**

```bash
poe gpu_run training/train_head.py
```

**VSCode Devcontainer:**

```bash
python training/train_head.py
```

**Local uv:**

```bash
uv run training/train_head.py
```

Checkpoints are saved to `checkpoints/` as `head_epoch_<NNN>.pth`. To run inference with a freshly trained checkpoint, pass it via `--weights_path`.

---

## Project Structure

```
SiamRAM/
├── models/                       # Tracker implementations
│   ├── SiamRAM.py                # Main SiamRAMTracker class
│   ├── ram_memory.py             # AppearanceMemory and DRM bank
│   ├── motion_model.py           # BBoxEKF (Extended Kalman Filter)
│   └── SiamABC/                  # Siamese base tracker
├── utils/                        # Shared utilities (IoU, descriptors, cosine sim, losses, etc.)
├── config/
│   ├── inference_config.yaml
│   └── training_config.yaml
├── data_prep/
│   └── build_dataset_index.py   # Decodes videos to frames and builds the CSV index
├── training/
│   └── train_head.py            # Fine-tunes the SiamABC tracking head
├── vis/                          # Visualisation tools
├── containers/                   # Docker Compose files and environment verification
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── docker-compose.gpu.yml
│   ├── docker-compose.cpu.yml
│   └── test.py
├── docs/
│   ├── install-CUDA.md
│   ├── install-CPU.md
│   ├── system_description.pdf
│   └── system_diagram.png
├── data/                         # Raw videos, annotations, manifest
├── data_imgs/                    # Extracted frames (generated by build_dataset_index.py)
├── checkpoints/                  # Model weights and download scripts
│   ├── download_checkpoints.py
│   ├── download-checkpoints.sh
│   └── download-checkpoints.bat
├── pyproject.toml                # uv dependencies and poe tasks (GPU)
├── pyproject.cpu.toml            # uv dependencies (CPU-only)
├── requirements.txt              # pip dependencies (device-agnostic)
└── run_inference.py              # Main inference entry point
```

---

## Authors

- Mohammed Metwally — [@MohammedAhmed124](https://github.com/MohammedAhmed124)
- Yousif Abdulhafiz — [@ysif9](https://github.com/ysif9)
- Ahmed Lotfy — [@alofty25](https://github.com/alofty25)
- Philopater Guirgis — [@Philodoescode](https://github.com/Philodoescode)
- Soliman Elhassanein — [@SolimanElhassanein](https://github.com/Soliman-Elhassanein)

---

## Citation

If you find SiamRAM useful in your research, please cite:

```bibtex
@misc{siamram2026,
  author       = {Metwally, Mohammed and Abdulhafiz, Yousif and Lotfy, Ahmed and Guirgis, Philopater and Elhassanein, Soliman},
  title        = {{SiamRAM}: Robust Long-Term Visual Tracking with Distractor-Aware Reacquisition Memory},
  year         = {2026},
  howpublished = {\url{https://github.com/MohammedAhmed124/SiamRAM}},
}
```