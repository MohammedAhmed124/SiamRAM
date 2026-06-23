# Jetson Nano Setup Guide (SSH)

Step-by-step commands to run after SSH'ing into a Jetson Orin Nano with JetPack 6.x.

## Prerequisites (already on-device via JetPack 6.x)

- Python 3.10
- CUDA 12.6
- TensorRT 10.x
- cuDNN 9.x

Verify JetPack is healthy before starting:

```bash
nvidia-smi
```

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/MohammedAhmed124/SiamRAM.git
cd SiamRAM
```

If the repo is already on the device, just `cd` into it and pull:

```bash
cd SiamRAM
git pull
```

---

## Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Verify:

```bash
uv --version
```

---

## Step 3 — Install dependencies

The main `pyproject.toml` targets x86 CUDA 12.8 and will not work on Jetson.
Use `pyproject.nano.toml` instead, which:
- Pulls `torch`, `torchvision`, and `torch-tensorrt` from NVIDIA's Jetson AI Lab index (ARM64, CUDA 12.6)
- Excludes `triton` (no ARM64 wheel exists)

```bash
cp pyproject.nano.toml pyproject.toml
uv sync
```

> **Note:** Do not use `uv sync --frozen`. The repo lock file was generated on x86 and has no Jetson wheel entries. `uv sync` re-resolves fresh from the Jetson AI Lab index.

This step takes a few minutes on first run.

---

## Step 4 — Verify the environment

```bash
uv run containers/test.py
```

Expected output:

```
python: 3.10.x
platform: linux
torch: 2.x.x
cuda available: True
cuda device count: 1
gpu 0: Orin
nvidia-smi: ok
```

If `cuda available: False`, TRT compilation will fail later. Fix CUDA before continuing.

---

## Step 5 — Download checkpoints

```bash
uv run python checkpoints/download_checkpoints.py
```

This downloads three files into `checkpoints/`:
- `inference_checkpoint.pth` — SiamABC weights (required for inference)
- `yolo11n.pt` — YOLO re-detection weights (required for inference)
- `SiamABC_init_checkpoint.pth` — backbone init weights (only needed for training)

---

## Step 6 — Run inference

Place your data under `data/` following this layout:

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

Then run:

```bash
uv run run_inference.py
```

Or with explicit paths:

```bash
uv run run_inference.py \
    --data_dir data/ \
    --manifest_path data/metadata/contestant_manifest.json \
    --weights_path checkpoints/inference_checkpoint.pth \
    --outputs_dir outputs/SiamRAM \
    --submission_csv submission.csv
```

> **First-run TRT compilation:** On the very first inference call, TRT engines are built from the checkpoint. This takes **3–5 minutes** on Orin Nano (longer than desktop). Subsequent runs reuse the compiled engines and start immediately.

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Root directory of videos and annotations |
| `--manifest_path` | `data/metadata/contestant_manifest.json` | Competition manifest JSON |
| `--weights_path` | `checkpoints/inference_checkpoint.pth` | SiamABC checkpoint |
| `--yaml_config_path` | `config/inference_config.yaml` | Inference config |
| `--outputs_dir` | `outputs/SiamRAM` | Per-video prediction output directory |
| `--model_size` | `M` | Model size: `S`, `M`, or `L` |
| `--lambda_tta` | `0.1` | TTA lambda for the base tracker |
| `--submission_csv` | `submission.csv` | Final submission CSV path |

---

## Troubleshooting

### `uv sync` fails with "no matching distribution"

Make sure you copied the nano config first:

```bash
cp pyproject.nano.toml pyproject.toml
uv sync
```

### CUDA not available after install

Check that JetPack is properly installed and the CUDA runtime is on `LD_LIBRARY_PATH`:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
ls /usr/local/cuda/lib64/libcudart*
```

### TRT engine build fails

Ensure TensorRT is installed via JetPack (not pip):

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

If that fails, re-flash JetPack 6.x using NVIDIA SDK Manager.

### Out of memory during TRT build

TRT engine compilation is memory-intensive. Close other processes before running:

```bash
sudo systemctl stop <any-heavy-service>
uv run run_inference.py
```
