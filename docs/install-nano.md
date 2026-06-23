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

> **Already have torch/torchvision/torch-tensorrt on the device?** If JetPack (or
> your rental provider) pre-installed PyTorch and TensorRT, skip this guide and use
> the [pre-installed JetPack guide](install-nano-system.md) — it reuses the system
> stack directly instead of downloading wheels. This guide is for a base JetPack
> flash that has CUDA/TRT/cuDNN but **not** torch.

## Step 3 — Create a system-site-packages venv and install torch from the Jetson index

`pyproject.nano.toml` no longer declares `torch`/`torchvision`/`torch-tensorrt` —
it expects them to be visible in the venv already (so it works identically whether
they come from JetPack or from the Jetson AI Lab index). On a base flash without
torch, install them into a `--system-site-packages` venv first.

Pick the Jetson AI Lab index path that matches your JetPack CUDA minor:

| JetPack | CUDA | Index path |
|---|---|---|
| 6.0 | 12.2 | `https://pypi.jetson-ai-lab.dev/jp6/cu122` |
| 6.1 | 12.4 | `https://pypi.jetson-ai-lab.dev/jp6/cu124` |
| 6.2 | 12.6 | `https://pypi.jetson-ai-lab.dev/jp6/cu126` |

Check your CUDA version:

```bash
cat /usr/local/cuda/version.json 2>/dev/null | grep -m1 version || nvcc --version
```

Then (substituting your `cuXXX` path):

```bash
uv venv --system-site-packages
source .venv/bin/activate
uv pip install torch torchvision torch-tensorrt \
    --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

> If the wrong `cuXXX` path is used, the torch wheel will fail to load its CUDA
> libraries at import time.

## Step 4 — Install the project

The main `pyproject.toml` targets x86 CUDA 12.8 and will not work on Jetson. Use
`pyproject.nano.toml`, which skips torch/torchvision/torch-tensorrt/opencv (provided
above / by the system), excludes `triton` (no ARM64 wheel exists), and builds
`torchreid`/`mobile-cv` against the installed torch:

```bash
cp pyproject.nano.toml pyproject.toml
uv pip install --no-build-isolation .
```

This step takes a few minutes on first run (torchreid compiles a Cython extension).

> **Why not `uv sync`:** `uv sync` re-resolves the entire graph and would pull torch
> (a transitive dep of ultralytics, timm, …) from PyPI, shadowing the Jetson build.
> `uv pip install` respects the torch already in the `--system-site-packages` venv.

> **If `import torch_tensorrt` later fails with `No module named 'tensorrt'`:**
> JetPack installs TensorRT's Python bindings as a *system apt* package. The
> `--system-site-packages` venv normally exposes it; if not, install the matching
> wheel into the venv:
>
> ```bash
> uv pip install tensorrt --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
> ```

---

## Step 5 — Verify the environment

Run inside the activated venv from Step 3 (use plain `python`, not `uv run` — the
project is installed with `uv pip`, not project-mode `uv sync`):

```bash
python containers/test.py
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

## Step 6 — Download checkpoints

```bash
python checkpoints/download_checkpoints.py
```

This downloads three files into `checkpoints/`:
- `inference_checkpoint.pth` — SiamABC weights (required for inference)
- `yolo11n.pt` — YOLO re-detection weights (required for inference)
- `SiamABC_init_checkpoint.pth` — backbone init weights (only needed for training)

---

## Step 7 — Run inference

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
python run_inference.py
```

Or with explicit paths:

```bash
python run_inference.py \
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

### `uv pip install` fails with "no matching distribution" for torch

`pyproject.nano.toml` does not declare torch — it must already be in the venv. Make
sure you created the venv with `--system-site-packages` and installed torch from the
Jetson index (Step 3) before running `uv pip install --no-build-isolation .`.

### `uv pip install` tries to build torchreid before torch is present

Confirm the venv is activated and `python -c "import torch"` works inside it. The
`no-build-isolation-package` setting builds `torchreid`/`mobile-cv` against that torch,
so torch must import before this step.

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
python run_inference.py
```
