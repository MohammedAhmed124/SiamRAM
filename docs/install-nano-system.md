# Jetson Nano Setup Guide — Using Pre-installed JetPack Environment

Use this guide when your Jetson already has PyTorch, TorchVision, TensorRT, and Docker
installed via JetPack 6.x. You do **not** need to reinstall them from pip.

Two paths are covered:

- [Path A — System Python venv](#path-a--system-python-venv) — no Docker, uses JetPack's system Python directly
- [Path B — NVIDIA L4T Docker container](#path-b--nvidia-l4t-docker-container) — uses NVIDIA's pre-built Jetson container

---

## Before you start — verify JetPack is healthy

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Both should succeed. If `torch` import fails, PyTorch was not installed by JetPack — use the [install-nano.md](install-nano.md) guide instead.

---

## Path A — System Python venv

This creates a virtual environment that inherits JetPack's system packages (torch, torchvision, tensorrt, torch-tensorrt) and adds only the remaining project dependencies on top.

### Step 1 — Clone the repo

```bash
git clone https://github.com/MohammedAhmed124/SiamRAM.git
cd SiamRAM
```

If already cloned:

```bash
cd SiamRAM
git pull
```

### Step 2 — Create a system-site-packages venv

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

The `--system-site-packages` flag makes torch, torchvision, and tensorrt visible inside the venv without reinstalling them.

Verify torch is available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3 — Check torch-tensorrt

```bash
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

If that fails (not bundled in your JetPack version), install it from NVIDIA's Jetson index:

```bash
pip install torch-tensorrt --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

### Step 4 — Install remaining dependencies

```bash
pip install \
    "albumentations>=2.0.8" \
    "coloredlogs>=15.0.1" \
    "easydict>=1.13" \
    "einops>=0.8.2" \
    "fire>=0.7.1" \
    "gdown>=6.0.0" \
    "got10k>=0.1.3" \
    "hydra-core>=1.3.2" \
    "imageio>=2.37.3" \
    "ipykernel>=7.2.0" \
    "ipython>=8.39.0" \
    "ipywidgets>=8.1.8" \
    "jpeg4py>=0.1.4" \
    "lmdb>=2.2.0" \
    "matplotlib>=3.10.9" \
    "numpy>=2.2.6" \
    "opencv-python>=4.13.0" \
    "pandas>=2.3.3" \
    "pillow>=12.2.0" \
    "pycocotools>=2.0.11" \
    "pytorch-toolbelt>=0.8.0" \
    "pyyaml>=6.0.3" \
    "scikit-learn>=1.7.2" \
    "scipy>=1.15.3" \
    "shapely>=2.1.2" \
    "six>=1.17.0" \
    "spikingjelly>=0.0.0.0.14" \
    "tensorboard>=2.16.0" \
    "tensorboardx>=2.6.5" \
    "timm>=1.0.26" \
    "torchmetrics>=1.9.0" \
    "tqdm>=4.67.3" \
    "typing-extensions>=4.15.0" \
    "ultralytics>=8.4.41" \
    "wandb>=0.26.1" \
    "omegaconf>=2.3.0" \
    "cython>=3.0.0" \
    "h5py>=3.10.0" \
    "future>=1.0.0" \
    "yacs>=0.1.8" \
    "chardet>=5.2.0"
```

### Step 5 — Install git-sourced dependencies

```bash
pip install git+https://github.com/facebookresearch/mobile-vision.git
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

`torchreid` builds a Cython extension — this takes a minute.

### Step 6 — Verify the environment

```bash
python containers/test.py
```

Expected:

```
torch: 2.x.x
cuda available: True
cuda device count: 1
gpu 0: Orin
```

### Step 7 — Download checkpoints

```bash
python checkpoints/download_checkpoints.py
```

### Step 8 — Run inference

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

> **First-run TRT compilation:** Takes 3–5 minutes on Orin Nano while the TRT engines are built. Subsequent runs start immediately.

To deactivate the venv when done:

```bash
deactivate
```

---

## Path B — NVIDIA L4T Docker container

Use this path if you prefer a fully containerised setup. JetPack ships Docker and the NVIDIA Container Runtime, so GPU access inside containers works out of the box.

### Step 1 — Clone the repo

```bash
git clone https://github.com/MohammedAhmed124/SiamRAM.git
cd SiamRAM
```

### Step 2 — Pull the L4T PyTorch container

Find the correct tag for your JetPack version on [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch). For JetPack 6.x the tags follow the pattern `r36.x.x-pth2.x.x-py3`:

```bash
docker pull nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3
```

Replace `r36.4.0-pth2.3-py3` with the latest tag that matches your JetPack version.

### Step 3 — Start the container with the repo mounted

```bash
docker run --runtime nvidia -it --rm \
    -v $(pwd):/app \
    -w /app \
    nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3 \
    bash
```

All steps below run **inside the container**.

### Step 4 — Verify torch and CUDA inside the container

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 5 — Check torch-tensorrt

```bash
python3 -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

If it is missing:

```bash
pip install torch-tensorrt --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

### Step 6 — Install remaining dependencies

```bash
pip install \
    "albumentations>=2.0.8" \
    "coloredlogs>=15.0.1" \
    "easydict>=1.13" \
    "einops>=0.8.2" \
    "fire>=0.7.1" \
    "gdown>=6.0.0" \
    "got10k>=0.1.3" \
    "hydra-core>=1.3.2" \
    "imageio>=2.37.3" \
    "ipykernel>=7.2.0" \
    "ipython>=8.39.0" \
    "ipywidgets>=8.1.8" \
    "jpeg4py>=0.1.4" \
    "lmdb>=2.2.0" \
    "matplotlib>=3.10.9" \
    "numpy>=2.2.6" \
    "opencv-python>=4.13.0" \
    "pandas>=2.3.3" \
    "pillow>=12.2.0" \
    "pycocotools>=2.0.11" \
    "pytorch-toolbelt>=0.8.0" \
    "pyyaml>=6.0.3" \
    "scikit-learn>=1.7.2" \
    "scipy>=1.15.3" \
    "shapely>=2.1.2" \
    "six>=1.17.0" \
    "spikingjelly>=0.0.0.0.14" \
    "tensorboard>=2.16.0" \
    "tensorboardx>=2.6.5" \
    "timm>=1.0.26" \
    "torchmetrics>=1.9.0" \
    "tqdm>=4.67.3" \
    "typing-extensions>=4.15.0" \
    "ultralytics>=8.4.41" \
    "wandb>=0.26.1" \
    "omegaconf>=2.3.0" \
    "cython>=3.0.0" \
    "h5py>=3.10.0" \
    "future>=1.0.0" \
    "yacs>=0.1.8" \
    "chardet>=5.2.0"
```

### Step 7 — Install git-sourced dependencies

```bash
pip install git+https://github.com/facebookresearch/mobile-vision.git
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### Step 8 — Verify the environment

```bash
python3 containers/test.py
```

### Step 9 — Download checkpoints

```bash
python3 checkpoints/download_checkpoints.py
```

### Step 10 — Run inference

```bash
python3 run_inference.py
```

Or with explicit paths:

```bash
python3 run_inference.py \
    --data_dir data/ \
    --manifest_path data/metadata/contestant_manifest.json \
    --weights_path checkpoints/inference_checkpoint.pth \
    --outputs_dir outputs/SiamRAM \
    --submission_csv submission.csv
```

> **Note:** Since the repo is bind-mounted (`-v $(pwd):/app`), outputs written inside the container (checkpoints, submission CSV, outputs folder) are immediately visible on the host at the same paths.

> **Persistence:** The container is started with `--rm`, so it is deleted when you exit. The repo mount keeps your data safe, but any pip installs done inside are lost on exit. To persist the environment, remove `--rm` and commit the container, or build a custom image on top of the L4T base.

---

## Troubleshooting

### `import torch_tensorrt` fails after pip install

The Jetson AI Lab `torch-tensorrt` wheel links against the system TensorRT. If TRT shared libraries are not on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib:$LD_LIBRARY_PATH
python -c "import torch_tensorrt"
```

Add the export to `~/.bashrc` to make it permanent.

### Docker: `--runtime nvidia` not recognised

Confirm the NVIDIA Container Runtime is installed:

```bash
docker info | grep -i runtime
```

If `nvidia` is not listed, reinstall the container runtime:

```bash
sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
```

### torchreid build fails (Cython error)

Ensure `gcc` and `python3-dev` are present:

```bash
sudo apt-get install -y gcc python3.10-dev
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```
