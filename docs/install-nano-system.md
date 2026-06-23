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

### Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### Step 3 — Create a system-site-packages venv

```bash
uv venv --system-site-packages
source .venv/bin/activate
```

The `--system-site-packages` flag makes JetPack's torch, torchvision, torch-tensorrt,
tensorrt, and opencv visible inside the venv without reinstalling them. This is the
key to reusing the pre-installed stack: because the project is installed with
`uv pip install` (not `uv sync`), uv treats those system packages as already
satisfied and will not re-download multi-GB Jetson-index wheels.

Verify torch is available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 4 — Check torch-tensorrt

```bash
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

If that fails (not bundled in your JetPack version), install it from NVIDIA's Jetson index:

```bash
uv pip install torch-tensorrt --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

### Step 5 — Install project dependencies

`pyproject.nano.toml` is configured for exactly this scenario: it declares everything
*except* torch/torchvision/torch-tensorrt/opencv (which come from the system) and marks
`torchreid` and `mobile-cv` as no-build-isolation packages so they compile against the
system torch. Install it with:

```bash
cp pyproject.nano.toml pyproject.toml
uv pip install --no-build-isolation .
```

This installs the remaining PyPI dependencies plus the two git-sourced packages
(`mobile-cv`, `torchreid`). `torchreid` builds a Cython extension against the system
torch — this takes a minute.

> **Why `uv pip install` and not `uv sync`:** `uv sync` strictly reconciles the venv to
> the full resolved graph and would re-resolve torch (a transitive dep of ultralytics,
> timm, torchmetrics, …) from PyPI, shadowing the JetPack build. `uv pip install` respects
> packages already visible via `--system-site-packages` and skips them.

### Step 6 — Verify the environment

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected:

```
torch: 2.x.x
cuda available: True
device: Orin
```

### Step 7 — Download checkpoints

`inference.py` fetches these on first run, but you can pre-download them:

```bash
python download.py
```

### Step 8 — Run inference

`inference.py` takes a manifest JSON, the split name inside it, and the output CSV
path, with each sequence's `video_path` / `annotation_path` read relative to the
launch directory — so run it from the folder holding the dataset:

```bash
python inference.py test.json public_lb submission.csv
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

### Step 6 — Install project dependencies

The container's system Python already provides torch/torchvision/torch-tensorrt/opencv,
so install the project the same way as Path A — `pyproject.nano.toml` skips those and
builds `torchreid`/`mobile-cv` against the container torch. Inside the container:

```bash
pip install uv
cp pyproject.nano.toml pyproject.toml
uv pip install --system --no-build-isolation .
```

`--system` installs into the container's Python (there is no separate venv inside the
container). This also pulls the two git-sourced packages (`mobile-cv`, `torchreid`).

### Step 7 — Verify the environment

```bash
python3 -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

### Step 8 — Download checkpoints

```bash
python3 download.py
```

### Step 9 — Run inference

Run from the folder holding the dataset (each sequence's `video_path` /
`annotation_path` in the manifest is read relative to the launch directory):

```bash
python3 inference.py test.json public_lb submission.csv
```

> **Note:** Since the repo is bind-mounted (`-v $(pwd):/app`), outputs written inside the container (the checkpoints folder and submission CSV) are immediately visible on the host at the same paths.

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
