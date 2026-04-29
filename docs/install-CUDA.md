# CUDA Installation Guide

This guide sets up SiamRAM with CUDA (GPU acceleration).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option 1: Native Docker](#option-1-native-docker)
- [Option 2: VSCode Devcontainer](#option-2-vscode-devcontainer-recommended)
- [Option 3: Local Installation with uv](#option-3-local-installation-with-uv)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required software

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install `uv`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### GPU requirement (important)

> **Important:** This project uses CUDA 12.8 + cuDNN runtime. Your host NVIDIA driver must be **>= 570.x**.

Check your driver:

```bash
nvidia-smi
```

The `Driver Version` shown by `nvidia-smi` must be `570.x` or newer.

## Option 1: Native Docker

Use this path if you want to run from terminal without devcontainers.

### Step 1: Install poe (lightweight runner)

```bash
uv tool install poethepoet
poe --help
```

### Step 2: Build and start the GPU container

```bash
poe gpu_setup
```

### Step 3: Verify Docker GPU access

```bash
poe gpu_verify
```

If this fails, fix Docker GPU runtime/NVIDIA Container Toolkit, then run `poe gpu_setup` again.

### Step 4: Run a script in the GPU container

```bash
poe gpu_run test.py
```

## Option 2: VSCode Devcontainer (Recommended)

### Step 1: Install required software

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [Visual Studio Code](https://code.visualstudio.com/)
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Step 2: Open the project in devcontainer

1. Open this project in VSCode
2. Press `Ctrl+Shift+P`
3. Run **Dev Containers: Reopen in Container**
4. Wait for the build to finish

The default `.devcontainer/devcontainer.json` uses the GPU compose setup.

### Step 3: Verify inside container

```bash
python test.py
```

Expected result: CUDA available is `True` and at least one GPU is detected.

## Option 3: Local Installation with uv

Use this path if you want to run directly on your host Python environment.
Plain `uv sync` defaults to CPU torch, so CUDA setup must use the GPU group.

### Step 1: Install project dependencies

```bash
uv sync --group gpu
```

### Step 2: Verify environment

```bash
uv run python test.py
```

### Optional: Match your local CUDA wheel channel (`cuXXX`)

Local install can target different CUDA wheels by changing the `gpu` group torch sources in `pyproject.toml`.

- Current default is `cu128` (CUDA 12.8 wheels).
- If your machine needs a different wheel, switch to another `cuXXX` index (example: `cu124`), then run:

```bash
uv sync --group gpu
```

## Troubleshooting

### Docker GPU check fails

1. Re-check host driver with `nvidia-smi`
2. Make sure Docker Desktop GPU support is enabled
3. Re-run:

```bash
docker compose -f containers/docker-compose.gpu.yml run --rm gpu nvidia-smi
```

### Torch/CUDA not detected in container

Use the project verify command:

```bash
poe gpu_verify
```
