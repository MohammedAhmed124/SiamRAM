# CPU-only Installation Guide

This guide sets up SiamRAM without GPU/CUDA.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option 1: Native Docker](#option-1-native-docker)
- [Option 2: VSCode Devcontainer](#option-2-vscode-devcontainer-recommended)
- [Option 3: Local Installation with uv](#option-3-local-installation-with-uv)
- [Troubleshooting](#troubleshooting)

## Prerequisites

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

## Option 1: Native Docker

Use this path if you want to run from terminal without devcontainers.

### Step 1: Install poe (lightweight runner)

```bash
uv tool install poethepoet
poe --help
```

### Step 2: Build and start the CPU container

```bash
poe cpu_setup
```

### Step 3: Run a script in the CPU container

```bash
poe cpu_run test.py
```

## Option 2: VSCode Devcontainer (Recommended)

### Step 1: Install required software

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [Visual Studio Code](https://code.visualstudio.com/)
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Step 2: Open the CPU devcontainer

1. Open this project in VSCode
2. Press `Ctrl+Shift+P`
3. Run **Dev Containers: Open Folder in Container...**
4. Choose `.devcontainer/cpu/devcontainer.json`
5. Wait for the build to finish

### Step 3: Verify inside container

```bash
python test.py
```

Expected result: CUDA available is `False`.

## Option 3: Local Installation with uv

Use this path if you want to run directly on your host Python environment.
CPU installs in this repo are defined in `pyproject.cpu.toml`.

### Step 1: Install project dependencies

Linux/macOS:

```bash
cp pyproject.cpu.toml pyproject.toml
uv sync
```

Windows PowerShell:

```powershell
Copy-Item pyproject.cpu.toml pyproject.toml -Force
uv sync
```

### Step 2: Verify environment

```bash
uv run python test.py
```

Expected result: CUDA available is `False`.

### Step 3: Restore GPU pyproject (optional)

If you want to switch back to the CUDA-default setup:

```bash
git checkout -- pyproject.toml uv.lock
```

## Troubleshooting

### Container verification fails

Run the project verify command:

```bash
poe cpu_verify
```
