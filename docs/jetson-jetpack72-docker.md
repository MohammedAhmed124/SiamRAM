# Jetson Orin Nano — JetPack 7.2 Docker image

`Dockerfile.jetson` + `docker-compose.jetson.yml` target **JetPack 7.2**
(L4T r39.2, Ubuntu 24.04, CUDA 13.2.1 host, TensorRT 10.16, kernel 6.8) on the
Jetson Orin Nano. This replaces the bare-metal JetPack 6 flow in
`jetson-orin-nano-runbook.md` for the MTC-AIC4 Phase 3 Docker submission; the
runbook's tracker-level guidance (memory limits, NVDEC pipeline, engine cache)
still applies inside the container.

## Verified stack

Every load-bearing claim was verified against primary sources (July 2026):

| Component | Version | Verification |
|---|---|---|
| JetPack 7.2 | L4T r39.2, CUDA 13.2.1, TensorRT 10.16.2, cuDNN 9.20, Ubuntu 24.04 | [JetPack 7.2 archive page](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2). First 7.x release supporting Orin (released 2026-06-02); 7.0/7.1 were Thor-only. |
| Base image | `nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04` (arm64), digest-pinned `@sha256:c3fde347...` | Tag exists with an arm64 variant (~1.76 GB); the Dockerfile pins the manifest-list digest so the tag can't silently move. Standard CUDA images are the official JetPack 7 container path ([Orin Nano devkit Docker docs](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_docker.html)); `l4t-base` is frozen at JetPack 6 (r36.2.0). CUDA 13.0 userspace on the 13.2 host driver is same-major compatible. |
| PyTorch | `torch==2.9.1+cu130`, `torchvision==0.24.1` (cu130 index) | aarch64 cp312 wheels confirmed present on the [cu130 index](https://download.pytorch.org/whl/cu130/torch/); the aarch64 torchvision 0.24.1 wheel there has no `+cu130` local tag, so the index must be the primary one. Work on Jetson because CUDA 13.2/JetPack 7.2 unified the SBSA and Jetson aarch64 platforms ([CUDA 13.2 blog](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)). |
| TensorRT | `10.16.1.11-1+cuda13.2` via apt | Exact deb version confirmed in the [ubuntu2404/**sbsa** CUDA repo](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/) index (note: the `ubuntu2404/arm64` path carries **no** TensorRT). Same 10.16 line / `libnvinfer.so.10` ABI as the device's 10.16.2.10. |
| onnx | `1.18.0` | PyTorch 2.9.1's CI-tested onnx version, and exactly the release the TensorRT 10.16 ONNX parser supports ([onnx-tensorrt release/10.16-GA](https://github.com/onnx/onnx-tensorrt/blob/release/10.16-GA/README.md)). Tracker exports at opset 17 — inside both ranges. cp312 aarch64 wheel confirmed on PyPI. |
| OpenCV | Ubuntu 24.04 `python3-opencv` 4.6.0 | `libopencv-videoio406t64` depends on `libgstreamer1.0-0` and FFmpeg libs — GStreamer + FFmpeg support confirmed. pip `opencv-python` has no GStreamer and must never be installed on top. |
| NVDEC GStreamer | `nvidia-l4t-gstreamer` 39.2.0 (+ multimedia/camera/nvsci/core) | Downloaded the actual r39.2 debs and listed contents: `libgstnvvideo4linux2.so` (**nvv4l2decoder**) and `libgstnvvidconv.so` (**nvvidconv**) confirmed inside. JetPack 7's OTA repos: `repo.download.nvidia.com/jetson/common` and `.../jetson/som`, suite `r39.2`. |

Not used, deliberately: `onnxruntime-gpu` (no aarch64 wheel exists, official or
NVIDIA index — the tracker drives TensorRT directly), `torch_tensorrt`
(aarch64 uses the raw ONNX→TRT backend), `pynvvideocodec` (dGPU-only).

## Build

On the Jetson (native arm64):

```bash
docker compose -f docker-compose.jetson.yml build
```

On an x86_64 machine, via qemu (one-time binfmt setup, then buildx):

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx build --platform linux/arm64 -f Dockerfile.jetson -t siamram-jetson --load .
```

The Dockerfile itself contains nothing qemu-specific — the submission artifact
is the same either way. Expect the qemu build to be slow on the torchreid
Cython compile step; that is emulation, not a hang.

## Run

```bash
# 1. Stack certification: exact package versions, allowlisted pip check,
#    cv2+GStreamer, then (--gpu) a real ONNX->TensorRT engine build+run and an
#    NVDEC decode through the same pipeline shape predictor.py uses.
docker compose -f docker-compose.jetson.yml run --rm jetson python docker/verify_stack.py --gpu

# 2. Grader entrypoint
docker compose -f docker-compose.jetson.yml run --rm jetson \
    python inference.py test.json public_lb submission.csv
```

Requirements on the host, all shipped with the JetPack 7.2 ISO: Docker with
the NVIDIA runtime configured (`sudo nvidia-ctk runtime configure
--runtime=docker`). Compose uses `runtime: nvidia` because **`--gpus all` is
not supported on the Jetson iGPU** — plain `docker run` needs
`--runtime nvidia` too.

Checkpoints download on first run (network needed once) into `./checkpoints`;
TensorRT engines build on first run into `./trt_cache`. Both are volume-mounted
so subsequent runs skip the slow parts. Engines are fingerprinted with the
TensorRT version and GPU, so a cache built by this container is only reused by
this container — never copy `trt_cache/` between images or devices.

## Known caveats

- **sm_87 kernels**: the cu130 PyTorch wheels ship no native Orin (sm_87)
  SASS; they run through sm_80 binary compatibility. Community reports note
  rare NaN issues in exotic attention/conv paths on Orin. Low risk here — the
  performance-critical inference runs in TensorRT engines compiled natively
  on-device, with torch handling preprocessing and ONNX export — but validate
  `check_submission.py` output on the device before submitting.
- **TRT 10.16.1.11 (container) vs 10.16.2.10 (device apt)**: same minor
  version and ABI; engines are built *inside* the container by the same
  library that runs them, so no cross-version engine reuse ever happens.
- **NVDEC is best-effort**: if the L4T plugins can't load on a given host
  (driver injection differences), `predictor.py` logs it and falls back to CPU
  decode — slower on the 4K sequences but functionally identical output.
- **No container memory limit**: NVDEC NVMM surfaces and TensorRT engine
  builds share unified memory; a cgroup cap triggers
  `NvMapMemAllocInternalTagged ... error 12` (runbook §6).
