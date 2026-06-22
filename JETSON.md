# Running SiamRAM on a Jetson Orin Nano (Docker)

This is the guide for running the submission on a Jetson Orin Nano using Docker.
It is separate from the normal `requirements.txt` flow because a Jetson is an
ARM64 board with its own CUDA / PyTorch / OpenCV stack (JetPack / L4T), which is
not the same as the x86 + CUDA wheels the plain `requirements.txt` is pinned to.

The relevant files:

- `Dockerfile.jetson` — the image definition (starts from an L4T PyTorch base).
- `requirements.jetson.txt` — only the pure-Python packages we add on top.
- `docker-compose.jetson.yml` — a convenience wrapper for build / run.

## Step 1 — Find your JetPack version

Run this on the Orin Nano:

```bash
cat /etc/nv_tegra_release
```

The `R36` / `R35` number tells you which L4T (and therefore which base image)
to use:

| JetPack | L4T   | Base image tag to use            |
|---------|-------|----------------------------------|
| 6.x     | r36.x | `dustynv/l4t-ml:r36.2.0` (default) |
| 5.x     | r35.x | `dustynv/l4t-ml:r35.4.1`    |

If your exact `r36.x` / `r35.x` differs, pick the closest published
`dustynv/l4t-ml` tag (see hub.docker.com/r/dustynv/l4t-ml/tags).

We use the `l4t-ml` image because it bundles torch, torchvision, the scientific
stack, and an OpenCV built with CUDA + GStreamer, so hardware NVDEC works
without extra work. The build will fail fast (on purpose) if the tag you pick
doesn't actually have those.

## Step 2 — Turn on hardware decoding (optional but recommended)

The video decoder is chosen in `src/config/inference_config_944f2b8.yaml`:

```yaml
runtime:
  use_nvdec: false   # change to true on the Jetson
```

- `false` (default): the normal OpenCV / CPU decoder. This is the safe default
  and is what the x86 submission uses.
- `true`: decode on the Jetson's NVDEC hardware block. This is the whole point
  of running on the Jetson, so set it to `true` here.

Setting it to `true` is safe even off-Jetson: the code only takes the hardware
path when the `nvv4l2decoder` plugin is actually present, and otherwise falls
back to the CPU decoder, so it never crashes if you build the image somewhere
without the plugin.

## Step 3 — Build

With compose (edit `L4T_BASE` in the compose file first if your JetPack differs):

```bash
docker compose -f docker-compose.jetson.yml build
```

Or directly with Docker, overriding the base for a different JetPack:

```bash
docker build -f Dockerfile.jetson \
    --build-arg L4T_BASE=dustynv/l4t-pytorch:r36.2.0 \
    -t siamram-jetson .
```

The build runs a self-check at the end. If torch is not a CUDA build, or if
OpenCV lost its GStreamer support, **the build fails on purpose** with a clear
message, so you never get a silently broken image.

## Step 4 — Run the evaluation

```bash
docker compose -f docker-compose.jetson.yml up -d
docker compose -f docker-compose.jetson.yml exec jetson \
    python3 inference.py test.json public_test submission.csv
```

(`inference.py` downloads the checkpoints on first run, so the device needs
network access the first time.)

## Honest status — please read

This image is built the correct, standard way for a Jetson, but it has **not**
been built or run on a physical Orin Nano from here, so treat it as "ready to
validate on the device", not "certified". Two things to confirm on the board:

1. **The base tag matches your JetPack.** An r36 image will not run correctly on
   an r35 board and vice versa. Step 1 tells you which to use.
2. **Build it once on the Orin Nano.** The self-check in `Dockerfile.jetson`
   will immediately tell you if torch lost CUDA or OpenCV lost GStreamer.

### If the OpenCV / GStreamer check fails

It almost always means a pip package (commonly `opencv-python-headless`, pulled
in by some other library) overwrote the base image's OpenCV. Fix it by removing
the pip copy so the base's GStreamer-enabled OpenCV is used again:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
```

then rebuild. `requirements.jetson.txt` is already written to avoid this (it
leaves OpenCV and `albumentations` out), but third-party version bumps can
reintroduce it, which is exactly why the build self-check is there.
