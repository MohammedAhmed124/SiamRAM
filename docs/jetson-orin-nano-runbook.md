# SiamRAM on Jetson Orin Nano

This runbook sets up and runs SiamRAM in a managed Jetson Orin Nano workspace
with JetPack 6.x. It assumes the system Python already provides CUDA, TensorRT,
PyTorch, OpenCV, the NVIDIA GStreamer plugins, and the native build tools needed
by Python packages. It also assumes `uv` is available. The `python3-venv` system
package is not required. No root access or `sudo` commands are required. Run
every command from an SSH shell on the Jetson.

The deployment deliberately does not install Torch-TensorRT. On aarch64,
SiamRAM automatically exports its PyTorch modules to ONNX and builds native
TensorRT engines using JetPack's `tensorrt` Python bindings. YOLO uses
Ultralytics' native TensorRT engine export.

## 1. Verify the preinstalled JetPack stack

```bash
uname -m
python3 --version
uv --version
nvcc --version

python3 - <<'PY'
import cv2
import tensorrt
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("TensorRT:", tensorrt.__version__)
print("OpenCV:", cv2.__version__)
print(
    "OpenCV GStreamer:",
    any("GStreamer" in line and "YES" in line.upper()
        for line in cv2.getBuildInformation().splitlines()),
)
PY

gst-inspect-1.0 nvv4l2decoder >/dev/null && echo "nvv4l2decoder: OK"
gst-inspect-1.0 nvvidconv >/dev/null && echo "nvvidconv: OK"
```

Expected results:

- `uname -m` prints `aarch64`.
- CUDA availability is `True`.
- TensorRT imports successfully.
- OpenCV GStreamer is `True`.
- Both GStreamer checks print `OK`.

Torch-TensorRT may be absent. That is expected and is not an error.

## 2. Clone the project

```bash
cd ~/my_project
git clone https://github.com/MohammedAhmed124/SiamRAM.git
cd SiamRAM
```

If the repository already exists:

```bash
cd ~/my_project/SiamRAM
git pull
```

Use the branch or commit containing `trt_engine/raw_trt.py` and
`requirements.jetson.txt`. Confirm both exist:

```bash
test -f src/models/SiamABC/tracker/trt_engine/raw_trt.py
test -f requirements.jetson.txt
```

## 3. Create the Python environment

The virtual environment must expose the packages already configured in the
system Python. A normal isolated venv would hide TensorRT and the
GStreamer-enabled OpenCV build. Do not install or replace CUDA, PyTorch,
TensorRT, OpenCV, GStreamer, or NVIDIA multimedia packages.

Create the environment with `uv`, exposing the system Python packages, then
activate it. `uv` creates the environment directly and does not depend on
`python3 -m venv` or the missing `python3-venv` package.

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
source .venv/bin/activate
```

Install the build tools first because the remaining source packages are built
without isolation, then install the safe dependency set. Use `uv pip`; a
uv-created environment does not need to contain the `pip` module.

```bash
uv pip install --python .venv/bin/python 'setuptools<60' wheel
uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -r requirements.jetson.txt
```

`--no-deps` is required here. Unlike pip, uv resolves transitive dependencies
without treating the JetPack packages exposed through `--system-site-packages`
as satisfying the solve. Without this flag, `timm` and `ultralytics-thop` make
uv download PyPI builds of Torch and TorchVision, and other packages can make it
replace NumPy or Pillow. `requirements.jetson.txt` therefore lists the safe
runtime dependency closure explicitly.

Install Ultralytics and torchreid without their declared dependencies. Both
projects declare `opencv-python`; allowing pip to resolve it would replace
JetPack's GStreamer-enabled OpenCV wheel.

```bash
uv pip install --python .venv/bin/python \
  --no-deps ultralytics==8.4.45

python -c 'import Cython, numpy, scipy; print("torchreid build inputs: OK")'

uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation \
  'git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50'
```

The torchreid command also requires `--no-build-isolation`: its legacy
`setup.py` imports NumPy while building. An isolated uv build environment does
not inherit JetPack's system NumPy and fails with `No module named 'numpy'`.

Verify imports and make sure OpenCV still has GStreamer:

```bash
python - <<'PY'
import cv2
import mobile_cv
import numpy
import onnx
from pathlib import Path
import tensorrt
import torch
import torchreid
import ultralytics

assert torch.cuda.is_available()
assert any("GStreamer" in line and "YES" in line.upper()
           for line in cv2.getBuildInformation().splitlines())
venv = (Path.cwd() / ".venv").resolve()
for name, module in {"torch": torch, "numpy": numpy, "cv2": cv2}.items():
    module_path = Path(module.__file__).resolve()
    assert venv not in module_path.parents, (
        f"{name} shadows the JetPack package: {module_path}"
    )
print("Python environment: OK")
print("torch:", torch.__version__)
print("TensorRT:", tensorrt.__version__)
print("ONNX:", onnx.__version__)
print("OpenCV:", cv2.__version__, cv2.__file__)
PY
```

If the GStreamer assertion fails, remove `.venv`, recreate it with the `uv venv
--system-site-packages` command above, and repeat the install commands. Do not
fix it by installing `opencv-python` or `opencv-python-headless`.

## 4. Download checkpoints

```bash
python download.py
ls -lh checkpoints/model.pth \
       checkpoints/yolo11n.pt \
       checkpoints/osnet_x0_25_imagenet.pth
```

`load_model()` also invokes the downloader, so rerunning it is safe; existing
valid files are skipped.

## 5. Download the evaluation subset

The public Drive folder contains 21 files: eight MP4 videos, eight annotations,
and five metadata files. Download it directly into `data_subset`:

```bash
python -m gdown \
  'https://drive.google.com/drive/folders/1c4S9LQMPPvI47w_kq77Ba_ZwEULdsTQh' \
  --folder -O data_subset
```

The supplied manifest filename contains the original `contestent` spelling.
Copy it to a convenient root-level name without modifying its contents:

```bash
cp data_subset/metadata/contestent_manifest.json data_subset_manifest.json
```

Validate all eight sequences and their paths:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data_subset_manifest.json").read_text())
sequences = manifest["public_lb"]
assert len(sequences) == 8, len(sequences)

for name, info in sequences.items():
    for key in ("video_path", "annotation_path"):
        path = Path(info[key])
        assert path.is_file(), f"Missing {name}: {path}"

frames = sum(int(info["n_frames"]) for info in sequences.values())
assert frames == 11259, frames
print(f"Subset: {len(sequences)} sequences, {frames} frames — OK")
PY
```

The subset contains:

| Sequence | Frames |
|---|---:|
| `dataset1/Car_video` | 585 |
| `dataset1/cows` | 489 |
| `dataset2/RcCar4` | 699 |
| `dataset2/Sheep2` | 251 |
| `dataset3/jogging1` | 324 |
| `dataset3/truck` | 1,871 |
| `dataset4/group2` | 2,683 |
| `dataset4/person19` | 4,357 |

## 6. Validate the Jetson decoder path

The application checks OpenCV's GStreamer support and all required pipeline
elements. It then opens the pipeline, decodes one frame to prove delayed
GStreamer linking succeeded, and returns that prefetched frame as frame zero.
This avoids losing the first frame. The appsink uses `drop=false`, so frames are
not discarded when inference is slower than decoding. The pipeline requests no
extra decoder surfaces and retains only one decoded appsink frame, which keeps
peak memory lower for the 2160x3840 videos in this subset.

Run the application-level probe on one downloaded video:

```bash
python - <<'PY'
import predictor

path = "data_subset/dataset1/cows/cows.mp4"
predictor._USE_NVDEC = True
assert predictor._jetson_gstreamer_available(), "Jetson GStreamer probe failed"

cap = predictor._open_video(path)
ok, frame = cap.read()
print("capture:", type(cap).__name__)
print("first frame:", ok, None if frame is None else frame.shape)
cap.release()
assert ok and frame is not None
PY
```

The output should include:

```text
[SiamRAM][DECODE] ready Using Jetson nvv4l2decoder through OpenCV/GStreamer
```

If it instead reports CPU fallback, rerun the checks in section 1 and confirm
that uv did not install a second OpenCV package inside `.venv`.

If the GStreamer log contains `NvMapMemAllocInternalTagged` with `error 12`, the
decoder was found but Jetson could not allocate its required NVMM surfaces. Run
these unprivileged checks before loading any models:

```bash
free -h
ps -eo pid,rss,comm --sort=-rss | head -15

test -f /sys/fs/cgroup/memory.current && \
  cat /sys/fs/cgroup/memory.current
test -f /sys/fs/cgroup/memory.max && \
  cat /sys/fs/cgroup/memory.max

gst-launch-1.0 -e \
  filesrc location=data_subset/dataset1/cows/cows.mp4 ! \
  qtdemux ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  h264parse ! nvv4l2decoder num-extra-surfaces=0 ! \
  fakesink sync=false
```

If the standalone `gst-launch-1.0` command reports the same allocation error,
the problem is below OpenCV and SiamRAM. Stop other GPU processes or restart the
workspace/device, then run this decoder probe before TensorRT engines or models
are loaded. Also check whether `memory.max` imposes a container limit. Do not add
`disable-dpb=true`: `cows.mp4` contains B-frames, and NVIDIA supports that option
only for streams without B-frames. CPU fallback remains correct but is slower.

## 7. Run inference

Raw TensorRT is selected automatically on aarch64. Set it explicitly for the
first run so the intended backend is unambiguous:

```bash
export SIAMRAM_TRT_BACKEND=raw
export SIAMRAM_TRT_WORKSPACE_MB=256
export SIAMRAM_TRT_TACTIC_DRAM_MB=512
```

The memory limits are also the automatic defaults on aarch64. They prevent the
TensorRT optimizer's embedded-device tactic pool from consuming most of the
Jetson's unified memory during engine compilation. They affect first-run engine
selection, not model precision or output shapes.

Run the eight-sequence subset:

```bash
python inference.py \
  data_subset_manifest.json \
  public_lb \
  subset_predictions.csv
```

The first run exports and compiles these components:

- SiamABC template and search backbones
- Two attention-neck shapes
- Two connector variants
- OSNet
- YOLO

The native engines are cached under `trt_cache/`. Later runs reuse them. Never
copy `.engine` files from another GPU or JetPack/TensorRT version; TensorRT
engines are hardware- and version-specific.

## 8. Validate the result

The Drive folder's `sample_submission.csv` covers the full competition dataset,
not just this eight-sequence subset, so do not pass it to `check_submission.py`.
Validate the subset output against its manifest instead:

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

manifest = json.loads(Path("data_subset_manifest.json").read_text())["public_lb"]
expected = {
    f"{sequence}_{frame}"
    for sequence, info in manifest.items()
    for frame in range(int(info["n_frames"]))
}

with Path("subset_predictions.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

actual = {row["id"] for row in rows}
assert len(rows) == 11259, len(rows)
assert actual == expected, (
    f"missing={len(expected - actual)}, extra={len(actual - expected)}"
)
print("subset_predictions.csv: 11,259 rows and all IDs match — OK")
PY
```

## Troubleshooting

### `import tensorrt` fails inside `.venv`

The environment was probably created without system packages. Recreate it:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages .venv
source .venv/bin/activate
```

### uv tries to download Torch, TorchVision, NumPy, or OpenCV

Stop the installation. The requirements command is missing `--no-deps`, or an
older version of this runbook was used. Recreate the environment to remove any
packages that may already shadow JetPack, then repeat section 3 exactly:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python 'setuptools<60' wheel
uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -r requirements.jetson.txt
```

### Torch-TensorRT is missing

No action is required. The Jetson path uses ONNX and raw TensorRT.

### TensorRT compilation runs out of memory

Messages saying TensorRT is `Skipping tactic` are recoverable while compilation
continues; wait unless the process exits or reports that it failed to build the
serialized engine. If the build does fail, stop unrelated workloads, remove
incomplete caches, lower the two builder pools, and try again:

```bash
rm -rf trt_cache
mkdir trt_cache
export SIAMRAM_TRT_BACKEND=raw
export SIAMRAM_TRT_WORKSPACE_MB=128
export SIAMRAM_TRT_TACTIC_DRAM_MB=256
python inference.py data_subset_manifest.json public_lb subset_predictions.csv
```

If TensorRT cannot find an implementation at those limits, retry with the
defaults (`256` and `512`). Build the engines before opening a video so NVDEC
surfaces and TensorRT tactic profiling do not compete for unified memory.

### Google Drive throttles the download

Resume the folder download with the same gdown command. Existing completed files
are reused. If Google refuses anonymous access, verify that the folder remains
shared as “Anyone with the link.”

### Force a clean engine rebuild

Delete `trt_cache/`, or temporarily set `rebuild_trt_cache: True` in
`src/config/inference_config.yaml`. Restore it to `False` after the successful
build.

## References

- [NVIDIA Jetson accelerated GStreamer guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
- [gdown folder-download documentation](https://github.com/wkentaro/gdown)
