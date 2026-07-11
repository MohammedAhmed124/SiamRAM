# SiamRAM on Jetson Orin Nano

This runbook sets up and runs SiamRAM in a managed Jetson Orin Nano workspace
with JetPack 6.x. It assumes the system Python already provides CUDA, TensorRT,
PyTorch, OpenCV, the NVIDIA GStreamer plugins, and the native build tools needed
by Python packages. No root access or `sudo` commands are required. Run every
command from an SSH shell on the Jetson.

The deployment deliberately does not install Torch-TensorRT. On aarch64,
SiamRAM automatically exports its PyTorch modules to ONNX and builds native
TensorRT engines using JetPack's `tensorrt` Python bindings. YOLO uses
Ultralytics' native TensorRT engine export.

## 1. Verify the preinstalled JetPack stack

```bash
uname -m
python3 --version
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

Create and activate the environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Install the safe dependency set:

```bash
python -m pip install -r requirements.jetson.txt --no-build-isolation
```

Install Ultralytics and torchreid without their declared dependencies. Both
projects declare `opencv-python`; allowing pip to resolve it would replace
JetPack's GStreamer-enabled OpenCV wheel.

```bash
python -m pip install --no-deps ultralytics==8.4.45

python -m pip install --no-deps \
  'git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50'
```

Verify imports and make sure OpenCV still has GStreamer:

```bash
python - <<'PY'
import cv2
import mobile_cv
import onnx
import tensorrt
import torch
import torchreid
import ultralytics

assert torch.cuda.is_available()
assert any("GStreamer" in line and "YES" in line.upper()
           for line in cv2.getBuildInformation().splitlines())
print("Python environment: OK")
print("torch:", torch.__version__)
print("TensorRT:", tensorrt.__version__)
print("ONNX:", onnx.__version__)
print("OpenCV:", cv2.__version__, cv2.__file__)
PY
```

If the GStreamer assertion fails, remove `.venv`, recreate it with
`--system-site-packages`, and repeat the commands above. Do not fix it by
installing `opencv-python` or `opencv-python-headless`.

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
not discarded when inference is slower than decoding.

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
that pip did not install a second OpenCV package inside `.venv`.

## 7. Run inference

Raw TensorRT is selected automatically on aarch64. Set it explicitly for the
first run so the intended backend is unambiguous:

```bash
export SIAMRAM_TRT_BACKEND=raw
```

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
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### Torch-TensorRT is missing

No action is required. The Jetson path uses ONNX and raw TensorRT.

### TensorRT compilation runs out of memory

Stop unrelated workloads, keep swap enabled, remove incomplete caches, and try
again:

```bash
rm -rf trt_cache
mkdir trt_cache
export SIAMRAM_TRT_BACKEND=raw
python inference.py data_subset_manifest.json public_lb subset_predictions.csv
```

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
