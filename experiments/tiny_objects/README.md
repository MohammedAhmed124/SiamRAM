# Tiny Objects: Telescope Experiment

This isolated experiment tests a Telescope-style foveation frontend/backend for
single-object tracking of tiny targets. It keeps the tracker itself as a black
box and wraps it with a hyperbolic foveation module based on the Telescope
method described in `arxiv.org_html_2604.06332v1.md`.

The A/B comparison is:

- `baseline`: the normal SiamRAM/SiamABC tracker.
- `telescope_tiny_objects`: the previous accepted target box becomes the next
  foveation focus, the tracker predicts in foveated space, and the wrapper maps
  the box back to Euclidean coordinates for output and scoring.

Bounding boxes are mapped with a local Jacobian/tangent approximation at the
box center instead of corner projection. This preserves aspect ratio and avoids
the box growth that non-linear corner min/max projection can introduce.

## Run

Quick smoke test with annotated videos:

```powershell
.\.venv\Scripts\python.exe experiments\tiny_objects\tiny_objects_ab_test.py --tracker-kind siamabc --no-trt --allow-cpu --max-frames 5 --output-video
```

Full CUDA run:

```powershell
.\.venv\Scripts\python.exe experiments\tiny_objects\tiny_objects_ab_test.py --no-trt --output-video
```

Use a custom sequence:

```powershell
.\.venv\Scripts\python.exe experiments\tiny_objects\tiny_objects_ab_test.py --sequence-dir "E:\path\to\sequence" --output-video
```

Useful foveation controls:

```powershell
--foveation-alpha 3.5
--foveation-p 2.0
--foveation-beta 5.0
--foveation-radius-ema-alpha 0.2
```

The dense inverse grid uses PyTorch when available and caches the normalized
destination grid per frame size/device. Pixel remapping still uses OpenCV, with
`BORDER_REPLICATE` by default to avoid reflected-edge ghost objects.

## Outputs

Artifacts are written under:

```text
outputs/experiments/tiny_objects/<sequence>_<timestamp>/
```

Important files:

```text
comparison_report.md
comparison_summary.json
baseline/<video>.txt
telescope_tiny_objects/<video>.txt
baseline/tracking_metrics_by_frame.csv
telescope_tiny_objects/tracking_metrics_by_frame.csv
telescope_tiny_objects/tiny_telescope_log.csv
baseline/*_annotated.mp4                    only with --output-video
telescope_tiny_objects/*_annotated.mp4      only with --output-video
```

## Overlay Legend

The variant annotated video includes:

- GT box
- predicted box
- foveation focus box
- foveated search box in the inset
- dynamic template thumbnail crop when available
- frame number, score, IoU
- foveation center, radius rings, estimated gain, raw/EMA radius, inverse roundtrip error
- dynamic template enabled/refresh status, memory size, best score, and update threshold
- top-right foveated inset showing the actual frame sent to the tracker

The current implementation focuses on the Telescope frontend/backend around the
existing tracker. The paper's SAM3/Perception Encoder swap is left as a backbone
integration point because this repository does not ship SAM3 weights or a SAM3
feature API.
