# Running the benchmark sweep on Modal

`modal_app.py` runs `bench/run_tracker.py` over benchmark datasets on cloud GPUs and
scores them with `bench/eval.py`.

## Setup

```bash
pip install modal
modal token new
```

## Volumes

| Volume | Mount | Holds |
|---|---|---|
| `siamram-data` | `/data` | benchmark datasets, one dir per dataset (`/data/<name>`) |
| `siamram-results` | `/results` | `<tracker>/<dataset>/<seq>.txt`, `metrics/`, `trt_cache/` |

Both are created on first use.

## Getting datasets in

Automatic, for anything `bench/download.py` supports:

```bash
modal run modal_app.py::download_dataset --name dtb70
```

Manual, for the ones behind a login or a Google Drive link (UAVDT, VisDrone, VOT-LT2021).
Download and unpack locally, then push the frame directories up:

```bash
modal volume put siamram-data ./local/DTB70 /dtb70
modal volume put -f siamram-data ./local/UAV123 /uav123   # -f overwrites
modal volume ls siamram-data /
```

The remote path must be `/<dataset-name>` — that is the `--data-root` the tracker is
given. Upload the original JPEG frame directories, never the MP4 re-encodes.

## Launching a sweep

```bash
modal run modal_app.py \
  --datasets dtb70,uav123,uav20l \
  --configs inference_config.yaml,inference_config_tiny.yaml \
  --gpu A10G
```

`--datasets` and `--configs` are comma-separated; the run is the cross product. The
tracker name for each config is its filename stem (`inference_config`,
`inference_config_tiny`), which is also the results subdirectory and the label in the
metrics table.

Order of operations: `warm_trt_cache` runs once per config to build the TensorRT engines
and commit them to the volume, then `run_benchmark` fans out over every
(dataset x config) pair, then `evaluate` scores each dataset. Markdown tables print to
the terminal and the CSVs land in `./bench_results/`.

### TensorRT cache

Engines are SM-architecture specific — an A10G engine will not load on an H100 — so the
cache lives at `/results/trt_cache/<gpu-model>/`. Switching `--gpu` pays a fresh build
once and then reuses it. To force a rebuild, delete that directory:

```bash
modal volume rm -r siamram-results /trt_cache/nvidia_a10g
```

### Config overrides applied on Modal

Each run gets a patched copy of the config in the container: `runtime.use_nvdec: false`
(benchmark data is JPEG frame directories, so NVDEC never applies) and
`trt_engine.trt_cache_dir` pointed at the GPU-keyed volume path. Nothing under `src/` is
modified.

## Cost

The plan puts a full sweep at ~600k frames per configuration; at ~175 FPS that is roughly
**1 GPU-hour per config**. Ten configs (7 ablation rows + full + two SiamABC baselines)
is ~10 GPU-hours, plus a few minutes of engine-build time per GPU type. At A10G on-demand
pricing that lands in the low tens of dollars. H100 is faster per frame but not cheaper
per frame for this workload — A10G is the default for a reason.

Downloads run on CPU-only containers, so a slow 250 GB fetch costs almost nothing.

## Pulling results down

```bash
modal volume get siamram-results /metrics ./bench_results
modal volume get siamram-results /inference_config ./raw_predictions   # per-seq .txt
```

The per-sequence `.txt` files are the archive reviewers ask for — keep them.

## Notes

- `modal run` is enough for a sweep. `modal deploy` only matters if you want to call
  these functions from other code.
- The image build (torch 2.11+cu128, TensorRT 10.15, two git deps) takes ~10 minutes the
  first time and is cached afterwards.
- `checkpoints/` ships with the image, so new weights mean a rebuild.
