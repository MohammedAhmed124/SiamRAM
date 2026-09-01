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
| `siamram-results` | `/results` | `<dataset>/<tracker>/<seq>.txt`, `metrics/`, `trt_cache/` |

Both are created on first use.

## Getting datasets in

Automatic, for anything `bench/download.py` supports:

```bash
modal run modal_app.py::download_dataset --name dtb70
```

Manual, for DTB70 only - its images are on Baidu Pan with no mirror. Download and
unpack locally, then push the frame directories up:

```bash
modal volume put siamram-data ./local/DTB70 /dtb70
modal volume put -f siamram-data ./local/DTB70 /dtb70   # -f overwrites
modal volume ls siamram-data /
```

The remote path must be `/<dataset-name>` — that is the `--data-root` the tracker is
given. Upload the original JPEG frame directories, never the MP4 re-encodes.

## Launching a sweep

```bash
modal run modal_app.py \
  --datasets uav123,uav123_10fps,visdrone_sot,dtb70,lasot,trackingnet \
  --configs inference_config.yaml \
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

### Volumes have an inode limit, not a size limit

A Modal Volume holds at most **500,000 files and directories** and returns `ENOSPC` past it.
`df` will not warn you - volumes do not report capacity at the filesystem level. They are also
only fast below ~50,000 files.

That is why LaSOT is stored as one tar per category (70 files) rather than ~700,000 loose
JPEGs, and why `_stage()` unpacks it to container-local disk before a run. Datasets that fit
comfortably - UAV123 at ~113k files, UAV123@10fps at ~38k - stay as plain directories and are
read straight off the volume.

Check what a volume holds:

```bash
modal volume ls siamram-data /
```

If a run ever dies with "No space left on device", count files before assuming it is bytes.

### TrackingNet is different

TrackingNet withholds its test ground truth, so `evaluate` cannot score it. For that dataset
the run ends by building an eval-server submission zip instead:

```
/results/metrics/trackingnet_<tracker>.zip
```

Pull it down and upload it to the TrackingNet EvalAI portal by hand. Submissions are
rate-limited, so treat TrackingNet as a once-at-the-end column, not something to iterate on.

### Running several datasets in parallel

Different datasets write to different paths (`/results/<dataset>/<tracker>`), so parallel
`modal run` invocations do not collide **except on the TensorRT cache**. From cold, several
containers would build engines into the same `/results/trt_cache/<gpu-model>/` at once.

So: **one person runs a single dataset to completion first.** That builds and commits the
engines. Everyone else starts after that, and their runs load the cache instead of building it.

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

The six datasets total ~1.10M frames per configuration; at ~175 FPS that is roughly
**1.75 GPU-hours per config**. Ten configs (7 ablation rows + full + two SiamABC baselines)
is ~10 GPU-hours, plus a few minutes of engine-build time per GPU type. At A10G on-demand
pricing that lands in the low tens of dollars. H100 is faster per frame but not cheaper
per frame for this workload — A10G is the default for a reason.

Downloads run on CPU-only containers, so LaSOT's slow 248 GB fetch costs almost nothing.

## Pulling results down

```bash
modal volume get siamram-results /metrics ./bench_results
modal volume get siamram-results /uav123 ./raw_predictions   # per-seq .txt
```

The per-sequence `.txt` files are the archive reviewers ask for — keep them.

## Notes

- `modal run` is enough for a sweep. `modal deploy` only matters if you want to call
  these functions from other code.
- The image build (torch 2.11+cu128, TensorRT 10.15, two git deps) takes ~10 minutes the
  first time and is cached afterwards.
- `checkpoints/` ships with the image, so new weights mean a rebuild.
