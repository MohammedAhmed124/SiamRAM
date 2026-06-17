# Jetson Orin Nano Optimization Report

Date: 2026-06-17

Status: exploration only. No runtime behavior is changed by this document.

This report supersedes the earlier optimization draft. It was rewritten after reading
`docs/JETSON_ORIN_NANO_OPTIMIZATION_AUDIT.md` and then re-checking the strongest
claims directly against the repository. The goal is not just to list possible speedups,
but to give an implementation-grade backlog for getting SiamRAM onto a Jetson Orin
Nano without changing tracking behavior.

## Executive Verdict

The codebase is already pointed in the right direction for Jetson:

- SiamABC has TensorRT backbone/connect paths with cache fingerprinting.
- OSNet has TensorRT, FP16, and side-stream overlap support.
- Runtime frame size is capped by `max_proc_long_edge`.
- YOLO input is already small at `yolo_imgsz: 320`.
- Descriptor work is bounded by `descriptor_stride: 2` and `osnet_max_candidate_batch: 5`.
- Camera motion defaults to the cheaper `classic` mode.
- The non-video-output inference path preallocates bbox output and avoids visualization cost.

The best next optimizations are plumbing around the algorithms, not algorithm changes:

1. Overlap frame decode/read with tracking compute.
2. Stop reloading YOLO for every clip while preserving fresh sequence state.
3. Remove per-frame CUDA synchronization points in SiamABC decode/postprocess.
4. Make Jetson profiling include decode time, TRT cache behavior, RSS, and CUDA memory.
5. Split Jetson runtime packaging away from workstation/dev/training dependencies.

The high-priority work can be behavior-preserving if validated with the existing golden
regression harness and per-frame bbox diffs.

## Safety Classes

Use this classification before implementing anything:

| Class | Meaning | Examples |
|---|---|---|
| A | Should be byte-identical if implemented correctly | frame prefetch with ordered output, YOLO object cache with fresh tracker state, profiler changes, Docker ignore changes |
| B | Should preserve decisions, but may need numeric parity checks | removing CUDA syncs, coalescing GPU postprocess, `cudnn.benchmark` behind a flag |
| C | Intentionally behavior-changing or accuracy-risky | FP16 YOLO TRT, INT8, TF32, SiamABC M to S, lower `max_proc_long_edge`, lower `yolo_imgsz`, higher `descriptor_stride` |

The first implementation pass should stay in Class A and the lowest-risk parts of
Class B.

## Top Backlog

| Rank | Class | Item | Expected Benefit | Evidence | Effort |
|---:|---|---|---|---|---|
| 1 | A | Add one-frame/two-frame input prefetch | High when JPEG/video decode is nontrivial | `vis/test_model.py:370-417`, `profile_components.py:327-342` | Low-Med |
| 2 | A | Cache/inject YOLO model object per process | High for full split runs with many clips | `run_inference.py:1425-1438`, `run_inference.py:1570-1572`, `tracker.py:389-392` | Med |
| 3 | B | Remove TRT SiamABC `lam.item()` sync | Medium-High per-frame latency cleanup | `siamabc.py:487-490` | Low |
| 4 | B | Collapse SiamABC decode/postprocess host syncs | Medium per-frame latency cleanup | `box_coder.py:395-398`, `SiamABC_Tracker.py:954-995`, `base_tracker.py:276`, `base_tracker.py:336` | Med |
| 5 | A | Make profiler production-faithful | High measurement value | `profile_components.py:269-277`, `profile_components.py:457-472` | Low |
| 6 | A | Remove dead albumentations runtime dependency | Startup/deploy simplification | `base_tracker.py:14`, `base_tracker.py:107-191`, `base_tracker.py:214-222` | Low |
| 7 | A | Stream submission CSV instead of pandas materialization | Memory reduction on 4 GB/8 GB devices | `run_inference.py:860-897`, `run_inference.py:1650`, `run_inference.py:1688` | Low-Med |
| 8 | A | Split Jetson runtime deps/container | Deployment blocker reduction | `pyproject.toml:7-57`, `containers/Dockerfile.gpu:1` | Med |
| 9 | A | Tighten Docker ignores for nested TRT/ReID caches | Smaller context, fewer stale engine mistakes | `.dockerignore:22-25`, `.gitignore:125-132`, TRT cache `.ts` paths | Low |
| 10 | A/B | Reuse occlusion candidate descriptors | Less OSNet work during reacquisition | `occlusion_recovery.py:367-375`, `memory.py:224`, `memory.py:303` | Med |
| 11 | A | Add central CPU/OpenCV/PyTorch thread knobs | Lower latency variance under Orin CPU limits | no current thread controls found outside docs | Low |
| 12 | A | Add no-video/profile-fast mode to `run_single_video.py` | Cleaner local perf testing | `run_single_video.py` always requests output video | Low |

## Source Evidence Index

| Area | Evidence |
|---|---|
| Serialized frame read/decode | `vis/test_model.py:370-417` reads with `cap.read()`/`cv2.imread()` immediately before `tracker.update(frame)` |
| Profiler excludes read/decode from frame timing | `profile_components.py:327-342` yields decoded frames before timing starts at `profile_components.py:461` |
| Fresh wrapper per clip | `run_inference.py:1425-1438` logs fresh wrapper mode, `run_inference.py:1570-1572` constructs per clip |
| YOLO reload | `models/siamram/tracker.py:389-392` calls `YOLO(yolo_weights)` in constructor |
| OSNet already singleton | `utils/utils.py:413`, `utils/utils.py:675-697` lazy-build one `_OSNET_EXTRACTOR` |
| TRT TTA sync | `models/SiamABC/tracker/trt_engine/siamabc.py:487-490` uses `float(lam.item())` |
| Box coder syncs | `utils/box_coder.py:395-398` uses two `.item()` calls after `argmax` |
| SiamABC postprocess syncs | `SiamABC_Tracker.py:954`, `SiamABC_Tracker.py:961`, `SiamABC_Tracker.py:974`, `SiamABC_Tracker.py:995`, `base_tracker.py:276`, `base_tracker.py:336` |
| Dead albumentations path | `base_tracker.py:14`, transform construction at `base_tracker.py:107-191`, unused transform arg at `base_tracker.py:214-222` |
| Current Jetson-friendly config | `config/inference_config_experimental.yaml:23`, `:131`, `:142`, `:150`, `:184`, `:191`, `:214` |
| Workstation CUDA dependency profile | `pyproject.toml:45-50`, CUDA 12.8 index at `pyproject.toml:90-93` |
| x86/workstation-style GPU image | `containers/Dockerfile.gpu:1` uses `nvidia/cuda:12.8.1-base-ubuntu22.04` |
| Docker context gaps | `.dockerignore:22-25` only top-level checkpoint globs, `.gitignore:125-132` shows broader patterns |
| YOLO compile path | typo key in config at `config/inference_config_experimental.yaml:150`; FP32 export at `tracker.py:3556` |
| Golden regression harness | `tests/test_regression.py:1-26`, exact bbox and mode checks with score tolerance |

## Hot Path Map

Normal frame update, simplified:

1. Read/decode frame on CPU.
2. Optional prescale on CPU.
3. Camera motion estimate on CPU using downscaled grayscale optical flow.
4. EKF predict on CPU.
5. SiamABC crop/preprocess/forward/decode on GPU with several host syncs.
6. Optional YOLO detectability/class probes.
7. Optional OSNet descriptor extraction, strided and capped.
8. EKF update, appearance memory, and bookkeeping on CPU.

Important Jetson implications:

- The Orin Nano CPU is limited enough that serialized JPEG/video decode can be visible.
- CPU optical flow and decode compete with Python and OpenCV thread pools.
- Unified memory makes host-device copies less PCIe-like, but `.item()` and `.cpu()`
  still force stream synchronization. The stall is the problem.
- DLA work is not a good target for this tracker on Orin Nano. Keep focus on CUDA/TensorRT,
  CPU scheduling, memory pressure, and I/O overlap.

## Current Settings Worth Preserving

| Config | Current Value | Why It Helps |
|---|---:|---|
| `trt_engine.trt_compile_siamabc` | `True` | Main tracker has a TensorRT path |
| `trt_engine.backbone_mode` | `dynamic_fp16` | FP16 backbone lowers compute and bandwidth |
| `trt_engine.trt_compile_osnet` | `True` | ReID backbone has a TensorRT path |
| `trt_engine.osnet_fp16` | `True` | Descriptor model runs lighter |
| `trt_engine.osnet_async_overlap` | `True` | Descriptor work can overlap with SiamABC |
| `ram_tracker.runtime.max_proc_long_edge` | `1280` | Bounds CPU/GPU pixel work |
| `ram_tracker.yolo.yolo_imgsz` | `320` | Keeps YOLO detector small |
| `ram_tracker.yolo.osnet_max_candidate_batch` | `5` | Bounds ReID work per detector pass |
| `ram_tracker.yolo.descriptor_stride` | `2` | Avoids per-frame normal-path descriptors |
| `ram_tracker.camera_motion.homography_mode` | `classic` | Keeps camera motion on the cheap path |
| `ram_tracker.occlusion.cand_collection_frames` | `1` | Bounds recovery collection cost |
| `ram_tracker.occlusion.reacq_confirm_frames` | `1` | Bounds confirmation delay and compute |

## Detailed Findings

### 1. P0 Class A: Add Input Prefetch

Relevant files:

- `vis/test_model.py`
- `profile_components.py`

In the fast inference loop, `_read_next_frame()` blocks on `cap.read()` or `cv2.imread()`
and only then calls `tracker.update(frame)`. The profiler has the same shape: the frame
is yielded by `_frame_reader()` before timing starts around `tracker.update(frame)`.

That means decode/read and GPU tracking are serialized. On a workstation this can be
masked by a strong CPU. On Orin Nano, especially with image sequences or high-resolution
video, decode can become a visible part of frame time.

Behavior-preserving implementation:

- Add a single producer thread with a queue depth of 1 or 2.
- Producer reads/decompresses frame N+1 while the main thread tracks frame N.
- Preserve exact frame order and termination behavior.
- Propagate read errors in the same place the current loop would stop.
- Keep this behind a small helper, for example `PrefetchFrameReader`, so both
  `vis/test_model.py` and `profile_components.py` can use it.

Validation:

- Run `python -m tests.test_regression`.
- Compare per-frame bbox text for a short sequence before/after.
- Profile wall-clock FPS, not just `tracker.update()` time.
- Keep the old synchronous reader selectable for debugging.

Why this is better than tuning models first:

- It can improve throughput without touching tracker scores, thresholds, models, or
  candidate logic.
- It makes the profiler more honest because decode time becomes measurable and hideable.

### 2. P0 Class A: Cache YOLO Object Without Reusing Tracker State

Relevant files:

- `run_inference.py`
- `models/siamram/tracker.py`

The default split runner constructs a fresh `SiamRAMExperimentTracker` per clip unless
`--reuse_tracker` is enabled. Fresh wrappers are good for sequence isolation, but each
wrapper constructor loads YOLO:

```python
self.yolo = YOLO(yolo_weights)
```

OSNet does not have the same problem because it is already process-global through
`_OSNET_EXTRACTOR`.

Behavior-preserving implementation:

- Add a process-local YOLO cache keyed by:
  - resolved weights path
  - device
  - `yolo_imgsz`
  - compiled engine mode
  - precision/engine fingerprint if TensorRT YOLO is used later
- Let `SiamRAMExperimentTracker` accept an optional `yolo_model` object.
- Keep all sequence-local state fresh by still constructing a new wrapper per clip.
- Do not rely on `--reuse_tracker` as the main optimization, because that can share
  more state than intended.

Validation:

- `python -m tests.test_regression`
- Two-clip smoke run with fresh wrapper plus cached YOLO.
- Diff generated bbox files before/after.
- Log YOLO load count in a temporary debug run and confirm it drops from per-clip to once.

### 3. P0 Class B: Remove Per-Frame `lam.item()` In TRT SiamABC

Relevant files:

- `models/SiamABC/tracker/trt_engine/siamabc.py`
- `models/SiamABC/tracker/SiamABC_Tracker.py`

`TRTSiamABCNet.track()` dispatches the connect engine using:

```python
lam_val=float(lam.item())
```

Because `lam` is a CUDA tensor, `.item()` synchronizes the stream every frame.

Behavior-preserving implementation:

- Store a Python-side TTA mode/value when `SiamABCTracker.set_tta()` updates the tensor.
- Pass that Python value to `_dispatch_connect()`.
- Keep the CUDA tensor for math if needed, but never read it back to decide dispatch.

Validation:

- `python -m tests.test_regression`
- Run `benchmark_siamabc_backbone.py` and compare parity.
- Profile to confirm the sync is gone.

### 4. P1 Class B: Collapse SiamABC Decode/Postprocess Syncs

Relevant files:

- `utils/box_coder.py`
- `models/SiamABC/tracker/SiamABC_Tracker.py`
- `models/SiamABC/tracker/base_tracker.py`

Current sync points include:

| Sync | Why It Stalls |
|---|---|
| `lam.item()` | Reads CUDA scalar for Python dispatch |
| two `flat_idx...item()` calls | Reads row/column separately after GPU argmax |
| `cls_score.squeeze().cpu().numpy()` | Copies score map before peak score lookup |
| `decoded_info.bbox.cpu().numpy()` | Copies decoded bbox |
| `penalty.cpu().numpy()` | Only when smoothing is enabled, but still a sync in that mode |
| `iou_score_map...cpu().numpy()` | Only when IoU map is active |
| `sim_score_raw.item()` | Reads similarity scalar if present |

Low-risk first step:

- Replace the two box-coder `.item()` calls with one:

```python
flat = int(flat_idx.item())
r_max, c_max = divmod(flat, W)
```

Higher-value step:

- Keep peak selection, peak score lookup, bbox extraction, optional IoU lookup, and
  optional smoothing math on GPU.
- Copy one compact tensor/tuple to CPU at the end.
- Preserve current rounding, clipping, and threshold behavior exactly.

Validation:

- Golden regression.
- Additional per-frame score comparison around occlusion thresholds.
- Benchmark normal frames and occlusion-verification frames separately.

### 5. P1 Class A: Make Profiler Production-Faithful

Relevant file:

- `profile_components.py`

Current gaps:

- `get_trt_tracker()` is called without the same cache controls exposed by `run_inference.py`.
- Decode/read time is outside the measured frame bucket.
- Memory pressure is not reported.
- Cache hit/miss/rebuild status is not surfaced.

Implementation:

- Add `--trt_cache_dir`, `--rebuild_trt_cache`, and `--use_existing_trt_cache`.
- Pass cache settings into the TRT tracker builder.
- Add timing buckets for frame read/decode and optional output/write.
- Report process RSS, peak CUDA allocated/reserved memory, and whether caches were reused.
- Optionally emit JSON/CSV so Jetson runs can be compared over time.

Validation:

- Tooling-only change, so tracker outputs should not change.
- Run one warm-cache profile and one cold-cache profile to confirm the report separates
  compile/setup cost from steady-state tracking.

### 6. P1 Class A: Drop Dead `albumentations` From Runtime Path

Relevant file:

- `models/SiamABC/tracker/base_tracker.py`

`albumentations` is imported and three transform callables are constructed, but
`_preprocess_image(image, transform=None)` never uses the `transform` argument. The real
normalization is done directly on GPU.

Behavior-preserving implementation:

- Remove the `albumentations` import from the runtime tracker path.
- Remove `_get_default_transform()` and the transform attributes, or leave harmless
  `None` placeholders if external code expects those attributes.
- Remove `albumentations` from the Jetson runtime dependency set.

Validation:

- Golden regression should be identical because the transform is already ignored.
- Import/startup timing should improve slightly.
- Jetson dependency resolution becomes simpler.

### 7. P1 Class A: Stream Submission CSV

Relevant file:

- `run_inference.py`

The current submission path reads bbox files into Python lists, builds pandas DataFrames,
does duplicate validation through pandas, then writes CSV. That is fine on a workstation,
but unnecessary on an 8 GB or 4 GB Jetson run.

Behavior-preserving implementation:

- Stream rows with `csv.DictWriter`.
- Maintain a `set` of ids for duplicate validation.
- For `--override_csv`, stream preserved rows and rebuilt rows instead of constructing
  two full DataFrames.
- Keep pandas in evaluation/offline tools if desired, but do not require it for the
  main Jetson inference path.

Validation:

- Generate small submissions with both implementations.
- Compare parsed rows and column order.
- Preserve numeric formatting intentionally. If exact byte parity is required, match
  pandas' float formatting or explicitly decide the competition accepts equivalent CSV.

### 8. P1 Class A: Jetson Runtime Dependency And Container Split

Relevant files:

- `pyproject.toml`
- `pyproject.cpu.toml`
- `containers/Dockerfile.gpu`

The current GPU environment pins workstation CUDA packages (`torch==2.11.0`,
`torchvision==0.26.0`, `torch-tensorrt==2.11.0`, `triton==3.6.0`) from the CUDA 12.8
PyTorch index. The GPU Dockerfile starts from `nvidia/cuda:12.8.1-base-ubuntu22.04`.

That is not a Jetson deployment profile. Jetson needs JetPack/L4T-compatible PyTorch,
TensorRT, and Torch-TensorRT builds. The main project dependencies also include many
dev/training/notebook packages that should not be in the runtime image.

Implementation:

- Add `docs/install-Jetson-Orin-Nano.md` or `pyproject.jetson.toml`.
- Split dependencies into:
  - runtime inference
  - evaluation/reporting
  - training/dev/notebooks
- Use a JetPack/L4T-compatible base image or document native install.
- Rebuild TRT engines on-device into a mounted writable cache.
- Keep datasets, outputs, and engine caches outside the image.

Validation:

- Fresh Jetson environment can import `torch`, `torchvision`, `tensorrt`, `torch_tensorrt`,
  `cv2`, `ultralytics`, and the tracker modules.
- `python containers/test.py`
- One warm-cache split smoke test.

### 9. P2 Class A: Tighten Docker Context Rules

Relevant files:

- `.dockerignore`
- `.gitignore`

Large local artifact directories observed:

- `data_imgs`: about 68 GB
- `data`: about 25 GB
- `outputs`: about 5.4 GB
- `checkpoints`: about 797 MB

`.dockerignore` excludes `data`, `data_imgs`, `outputs`, and some top-level checkpoint
globs, but not all nested caches. `.gitignore` already has broader checkpoint patterns,
including `.ts` and `.ep`.

Implementation:

- Add recursive Docker ignores:

```gitignore
checkpoints/**/*.pt
checkpoints/**/*.pth
checkpoints/**/*.onnx
checkpoints/**/*.engine
checkpoints/**/*.plan
checkpoints/**/*.ts
checkpoints/**/*.ep
checkpoints/reid/
checkpoints/trt_engines/
```

- Mount caches explicitly on Jetson instead of baking them into images.
- Treat workstation-built TensorRT engines as disposable; rebuild on the target.

Validation:

- Compare Docker build context size before/after.
- Confirm required small config/source files are still included.

### 10. P2 Class A/B: Reuse Occlusion Candidate Descriptors

Relevant files:

- `models/siamram/occlusion_recovery.py`
- `models/siamram/memory.py`

During occlusion collection, candidates are stored as `CandidateRecord` with descriptors.
Later RAM/DRM matching receives only bboxes and re-extracts descriptors from the frame.
That duplicates OSNet work during reacquisition bursts.

Implementation:

- Add descriptor-aware variants to `AppearanceMemory.match()` and `drm_match()`, or let
  them accept `(bbox, descriptor)` records.
- Use provided descriptors when present.
- Fall back to `_extract_descriptor()` for callers that only have bboxes.
- Preserve candidate ordering and filtering.

Validation:

- Golden regression.
- Focused occlusion sequence test.
- Instrument OSNet extraction count during occlusion before/after.

### 11. P2 Class A/B: Add Thread Controls

Relevant files:

- Runtime entrypoints
- Jetson install docs

There are no code-level controls for OpenCV/PyTorch/BLAS thread counts. On Orin Nano,
OpenCV decode, optical flow, Python, and model runtimes can oversubscribe CPU cores and
increase latency variance.

Implementation:

- Add environment-driven startup controls:

```python
cv2.setNumThreads(int(os.getenv("SIAMRAM_OPENCV_THREADS", "0")))
torch.set_num_threads(int(os.getenv("SIAMRAM_TORCH_THREADS", "1")))
```

- Document shell-level knobs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

- Tune on-device. Do not hard-code a magic number without measurements.

Validation:

- Compare mean, p95, and p99 frame time over a long warm run.
- Watch `tegrastats`/`jtop` for CPU saturation and thermal throttling.

### 12. P2 Class A/B: `cudnn.benchmark` Startup Knob

No code path currently sets `torch.backends.cudnn.benchmark` outside documentation.
The model shapes are mostly fixed, so cuDNN autotune may help non-TRT convolution paths.

Nuance:

- This is usually safe for inference, but it can change selected kernels and therefore
  tiny floating-point details.
- Do not silently enable it if exact golden parity is required.
- Add a CLI/env knob, default it based on validation results, and keep a deterministic
  escape hatch.

Implementation:

```python
torch.backends.cudnn.benchmark = bool(os.getenv("SIAMRAM_CUDNN_BENCHMARK", "1") == "1")
```

Validation:

- Golden regression on the target device.
- Profile warm runs with and without the knob.

### 13. P3 Class A: Make `run_single_video.py` Usable For Fast Tests

Relevant file:

- `run_single_video.py`

The helper is useful for qualitative review, but always enabling output video makes it
a poor speed test. Visualization and encoding can dominate the run.

Implementation:

- Add `--no_output_video` or `--profile_fast`.
- Keep visualization as the default if that is the script's purpose.
- Print a warning when users request timing with video output enabled.

### 14. P3 Class A/B: Safer YOLO TensorRT Path

Relevant files:

- `config/inference_config_experimental.yaml`
- `models/siamram/tracker.py`

The config key is misspelled as `copile_yolo`, and the engine path is a simple
`.pt` to `.engine` replacement. Export currently uses `half=False` and `device=0`.

Safe implementation:

- Support `compile_yolo` as an alias while keeping `copile_yolo` backward-compatible.
- Fingerprint YOLO engines by weights, image size, precision, device, TensorRT version,
  and JetPack/L4T environment.
- Keep YOLO TensorRT disabled by default until detector parity is measured.

Behavior-changing follow-up:

- Test FP16 YOLO TRT (`half=True`) separately. It may be much faster, but it can change
  detector scores and boxes.

## What Not To Do First

Do not start with these unless you are explicitly trading accuracy for speed:

- SiamABC model size `M` to `S`.
- Lowering `max_proc_long_edge`.
- Lowering `yolo_imgsz`.
- Increasing `descriptor_stride`.
- Disabling YOLO, OSNet, detectability probing, or occlusion recovery.
- Enabling INT8 without calibration and metric validation.
- Enabling TF32 on the FP32 score-sensitive stages without checking threshold behavior.
- Treating `run_single_video.py` with output video as a performance benchmark.
- Shipping workstation-built TRT caches to Jetson.
- Spending time on DLA placement for this Orin Nano target.
- Chasing pinned-memory tricks before fixing serialized decode and CUDA syncs.

## Implementation Plan

### Phase 0: Measurement First

1. Add production TRT cache args to `profile_components.py`.
2. Add read/decode timing.
3. Add RSS and CUDA peak memory reporting.
4. Build TRT caches once on-device.
5. Save golden outputs and baseline profile JSON.

Exit gate:

- Warm-cache profiler runs successfully on Jetson.
- `python -m tests.test_regression` passes in the deployment environment.

### Phase 1: Class A Wins

1. Add ordered frame prefetch.
2. Add YOLO model cache/injection.
3. Remove dead albumentations runtime path.
4. Stream submission output.
5. Add no-video fast mode for the helper script.
6. Tighten `.dockerignore`.

Exit gate:

- Per-frame bbox files match before/after for smoke clips.
- Full small split produces same parsed submission rows.
- Startup/load logs confirm YOLO loads once per process, not once per clip.

### Phase 2: Class B Hot-Path Cleanup

1. Remove `lam.item()` dispatch sync.
2. Collapse box-coder double `.item()` to one.
3. Move SiamABC peak/bbox/score postprocess toward one host transfer.
4. Add `cudnn.benchmark` as an opt-in or validated default.
5. Add thread-count knobs and tune on-device.

Exit gate:

- Golden regression passes.
- Score differences, if any, are below the accepted threshold and do not change
  occlusion state.
- Profile shows lower p95 latency, not just lower mean latency.

### Phase 3: Deployment Hardening

1. Add Jetson install/container documentation.
2. Split runtime/dev/eval dependencies.
3. Document clock/thermal/profile methodology.
4. Fingerprint YOLO TensorRT engines before enabling compiled YOLO.
5. Evaluate behavior-changing speed knobs only after the safe backlog is done.

## Validation Commands

Basic import/environment smoke:

```bash
python containers/test.py
```

Golden behavior check:

```bash
python -m tests.test_regression
```

Config/cache tests:

```bash
pytest tests/test_config_schema.py tests/test_siamabc_trt_cache.py
```

SiamABC TensorRT mode/cache benchmark:

```bash
uv run benchmark_siamabc_backbone.py --cache-dir checkpoints/trt_engines --max-frames 120
```

Production split smoke:

```bash
uv run run_inference.py --run_split public_lb --max_sequences 1 --use_existing_trt_cache
```

Future production-faithful component profile, after profiler args are added:

```bash
uv run profile_components.py \
  --video_key dataset5/person19_3 \
  --frames 300 \
  --warmup_frames 30 \
  --trt_cache_dir checkpoints/trt_engines \
  --use_existing_trt_cache
```

If `uv` is not used on Jetson, run the same scripts through the active Python
environment.

## Jetson Measurement Notes

Before trusting any number:

- Use the target Jetson, not a workstation.
- Build TensorRT engines on the target once, then profile warm-cache runs.
- Keep the cache directory writable and outside the Docker image.
- Watch clocks, temperature, RAM, and swap with `tegrastats` or `jtop`.
- Record mean, median, p95, and p99 frame time.
- Report normal and occlusion frames separately.
- Report decode/read time separately from `tracker.update()`.
- Run enough clips to expose per-sequence startup costs.

## Final Recommendation

The first real implementation sprint should be:

1. Production-faithful profiler.
2. Ordered frame prefetch.
3. YOLO object cache with fresh tracker state.
4. `lam.item()` removal.
5. Box-coder one-sync cleanup.
6. Dead albumentations removal.
7. Streaming submission writer.
8. Jetson dependency/container split.

That sequence targets the highest-value Orin Nano bottlenecks while preserving tracker
behavior and giving clear validation gates after every step.
