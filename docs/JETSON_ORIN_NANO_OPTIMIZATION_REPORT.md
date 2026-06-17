# Jetson Orin Nano Optimization Exploration

Date: 2026-06-17

Scope: exploratory analysis only. This report does not change runtime behavior. It documents optimization opportunities that appear likely to help when running SiamRAM on a Jetson Orin Nano, with emphasis on changes that can preserve current tracking behavior.

## Executive Summary

The tracker already contains several Jetson-friendly optimizations:

- SiamABC backbone TensorRT support with cache fingerprinting.
- OSNet TensorRT support, FP16 support, and optional async overlap.
- A capped processing resolution through `ram_tracker.runtime.max_proc_long_edge`.
- A small YOLO inference size through `ram_tracker.yolo.yolo_imgsz`.
- Bounded ReID work through `descriptor_stride` and `osnet_max_candidate_batch`.
- Classic camera motion mode using low-resolution optical flow.
- A fast inference path in `vis/test_model.py` when `output_video=False`.

The best easy wins are not algorithm changes. They are mostly runtime plumbing:

1. Reuse heavy model objects, especially YOLO, without reusing sequence state.
2. Remove avoidable CUDA synchronization points in the per-frame hot path.
3. Make Jetson profiling measure the same cache and decode path used in production.
4. Split Jetson runtime dependencies and deployment packaging from workstation/dev dependencies.
5. Reduce submission and reporting memory overhead for full-split runs.

None of the high-priority recommendations require changing the tracker decision logic or expected outputs if implemented carefully and validated with the existing regression harness.

## Priority Table

| Priority | Finding | Why It Matters On Orin Nano | Behavior Risk | Suggested Action |
|---|---|---|---|---|
| P0 | Per-clip YOLO model construction in the default fresh-wrapper path | Model load/initialization is expensive, and full split runs contain many clips | Low if state is not shared | Cache or inject the YOLO model while still creating fresh tracker state per sequence |
| P0 | Per-frame CUDA scalar synchronization in TensorRT SiamABC dispatch | Synchronizing a CUDA scalar every frame can stall the Orin GPU/CPU pipeline | Low | Replace `float(lam.item())` with a Python-side TTA/connect-mode flag |
| P1 | Profiling does not fully match production cache/decode behavior | Optimization choices can be misleading if TRT caches and video decode are excluded | Low | Add cache arguments, decode/read timing, RSS, and CUDA memory stats to `profile_components.py` |
| P1 | Submission building uses pandas/list materialization | Large split output can increase memory pressure on an 8 GB class device | Low | Stream CSV rows with stdlib `csv` or incrementally merge outputs |
| P1 | Occlusion recovery may re-extract descriptors already computed for candidates | OSNet/ReID can become expensive during occlusion and reacquisition bursts | Low to Medium | Pass cached candidate descriptors into DRM/memory matching |
| P1 | SiamABC postprocess copies small maps/scalars from GPU to CPU each frame | Small copies still cause synchronization and limit overlap | Low to Medium | Keep argmax/score postprocess on GPU and copy only final compact results |
| P2 | Deployment dependencies are workstation/CUDA oriented | Jetson needs JetPack/L4T-compatible PyTorch/TensorRT packages | Medium operational risk | Add a Jetson runtime profile or install doc separate from dev/training dependencies |
| P2 | `.dockerignore` excludes some large top-level artifacts but not all nested caches | Docker context can accidentally include hundreds of MB of nested checkpoint/cache data | Low | Recursively ignore nested TRT/ReID caches or mount them explicitly |
| P2 | YOLO TensorRT config uses typo key and simple engine path | Harder to enable and safely cache YOLO engines | Low if alias-only | Support `compile_yolo` alias and make engine naming cache-aware |
| P3 | CPU/OpenCV thread counts are not centrally controlled | Orin has limited CPU headroom, and oversubscription hurts latency | Low | Add documented env/profile knobs for `OMP_NUM_THREADS`, OpenCV threads, and PyTorch threads |
| P3 | Performance helper script always emits video | Video writing and overlays hide tracker speed | Low | Add a fast/no-video mode to `run_single_video.py` |

## Current Jetson-Friendly Settings To Preserve

These settings in `config/inference_config_experimental.yaml` are good defaults for the Orin Nano target and should not be casually relaxed:

| Config | Current Value | Why It Helps |
|---|---:|---|
| `trt_engine.trt_compile_siamabc` | `True` | Uses TensorRT for the main SiamABC path |
| `trt_engine.backbone_mode` | `dynamic_fp16` | Uses an FP16 TensorRT backbone path |
| `trt_engine.trt_compile_osnet` | `True` | Uses TensorRT for the OSNet ReID extractor |
| `trt_engine.osnet_fp16` | `True` | Reduces OSNet compute and memory bandwidth |
| `trt_engine.osnet_async_overlap` | `True` | Allows descriptor work to overlap with SiamABC work |
| `ram_tracker.runtime.max_proc_long_edge` | `1280` | Caps per-frame pixel work |
| `ram_tracker.yolo.yolo_imgsz` | `320` | Keeps detector input small |
| `ram_tracker.yolo.osnet_max_candidate_batch` | `5` | Bounds descriptor work per detector pass |
| `ram_tracker.yolo.descriptor_stride` | `2` | Avoids descriptor extraction every frame |
| `ram_tracker.camera_motion.homography_mode` | `classic` | Uses the cheaper low-resolution affine/flow path |
| `ram_tracker.occlusion.cand_collection_frames` | `1` | Bounds occlusion candidate accumulation |
| `ram_tracker.occlusion.reacq_confirm_frames` | `1` | Bounds confirmation delay and compute |

## Detailed Findings

### P0: Avoid Per-Clip YOLO Reload While Keeping Fresh Tracker State

Relevant files:

- `run_inference.py`
- `models/siamram/tracker.py`

The default batch inference path builds the base SiamABC tracker once, then creates a fresh `SiamRAMExperimentTracker` wrapper for each sequence unless `--reuse_tracker` is enabled. Fresh wrappers are safer for state isolation, but the wrapper constructor loads YOLO from `yolo_weights` each time.

That means a full split run can repeatedly initialize the same detector. On Orin, this can dominate startup time per clip and add memory churn.

Recommended behavior-preserving fix:

- Add a small YOLO model cache keyed by weights path, device, image size, and compiled/engine mode.
- Inject the cached model into each fresh tracker wrapper.
- Continue resetting all sequence-local tracking state in `initialize()`.

This preserves the safer "fresh tracker per sequence" behavior while avoiding repeated detector construction.

Validation:

- Compare outputs against `tests/test_regression.py`.
- Run a small multi-sequence split once with the old path and once with cached YOLO, checking submission parity.
- Measure wall-clock startup time per sequence.

Notes:

- `--reuse_tracker` already avoids some reload overhead, but the current help text warns that sequence-local state issues may make clips depend on previous clips. A shared YOLO object is a safer target than full wrapper reuse.

### P0: Remove Per-Frame CUDA Scalar Sync In TensorRT SiamABC Dispatch

Relevant file:

- `models/SiamABC/tracker/trt_engine/siamabc.py`

In the TensorRT SiamABC path, connect-engine dispatch uses a CUDA scalar read similar to:

```python
float(lam.item())
```

Reading `.item()` from a CUDA tensor forces CPU/GPU synchronization. Because this happens in the frame update path, it can serialize work and reduce overlap on Jetson.

Recommended behavior-preserving fix:

- Track the TTA/connect mode as Python state when the tracker calls `set_tta()`.
- Dispatch the zero-lambda or lambda connect engine from that Python flag instead of reading a CUDA tensor.
- Keep the tensor value for computation, but do not use `.item()` to decide runtime control flow.

Validation:

- Run `tests/test_regression.py`.
- Run `benchmark_siamabc_backbone.py` before/after and compare parity.
- Use a profiler to confirm the synchronization point disappears.

### P1: Make Profiling Representative Of Production On Orin

Relevant file:

- `profile_components.py`

The profiling script is valuable, but it should more closely mirror real Jetson production runs. During inspection, the script did not appear to pass the same `engine_cache_dir`/rebuild policy into `get_trt_tracker()` that `run_inference.py` uses. Its hot timing also focuses on `tracker.update(frame)` and does not expose video decode/read time as a first-class bucket.

Recommended fix:

- Add `--trt_cache_dir`, `--rebuild_trt_cache`, and `--use_existing_trt_cache` to `profile_components.py`.
- Pass those cache settings into tracker construction.
- Report separate timings for:
  - frame read/decode
  - preprocessing/camera motion
  - SiamABC update
  - YOLO
  - OSNet/ReID
  - occlusion recovery
  - output/write time, if enabled
- Report:
  - process RSS
  - peak CUDA allocated/reserved memory
  - TensorRT cache hit/miss/rebuild status

This is low risk because it changes tooling, not tracker behavior.

### P1: Stream Submission Output Instead Of Building Large DataFrames

Relevant file:

- `run_inference.py`

The submission assembly path uses pandas and list/DataFrame materialization. On a workstation this is convenient. On an Orin Nano, it adds import cost and can create avoidable memory spikes for full split runs.

Recommended behavior-preserving fix:

- Write submission rows incrementally with Python's `csv` module.
- Preserve the exact column order and formatting currently produced by pandas.
- Keep pandas-based utilities for offline analysis if useful, but avoid requiring pandas in the main Jetson runtime path.

Validation:

- Generate a small submission with both paths.
- Compare files byte-for-byte if formatting is preserved, or compare parsed rows if pandas formatting differs.

### P1: Reuse Candidate Descriptors During Occlusion Recovery

Relevant files:

- `models/siamram/occlusion_recovery.py`
- `models/siamram/memory.py`

The occlusion recovery flow collects candidate records that can already include descriptors. Later memory/DRM matching paths can re-extract descriptors for candidate boxes from the same frame. This duplicates OSNet work during the exact frames where reacquisition can become compute-heavy.

Recommended behavior-preserving fix:

- Add descriptor-aware matching entry points in `AppearanceMemory`.
- Let recovery pass candidate descriptors forward when they are already available.
- Fall back to extraction when descriptors are missing.

Validation:

- Run golden regression tests.
- Add a focused occlusion scenario that checks the same candidate is selected before and after.
- Profile OSNet call count in occlusion frames.

### P1: Reduce GPU-To-CPU Synchronization In SiamABC Postprocess

Relevant files:

- `models/SiamABC/tracker/SiamABC_Tracker.py`
- `models/SiamABC/tracker/base_tracker.py`

SiamABC postprocess copies several small values/maps to CPU every frame: decoded bounding boxes, class score maps, optional IoU score maps, and scalar scores. The tensors are small, but every CPU copy can synchronize the GPU stream.

Recommended behavior-preserving fix:

- Keep score-map peak selection and score arithmetic on GPU.
- Copy only the final compact result to CPU: bbox, best score, and any required diagnostics.
- Preserve current numerical logic and thresholds.

Validation:

- Run `tests/test_regression.py`.
- Compare per-frame bbox and score values on a representative sequence.
- Profile synchronization count/latency before and after.

### P2: Add A Jetson Runtime Dependency And Container Profile

Relevant files:

- `pyproject.toml`
- `pyproject.cpu.toml`
- `containers/Dockerfile.gpu`
- `containers/docker-compose.gpu.yml`

The current GPU environment is oriented around workstation CUDA packages, including `torch==2.11.0`, `torchvision==0.26.0`, `torch-tensorrt==2.11.0`, `triton==3.6.0`, and a CUDA 12.8 package index. The GPU Dockerfile starts from `nvidia/cuda:12.8.1-base-ubuntu22.04`.

Jetson deployments usually need JetPack/L4T-compatible base images and NVIDIA-provided or Jetson-compatible PyTorch/TensorRT packages. The current dependency set also includes many dev/training/notebook packages in the main runtime environment, such as IPython/Jupyter tools, matplotlib, pandas, tensorboard, wandb, scikit-learn, and training-oriented libraries.

Recommended operational fix:

- Add a Jetson-specific install document or `pyproject.jetson.toml`.
- Separate runtime dependencies from dev/training/evaluation dependencies.
- Use a Jetson/L4T-compatible base image or document native install steps.
- Keep TensorRT engine caches outside the image and mount them at runtime.

Behavior risk is operational rather than algorithmic, but this is important before deployment.

### P2: Tighten Docker Context Excludes For Nested Caches

Relevant files:

- `.dockerignore`
- `.gitignore`

The workspace contains large local artifacts:

- `data_imgs`: about 68 GB
- `data`: about 25 GB
- `outputs`: about 5.4 GB
- `checkpoints`: about 797 MB

The current `.dockerignore` excludes several large top-level directories and checkpoint extensions, which is good. However, the checkpoint rules are not fully recursive for nested cache locations such as `checkpoints/reid` and `checkpoints/trt_engines`.

Recommended behavior-preserving fix:

- Recursively ignore generated cache artifacts in Docker contexts:
  - `checkpoints/**/*.engine`
  - `checkpoints/**/*.plan`
  - `checkpoints/**/*.onnx`
  - `checkpoints/**/*.ts`
  - `checkpoints/**/*.ep`
  - `checkpoints/reid/`
  - `checkpoints/trt_engines/`
- Mount or pre-seed caches explicitly when needed.

This does not change tracker behavior, but it makes Jetson builds and copies less fragile.

### P2: Make YOLO TensorRT Enablement Safer

Relevant file:

- `models/siamram/tracker.py`

The config key for YOLO compile appears as `copile_yolo`, and the YOLO engine export path is a simple replacement from `.pt` to `.engine`. Export also uses fixed device assumptions.

Recommended low-risk fixes:

- Support `compile_yolo` as an alias while retaining `copile_yolo` for backward compatibility.
- Make YOLO engine cache names include relevant build inputs such as image size, precision, device/JetPack/TensorRT version, and weights fingerprint.
- Keep YOLO TensorRT disabled by default until detector parity is validated on Orin.

This is mainly a reliability improvement. Actually enabling YOLO TensorRT can change detector numerics and should be treated as a benchmarked behavior change.

### P3: Add Central CPU Thread Controls

Relevant files:

- Runtime entrypoints and install docs

Jetson performance can suffer when OpenCV, PyTorch, NumPy, BLAS, and video decode compete for CPU threads. There does not appear to be a central Jetson runtime profile for thread counts.

Recommended fix:

- Document or set Jetson profile defaults for:
  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `NUMEXPR_NUM_THREADS`
  - `torch.set_num_threads(...)`
  - `cv2.setNumThreads(...)`
- Add a CLI/env knob rather than hard-coding a single value.

This is behavior-preserving if only thread scheduling changes.

### P3: Add A Fast Mode To `run_single_video.py`

Relevant file:

- `run_single_video.py`

`run_single_video.py` always enables `--output_video`, which is useful for qualitative review but poor for measuring tracker speed. Video writing, overlays, and visualization panels hide the real runtime cost.

Recommended fix:

- Add a `--no_output_video` or `--profile_fast` flag.
- Default can remain unchanged if the script is meant for visualization.
- Document that performance runs should use `output_video=False`.

## Things Not To Change First

These may improve speed, but they can change tracking behavior or accuracy. They should be benchmarked only after the low-risk items above.

- Do not switch from SiamABC model size `M` to `S` without accuracy evaluation.
- Do not disable YOLO, OSNet, detectability probing, or occlusion recovery as a first optimization.
- Do not increase `descriptor_stride` beyond the current value without reacquisition testing.
- Do not replace `homography_mode: classic` with a heavier accurate mode for Jetson runs.
- Do not use `run_single_video.py` with video output as a performance benchmark.
- Do not enable YOLO TensorRT solely for speed without checking detector parity.

## Suggested Orin Validation Loop

Run these on the target Jetson after the environment is installed and TensorRT caches are placed on a writable local disk:

```bash
python containers/test.py
```

Build or validate SiamABC TensorRT cache behavior:

```bash
uv run benchmark_siamabc_backbone.py --cache-dir checkpoints/trt_engines --max-frames 120
```

Run component profiling after adding production cache arguments to the profiler:

```bash
uv run profile_components.py --video_key dataset5/person19_3 --frames 300 --warmup_frames 30 --trt_cache_dir checkpoints/trt_engines --use_existing_trt_cache
```

Run a small split smoke test:

```bash
uv run run_inference.py --run_split public_lb --max_sequences 1 --use_existing_trt_cache
```

Run behavior checks before and after any optimization:

```bash
python -m tests.test_regression
pytest tests/test_config_schema.py tests/test_siamabc_trt_cache.py
```

If `uv` is not used on Jetson, run the same scripts through the active Python environment.

## Suggested Implementation Order

### Phase 0: Measurement And Safety

1. Make `profile_components.py` use the same TensorRT cache settings as production.
2. Add decode/read timing, RSS, and CUDA memory reporting.
3. Record a baseline on the Jetson with caches already built.
4. Run the existing regression harness and save outputs for comparison.

### Phase 1: Low-Risk Runtime Wins

1. Cache/inject the YOLO model while preserving fresh per-sequence tracker state.
2. Remove the per-frame `lam.item()` synchronization in TensorRT SiamABC dispatch.
3. Add a fast/no-video option to `run_single_video.py`.
4. Stream submission CSV rows instead of building large DataFrames.

### Phase 2: Hot-Path Cleanup

1. Keep more SiamABC postprocess logic on GPU and copy only final results.
2. Pass occlusion candidate descriptors into DRM/memory matching.
3. Add central CPU/OpenCV/PyTorch thread controls.

### Phase 3: Deployment Hardening

1. Add Jetson-specific install/container documentation.
2. Split runtime dependencies from dev/training/evaluation dependencies.
3. Tighten Docker ignore rules for nested caches.
4. Make YOLO TensorRT cache naming/fingerprinting explicit before enabling it.

## Quick Risk Matrix

| Change | Expected Speed/Memory Benefit | Output Parity Risk | Implementation Effort |
|---|---:|---:|---:|
| YOLO object cache/injection | High for multi-clip runs | Low | Medium |
| Remove `lam.item()` sync | Medium to High | Low | Low to Medium |
| Profiler cache/decode/RSS updates | High measurement value | None | Low |
| Stream submission CSV | Medium memory benefit | Low | Low |
| Candidate descriptor reuse | Medium in occlusion-heavy clips | Low to Medium | Medium |
| GPU-side SiamABC postprocess | Medium | Low to Medium | Medium |
| Jetson dependency split | High deployment value | None at algorithm level | Medium |
| Recursive Docker cache ignores | Medium build/copy value | None | Low |
| YOLO compile alias/cache naming | Low now, higher if enabled | Low for alias only | Low |
| Thread profile knobs | Low to Medium | Low | Low |

## Bottom Line

The repo is already pointed in a good direction for Jetson because the expensive visual models have TensorRT paths and the tracker has several bounded-compute settings. The most attractive next work is to remove avoidable overhead around the algorithms rather than changing the algorithms themselves.

The first implementation pass should target:

1. Shared YOLO model object with fresh sequence-local tracker state.
2. Removal of the TensorRT `lam.item()` per-frame sync.
3. Production-faithful Jetson profiling.
4. Streaming submission output.

Those changes are likely to improve startup, latency, and memory behavior on the Orin Nano while keeping the observable tracking behavior stable.
