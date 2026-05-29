# Camera Motion GMC Experiment

This isolated experiment ports Philo's masked KLT/RANSAC camera-motion estimator
and valid-only GMC search-prior adapter into SiamRAM.

It compares two tracker runs on the same sequence:

- `baseline`: the tracker searches from its normal previous prediction.
- `gmc_prior_valid_only`: the tracker first estimates camera/background motion, warps the previous target box by that motion, and uses the warped box as the next search prior only when the motion estimate passes the original validity gates.

Default target:

```text
E:\Coding projects\AIC-4-Hackathon\contest_release\dataset2\Animal4
```

## Why Camera Motion Helps

A Siamese tracker usually searches near the previous target box. That works when the camera is stable or the object moves smoothly. It fails when the camera itself moves quickly: the object may appear far away from its previous image location even if the object did not actually move much in the world.

Camera motion compensation tries to answer this question:

```text
How did the background move from frame t-1 to frame t?
```

If the background moved left by 30 pixels, then a stationary target will also appear shifted left by about 30 pixels. The tracker should therefore search around the camera-shifted target box, not the stale old box.

Simple view:

```text
Without GMC

Frame t-1 target box       Frame t after camera pan
       [target]                    camera moved
          |                             |
          v                             v
Tracker searches near old box --->   [old search area]     target is elsewhere
                                      miss / drift risk

With GMC

Frame t-1 target box       Estimate background motion       Frame t search prior
       [target]      --->  dx, dy, scale, rotation   --->   [warped target box]
                                                                  |
                                                                  v
                                                         tracker searches here
```

In other words, GMC does not replace the tracker. It gives the tracker a better starting point for where to look.

## Ported Motion Pipeline

The camera-motion estimator follows the original source logic:

```text
previous frame + current frame
        |
        v
mask out target area and previous dynamic residual mask
        |
        v
detect background corners on a grid with Shi-Tomasi
        |
        v
track corners with pyramidal KLT optical flow
        |
        v
filter KLT tracks with forward/backward error
        |
        v
fit transform with RANSAC
        |
        v
validate confidence gates
        |
        v
if valid: use transform; if invalid: fallback internally, but the A/B prior remains valid-only
        |
        v
warp previous bbox into current frame and set tracker search prior
```

The confidence gates check things like:

- enough KLT tracks survived
- enough RANSAC inliers exist
- inlier ratio is high enough
- inliers cover enough grid cells
- median residual is small enough
- translation, scale, and rotation are physically plausible

This is why the experiment is called `valid_only`: the tracker search prior is shifted only when the motion estimate is accepted by these gates.

## What The Overlay Shows

When `--output-video` is enabled, the GMC video includes:

- `GT`: ground-truth box from `annotation.txt`
- `PRED`: tracker prediction
- `GMC prior`: previous prediction warped by estimated camera motion
- green KLT dots/lines: RANSAC inlier background tracks used by the camera-motion estimate
- red KLT dots: rejected/outlier KLT tracks
- HUD values: `dx`, `dy`, scale, rotation, inliers, inlier ratio, residual, and whether GMC was used

The useful thing to watch is whether the yellow/orange `GMC prior` lands closer to the true object before the tracker updates. If it does, the tracker starts its visual search in a much better region.

## Motion Diagnostics Only

Use this when you want to inspect the camera-motion estimator without running the neural tracker:

```powershell
.\.venv\Scripts\python.exe experiments\camera_motion_gmc\camera_motion_experiment.py --max-frames 20
```

This writes a motion log and an annotated motion video under:

```text
outputs/experiments/camera_motion_gmc_smoke
```

or under the default camera-motion output directory if no output dir is supplied.

## Tracker A/B Test

Run the full A/B test:

```powershell
.\.venv\Scripts\python.exe experiments\camera_motion_gmc\tracker_gmc_prior_ab_test.py --no-trt 
```

Add annotated MP4 videos:

```powershell
.\.venv\Scripts\python.exe experiments\camera_motion_gmc\tracker_gmc_prior_ab_test.py --no-trt  --output-video
```

Quick smoke run:

```powershell
.\.venv\Scripts\python.exe experiments\camera_motion_gmc\tracker_gmc_prior_ab_test.py --tracker-kind siamabc --no-trt  --max-frames 2 --output-video
```

## GPU Notes

The neural tracker is required to run on CUDA by default. The script logs the active GPU, for example:

```text
Using CUDA device 0: NVIDIA GeForce RTX 3060 Laptop GPU
```

The OpenCV KLT/RANSAC motion stage is more subtle. The standard `opencv-python` wheel often exposes a `cv2.cuda` namespace but does not include CUDA-enabled optical flow or affine/homography RANSAC bindings. This environment currently reports no OpenCV CUDA device and no CUDA KLT/RANSAC functions.

For that reason, the A/B runner now refuses to run unless OpenCV GPU motion support is available. If you intentionally want the original CPU OpenCV motion path for debugging, pass:

```powershell

```

This preserves the exact original Philo logic. Rewriting RANSAC in CUDA/Torch would be a different implementation, not an exact port of the OpenCV RANSAC behavior.

## Outputs

Artifacts are written under:

```text
outputs/experiments/camera_motion_gmc/<sequence>_<timestamp>/
```

Important files:

```text
comparison_report.md
comparison_summary.json
baseline/Animal4_24.txt
gmc_prior_valid_only/Animal4_24.txt
baseline/tracking_metrics_by_frame.csv
gmc_prior_valid_only/tracking_metrics_by_frame.csv
gmc_prior_valid_only/gmc_prior_log.csv
baseline/*_annotated.mp4                  only with --output-video
gmc_prior_valid_only/*_annotated.mp4      only with --output-video
```

## How To Read The Metrics

The most important comparison fields are:

- `mean_iou`: average overlap between prediction and ground truth; higher is better
- `precision_20px`: fraction of frames where prediction center is within 20 pixels of GT center; higher is better
- `mean_center_error`: average center distance in pixels; lower is better
- `failure_count_iou_0`: frames with zero overlap; lower is better
- `gmc_prior_used_ratio`: fraction of frames where the valid camera-motion prior was applied
- `gmc_valid_ratio`: fraction of frames where the camera-motion estimate passed gates
- `mean_iou_delta_on_gmc_used_frames`: how much IoU changed on frames where GMC actually affected the search prior

Example interpretation:

```text
mean_iou:          0.12 -> 0.63
precision_20px:    0.16 -> 0.81
center error:       316px -> 14px
IoU failures:       141 -> 7
```

This means the camera-motion prior kept the tracker searching near the target after camera movement, instead of drifting around the old image location.

## Known Tradeoffs

GMC improves tracking when camera motion is a major source of apparent target displacement. It can regress when the camera-motion estimate is wrong, when background texture is weak, or when dynamic objects dominate the KLT tracks.

The added KLT/RANSAC stage also increases runtime. Check `gmc_mean_motion_runtime_ms` and `mean_runtime_ms` in the summary to decide whether the accuracy gain is worth the latency cost.
