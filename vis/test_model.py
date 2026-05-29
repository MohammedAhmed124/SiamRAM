"""
Full-video inference and performance visualisation for SiamRAM.

This module provides the main entry point for running the tracker on video
files, rendering annotated outputs with side-by-side template and search
crops, and generating detailed latency reports.
"""

import gc
import os
from collections import deque
from typing import List

import cv2
import numpy as np
from numpy._typing import NDArray

from utils.utils import _iou

PANEL_W = 240
GAP = 8
FPS_DEFAULT = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

C_GT = (255, 120, 0)
C_SEARCH = (0, 200, 255)
C_UPDATE = (0, 200, 80)
C_FRAME = (180, 180, 180)


def _load_sequence_frame_paths(
    sequence_dir: str,
) -> list[str]:
    """
    Return sorted image paths from a frame-directory sequence.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    frame_paths = [
        os.path.join(sequence_dir, name)
        for name in sorted(os.listdir(sequence_dir))
        if name.lower().endswith(exts)
    ]
    return frame_paths


def _confidence_color(
    score: float,
):
    """
    Inputs:
        score - float tracker confidence, expected in [0.0, 1.0]

    What it does:
        Maps a confidence score to a BGR colour that shifts from red (low
        confidence) through to green (high confidence). Clamps the input to
        [0, 1] before mapping so out-of-range values don't crash.

    Outputs:
        tuple (B, G, R) — OpenCV BGR colour

    Why / where:
        Called when drawing the predicted bounding box in run_inference. Gives
        an instant visual signal of how confident the tracker is without having
        to read the score number on the frame.
    """
    s = max(0.0, min(1.0, score))
    return (0, int(255 * s), int(255 * (1.0 - s)))


def _stamp_panel(
    panel: np.ndarray,
    label: str,
    updated: bool,
) -> None:
    """
    Inputs:
        panel   - np.ndarray (H x W x 3) the side panel image to stamp in place
        label   - str text to display in the header bar
        updated - bool True if the panel content was refreshed this frame

    What it does:
        Draws a dark header bar across the top 22 pixels of the panel, then
        renders the label text on it. The bar is dark green when updated and
        near-black when stale, giving a quick visual indicator of whether the
        template or search panel changed this frame.

    Outputs:
        None. Modifies panel in place.

    Why / where:
        Called from _refresh_panels and from the stale-panel branch in the
        run_inference main loop. Keeping the stamp logic here means both paths
        produce a consistent header without duplicating drawing code.
    """
    bar_color = (0, 70, 0) if updated else (30, 30, 30)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), bar_color, -1)
    cv2.putText(panel, label, (5, 15), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _resize_into(
    src: np.ndarray,
    dst: np.ndarray,
) -> None:
    """
    Inputs:
        src - np.ndarray source image, any size
        dst - np.ndarray destination array whose shape defines the target size

    What it does:
        Resizes src to exactly match dst's (width, height) and copies the result
        directly into dst. Works in place — no new array is allocated for the
        final result.

    Outputs:
        None. dst is overwritten with the resized content.

    Why / where:
        Used in _refresh_panels to fit the template and search crops into the
        fixed-size side panel buffers. Using copyto avoids an extra allocation
        on every frame, which matters at 30+ fps.
    """
    np.copyto(dst, cv2.resize(src, (dst.shape[1], dst.shape[0])))


def _refresh_recovery_panel(
    tracker,
    panel: np.ndarray,
) -> None:
    """
    Renders the last successful recovery (occlusion or distractor) into the
    panel: a resized crop of the recovered patch up top, then sim/iou/score
    stats and the thresholds that gated the decision. Reads
    `tracker._last_recovery_patch` and `tracker._last_recovery_info` which are
    populated by the tracker's `_record_recovery` helper.
    """
    panel.fill(0)
    patch = getattr(tracker, "_last_recovery_patch", None)
    info = getattr(tracker, "_last_recovery_info", None)
    if patch is None or info is None:
        _stamp_panel(panel, "RECOVERY  -", updated=False)
        return

    header_h = 22
    footer_h = 80
    img_h = panel.shape[0] - header_h - footer_h
    img_w = panel.shape[1]
    if img_h > 10 and patch.size > 0:
        img_panel = panel[header_h: header_h + img_h, :img_w]
        np.copyto(img_panel, cv2.resize(patch, (img_w, img_h)))

    mode = str(info.get("mode", "")).upper()
    _stamp_panel(panel, f"REC {mode}  F:{info.get('frame_idx', 0)}", updated=True)

    sim = float(info.get("sim", float("nan")))
    iou = float(info.get("iou_held", float("nan")))
    score = float(info.get("score", float("nan")))
    thr = info.get("thresholds", {})

    def _fmt(v: float) -> str:
        return "n/a" if not np.isfinite(v) else f"{v:.2f}"

    lines = [
        f"sim={_fmt(sim)}  iou={_fmt(iou)}",
        f"score={_fmt(score)}",
        f"sel>={float(thr.get('sel_min_sim', 0.0)):.2f}  "
        f"min>={float(thr.get('min_sim', 0.0)):.2f}",
        f"reacq>={float(thr.get('reacq', 0.0)):.2f}",
    ]
    y0 = panel.shape[0] - footer_h + 16
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (5, y0 + i * 16),
            FONT,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def _draw_legend(
    canvas: np.ndarray,
    x: int,
    y: int,
    include_distractor: bool = False,
) -> None:
    """
    Inputs:
        canvas - np.ndarray (H x W x 3) the full composite frame to draw onto
        x      - int left edge of the legend in pixels
        y      - int top edge of the legend in pixels

    What it does:
        Draws a small colour-coded legend in the side panel area showing what
        each bbox colour means — GT, predicted at various confidence levels,
        search region, and YOLO ROI.

    Outputs:
        None. Modifies canvas in place.

    Why / where:
        Called once per frame in run_inference after the main drawing is done.
        Positioned in the lower portion of the side panel so it doesn't overlap
        the template and search crop panels above it.
    """
    entries = [
        (C_GT, "GT bbox (init)"),
        ((0, 255, 0), "Pred  conf=1.0"),
        ((0, 128, 255), "Pred  conf=0.5"),
        ((0, 0, 255), "Pred  conf=0.0"),
        (C_SEARCH, "Search region"),
        ((0, 165, 255), "YOLO ROI"),
    ]
    if include_distractor:
        entries.insert(4, ((0, 0, 255), "Pred  distractor reject"))
    for i, (color, text) in enumerate(entries):
        iy = y + i * 16
        cv2.rectangle(canvas, (x, iy - 9), (x + 12, iy + 3), color, -1)
        cv2.putText(
            canvas, text, (x + 16, iy), FONT, 0.35, (210, 210, 210), 1, cv2.LINE_AA
        )


def _draw_velocity_curves(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    expected_hist: List[float],
    actual_hist: List[float],
    distractor_hist: List[bool] | None = None,
) -> None:
    """
    Draw overlapping spike-logic curves.

    expected_hist is the spike baseline norm history.
    actual_hist is the current camera-compensated step norm history.
    NaN values are allowed and are treated as gaps in the line.
    """
    if w < 40 or h < 30:
        return

    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(canvas.shape[1] - 1, x + w))
    y2 = int(min(canvas.shape[0] - 1, y + h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return

    pw = x2 - x1
    ph = y2 - y1

    C_BG = (22, 22, 22)
    C_GRID = (55, 55, 55)
    C_AXIS = (110, 110, 110)
    C_EXPECTED = (255, 190, 0)
    C_ACTUAL = (255, 120, 220)
    C_TEXT = (220, 220, 220)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), C_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C_AXIS, 1)

    n_grid = 4
    for i in range(1, n_grid):
        gy = y1 + int(i * ph / n_grid)
        cv2.line(canvas, (x1 + 1, gy), (x2 - 1, gy), C_GRID, 1)

    if distractor_hist:
        n_mask = min(len(distractor_hist), len(expected_hist), len(actual_hist))
        if n_mask >= 2:
            mask = list(distractor_hist)[-n_mask:]
            i = 0
            while i < n_mask:
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n_mask and mask[j + 1]:
                    j += 1
                t0 = float(i) / float(n_mask - 1)
                t1 = float(j) / float(n_mask - 1)
                sx = x1 + int(t0 * (pw - 1))
                ex = x1 + int(t1 * (pw - 1))
                cv2.rectangle(canvas, (sx, y1 + 1), (ex, y2 - 1), (0, 180, 210), -1)
                i = j + 1

    valid_vals = [
        float(v)
        for v in list(expected_hist) + list(actual_hist)
        if np.isfinite(v) and float(v) >= 0.0
    ]
    # Use full-range scaling so extreme velocity spikes remain visible
    # instead of being compressed by percentile capping.
    vmax = max(1.0, float(max(valid_vals)) * 1.08) if valid_vals else 1.0

    title = "Spike Curves (norm)"
    cv2.putText(
        canvas,
        title,
        (x1 + 6, y1 + 14),
        FONT,
        0.38,
        C_TEXT,
        1,
        cv2.LINE_AA,
    )

    def _plot_series(vals: List[float], color) -> None:
        n = len(vals)
        if n < 2:
            return
        prev_pt = None
        for i in range(n):
            v = float(vals[i])
            if not np.isfinite(v):
                prev_pt = None
                continue
            t = 0.0 if n == 1 else float(i) / float(n - 1)
            px = x1 + int(t * (pw - 1))
            py = y2 - 1 - int(np.clip(v / vmax, 0.0, 1.0) * (ph - 1))
            pt = (px, py)
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, pt, color, 2, cv2.LINE_AA)
            prev_pt = pt

    _plot_series(expected_hist, C_EXPECTED)
    _plot_series(actual_hist, C_ACTUAL)

    cv2.line(canvas, (x1 + 7, y2 - 14), (x1 + 22, y2 - 14), C_EXPECTED, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Baseline",
        (x1 + 26, y2 - 10),
        FONT,
        0.34,
        C_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (x1 + 94, y2 - 14), (x1 + 109, y2 - 14), C_ACTUAL, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Step",
        (x1 + 113, y2 - 10),
        FONT,
        0.34,
        C_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (x1 + 166, y2 - 18), (x1 + 179, y2 - 9), (0, 180, 210), -1)
    cv2.putText(
        canvas,
        "Distractor",
        (x1 + 183, y2 - 10),
        FONT,
        0.34,
        C_TEXT,
        1,
        cv2.LINE_AA,
    )

    exp_now = float(expected_hist[-1]) if expected_hist and np.isfinite(expected_hist[-1]) else np.nan
    act_now = float(actual_hist[-1]) if actual_hist and np.isfinite(actual_hist[-1]) else np.nan
    ratio_now = (
        float(act_now / (exp_now + 1e-6))
        if np.isfinite(exp_now) and np.isfinite(act_now)
        else np.nan
    )
    ratio_txt = f"{ratio_now:.2f}" if np.isfinite(ratio_now) else "n/a"
    cv2.putText(
        canvas,
        f"B:{exp_now:.2f}  S:{act_now:.2f}  R:{ratio_txt}  max:{vmax:.2f}",
        (x1 + 6, y1 + 30),
        FONT,
        0.34,
        C_TEXT,
        1,
        cv2.LINE_AA,
    )


def _refresh_panels(
    tracker,
    dyn_image,
    panel_template,
    panel_search,
    frame_idx,
    updated,
):
    """
    Inputs:
        tracker        - inner SiamABC tracker instance (not the EdgeDAMTracker wrapper)
        dyn_image      - np.ndarray (H x W x 3) the dynamic template image stored
                         inside the tracker
        panel_template - np.ndarray (PANEL_W x PANEL_W x 3) buffer to write template crop into
        panel_search   - np.ndarray (PANEL_W x PANEL_W x 3) buffer to write search crop into
        frame_idx      - int current frame number, shown in the template panel header
        updated        - bool whether the dynamic template changed this frame

    What it does:
        Extracts the template crop and search context crop from the tracker's
        internal dynamic image using its own bbox and config parameters, resizes
        both into the respective panel buffers, and stamps a header label on each.

    Outputs:
        None. panel_template and panel_search are updated in place.

    Why / where:
        Called at initialisation and then each frame in run_inference whenever
        the dynamic bbox changes. Keeping it out of run_inference makes the
        crop + resize + stamp sequence reusable and keeps the main loop readable.
    """
    from utils.utils import extend_bbox, get_extended_crop

    dyn_bbox = tracker.running_dynamic_bbox
    cfg = tracker.tracking_config
    ih, iw = dyn_image.shape[:2]
    pad_val = np.mean(dyn_image, axis=(0, 1))

    t_ctx = extend_bbox(
        dyn_bbox, image_width=iw, image_height=ih, offset=cfg["template_bbox_offset"]
    )
    t_crop, _, _ = get_extended_crop(
        image=dyn_image,
        bbox=dyn_bbox,
        context=t_ctx,
        crop_size=cfg["template_size"],
        padding_value=pad_val,
    )
    _resize_into(t_crop, panel_template)
    _stamp_panel(panel_template, f"TEMPLATE  F:{frame_idx}", updated=updated)

    s_ctx = extend_bbox(
        dyn_bbox, image_width=iw, image_height=ih, offset=cfg["search_context"]
    )
    s_crop, _, _ = get_extended_crop(
        image=dyn_image,
        bbox=dyn_bbox,
        context=s_ctx,
        crop_size=cfg["instance_size"],
        padding_value=pad_val,
    )
    _resize_into(s_crop, panel_search)
    _stamp_panel(panel_search, "SEARCH CTX", updated=updated)


def run_inference(
    initial_bbox: List[int] | NDArray[np.integer],
    video_path: str,
    tracker,
    output_path: str = "outputs/tracked_video.mp4",
    output_video: bool = False,
):
    is_dam = hasattr(tracker, "tracker")
    inner_tracker = tracker.tracker if is_dam else tracker

    initial_bbox = np.array(initial_bbox).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    head, tail = os.path.split(output_path)
    bbox_dir = os.path.join(head, "bboxes")
    os.makedirs(bbox_dir, exist_ok=True)
    bbox_file = os.path.join(bbox_dir, os.path.splitext(tail)[0] + ".txt")

    cap = None
    frame_paths: list[str] = []
    frame_ptr = 0
    if os.path.isdir(video_path):
        frame_paths = _load_sequence_frame_paths(video_path)
        if not frame_paths:
            raise RuntimeError(f"No image frames found in directory: {video_path}")
        first_bgr = cv2.imread(frame_paths[0])
        if first_bgr is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        frame_ptr = 1
        fps = FPS_DEFAULT
        total_frames = len(frame_paths)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_bgr = cap.read()
        if not ret or first_bgr is None:
            cap.release()
            raise RuntimeError(f"Cannot read first frame: {video_path}")

    def _read_next_frame():
        nonlocal frame_ptr
        if cap is not None:
            return cap.read()
        if frame_ptr >= len(frame_paths):
            return False, None
        frame = cv2.imread(frame_paths[frame_ptr])
        frame_ptr += 1
        if frame is None:
            return False, None
        return True, frame

    def _release_source():
        if cap is not None:
            cap.release()

    h, w = first_bgr.shape[:2]

    if not output_video:
        import time

        tracker.initialize(first_bgr, initial_bbox)

        if total_frames > 0:
            tracked_bboxes = np.empty((total_frames, 4), dtype=np.int32)
            tracked_bboxes[0] = initial_bbox
            bbox_idx = 1
            use_preallocated = True
        else:
            tracked_bboxes = [initial_bbox]
            use_preallocated = False

        _update = tracker.update

        frame_times: list[float] = []
        frame_idx = 1

        try:
            while True:
                ret, frame = _read_next_frame()
                if not ret or frame is None:
                    break

                t0 = time.perf_counter()
                result = _update(frame)
                frame_times.append((time.perf_counter() - t0) * 1000.0)

                bbox = result[0]

                if use_preallocated:
                    if bbox_idx < total_frames:
                        tracked_bboxes[bbox_idx] = bbox
                    else:

                        tracked_bboxes = list(tracked_bboxes[:bbox_idx])
                        tracked_bboxes.append(bbox)
                        use_preallocated = False
                else:
                    tracked_bboxes.append(bbox)

                bbox_idx += 1
                frame_idx += 1

        finally:
            _release_source()

        if use_preallocated:
            tracked_bboxes = tracked_bboxes[:bbox_idx]

        np.savetxt(bbox_file, tracked_bboxes, fmt="%d", delimiter=" ")

        def _stats_fast(
            name,
            arr,
        ):
            if not arr:
                print(f"{name:20s}  no data")
                return
            a = np.array(arr)
            print(
                f"{name:20s}  n={len(a):4d}  "
                f"mean={a.mean():.1f}ms  "
                f"med={np.median(a):.1f}ms  "
                f"p95={np.percentile(a, 95):.1f}ms  "
                f"p99={np.percentile(a, 99):.1f}ms  "
                f"min={a.min():.1f}ms  "
                f"max={a.max():.1f}ms  "
                f"fps={1000 / a.mean():.1f}"
            )

        print("\n─── Latency Report (fast inference) ──────────────────────")
        _stats_fast("ALL FRAMES", frame_times)
        print("──────────────────────────────────────────────────────────\n")

        gc.collect()
        return

    if output_video:
        C_OCCLUDED = (0, 0, 220)
        C_DISTRACTOR = (0, 0, 255)
        C_REAL_HYP = (0, 255, 0)
        C_YOLO_CANDIDATE = (0, 200, 255)
        C_YOLO_DISTRACTOR = (0, 0, 180)
        C_STATUS_OCC = (0, 0, 200)
        C_STATUS_OK = (0, 180, 0)
        C_STATUS_DIST = (0, 0, 255)
        C_STATUS_TEXT = (255, 255, 255)
        show_velocity_overlay = bool(
            is_dam
            and (
                hasattr(tracker, "_spike_reject_enabled")
                or hasattr(tracker, "_jump_reject_enabled")
            )
        )
        vel_strip_h = 190
        vel_strip_gap = 8
        expected_speed_hist = deque(maxlen=320)
        actual_speed_hist = deque(maxlen=320)
        distractor_mode_hist = deque(maxlen=320)

        top_h = max(h, PANEL_W * 3 + GAP * 2)
        extra_h = (vel_strip_gap + vel_strip_h) if show_velocity_overlay else 0
        canvas_h = top_h + extra_h
        total_w = w + PANEL_W
        y_off = (top_h - h) // 2

        canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)
        panel_template = np.zeros((PANEL_W, PANEL_W, 3), dtype=np.uint8)
        panel_search = np.zeros((PANEL_W, PANEL_W, 3), dtype=np.uint8)
        panel_recovery = np.zeros((PANEL_W, PANEL_W, 3), dtype=np.uint8)
        row_tmpl = np.s_[0:PANEL_W, w:total_w]
        row_search = np.s_[PANEL_W + GAP: PANEL_W * 2 + GAP, w:total_w]
        row_recovery = np.s_[
            PANEL_W * 2 + GAP * 2: PANEL_W * 3 + GAP * 2,
            w: total_w,
        ]
        vel_plot_x = 8
        vel_plot_y = top_h + vel_strip_gap
        vel_plot_w = max(40, total_w - 16)
        vel_plot_h = max(30, vel_strip_h - 2)

        avi_path = os.path.splitext(output_path)[0] + "_tmp.avi"
        writer = cv2.VideoWriter(
            avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (total_w, canvas_h)
        )

        def _overlay_bbox(
            bb,
            scale: float,
        ) -> tuple[int, int, int, int]:
            arr = np.asarray(bb, dtype=float).reshape(-1)
            if arr.size < 4:
                return 0, 0, 0, 0
            x = int(round(float(arr[0]) * scale))
            y = int(round(float(arr[1]) * scale))
            bw = int(round(float(arr[2]) * scale))
            bh = int(round(float(arr[3]) * scale))
            return x, y, bw, bh

        def _draw_status_pill(
            canvas: np.ndarray,
            mode: str,
        ) -> None:
            """
            Inputs:
                canvas       - np.ndarray (H x W x 3) the full composite canvas
                mode - str status mode: "tracking", "occluded", or "distractor"

            What it does:
                Draws a rounded-rectangle pill badge in the top-left corner of the
                frame area. The pill is red for occlusion recovery, yellow for
                distractor rejection, and green for normal tracking. Uses three draw
                calls — a rectangle for the body and two circles for the caps — to
                approximate rounded corners without requiring a custom shape routine.

            Outputs:
                None. Modifies canvas in place.

            Why / where:
                Called once per frame in the main loop after the bbox is drawn so
                the pill always sits on top. Defined as a closure inside run_inference
                so it captures y_off and the colour constants without needing them
                passed as arguments every call.
            """
            m = str(mode).strip().lower()
            if m == "occluded":
                label = "OCCLUDED"
                color = C_STATUS_OCC
            elif m == "distractor":
                label = "DISTRACTOR"
                color = C_STATUS_DIST
            else:
                label = "TRACKING"
                color = C_STATUS_OK
            px = 8
            py = y_off + 8
            ph = 28
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
            pw = max(140, tw + 28)
            r = ph // 2
            cv2.rectangle(canvas, (px + r, py), (px + pw - r, py + ph), color, -1)
            cv2.circle(canvas, (px + r, py + r), r, color, -1)
            cv2.circle(canvas, (px + pw - r, py + r), r, color, -1)
            cv2.putText(
                canvas,
                label,
                (px + (pw - tw) // 2, py + (ph + th) // 2 - 1),
                FONT,
                0.50,
                C_STATUS_TEXT,
                1,
                cv2.LINE_AA,
            )

        tracker.initialize(first_bgr, initial_bbox)
        _refresh_panels(
            inner_tracker,
            first_bgr,
            panel_template,
            panel_search,
            frame_idx=0,
            updated=True,
        )

        canvas.fill(0)
        canvas[y_off: y_off + h, :w] = first_bgr
        canvas[row_tmpl] = panel_template
        canvas[row_search] = panel_search
        _refresh_recovery_panel(tracker, panel_recovery)
        canvas[row_recovery] = panel_recovery
        bx, by, bw, bh = map(int, initial_bbox)
        cv2.rectangle(canvas, (bx, by + y_off), (bx + bw, by + bh + y_off), C_GT, 2)
        _draw_status_pill(canvas, mode="tracking")
        cv2.putText(
            canvas, "F:0  INIT", (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA
        )
        if not show_velocity_overlay:
            _draw_legend(canvas, w + 4, canvas_h - 90)
        if show_velocity_overlay:
            expected_speed_hist.append(np.nan)
            actual_speed_hist.append(np.nan)
            distractor_mode_hist.append(False)
            cv2.line(
                canvas,
                (0, top_h + vel_strip_gap // 2),
                (total_w - 1, top_h + vel_strip_gap // 2),
                (90, 90, 90),
                1,
            )
            _draw_velocity_curves(
                canvas,
                x=vel_plot_x,
                y=vel_plot_y,
                w=vel_plot_w,
                h=vel_plot_h,
                expected_hist=list(expected_speed_hist),
                actual_hist=list(actual_speed_hist),
                distractor_hist=list(distractor_mode_hist),
            )
        writer.write(canvas)

        last_dyn_bbox = inner_tracker.running_dynamic_bbox.copy()

    else:
        tracker.initialize(first_bgr, initial_bbox)

    tracked_bboxes = [initial_bbox]

    import time

    times_normal = []
    times_occlusion = []
    frame_times = []

    try:
        frame_idx = 1

        while True:
            ret, frame = _read_next_frame()
            if not ret or frame is None:
                break

            t0 = time.perf_counter()
            result = tracker.update(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            frame_times.append(elapsed_ms)

            bbox = result[0]
            score = result[1]
            in_occlusion = result[2] if is_dam else False
            yolo_dets = result[3] if (is_dam and len(result) > 3) else []
            visual_mode = "occluded" if in_occlusion else "tracking"
            visual_reason = ""
            visual_details = ""
            if is_dam and hasattr(tracker, "visual_mode"):
                raw_mode = str(getattr(tracker, "visual_mode", "")).strip().lower()
                if raw_mode == "distractor":
                    visual_mode = "distractor"
                elif not in_occlusion and raw_mode == "tracking":
                    visual_mode = raw_mode
                visual_reason = str(getattr(tracker, "visual_reason", "")).strip()
                visual_details = str(getattr(tracker, "visual_details", "")).strip()

            tracked_bboxes.append(bbox)

            if in_occlusion:
                times_occlusion.append(elapsed_ms)
            else:
                times_normal.append(elapsed_ms)

            if output_video:
                if show_velocity_overlay:
                    expected_speed = float(
                        getattr(tracker, "_spike_debug_baseline_norm", np.nan)
                    )
                    actual_speed = float(
                        getattr(tracker, "_spike_debug_speed_norm", np.nan)
                    )

                    expected_speed_hist.append(expected_speed)
                    actual_speed_hist.append(actual_speed)
                    distractor_mode_hist.append(visual_mode == "distractor")

                cur_dyn_bbox = inner_tracker.running_dynamic_bbox
                cur_dyn_obj = inner_tracker.running_dynamic_image
                template_updated = not np.array_equal(cur_dyn_bbox, last_dyn_bbox)

                if template_updated:
                    _refresh_panels(
                        inner_tracker,
                        cur_dyn_obj,
                        panel_template,
                        panel_search,
                        frame_idx=frame_idx,
                        updated=True,
                    )
                    last_dyn_bbox = cur_dyn_bbox.copy()
                else:
                    _stamp_panel(
                        panel_template,
                        f"Dynamic TEMPLATE  F:{frame_idx - 1}",
                        updated=False,
                    )
                    _stamp_panel(panel_search, "SEARCH CTX", updated=False)

                canvas.fill(0)
                canvas[y_off: y_off + h, :w] = frame
                canvas[row_tmpl] = panel_template
                canvas[row_search] = panel_search
                _refresh_recovery_panel(tracker, panel_recovery)
                canvas[row_recovery] = panel_recovery

                frame_scale = float(getattr(tracker, "_frame_scale", 1.0))
                if not np.isfinite(frame_scale) or frame_scale <= 1e-8:
                    overlay_scale = 1.0
                else:
                    overlay_scale = 1.0 / frame_scale

                if in_occlusion:
                    cv2.rectangle(
                        canvas, (w, 0), (total_w - 1, canvas_h - 1), C_OCCLUDED, 3
                    )
                elif template_updated:
                    cv2.rectangle(
                        canvas, (w, 0), (total_w - 1, canvas_h - 1), C_UPDATE, 2
                    )

                mapping = inner_tracker.tracking_state.mapping
                if mapping is not None:
                    mx, my, mw, mh = _overlay_bbox(mapping[:4], overlay_scale)

                    cv2.rectangle(
                        canvas,
                        (mx, my + y_off),
                        (mx + mw, my + mh + y_off),
                        C_SEARCH,
                        1,
                    )

                if in_occlusion and yolo_dets:
                    held = tracker.held_box if is_dam else None
                    for det in yolo_dets:
                        dx, dy, dw, dh = map(int, det)
                        is_dist = (
                            held is not None and _iou(det, held) >= tracker.tau_occ
                        )
                        color = C_YOLO_DISTRACTOR if is_dist else C_YOLO_CANDIDATE
                        cv2.rectangle(
                            canvas,
                            (dx, dy + y_off),
                            (dx + dw, dy + dh + y_off),
                            color,
                            1,
                        )
                        cv2.putText(
                            canvas,
                            "D" if is_dist else "Y",
                            (dx + 2, dy + y_off + 12),
                            FONT,
                            0.38,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                elif visual_mode == "distractor" and is_dam:
                    real_hyps = list(getattr(tracker, "_distractor_mode_visual_reals", []))
                    dist_hyps = list(
                        getattr(tracker, "_distractor_mode_visual_distractors", [])
                    )
                    for bb in real_hyps:
                        rx, ry, rw_, rh_ = _overlay_bbox(bb, overlay_scale)
                        cv2.rectangle(
                            canvas,
                            (rx, ry + y_off),
                            (rx + rw_, ry + rh_ + y_off),
                            C_REAL_HYP,
                            2,
                        )
                        cv2.putText(
                            canvas,
                            "REAL",
                            (rx + 2, max(12, ry + y_off - 4)),
                            FONT,
                            0.40,
                            C_REAL_HYP,
                            1,
                            cv2.LINE_AA,
                        )
                    for bb in dist_hyps:
                        dx, dy, dw_, dh_ = _overlay_bbox(bb, overlay_scale)
                        cv2.rectangle(
                            canvas,
                            (dx, dy + y_off),
                            (dx + dw_, dy + dh_ + y_off),
                            C_DISTRACTOR,
                            2,
                        )
                        cv2.putText(
                            canvas,
                            "D",
                            (dx + 2, max(12, dy + y_off - 4)),
                            FONT,
                            0.40,
                            C_DISTRACTOR,
                            1,
                            cv2.LINE_AA,
                        )

                bx, by, bw, bh = map(int, bbox)
                if visual_mode == "occluded":
                    pred_color = C_OCCLUDED
                elif visual_mode == "distractor":
                    pred_color = C_REAL_HYP
                else:
                    pred_color = _confidence_color(score)
                cv2.rectangle(
                    canvas, (bx, by + y_off), (bx + bw, by + bh + y_off), pred_color, 2
                )
                cv2.putText(
                    canvas,
                    f"{score:.2f}",
                    (bx, max(by + y_off - 4, 12)),
                    FONT,
                    0.45,
                    pred_color,
                    1,
                    cv2.LINE_AA,
                )

                _draw_status_pill(canvas, mode=visual_mode)

                if visual_mode == "occluded":
                    hud = f"F:{frame_idx}  [DAM RECOVERY]"
                elif visual_mode == "distractor":
                    hud = f"F:{frame_idx}  [DISTRACTOR REJECT]"
                elif template_updated:
                    hud = f"F:{frame_idx}  [TMPL UPDATE]"
                else:
                    hud = f"F:{frame_idx}"
                cv2.putText(
                    canvas, hud, (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA
                )
                if visual_mode == "distractor":
                    if visual_reason:
                        cv2.putText(
                            canvas,
                            visual_reason,
                            (156, y_off + 40),
                            FONT,
                            0.42,
                            C_DISTRACTOR,
                            1,
                            cv2.LINE_AA,
                        )
                    if visual_details:
                        cv2.putText(
                            canvas,
                            visual_details,
                            (156, y_off + 58),
                            FONT,
                            0.36,
                            C_DISTRACTOR,
                            1,
                            cv2.LINE_AA,
                        )

                if not show_velocity_overlay:
                    _draw_legend(
                        canvas,
                        w + 4,
                        canvas_h - 90,
                        include_distractor=(visual_mode == "distractor"),
                    )

                if is_dam and in_occlusion:
                    proc_frame = (
                        tracker._prescale_frame(frame)
                        if hasattr(tracker, "_prescale_frame")
                        else frame
                    )
                    rx, ry, rw, rh = tracker._get_yolo_search_roi(proc_frame)
                    rx, ry, rw, rh = _overlay_bbox((rx, ry, rw, rh), overlay_scale)
                    cv2.rectangle(
                        canvas,
                        (rx, ry + y_off),
                        (rx + rw, ry + rh + y_off),
                        (0, 165, 255),
                        1,
                    )
                    cv2.putText(
                        canvas,
                        "YOLO ROI",
                        (rx + 2, ry + y_off + 12),
                        FONT,
                        0.38,
                        (0, 165, 255),
                        1,
                        cv2.LINE_AA,
                    )
                elif is_dam and visual_mode == "distractor":
                    d_roi = getattr(tracker, "_distractor_mode_roi", None)
                    if d_roi is not None and len(d_roi) >= 4:
                        rx, ry, rw, rh = _overlay_bbox(d_roi[:4], overlay_scale)
                        cv2.rectangle(
                            canvas,
                            (rx, ry + y_off),
                            (rx + rw, ry + rh + y_off),
                            (0, 165, 255),
                            1,
                        )
                        cv2.putText(
                            canvas,
                            "DIST ROI",
                            (rx + 2, ry + y_off + 12),
                            FONT,
                            0.38,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )

                if show_velocity_overlay:
                    cv2.line(
                        canvas,
                        (0, top_h + vel_strip_gap // 2),
                        (total_w - 1, top_h + vel_strip_gap // 2),
                        (90, 90, 90),
                        1,
                    )
                    _draw_velocity_curves(
                        canvas,
                        x=vel_plot_x,
                        y=vel_plot_y,
                        w=vel_plot_w,
                        h=vel_plot_h,
                        expected_hist=list(expected_speed_hist),
                        actual_hist=list(actual_speed_hist),
                        distractor_hist=list(distractor_mode_hist),
                    )

                writer.write(canvas)

            frame_idx += 1

    finally:
        _release_source()
        if output_video:
            writer.release()

    if output_video and avi_path != output_path:
        os.replace(avi_path, output_path)

    with open(bbox_file, "w", encoding="utf-8") as f:
        for bb in tracked_bboxes:
            f.write(f"{bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

    if output_video:
        del canvas, panel_template, panel_search

    def _stats(
        name,
        arr,
    ):
        """
        Inputs:
            name - str label for this timing group, printed left-aligned
            arr  - list of float millisecond timings for this group

        What it does:
            Computes and prints mean, median, p95, p99, min, max, and effective
            FPS for the given timing array. Prints a single "no data" line if
            the list is empty.

        Outputs:
            None. Prints to stdout.

        Why / where:
            Called three times after the main loop to report latency for all
            frames, normal tracking frames, and occlusion frames separately.
            Defined here so it shares the numpy import without needing it passed in.
        """
        if not arr:
            print(f"{name:20s}  no data")
            return
        a = np.array(arr)
        print(
            f"{name:20s}  n={len(a):4d}  "
            f"mean={a.mean():.1f}ms  "
            f"med={np.median(a):.1f}ms  "
            f"p95={np.percentile(a, 95):.1f}ms  "
            f"p99={np.percentile(a, 99):.1f}ms  "
            f"min={a.min():.1f}ms  "
            f"max={a.max():.1f}ms  "
            f"fps={1000 / a.mean():.1f}"
        )

    print("\n─── Latency Report ───────────────────────────────────────")
    _stats("ALL FRAMES", frame_times)
    _stats("NORMAL TRACK", times_normal)
    _stats("OCCLUSION", times_occlusion)
    if times_occlusion:
        print(
            f"  occlusion frames: {len(times_occlusion)} "
            f"({100 * len(times_occlusion) / len(frame_times):.1f}% of total)"
        )
    print("──────────────────────────────────────────────────────────\n")

    gc.collect()
