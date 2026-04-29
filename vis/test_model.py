"""
Full-video inference and performance visualisation for SiamRAM.

This module provides the main entry point for running the tracker on video
files, rendering annotated outputs with side-by-side template and search
crops, and generating detailed latency reports.
"""
import gc
import os
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


def _confidence_color(score: float):
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


def _stamp_panel(panel: np.ndarray, label: str, updated: bool) -> None:
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


def _resize_into(src: np.ndarray, dst: np.ndarray) -> None:
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


def _draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
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
    for i, (color, text) in enumerate(entries):
        iy = y + i * 16
        cv2.rectangle(canvas, (x, iy - 9), (x + 12, iy + 3), color, -1)
        cv2.putText(canvas, text, (x + 16, iy), FONT, 0.35, (210, 210, 210), 1, cv2.LINE_AA)


def _refresh_panels(tracker, dyn_image, panel_template, panel_search,
                    frame_idx, updated):
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

    t_ctx = extend_bbox(dyn_bbox, image_width=iw, image_height=ih,
                        offset=cfg["template_bbox_offset"])
    t_crop, _, _ = get_extended_crop(image=dyn_image, bbox=dyn_bbox, context=t_ctx,
                                     crop_size=cfg["template_size"],
                                     padding_value=pad_val)
    _resize_into(t_crop, panel_template)
    _stamp_panel(panel_template, f"TEMPLATE  F:{frame_idx}", updated=updated)

    s_ctx = extend_bbox(dyn_bbox, image_width=iw, image_height=ih,
                        offset=cfg["search_context"])
    s_crop, _, _ = get_extended_crop(image=dyn_image, bbox=dyn_bbox, context=s_ctx,
                                     crop_size=cfg["instance_size"],
                                     padding_value=pad_val)
    _resize_into(s_crop, panel_search)
    _stamp_panel(panel_search, "SEARCH CTX", updated=updated)


def run_inference(
    initial_bbox: List[int] | NDArray[np.integer],
    video_path: str,
    tracker,
    output_path: str = "outputs/tracked_video.mp4",
    output_video: bool = True,
):
    """
    Inputs:
        initial_bbox - [x, y, w, h] ground-truth bounding box in the first frame
        video_path   - path to the input video file
        tracker      - EdgeDAMTracker instance (or a bare SiamABC tracker)
        output_path  - where to write the annotated output video;
                       also determines the bbox .txt file location
                       (ignored when output_video=False)
        output_video - if True, renders and saves the full annotated video with
                       side panels, overlays, and status pill; if False, only
                       runs tracking and writes the per-frame bbox .txt file.
                       Set to False when you need bbox outputs fast without the
                       overhead of video encoding and drawing.

    What it does:
        Initialises the tracker on the first frame, then loops over every
        subsequent frame calling tracker.update(). When output_video=True it
        also builds a composite canvas each frame (frame + two side panels),
        draws the predicted bbox, YOLO candidates, search ROI, status pill, HUD
        text, and legend, then encodes the frame into an AVI which is renamed to
        output_path at the end.

        Regardless of output_video, every predicted bbox is saved to a .txt file
        next to the video output (one bbox per line, space-separated x y w h).

        At the end prints a latency report broken down into normal tracking frames
        and occlusion recovery frames.

    Outputs:
        None. Side effects:
            - Writes tracked_bboxes to  <bbox_dir>/<video_name>.txt
            - If output_video=True, writes the annotated video to output_path
            - Prints a latency summary to stdout

    Why / where:
        Top-level entry point for evaluating or visualising the tracker on a
        video file. The output_video flag means the same function can be used
        both for quick evaluation runs (False — minimal overhead, fast) and for
        producing demo videos for inspection (True).
    """
    is_dam = hasattr(tracker, 'tracker')
    inner_tracker = tracker.tracker if is_dam else tracker

    initial_bbox = np.array(initial_bbox).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    head, tail = os.path.split(output_path)
    bbox_dir = os.path.join(head, "bboxes")
    os.makedirs(bbox_dir, exist_ok=True)
    bbox_file = os.path.join(bbox_dir, os.path.splitext(tail)[0] + ".txt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_bgr = cap.read()
    if not ret or first_bgr is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    h, w = first_bgr.shape[:2]

    if output_video:
        C_OCCLUDED = (0, 0, 220)
        C_YOLO_CANDIDATE = (0, 200, 255)
        C_YOLO_DISTRACTOR = (0, 0, 180)
        C_STATUS_OCC = (0, 0, 200)
        C_STATUS_OK = (0, 180, 0)
        C_STATUS_TEXT = (255, 255, 255)

        canvas_h = max(h, PANEL_W * 2 + GAP)
        total_w = w + PANEL_W
        y_off = (canvas_h - h) // 2

        canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)
        panel_template = np.zeros((PANEL_W, PANEL_W, 3), dtype=np.uint8)
        panel_search = np.zeros((PANEL_W, PANEL_W, 3), dtype=np.uint8)
        row_tmpl = np.s_[0: PANEL_W, w: total_w]
        row_search = np.s_[PANEL_W + GAP: PANEL_W * 2 + GAP, w: total_w]

        avi_path = os.path.splitext(output_path)[0] + "_tmp.avi"
        writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"),
                                 fps, (total_w, canvas_h))

        def _draw_status_pill(canvas: np.ndarray, in_occlusion: bool) -> None:
            """
            Inputs:
                canvas       - np.ndarray (H x W x 3) the full composite canvas
                in_occlusion - bool whether the tracker is currently in occlusion
                               recovery mode

            What it does:
                Draws a rounded-rectangle pill badge in the top-left corner of the
                frame area. The pill is red and reads "OCCLUDED" during occlusion
                recovery, green and reads "TRACKING" otherwise. Uses three draw
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
            label = "OCCLUDED" if in_occlusion else "TRACKING"
            color = C_STATUS_OCC if in_occlusion else C_STATUS_OK
            px = 8
            py = y_off + 8
            pw, ph = 140, 28
            r = ph // 2
            cv2.rectangle(canvas, (px + r, py), (px + pw - r, py + ph), color, -1)
            cv2.circle(canvas, (px + r, py + r), r, color, -1)
            cv2.circle(canvas, (px + pw - r, py + r), r, color, -1)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
            cv2.putText(canvas, label,
                        (px + (pw - tw) // 2, py + (ph + th) // 2 - 1),
                        FONT, 0.50, C_STATUS_TEXT, 1, cv2.LINE_AA)

        tracker.initialize(first_bgr, initial_bbox)
        _refresh_panels(inner_tracker, first_bgr, panel_template, panel_search,
                        frame_idx=0, updated=True)

        canvas.fill(0)
        canvas[y_off: y_off + h, :w] = first_bgr
        canvas[row_tmpl] = panel_template
        canvas[row_search] = panel_search
        bx, by, bw, bh = map(int, initial_bbox)
        cv2.rectangle(canvas, (bx, by + y_off), (bx + bw, by + bh + y_off), C_GT, 2)
        _draw_status_pill(canvas, in_occlusion=False)
        cv2.putText(canvas, "F:0  INIT", (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA)
        _draw_legend(canvas, w + 4, canvas_h - 90)
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
            ret, frame = cap.read()
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

            tracked_bboxes.append(bbox)

            if in_occlusion:
                times_occlusion.append(elapsed_ms)
            else:
                times_normal.append(elapsed_ms)

            if output_video:
                # cur_dyn_bbox = inner_tracker.running_dynamic_bbox
                # cur_dyn_obj = inner_tracker.running_dynamic_crop
                # template_updated = not np.array_equal(cur_dyn_bbox, last_dyn_bbox)

                # if template_updated:
                #     _refresh_panels(inner_tracker, cur_dyn_obj, panel_template,
                #                     panel_search, frame_idx=frame_idx, updated=True)
                #     last_dyn_bbox = cur_dyn_bbox.copy()
                # else:
                #     _stamp_panel(panel_template,
                #                  f"Dynamic TEMPLATE  F:{frame_idx - 1}", updated=False)
                #     _stamp_panel(panel_search, "SEARCH CTX", updated=False)

                canvas.fill(0)
                canvas[y_off: y_off + h, :w] = frame
                canvas[row_tmpl] = panel_template
                canvas[row_search] = panel_search

                if in_occlusion:
                    cv2.rectangle(canvas, (w, 0), (total_w - 1, canvas_h - 1), C_OCCLUDED, 3)
                # elif template_updated:
                #     cv2.rectangle(canvas, (w, 0), (total_w - 1, canvas_h - 1), C_UPDATE, 2)

                mapping = inner_tracker.tracking_state.mapping
                if mapping is not None:
                    mx, my, mw, mh = map(int, mapping)
                    cv2.rectangle(canvas, (mx, my + y_off), (mx + mw, my + mh + y_off),
                                  C_SEARCH, 1)

                if in_occlusion and yolo_dets:
                    held = tracker.held_box if is_dam else None
                    for det in yolo_dets:
                        dx, dy, dw, dh = map(int, det)
                        is_dist = held is not None and _iou(det, held) >= tracker.tau_occ
                        color = C_YOLO_DISTRACTOR if is_dist else C_YOLO_CANDIDATE
                        cv2.rectangle(canvas, (dx, dy + y_off), (dx + dw, dy + dh + y_off),
                                      color, 1)
                        cv2.putText(canvas, "D" if is_dist else "Y",
                                    (dx + 2, dy + y_off + 12), FONT, 0.38, color, 1, cv2.LINE_AA)

                bx, by, bw, bh = map(int, bbox)
                pred_color = C_OCCLUDED if in_occlusion else _confidence_color(score)
                cv2.rectangle(canvas, (bx, by + y_off), (bx + bw, by + bh + y_off),
                              pred_color, 2)
                cv2.putText(canvas, f"{score:.2f}", (bx, max(by + y_off - 4, 12)),
                            FONT, 0.45, pred_color, 1, cv2.LINE_AA)

                _draw_status_pill(canvas, in_occlusion=in_occlusion)

                if in_occlusion:
                    hud = f"F:{frame_idx}  [DAM RECOVERY]"
                # elif template_updated:
                #     hud = f"F:{frame_idx}  [TMPL UPDATE]"
                else:
                    hud = f"F:{frame_idx}"
                cv2.putText(canvas, hud, (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA)

                _draw_legend(canvas, w + 4, canvas_h - 90)

                if is_dam and in_occlusion:
                    rx, ry, rw, rh = tracker._get_yolo_search_roi(frame)
                    cv2.rectangle(canvas,
                                  (rx, ry + y_off),
                                  (rx + rw, ry + rh + y_off),
                                  (0, 165, 255), 1)
                    cv2.putText(canvas, "YOLO ROI",
                                (rx + 2, ry + y_off + 12),
                                FONT, 0.38, (0, 165, 255), 1, cv2.LINE_AA)

                writer.write(canvas)

            frame_idx += 1

    finally:
        cap.release()
        if output_video:
            writer.release()

    if output_video and avi_path != output_path:
        os.replace(avi_path, output_path)

    with open(bbox_file, "w", encoding="utf-8") as f:
        for bb in tracked_bboxes:
            f.write(f"{bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

    if output_video:
        del canvas, panel_template, panel_search

    def _stats(name, arr):
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
        print(f"{name:20s}  n={len(a):4d}  "
              f"mean={a.mean():.1f}ms  "
              f"med={np.median(a):.1f}ms  "
              f"p95={np.percentile(a, 95):.1f}ms  "
              f"p99={np.percentile(a, 99):.1f}ms  "
              f"min={a.min():.1f}ms  "
              f"max={a.max():.1f}ms  "
              f"fps={1000 / a.mean():.1f}")

    print("\n─── Latency Report ───────────────────────────────────────")
    _stats("ALL FRAMES", frame_times)
    _stats("NORMAL TRACK", times_normal)
    _stats("OCCLUSION", times_occlusion)
    if times_occlusion:
        print(f"  occlusion frames: {len(times_occlusion)} "
              f"({100 * len(times_occlusion) / len(frame_times):.1f}% of total)")
    print("──────────────────────────────────────────────────────────\n")

    gc.collect()
