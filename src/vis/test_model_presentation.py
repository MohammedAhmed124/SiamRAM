"""
Presentation-oriented full-video inference visualisation for SiamRAM.

This module provides the main entry point for running the tracker on video
files, rendering annotated outputs with side-by-side template and search
crops, and generating detailed latency reports.
"""

import gc
import csv
import os
from collections import deque
from typing import List

import cv2
import numpy as np
from numpy._typing import NDArray

from utils.console import SiamRAMProgressLine, siamram_latency
from utils.utils import _iou


def _latency_stats(arr):
    """Summarize a list of per-frame millisecond timings for siamram_latency.

    Returns ``None`` for an empty list (rendered as "no data"), otherwise a
    dict with the keys siamram_latency expects: n, mean, med, p95, p99, min,
    max, fps.
    """
    if not arr:
        return None
    a = np.array(arr)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "med": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "fps": 1000.0 / float(a.mean()),
    }

PANEL_W = 240
GAP = 8
FPS_DEFAULT = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

C_GT = (255, 120, 0)
C_UPDATE = (0, 200, 80)
C_FRAME = (180, 180, 180)
C_YOLO_CANDIDATE = (0, 255, 255)
C_OBJECT_MOTION = (255, 80, 255)
C_CAMERA_MOTION = (255, 255, 0)
C_TRACKER_SEARCH = (255, 190, 0)
C_TEXT_SHADOW = (0, 0, 0)

PRESENTATION_PRED_THICKNESS = 4
PRESENTATION_YOLO_THICKNESS = 4
PRESENTATION_ROI_THICKNESS = 3


def _draw_box_with_outline(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    outline: tuple[int, int, int] = C_TEXT_SHADOW,
) -> None:
    cv2.rectangle(canvas, pt1, pt2, outline, max(thickness + 3, 1), cv2.LINE_AA)
    cv2.rectangle(canvas, pt1, pt2, color, max(thickness, 1), cv2.LINE_AA)


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y_top: int,
    color: tuple[int, int, int],
    scale: float = 0.46,
) -> None:
    if not text:
        return
    pad_x = 5
    pad_y = 4
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    box_w = tw + pad_x * 2
    box_h = th + base + pad_y * 2
    x = int(np.clip(x, 0, max(0, canvas.shape[1] - box_w - 1)))
    y_top = int(np.clip(y_top, 0, max(0, canvas.shape[0] - box_h - 1)))
    cv2.rectangle(canvas, (x, y_top), (x + box_w, y_top + box_h), (15, 15, 15), -1)
    cv2.rectangle(canvas, (x, y_top), (x + box_w, y_top + box_h), color, 1)
    cv2.putText(
        canvas,
        text,
        (x + pad_x, y_top + pad_y + th),
        FONT,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _clamp_point_to_frame(
    pt: np.ndarray,
    frame_rect: tuple[int, int, int, int],
    margin: int = 6,
) -> tuple[int, int]:
    x0, y0, x1, y1 = frame_rect
    return (
        int(np.clip(pt[0], x0 + margin, x1 - margin)),
        int(np.clip(pt[1], y0 + margin, y1 - margin)),
    )


def _draw_roi_edge_motion_arrow(
    canvas: np.ndarray,
    roi_rect: tuple[int, int, int, int],
    vector,
    color: tuple[int, int, int],
    label: str,
    frame_rect: tuple[int, int, int, int],
    side_offset: float,
    min_len: float = 50.0,
    max_len: float = 150.0,
    vector_scale: float = 6.0,
) -> None:
    if vector is None:
        return
    try:
        vec = np.asarray(vector, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return
    if vec.size < 2 or not np.all(np.isfinite(vec[:2])):
        return
    mag = float(np.hypot(vec[0], vec[1]))
    if mag < 0.2:
        return

    x1, y1, x2, y2 = roi_rect
    if x2 <= x1 or y2 <= y1:
        return
    unit = vec[:2] / mag
    perp = np.array([-unit[1], unit[0]], dtype=float)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    start_inside = center + perp * float(side_offset)
    start_inside[0] = np.clip(start_inside[0], x1 + 4, x2 - 4)
    start_inside[1] = np.clip(start_inside[1], y1 + 4, y2 - 4)

    tx = float("inf")
    if unit[0] > 1e-6:
        tx = (x2 - start_inside[0]) / unit[0]
    elif unit[0] < -1e-6:
        tx = (x1 - start_inside[0]) / unit[0]

    ty = float("inf")
    if unit[1] > 1e-6:
        ty = (y2 - start_inside[1]) / unit[1]
    elif unit[1] < -1e-6:
        ty = (y1 - start_inside[1]) / unit[1]

    edge_dist = min(v for v in (tx, ty) if v >= 0.0)
    start_pt = start_inside + unit * edge_dist
    length = float(np.clip(mag * vector_scale, min_len, max_len))
    end_pt = start_pt + unit * length

    sp = _clamp_point_to_frame(start_pt, frame_rect)
    ep = _clamp_point_to_frame(end_pt, frame_rect)
    cv2.arrowedLine(canvas, sp, ep, C_TEXT_SHADOW, 8, cv2.LINE_AA, tipLength=0.30)
    cv2.arrowedLine(canvas, sp, ep, color, 4, cv2.LINE_AA, tipLength=0.30)
    _draw_label(canvas, label, ep[0] + 8, ep[1] - 16, color, scale=0.44)


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


def _tracker_visual_mode(tracker, result, is_dam: bool) -> str:
    if not is_dam:
        return "tracking"
    in_occlusion = bool(result[2]) if len(result) > 2 else False
    if in_occlusion:
        return "occluded"
    return "tracking"


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


def _stamp_template_stats(
    panel: np.ndarray,
    trk: float,
    drm: float,
    sharp: float,
) -> None:
    h = panel.shape[0]
    strip_h = 36

    def _fmt(v: float) -> str:
        return "--" if not np.isfinite(v) else f"{v:.2f}"

    def _fmt_sharp(v: float) -> str:
        return "--" if not np.isfinite(v) else f"{v:.0f}"

    cv2.rectangle(panel, (0, h - strip_h), (panel.shape[1], h), (20, 20, 20), -1)
    cv2.putText(
        panel,
        f"trk={_fmt(trk)}  drm={_fmt(drm)}",
        (5, h - 22),
        FONT,
        0.40,
        (200, 230, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"sharp={_fmt_sharp(sharp)}",
        (5, h - 6),
        FONT,
        0.40,
        (200, 220, 230),
        1,
        cv2.LINE_AA,
    )


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
    drm = float(info.get("drm_score", float("nan")))
    thr = info.get("thresholds", {})

    def _fmt(v: float) -> str:
        return "n/a" if not np.isfinite(v) else f"{v:.2f}"

    # trk = SiamABC verification score (gated by reacq>=); drm = DRM composite
    # ranking score of the chosen candidate (occlusion only; phase-0 gate app>=).
    lines = [
        f"sim={_fmt(sim)}  iou={_fmt(iou)}",
        f"trk={_fmt(score)}  drm={_fmt(drm)}",
        f"reacq>={float(thr.get('reacq', 0.0)):.2f}  "
        f"app>={float(thr.get('app_match', 0.0)):.2f}",
        f"sel>={float(thr.get('sel_min_sim', 0.0)):.2f}  "
        f"min>={float(thr.get('min_sim', 0.0)):.2f}",
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
) -> None:
    """
    Inputs:
        canvas - np.ndarray (H x W x 3) the full composite frame to draw onto
        x      - int left edge of the legend in pixels
        y      - int top edge of the legend in pixels

    What it does:
        Draws a small colour-coded legend in the side panel area showing what
        each overlay colour means — GT, predicted boxes, motion arrows,
        search ROI, and YOLO candidates.

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
        (C_OBJECT_MOTION, "Object motion"),
        (C_CAMERA_MOTION, "Camera motion"),
        ((0, 165, 255), "Search ROI"),
        (C_YOLO_CANDIDATE, "YOLO candidate"),
    ]
    for i, (color, text) in enumerate(entries):
        iy = y + i * 16
        cv2.rectangle(canvas, (x, iy - 9), (x + 12, iy + 3), color, -1)
        cv2.putText(
            canvas, text, (x + 16, iy), FONT, 0.35, (210, 210, 210), 1, cv2.LINE_AA
        )


def _draw_image_panel(
    canvas: np.ndarray,
    x: int,
    y: int,
    size: int,
    image: np.ndarray | None,
    label: str,
) -> None:
    """Draw a large, high-contrast labelled image crop."""
    header_h = 24
    x2 = x + size
    y2 = y + header_h + size
    cv2.rectangle(canvas, (x, y), (x2, y + header_h), (30, 30, 30), -1)
    cv2.putText(
        canvas, label, (x + 7, y + 17), FONT, 0.46,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    if image is None or getattr(image, "size", 0) == 0:
        cv2.rectangle(canvas, (x, y + header_h), (x2, y2), (45, 45, 45), -1)
        cv2.putText(
            canvas, "n/a", (x + size // 2 - 14, y + header_h + size // 2),
            FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA,
        )
    else:
        canvas[y + header_h:y2, x:x2] = cv2.resize(image, (size, size))
    cv2.rectangle(canvas, (x, y), (x2, y2), (230, 230, 230), 2)


def _search_region_rect(
    tracker,
    image: np.ndarray,
    bbox,
    use_last_mapping: bool = True,
) -> np.ndarray | None:
    """Return the live search rectangle, preferring the exact inference mapping."""
    from utils.utils import extend_bbox

    if image is None or getattr(image, "size", 0) == 0:
        return None
    cfg = getattr(tracker, "tracking_config", None)
    if cfg is None:
        return None
    context = None
    state = getattr(tracker, "tracking_state", None)
    if use_last_mapping and state is not None:
        context = getattr(state, "mapping", None)
    if context is None:
        context = extend_bbox(
            bbox,
            image_width=image.shape[1],
            image_height=image.shape[0],
            offset=cfg["search_context"],
        )
    return np.asarray(context, dtype=float).copy()


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
    _stamp_template_stats(
        panel_template,
        trk=float(getattr(tracker, "_last_template_update_score", float("nan"))),
        drm=float(getattr(tracker, "_last_template_update_drm", float("nan"))),
        sharp=float(
            getattr(tracker, "_last_template_update_sharpness", float("nan"))
        ),
    )

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


def _extract_dynamic_template_crop(
    tracker,
    dyn_image,
) -> np.ndarray | None:
    from utils.utils import extend_bbox, get_extended_crop

    if dyn_image is None or getattr(dyn_image, "size", 0) == 0:
        return None
    dyn_bbox = getattr(tracker, "running_dynamic_bbox", None)
    cfg = getattr(tracker, "tracking_config", None)
    if dyn_bbox is None or cfg is None:
        return None

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
    return t_crop


def run_inference(
    initial_bbox: List[int] | NDArray[np.integer],
    video_path: str,
    tracker,
    output_path: str = "outputs/tracked_video.mp4",
    output_video: bool = False,
    main_frame_only: bool = False,
):
    is_dam = hasattr(tracker, "tracker")
    inner_tracker = tracker.tracker if is_dam else tracker

    initial_bbox = np.array(initial_bbox).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    head, tail = os.path.split(output_path)
    # Write the bbox predictions right next to the (optional) annotated video,
    # inside the same per-video folder — no separate bboxes/ sub-directory.
    bbox_file = os.path.join(head, os.path.splitext(tail)[0] + ".txt")
    progress_name = os.path.splitext(tail)[0] or os.path.basename(
        os.path.normpath(video_path)
    )

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
        import threading
        import queue

        progress = SiamRAMProgressLine(progress_name, total_frames=total_frames)
        tracker.initialize(first_bgr, initial_bbox)
        progress.update("tracking")

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

        # Frame prefetch: one background thread decodes the next frame while the
        # GPU tracks the current one, hiding JPEG/video decode behind compute. A
        # single producer + FIFO queue preserves exact frame order, so tracker
        # output is identical to the old inline-read loop. `None` is the
        # end-of-stream sentinel (also covers the first failed/short read, which
        # the old loop handled via `if not ret or frame is None: break`).
        frame_q: "queue.Queue" = queue.Queue(maxsize=4)
        stop_reading = threading.Event()
        reader_exc: list = []

        def _frame_reader() -> None:
            try:
                while not stop_reading.is_set():
                    ret, fr = _read_next_frame()
                    if not ret or fr is None:
                        break
                    # Wait for room, but wake periodically so a stop request
                    # (consumer finished/errored) can never deadlock the reader.
                    while not stop_reading.is_set():
                        try:
                            frame_q.put(fr, timeout=0.25)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:  # surface a hard decode error to the main thread
                reader_exc.append(exc)
            finally:
                # Reliably deliver the sentinel in the normal case; skip if the
                # consumer already asked us to stop (it is no longer reading).
                while not stop_reading.is_set():
                    try:
                        frame_q.put(None, timeout=0.25)
                        break
                    except queue.Full:
                        continue

        reader_thread = threading.Thread(
            target=_frame_reader, name="frame-prefetch", daemon=True
        )
        reader_thread.start()

        try:
            while True:
                frame = frame_q.get()
                if frame is None:
                    break

                t0 = time.perf_counter()
                result = _update(frame)
                frame_times.append((time.perf_counter() - t0) * 1000.0)
                progress.update(_tracker_visual_mode(tracker, result, is_dam))

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
            # Stop the reader and drain buffered frames so a producer blocked on
            # put() can exit, then join before releasing the capture (the reader
            # is the sole owner of the cap / imread cursor).
            stop_reading.set()
            try:
                while True:
                    frame_q.get_nowait()
            except queue.Empty:
                pass
            reader_thread.join(timeout=2.0)
            progress.finish()
            _release_source()

        if reader_exc:
            raise reader_exc[0]

        if use_preallocated:
            tracked_bboxes = tracked_bboxes[:bbox_idx]

        np.savetxt(bbox_file, tracked_bboxes, fmt="%d", delimiter=" ")

        siamram_latency(
            [("all", _latency_stats(frame_times))],
            header="latency report · fast inference",
        )

        gc.collect()
        return

    if output_video:
        C_OCCLUDED = (0, 0, 220)
        C_STATUS_OCC = (0, 0, 200)
        C_STATUS_OK = (0, 180, 0)
        C_STATUS_TEXT = (255, 255, 255)
        inner_tracker._viz_capture = True

        top_h = h if main_frame_only else max(h, PANEL_W * 3 + GAP * 2)
        stats_strip_h = 236
        stats_strip_gap = 0 if main_frame_only else 8
        stats_y0 = top_h + stats_strip_gap
        canvas_h = h if main_frame_only else top_h + stats_strip_gap + stats_strip_h
        total_w = w if main_frame_only else w + PANEL_W
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
        avi_path = os.path.splitext(output_path)[0] + "_tmp.avi"
        writer = cv2.VideoWriter(
            avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (total_w, canvas_h)
        )
        progress = SiamRAMProgressLine(progress_name, total_frames=total_frames)
        aux_root = os.path.join(head, "auxelaries")
        aux_dirs = {
            "tracked": os.path.join(aux_root, "01_tracked_bboxes"),
            "templates": os.path.join(aux_root, "02_dynamic_templates"),
            "yolo_events": os.path.join(aux_root, "03_yolo_events"),
        }
        for aux_dir in (aux_root, *aux_dirs.values()):
            os.makedirs(aux_dir, exist_ok=True)

        aux_fps = float(fps) if np.isfinite(float(fps)) and float(fps) > 0 else FPS_DEFAULT
        tracked_snapshot_stride = max(1, int(round(aux_fps / 5.0)))
        tracked_snapshot_limit = 50
        tracked_snapshot_count = 0
        yolo_candidate_count = 0
        template_snapshot_count = 0
        last_yolo_event_rel_dir: str | None = None
        aux_index_rows: list[dict[str, str]] = []

        def _frame_crop(image: np.ndarray, bbox) -> np.ndarray | None:
            if image is None or getattr(image, "size", 0) == 0:
                return None
            arr = np.asarray(bbox, dtype=float).reshape(-1)
            if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
                return None
            x, y, bw, bh = arr[:4]
            if bw <= 1.0 or bh <= 1.0:
                return None
            ih, iw = image.shape[:2]
            x1 = int(np.clip(np.floor(x), 0, iw))
            y1 = int(np.clip(np.floor(y), 0, ih))
            x2 = int(np.clip(np.ceil(x + bw), 0, iw))
            y2 = int(np.clip(np.ceil(y + bh), 0, ih))
            if x2 <= x1 or y2 <= y1:
                return None
            return image[y1:y2, x1:x2].copy()

        def _context_crop_and_origin(
            image: np.ndarray,
            bbox,
            context_frac: float = 0.12,
        ) -> tuple[np.ndarray | None, tuple[int, int]]:
            if image is None or getattr(image, "size", 0) == 0:
                return None, (0, 0)
            arr = np.asarray(bbox, dtype=float).reshape(-1)
            if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
                return None, (0, 0)
            x, y, bw, bh = arr[:4]
            if bw <= 1.0 or bh <= 1.0:
                return None, (0, 0)
            ih, iw = image.shape[:2]
            pad = max(8.0, context_frac * max(float(bw), float(bh)))
            x1 = int(np.clip(np.floor(x - pad), 0, iw))
            y1 = int(np.clip(np.floor(y - pad), 0, ih))
            x2 = int(np.clip(np.ceil(x + bw + pad), 0, iw))
            y2 = int(np.clip(np.ceil(y + bh + pad), 0, ih))
            if x2 <= x1 or y2 <= y1:
                return None, (0, 0)
            return image[y1:y2, x1:x2].copy(), (x1, y1)

        def _draw_bbox_on_crop(
            crop: np.ndarray,
            bbox,
            origin: tuple[int, int],
            color: tuple[int, int, int],
            thickness: int,
            label: str = "",
        ) -> None:
            arr = np.asarray(bbox, dtype=float).reshape(-1)
            if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
                return
            ox, oy = origin
            x = int(round(float(arr[0]) - ox))
            y = int(round(float(arr[1]) - oy))
            bw = int(round(float(arr[2])))
            bh = int(round(float(arr[3])))
            if bw <= 0 or bh <= 0:
                return
            ch, cw = crop.shape[:2]
            x1 = int(np.clip(x, 0, max(0, cw - 1)))
            y1 = int(np.clip(y, 0, max(0, ch - 1)))
            x2 = int(np.clip(x + bw, 0, max(0, cw - 1)))
            y2 = int(np.clip(y + bh, 0, max(0, ch - 1)))
            if x2 <= x1 or y2 <= y1:
                return
            cv2.rectangle(crop, (x1, y1), (x2, y2), C_TEXT_SHADOW, thickness + 3, cv2.LINE_AA)
            cv2.rectangle(crop, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            if label:
                _draw_label(crop, label, x1 + 2, max(0, y1 - 24), color, scale=0.40)

        def _annotated_roi_crop(
            image: np.ndarray,
            roi_bbox,
            detections,
        ) -> np.ndarray | None:
            crop, origin = _context_crop_and_origin(image, roi_bbox, context_frac=0.12)
            if crop is None:
                return None
            _draw_bbox_on_crop(
                crop,
                roi_bbox,
                origin,
                (0, 165, 255),
                PRESENTATION_ROI_THICKNESS,
                "Search ROI",
            )
            for det_idx, det in enumerate(detections or [], start=1):
                _draw_bbox_on_crop(
                    crop,
                    det,
                    origin,
                    C_YOLO_CANDIDATE,
                    PRESENTATION_YOLO_THICKNESS,
                    f"YOLO {det_idx}",
                )
            return crop

        def _bbox_text(bbox) -> str:
            if bbox is None:
                return ""
            arr = np.asarray(bbox, dtype=float).reshape(-1)
            if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
                return ""
            return " ".join(str(int(round(v))) for v in arr[:4])

        def _write_aux_image(
            category: str,
            rel_path: str,
            image: np.ndarray | None,
            frame_num: int,
            bbox=None,
            note: str = "",
        ) -> bool:
            if image is None or getattr(image, "size", 0) == 0:
                return False
            out_path = os.path.join(aux_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ok = bool(cv2.imwrite(out_path, image))
            if ok:
                aux_index_rows.append(
                    {
                        "category": category,
                        "frame": str(int(frame_num)),
                        "path": rel_path.replace(os.sep, "/"),
                        "bbox_xywh": _bbox_text(bbox),
                        "note": note,
                    }
                )
            return ok

        def _save_dynamic_template_snapshot(frame_num: int, dyn_image) -> None:
            nonlocal template_snapshot_count
            crop = _extract_dynamic_template_crop(inner_tracker, dyn_image)
            if _write_aux_image(
                "dynamic_template",
                os.path.join(
                    "02_dynamic_templates",
                    f"template_{template_snapshot_count + 1:04d}_frame_{frame_num:06d}.jpg",
                ),
                crop,
                frame_num,
                bbox=getattr(inner_tracker, "running_dynamic_bbox", None),
            ):
                template_snapshot_count += 1

        def _save_tracked_bbox_snapshot(frame_num: int, frame_img: np.ndarray, tracked_bbox) -> None:
            nonlocal tracked_snapshot_count
            if tracked_snapshot_count >= tracked_snapshot_limit:
                return
            if frame_num % tracked_snapshot_stride != 0 and frame_num != 0:
                return
            crop = _frame_crop(frame_img, tracked_bbox)
            if _write_aux_image(
                "tracked_bbox",
                os.path.join(
                    "01_tracked_bboxes",
                    f"tracked_{tracked_snapshot_count + 1:03d}_frame_{frame_num:06d}.jpg",
                ),
                crop,
                frame_num,
                bbox=tracked_bbox,
                note=f"sampled every {tracked_snapshot_stride} frames (~5 fps)",
            ):
                tracked_snapshot_count += 1

        def _save_yolo_aux_snapshots(
            frame_num: int,
            frame_img: np.ndarray,
            roi_bbox,
            detections,
            recovered: bool = False,
        ) -> None:
            nonlocal yolo_candidate_count, last_yolo_event_rel_dir
            if not detections:
                return
            event_name = (
                f"RECOVERED_frame_{frame_num:06d}"
                if recovered
                else f"frame_{frame_num:06d}"
            )
            event_rel_dir = os.path.join("03_yolo_events", event_name)
            last_yolo_event_rel_dir = event_rel_dir
            candidates_rel_dir = os.path.join(event_rel_dir, "candidates")
            roi_crop = _frame_crop(frame_img, roi_bbox)
            _write_aux_image(
                "yolo_roi_clean",
                os.path.join(event_rel_dir, "roi_clean.jpg"),
                roi_crop,
                frame_num,
                bbox=roi_bbox,
                note=f"clean Search ROI crop; {len(detections)} candidates detected",
            )
            annotated_roi = _annotated_roi_crop(frame_img, roi_bbox, detections)
            _write_aux_image(
                "yolo_roi",
                os.path.join(event_rel_dir, "roi.jpg"),
                annotated_roi,
                frame_num,
                bbox=roi_bbox,
                note=(
                    "RECOVERED frame; " if recovered else ""
                ) + "context crop with Search ROI and YOLO candidate boxes drawn",
            )
            for det_idx, det in enumerate(detections or [], start=1):
                cand_crop = _frame_crop(frame_img, det)
                if _write_aux_image(
                    "yolo_candidate",
                    os.path.join(
                        candidates_rel_dir,
                        f"candidate_{det_idx:02d}.jpg",
                    ),
                    cand_crop,
                    frame_num,
                    bbox=det,
                    note=f"global candidate #{yolo_candidate_count + 1}",
                ):
                    yolo_candidate_count += 1

        def _mark_last_yolo_event_recovered(recovered_frame: int) -> None:
            nonlocal last_yolo_event_rel_dir
            if not last_yolo_event_rel_dir:
                return
            base = os.path.basename(last_yolo_event_rel_dir)
            if base.startswith("RECOVERED"):
                return
            old_rel = last_yolo_event_rel_dir
            new_rel = os.path.join(
                "03_yolo_events",
                f"RECOVERED_BY_frame_{recovered_frame:06d}__{base}",
            )
            old_abs = os.path.join(aux_root, old_rel)
            new_abs = os.path.join(aux_root, new_rel)
            if not os.path.isdir(old_abs) or os.path.exists(new_abs):
                return
            os.rename(old_abs, new_abs)

            old_prefix = old_rel.replace(os.sep, "/") + "/"
            new_prefix = new_rel.replace(os.sep, "/") + "/"
            for row in aux_index_rows:
                path = row.get("path", "")
                if path.startswith(old_prefix):
                    row["path"] = new_prefix + path[len(old_prefix):]
                    note = row.get("note", "")
                    suffix = f"recovered by frame {int(recovered_frame)}"
                    row["note"] = f"{note}; {suffix}" if note else suffix
            last_yolo_event_rel_dir = new_rel

        def _write_aux_index() -> None:
            readme_path = os.path.join(aux_root, "README.txt")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(
                    "SiamRAM presentation auxiliary snapshots\n"
                    "========================================\n\n"
                    "01_tracked_bboxes/\n"
                    "  Crops from tracked boxes sampled at roughly 5 fps, capped at 50 images.\n\n"
                    "02_dynamic_templates/\n"
                    "  Dynamic-template crops saved on initialization and each template update.\n\n"
                    "03_yolo_events/frame_XXXXXX/\n"
                    "  One folder per frame where YOLO returned candidates.\n"
                    "  RECOVERED_frame_XXXXXX marks the frame where recovery committed.\n"
                    "  roi.jpg includes context, the Search ROI box, and YOLO boxes.\n"
                    "  roi_clean.jpg is the clean Search ROI crop without overlays.\n"
                    "  candidates/ contains one crop per YOLO box.\n\n"
                    "index.csv lists every saved image with category, frame, relative path, bbox, and note.\n"
                )
            index_path = os.path.join(aux_root, "index.csv")
            with open(index_path, "w", encoding="utf-8", newline="") as f:
                fieldnames = ["category", "frame", "path", "bbox_xywh", "note"]
                writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
                writer_csv.writeheader()
                writer_csv.writerows(aux_index_rows)

        # Projector-friendly initialization-template inset in the upper-right,
        # positioned just below the camera-status pill.
        viz_pad = 12
        viz_label_h = 24
        preferred_viz_sz = int(round(min(w, h) * 0.18))
        max_fitting_viz_sz = max(80, w - 2 * viz_pad)
        viz_sz = int(np.clip(min(preferred_viz_sz, max_fitting_viz_sz), 120, 220))
        viz_y = y_off + 44
        viz_x_template = w - viz_pad - viz_sz
        viz_enabled = (
            viz_x_template >= 0
            and viz_y + viz_label_h + viz_sz <= y_off + h
        )

        init_template_crop: np.ndarray | None = None
        current_search_rect: np.ndarray | None = None

        def _draw_viz_panels() -> None:
            """Draw the fixed initialization template."""
            if not viz_enabled:
                return
            _draw_image_panel(
                canvas, viz_x_template, viz_y, viz_sz,
                init_template_crop, "INIT TEMPLATE",
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

        def _draw_tracker_search_box(rect, scale: float) -> None:
            """Draw the exact model-search mapping in source-frame coordinates."""
            if rect is None:
                return
            arr = np.asarray(rect, dtype=float).reshape(-1)
            if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
                return
            sx, sy, sw, sh = _overlay_bbox(arr, scale)
            if sw <= 0 or sh <= 0:
                return
            _draw_box_with_outline(
                canvas,
                (sx, sy + y_off),
                (sx + sw, sy + sh + y_off),
                C_TRACKER_SEARCH,
                PRESENTATION_ROI_THICKNESS,
            )
            visible_w = max(0, min(w, sx + sw) - max(0, sx))
            visible_h = max(0, min(h, sy + sh) - max(0, sy))
            padded = visible_w != sw or visible_h != sh
            size_text = f"{sw}x{sh}" + (" PAD" if padded else "")
            _draw_label(
                canvas,
                f"TRACKER SEARCH  {size_text}",
                sx + 3,
                max(y_off + 42, sy + y_off + 3),
                C_TRACKER_SEARCH,
                scale=0.50,
            )

        def _scaled_vector(vec, scale: float):
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=float).reshape(-1)
            if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
                return None
            return arr[:2] * float(scale)

        def _ekf_velocity_vector(scale: float):
            if not is_dam:
                return None
            ekf = getattr(tracker, "ekf", None)
            vel = None
            if ekf is not None and hasattr(ekf, "get_velocity"):
                try:
                    vel = ekf.get_velocity()
                except (TypeError, ValueError, AttributeError):
                    vel = None
            if vel is None:
                vel = getattr(tracker, "velocity", None)
            return _scaled_vector(vel, scale)

        def _camera_motion_vector_at(point_xy, scale: float):
            if not is_dam:
                return None
            H = getattr(tracker, "_last_H", None)
            if H is None or not bool(getattr(tracker, "_last_H_reliable", False)):
                return None
            if not hasattr(tracker, "_camera_displacement_from_h_at_point"):
                return None
            point = np.asarray(point_xy, dtype=float).reshape(-1)
            if point.size < 2 or not np.all(np.isfinite(point[:2])):
                return None
            try:
                cam = tracker._camera_displacement_from_h_at_point(
                    H, float(point[0]), float(point[1])
                )
            except (TypeError, ValueError, AttributeError):
                return None
            return _scaled_vector(cam, scale)

        def _draw_roi_direction_arrows(
            roi_overlay: tuple[int, int, int, int],
            roi_proc: tuple[int, int, int, int],
            scale: float,
        ) -> None:
            ox, oy, ow, oh = roi_overlay
            px, py, pw, ph = roi_proc
            if ow <= 0 or oh <= 0 or pw <= 0 or ph <= 0:
                return
            roi_rect = (ox, oy + y_off, ox + ow, oy + y_off + oh)
            frame_rect = (0, y_off, w - 1, y_off + h - 1)
            offset = float(np.clip(min(ow, oh) * 0.16, 14.0, 34.0))
            object_vec = _ekf_velocity_vector(scale)
            camera_vec = _camera_motion_vector_at(
                (float(px + pw / 2.0), float(py + ph / 2.0)),
                scale,
            )
            _draw_roi_edge_motion_arrow(
                canvas,
                roi_rect,
                object_vec,
                C_OBJECT_MOTION,
                "Object motion",
                frame_rect,
                side_offset=-offset,
                vector_scale=7.0,
            )
            _draw_roi_edge_motion_arrow(
                canvas,
                roi_rect,
                camera_vec,
                C_CAMERA_MOTION,
                "Camera motion",
                frame_rect,
                side_offset=offset,
                vector_scale=6.0,
            )

        def _is_recovered_frame(frame_num: int) -> bool:
            info = getattr(tracker, "_last_recovery_info", None) if is_dam else None
            if not isinstance(info, dict):
                return False
            try:
                return int(info.get("frame_idx", -1)) == int(frame_num)
            except (TypeError, ValueError):
                return False

        def _draw_status_pill(
            canvas: np.ndarray,
            mode: str,
        ) -> None:
            """
            Inputs:
                canvas       - np.ndarray (H x W x 3) the full composite canvas
                mode - str status mode: "tracking" or "occluded"

            What it does:
                Draws a rounded-rectangle pill badge in the top-left corner of the
                frame area. The pill is red for occlusion recovery and green
                for normal tracking. Uses three draw
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

        def _draw_camera_motion_pill(
            canvas: np.ndarray,
            heavy: bool,
            disp: float,
        ) -> None:
            """
            Draws a pill in the top-right of the frame area reporting the
            camera-motion gate's verdict for this frame:
              - red  "CAMERA HIGH"   when heavy camera motion is flagged,
              - green "CAMERA STABLE" otherwise.
            `disp` is the per-frame camera displacement (px) appended for context.
            Mirrors _draw_status_pill's rounded-rectangle style.
            """
            if heavy:
                label = "CAMERA HIGH"
                color = (0, 0, 255)
            else:
                label = "CAMERA STABLE"
                color = C_STATUS_OK
            label = f"{label}  {disp:.0f}px"
            ph = 28
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
            pw = max(150, tw + 28)
            px = max(8, w - pw - 8)
            py = y_off + 8
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

        def _draw_stats_strip_base() -> None:
            cv2.line(
                canvas,
                (0, top_h + stats_strip_gap // 2),
                (total_w - 1, top_h + stats_strip_gap // 2),
                (90, 90, 90),
                1,
            )
            cv2.putText(
                canvas,
                "STATS",
                (w + 6, stats_y0 + 14),
                FONT,
                0.42,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

        tracker.initialize(first_bgr, initial_bbox)
        viz_init_frame = (
            getattr(tracker, "init_frame", None) if is_dam else first_bgr
        )
        if viz_init_frame is None:
            viz_init_frame = first_bgr
        viz_init_bbox = getattr(inner_tracker, "running_dynamic_bbox", initial_bbox)
        init_template_crop = _extract_dynamic_template_crop(
            inner_tracker, viz_init_frame
        )
        current_search_rect = _search_region_rect(
            inner_tracker, viz_init_frame, viz_init_bbox, use_last_mapping=False
        )
        _refresh_panels(
            inner_tracker,
            first_bgr,
            panel_template,
            panel_search,
            frame_idx=0,
            updated=True,
        )
        init_dyn_image = getattr(inner_tracker, "running_dynamic_image", None)
        _save_dynamic_template_snapshot(
            0, init_dyn_image if init_dyn_image is not None else first_bgr
        )
        _save_tracked_bbox_snapshot(0, first_bgr, initial_bbox)

        canvas.fill(0)
        canvas[y_off: y_off + h, :w] = first_bgr
        if not main_frame_only:
            canvas[row_tmpl] = panel_template
            canvas[row_search] = panel_search
            _refresh_recovery_panel(tracker, panel_recovery)
            canvas[row_recovery] = panel_recovery
        init_frame_scale = float(getattr(tracker, "_frame_scale", 1.0))
        init_overlay_scale = (
            1.0
            if not np.isfinite(init_frame_scale) or init_frame_scale <= 1e-8
            else 1.0 / init_frame_scale
        )
        _draw_tracker_search_box(current_search_rect, init_overlay_scale)
        bx, by, bw, bh = map(int, initial_bbox)
        _draw_box_with_outline(
            canvas,
            (bx, by + y_off),
            (bx + bw, by + bh + y_off),
            C_GT,
            3,
        )
        _draw_status_pill(canvas, mode="tracking")
        cv2.putText(
            canvas, "F:0  INIT", (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA
        )
        _draw_stats_strip_base()
        _draw_legend(canvas, 8, stats_y0 + 24)
        _draw_viz_panels()
        writer.write(canvas)
        progress.update("tracking")

        last_dyn_bbox = inner_tracker.running_dynamic_bbox.copy()
        last_roi_proc = None

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
                if not in_occlusion and raw_mode == "tracking":
                    visual_mode = raw_mode
                visual_reason = str(getattr(tracker, "visual_reason", "")).strip()
                visual_details = str(getattr(tracker, "visual_details", "")).strip()

            tracked_bboxes.append(bbox)

            if in_occlusion:
                times_occlusion.append(elapsed_ms)
            else:
                times_normal.append(elapsed_ms)
            progress.update(visual_mode)

            if output_video:
                viz_search_frame = (
                    tracker._prescale_frame(frame)
                    if is_dam and hasattr(tracker, "_prescale_frame")
                    else frame
                )
                viz_search_bbox = getattr(
                    getattr(inner_tracker, "tracking_state", None), "bbox", bbox
                )
                current_search_rect = _search_region_rect(
                    inner_tracker,
                    viz_search_frame,
                    viz_search_bbox,
                    use_last_mapping=True,
                )
                recovered_frame = _is_recovered_frame(frame_idx)
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
                    _save_dynamic_template_snapshot(frame_idx, cur_dyn_obj)
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
                if not main_frame_only:
                    canvas[row_tmpl] = panel_template
                    canvas[row_search] = panel_search
                    _refresh_recovery_panel(tracker, panel_recovery)
                    canvas[row_recovery] = panel_recovery

                frame_scale = float(getattr(tracker, "_frame_scale", 1.0))
                if not np.isfinite(frame_scale) or frame_scale <= 1e-8:
                    overlay_scale = 1.0
                else:
                    overlay_scale = 1.0 / frame_scale

                # During recovery the inner mapping may belong to a candidate
                # probe rather than this frame's normal tracker search. The
                # dedicated recovery ROI is drawn later, so suppress this box.
                if not in_occlusion:
                    _draw_tracker_search_box(current_search_rect, overlay_scale)

                if visual_mode != "occluded":
                    _save_tracked_bbox_snapshot(frame_idx, frame, bbox)

                proc_frame = None
                roi_proc = None
                if is_dam and (in_occlusion or (recovered_frame and yolo_dets)):
                    proc_frame = (
                        tracker._prescale_frame(frame)
                        if hasattr(tracker, "_prescale_frame")
                        else frame
                    )
                    if in_occlusion:
                        roi_proc = tracker._get_yolo_search_roi(proc_frame)
                        last_roi_proc = roi_proc
                    elif last_roi_proc is not None:
                        roi_proc = last_roi_proc
                    else:
                        roi_proc = tracker._get_yolo_search_roi(proc_frame)
                    rx, ry, rw, rh = _overlay_bbox(roi_proc, overlay_scale)
                    _save_yolo_aux_snapshots(
                        frame_idx,
                        frame,
                        (rx, ry, rw, rh),
                        yolo_dets,
                        recovered=recovered_frame,
                    )
                elif recovered_frame:
                    _mark_last_yolo_event_recovered(frame_idx)

                if in_occlusion and not main_frame_only:
                    cv2.rectangle(
                        canvas, (w, 0), (total_w - 1, canvas_h - 1), C_OCCLUDED, 3
                    )
                elif template_updated and not main_frame_only:
                    cv2.rectangle(
                        canvas, (w, 0), (total_w - 1, canvas_h - 1), C_UPDATE, 2
                    )

                if in_occlusion and yolo_dets:
                    for cand_idx, det in enumerate(yolo_dets, start=1):
                        dx, dy, dw, dh = map(int, det)
                        _draw_box_with_outline(
                            canvas,
                            (dx, dy + y_off),
                            (dx + dw, dy + dh + y_off),
                            C_YOLO_CANDIDATE,
                            PRESENTATION_YOLO_THICKNESS,
                        )
                        _draw_label(
                            canvas,
                            f"YOLO {cand_idx}",
                            dx + 2,
                            max(y_off + 4, dy + y_off - 28),
                            C_YOLO_CANDIDATE,
                            scale=0.42,
                        )

                if visual_mode != "occluded":
                    bx, by, bw, bh = map(int, bbox)
                    pred_color = _confidence_color(score)
                    _draw_box_with_outline(
                        canvas,
                        (bx, by + y_off),
                        (bx + bw, by + bh + y_off),
                        pred_color,
                        PRESENTATION_PRED_THICKNESS,
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
                if is_dam:
                    _draw_camera_motion_pill(
                        canvas,
                        heavy=bool(getattr(tracker, "_last_heavy_cam_motion", False)),
                        disp=float(getattr(tracker, "_last_heavy_cam_disp", 0.0)),
                    )

                if visual_mode == "occluded":
                    hud = f"F:{frame_idx}  [DAM RECOVERY]"
                elif template_updated:
                    hud = f"F:{frame_idx}  [TMPL UPDATE]"
                else:
                    hud = f"F:{frame_idx}"
                cv2.putText(
                    canvas, hud, (156, y_off + 22), FONT, 0.45, C_FRAME, 1, cv2.LINE_AA
                )

                if is_dam and in_occlusion:
                    if roi_proc is None:
                        proc_frame = (
                            tracker._prescale_frame(frame)
                            if hasattr(tracker, "_prescale_frame")
                            else frame
                        )
                        roi_proc = tracker._get_yolo_search_roi(proc_frame)
                        last_roi_proc = roi_proc
                    rx, ry, rw, rh = _overlay_bbox(roi_proc, overlay_scale)
                    _draw_box_with_outline(
                        canvas,
                        (rx, ry + y_off),
                        (rx + rw, ry + rh + y_off),
                        (0, 165, 255),
                        PRESENTATION_ROI_THICKNESS,
                    )
                    cv2.putText(
                        canvas,
                        "Search ROI",
                        (rx + 2, ry + y_off + 12),
                        FONT,
                        0.38,
                        (0, 165, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    _draw_roi_direction_arrows(
                        (rx, ry, rw, rh),
                        roi_proc,
                        overlay_scale,
                    )
                _draw_stats_strip_base()
                _draw_legend(canvas, 8, stats_y0 + 24)
                # Right-column readout stack, drawn LAST so the full-width
                # strip stays readable. Bottom-anchored inside the reserved stats band.
                stack_x0 = w + 4
                stack_x1 = total_w - 4
                stack_y = canvas_h - 6

                def _stack_box(y_bottom, line1, line2, fg):
                    if main_frame_only:
                        return y_bottom
                    y_top = y_bottom - 40
                    cv2.rectangle(
                        canvas, (stack_x0, y_top), (stack_x1, y_bottom), (20, 20, 20), -1
                    )
                    cv2.rectangle(
                        canvas, (stack_x0, y_top), (stack_x1, y_bottom), (70, 70, 70), 1
                    )

                    def _fit_text(text, y, scale):
                        max_w = max(20, stack_x1 - stack_x0 - 12)
                        while scale > 0.28:
                            (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
                            if tw <= max_w:
                                break
                            scale -= 0.03
                        cv2.putText(
                            canvas,
                            text,
                            (stack_x0 + 6, y),
                            FONT,
                            scale,
                            fg,
                            1,
                            cv2.LINE_AA,
                        )

                    _fit_text(line1, y_top + 16, 0.42)
                    _fit_text(line2, y_top + 33, 0.38)
                    return y_top - 6

                # (1) Adaptive DRM acceptance margin (only when drm_margin=="auto";
                # numeric drm_margin leaves _drm_margin_auto None -> not drawn).
                auto_margin = getattr(tracker, "_drm_margin_auto", None) if is_dam else None
                if auto_margin is not None:
                    ema = auto_margin.ema
                    cur = getattr(tracker, "_last_drm_self_score", None)
                    cur_txt = "--" if cur is None else f"{float(cur):.3f}"
                    m_l2 = (
                        f"(auto, warmup) cur={cur_txt}"
                        if ema is None
                        else f"(auto) ema={ema:.3f} cur={cur_txt} n={auto_margin.samples}"
                    )
                    stack_y = _stack_box(
                        stack_y, f"DRM margin: {auto_margin.value:.3f}", m_l2, (0, 220, 220)
                    )

                auto_reacq = (
                    getattr(tracker, "_reacq_threshold_auto", None) if is_dam else None
                )
                if auto_reacq is not None:
                    ema = auto_reacq.ema
                    cur = getattr(tracker, "_last_reacq_threshold_score", None)
                    cur_txt = "--" if cur is None else f"{float(cur):.3f}"
                    r_l2 = (
                        f"(auto, warmup) cur={cur_txt}"
                        if ema is None
                        else f"(auto) ema={ema:.3f} cur={cur_txt} n={auto_reacq.samples}"
                    )
                    stack_y = _stack_box(
                        stack_y,
                        f"Reacq thr: {auto_reacq.value:.3f}",
                        r_l2,
                        (80, 190, 255),
                    )

                auto_conf = (
                    getattr(tracker, "_conf_threshold_auto", None) if is_dam else None
                )
                if auto_conf is not None:
                    ema = auto_conf.ema
                    cur = getattr(tracker, "_last_conf_threshold_score", None)
                    cur_txt = "--" if cur is None else f"{float(cur):.3f}"
                    c_l2 = (
                        f"(auto, warmup) cur={cur_txt}"
                        if ema is None
                        else f"(auto) ema={ema:.3f} cur={cur_txt} n={auto_conf.samples}"
                    )
                    stack_y = _stack_box(
                        stack_y,
                        f"Occ entry: {auto_conf.value:.3f}",
                        c_l2,
                        (255, 180, 80),
                    )

                auto_dyn = (
                    getattr(inner_tracker, "_dynamic_update_threshold_auto", None)
                    if is_dam
                    else None
                )
                if auto_dyn is not None:
                    ema = auto_dyn.ema
                    cur = getattr(inner_tracker, "_last_dyn_self_score", None)
                    cur_txt = "--" if cur is None else f"{float(cur):.3f}"
                    t_l2 = (
                        f"(auto, warmup) cur={cur_txt}"
                        if ema is None
                        else f"(auto) ema={ema:.3f} cur={cur_txt} n={auto_dyn.samples}"
                    )
                    stack_y = _stack_box(
                        stack_y,
                        f"Tmpl admit: {auto_dyn.value:.3f}",
                        t_l2,
                        (255, 160, 60),
                    )

                if is_dam:
                    n_cur = int(getattr(inner_tracker, "N", 0))
                    win_cur = int(getattr(inner_tracker, "memory_window_size", 0))
                    stack_y = _stack_box(
                        stack_y,
                        f"Tmpl rate: N={n_cur} win={win_cur}",
                        "(fixed)",
                        (0, 210, 140),
                    )

                _draw_viz_panels()
                writer.write(canvas)

            frame_idx += 1

    finally:
        progress.finish()
        _release_source()
        if output_video:
            writer.release()
            _write_aux_index()

    if output_video and avi_path != output_path:
        os.replace(avi_path, output_path)

    with open(bbox_file, "w", encoding="utf-8") as f:
        for bb in tracked_bboxes:
            f.write(f"{bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

    if output_video:
        del canvas, panel_template, panel_search

    occ_pct = (
        100 * len(times_occlusion) / len(frame_times)
        if times_occlusion and frame_times
        else None
    )
    siamram_latency(
        [
            ("all", _latency_stats(frame_times)),
            ("track", _latency_stats(times_normal)),
            ("occl", _latency_stats(times_occlusion)),
        ],
        occlusion_pct=occ_pct,
    )

    gc.collect()
