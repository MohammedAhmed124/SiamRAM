"""
IPython display utilities for tracking visualisation.

This module provides functions to render tracking sequences and annotations
as HTML5 videos directly within Jupyter/IPython environments.
"""

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from PIL import Image
from matplotlib import animation


def show_video(
    video_folder_path,
    annotation_path,
    start_end_idx,
    n_frames_to_display=None,
    figsize=(12, 6),
):
    """
    Render a tracking sequence as an HTML5 video for Jupyter Notebooks.

    Args:
        video_folder_path (str): Path to the directory containing image frames.
        annotation_path (str): Path to the ground-truth annotation file.
        start_end_idx (Tuple[int, int]): Start and end frame indices.
        n_frames_to_display (Optional[int]): Number of frames to render.
        figsize (Tuple[int, int]): Size of the matplotlib figure.

    Returns:
        IPython.display.HTML: The rendered HTML5 video object.
    """
    start_frame, end_frame = start_end_idx
    all_frames = sorted(
        os.listdir(video_folder_path), key=lambda x: int(x.split(".")[0])
    )
    frames = all_frames[start_frame - 1: end_frame]
    bboxes = np.loadtxt(annotation_path, delimiter=",")

    max_valid_frames = min(len(frames), len(bboxes))

    if not n_frames_to_display:
        n_frames_to_display = max_valid_frames
    else:
        n_frames_to_display = min(n_frames_to_display, max_valid_frames)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    img_path = os.path.join(video_folder_path, frames[0])
    im = ax.imshow(Image.open(img_path))

    rect = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor="r", facecolor="none")
    ax.add_patch(rect)

    def update(
        frame_idx,
    ):
        frame_name = frames[frame_idx]
        img_path = os.path.join(video_folder_path, frame_name)
        with Image.open(img_path) as img:
            img_array = np.asarray(img.convert("RGBA"))
            im.set_data(img_array)
        x, y, w, h = bboxes[frame_idx]
        rect.set_xy((x, y))
        rect.set_width(w)
        rect.set_height(h)
        return [im, rect]

    ani = animation.FuncAnimation(
        fig, update, frames=range(n_frames_to_display), interval=50
    )

    plt.close()

    video_html = ani.to_html5_video()

    return HTML(video_html)
