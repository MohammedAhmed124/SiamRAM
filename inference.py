"""
Inference Module

This module provides functionality for running inference on video sequences using a pre-trained model.

This module provides:
- Running inference on video sequences
- Loading models
- Writing predictions to CSV

"""

import csv
import json
import sys
import time

import torch

from predictor import load_model, run_tracker

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def load_json(path):
    """ Load JSON file from the given path."""
    with open(path, "r") as f:
        data = json.load(f)

    return data


def write_csv(rows, output_path):
    """ Write rows to a CSV file at the given output path."""
    with open(output_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "id",
            "x",
            "y",
            "w",
            "h",
        ])

        for row in rows:
            writer.writerow(row)

def run_sequence(
    model,
    sequence_name,
    sequence_info,
):
    """
    Run tracking on a single sequence.
    """

    video_path = sequence_info["video_path"]

    annotation_path = sequence_info[
        "annotation_path"
    ]

    print(f"\nRunning: {sequence_name}")

    start_time = time.perf_counter()

    predictions = run_tracker(
        model=model,
        video_path=video_path,
        init_box_path=annotation_path,
    )

    elapsed = (
        time.perf_counter() - start_time
    )

    fps = (
        len(predictions) / elapsed
        if elapsed > 0
        else 0
    )

    print(f"Frames: {len(predictions)}")
    print(f"Time: {elapsed:.2f} sec")
    print(f"FPS: {fps:.2f}")

    return predictions

def infer_dataset():
    """
    Run inference on the entire dataset.
    """

    if len(sys.argv) != 4:

        print(
            "\nUsage:\n"
            "python inference.py "
            "input.json split output.csv\n"
        )

        sys.exit(1)

    input_json = sys.argv[1]

    split_name = sys.argv[2]

    output_csv = sys.argv[3]

    data = load_json(input_json)

    if split_name not in data:

        raise ValueError(
            f"Split '{split_name}' not found."
        )

    sequences = data[split_name]

    model = load_model(device=DEVICE)

    all_rows = []

    total_start = time.perf_counter()

    for (
        sequence_name,
        sequence_info,
    ) in sequences.items():

        predictions = run_sequence(
            model=model,
            sequence_name=sequence_name,
            sequence_info=sequence_info,
        )

        for pred in predictions:

            frame_idx = pred["frame_idx"]

            row_id = (
                f"{sequence_name}_{frame_idx}"
            )

            all_rows.append([
                row_id,
                pred["x"],
                pred["y"],
                pred["w"],
                pred["h"],
            ])

    total_elapsed = (
        time.perf_counter() - total_start
    )

    write_csv(
        rows=all_rows,
        output_path=output_csv,
    )

    print("\n" + "=" * 50)
    print("FINISHED")
    print("=" * 50)

    print(f"Sequences: {len(sequences)}")
    print(f"Predictions: {len(all_rows)}")

    print(
        f"Total Runtime: "
        f"{total_elapsed:.2f} sec"
    )

    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    infer_dataset()
