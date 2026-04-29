import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import OmegaConf
from models.SiamABC.tracker.trt_engine.siamabc import get_trt_tracker
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from models.SiamABC.tracker.tracker_setup import get_tracker
from models.SiamRAM import SiamRAMTracker
from utils.hydra import load_hydra_config_from_path
from vis.test_model import run_inference


manifest_path = "/home/moha/AIC-4/data_competition/metadata/contestant_manifest.json"

data_dir = "../../AIC-4/data_competition"
outputs_dir = "outputs/SiamDAM__test"
manifest_path = "/home/moha/AIC-4/data_competition/metadata/contestant_manifest.json"
weights_path = "/home/moha/AIC-for submission/SiamRAM/checkpoints/head_epoch_000.pth"
yaml_config_path = "config/inference_config.yaml" 


with open(manifest_path, "r") as f:
    manifest = json.load(f)
test_public_lb = manifest["public_lb"]
manifest.keys()





config = OmegaConf.load(yaml_config_path)


wrapped = get_trt_tracker(
    config=config,
    weights_path=weights_path,
    **config.trt_engine  
)


tracker = SiamRAMTracker(
    siam_tracker=wrapped,
    **config.ram_tracker 
)



with open(manifest_path, "r") as f:
    manifest = json.load(f)

test_public_lb = {
    k: v for k, v in manifest["public_lb"].items()
    if v["dataset"] in [
        # "dataset1",
        # "dataset2",
        # "dataset3",
        "dataset4",
        "dataset5"
        ]
}

print(f"Starting inference on {len(test_public_lb)} videos...")
for i, (key, value) in enumerate(test_public_lb.items()):
    video_path = os.path.join(data_dir, value["video_path"])
    ann_path = os.path.join(data_dir, value["annotation_path"])
    output_path = os.path.join(outputs_dir, value["video_path"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[{i+1}/{len(test_public_lb)}] Processing: {value['video_path']}")
    
    init_bbox = np.loadtxt(ann_path, delimiter=",", dtype=np.float32).tolist()
    if isinstance(init_bbox[0], list):
        init_bbox = init_bbox[0]

    run_inference(
        video_path=video_path,
        initial_bbox=init_bbox,
        tracker=tracker,
        output_path=output_path
    )

print("\nCompiling submission.csv...")
submission_df = defaultdict(list)

for key, value in test_public_lb.items():
    head, tail = os.path.split(os.path.join(outputs_dir, value["video_path"]))
    bbox_file = os.path.join(head, 'bboxes', os.path.splitext(tail)[0] + '.txt')

    if os.path.exists(bbox_file):
        with open(bbox_file, 'r') as f:
            lines = f.read().strip().split('\n')
            for frame_idx, line in enumerate(lines):
                x, y, w, h = line.strip().split()
                submission_df["id"].append(f"{key}_{frame_idx}")
                submission_df["x"].append(float(x))
                submission_df["y"].append(float(y))
                submission_df["w"].append(float(w))
                submission_df["h"].append(float(h))
    else:
        print(f"Warning: Bbox file missing for {key}")

submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv", index=False)

print("\nSubmission Preview:")
print(submission.head())
print(f"Total rows: {len(submission)}")