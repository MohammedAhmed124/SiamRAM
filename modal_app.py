"""Modal app that runs SiamRAM over tracking benchmarks on cloud GPUs.

    modal run modal_app.py --datasets dtb70,uav123 --configs inference_config.yaml
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "siamram-bench"
REPO = "/root/siamram"
DATA = "/data"
RESULTS = "/results"

_here = Path(__file__).parent

# Package set mirrors Dockerfile.gpu. `devel` (not `base`) because TensorRT engine
# builds need the full CUDA toolkit. Python 3.10 matches the pinned wheels in
# requirements.txt. The two git deps (mobile-cv, torchreid) install from that file.
# pynvvideocodec is stripped: benchmark data is JPEG frame directories, not video,
# so NVDEC is never used here and the runs force runtime.use_nvdec=false.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .entrypoint([])
    .apt_install(
        "build-essential",
        # torchreid compiles rank_cylib, and add_python's interpreter reports CC=clang.
        "clang",
        "ca-certificates",
        "git",
        "libglib2.0-0",
        "libgl1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libxcb1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster HF Hub transfers
    .add_local_file(_here / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands(
        "grep -v '^pynvvideocodec' /tmp/requirements.txt > /tmp/req.modal.txt",
        "pip install --no-cache-dir -r /tmp/req.modal.txt",
    )
    .add_local_dir(_here / "src", f"{REPO}/src")
    .add_local_dir(_here / "bench", f"{REPO}/bench")
    .add_local_dir(_here / "checkpoints", f"{REPO}/checkpoints")
    # data/ is not shipped, so bench/eval.py reads the committed manifest for its leakage check.
    .add_local_dir(_here / "splits", f"{REPO}/splits")
    # Baseline configs (t1_vanilla_siamabc.yaml and the ablation rows) live here.
    .add_local_dir(_here / "ablation" / "configs", f"{REPO}/ablation/configs")
    .add_local_file(_here / "predictor.py", f"{REPO}/predictor.py")
)

data_vol = modal.Volume.from_name("siamram-data", create_if_missing=True)
results_vol = modal.Volume.from_name("siamram-results", create_if_missing=True)

VOLUMES = {DATA: data_vol, RESULTS: results_vol}
GPU_TIMEOUT = 60 * 60 * 8  # ~168k frames is the largest single set in the plan

app = modal.App(APP_NAME, image=image)


def _run(*args: str) -> None:
    """Run a repo script with the repo root as cwd, raising on failure."""
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def _trt_cache() -> str:
    """TRT engine cache dir on the results volume, keyed by GPU model."""
    # Engines are SM-architecture specific: an A10G engine will not load on an H100,
    # so each GPU model gets its own cache directory.
    import re

    import torch

    name = re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name(0).lower()).strip("_")
    return f"{RESULTS}/trt_cache/{name}"


def _config(name: str) -> str:
    """Copy a repo config to /tmp with NVDEC off and the GPU-keyed TRT cache path."""
    import yaml

    src = Path(name)
    if not src.is_absolute():
        src = next(p for p in (Path(REPO) / "src" / "config" / name,
                               Path(REPO) / "ablation" / "configs" / name) if p.is_file())
    cfg = yaml.safe_load(src.read_text())
    cfg.setdefault("runtime", {})["use_nvdec"] = False
    cfg.setdefault("trt_engine", {})["trt_cache_dir"] = _trt_cache()
    out = Path("/tmp") / src.name
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


@app.function(
    volumes=VOLUMES,
    ephemeral_disk=1000 * 1000,
    timeout=60 * 60 * 12,
    # LaSOT and TrackingNet come from the HF Hub, which rate-limits anonymous requests.
    # Create it once:  modal secret create huggingface-secret HF_TOKEN=hf_...
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def download_dataset(name: str) -> str:
    """Fetch a benchmark into the dataset volume."""
    dest = f"{DATA}/{name}"
    _run("bench/download.py", "--dataset", name, "--dest", dest)
    data_vol.commit()
    return dest


@app.function(gpu="A10G", volumes=VOLUMES, timeout=GPU_TIMEOUT)
def warm_trt_cache(dataset: str, config: str) -> str:
    """Build and cache the TensorRT engines for this GPU by tracking one sequence."""
    data_vol.reload()
    results_vol.reload()
    _run(
        "bench/run_tracker.py",
        "--dataset", dataset,
        "--data-root", f"{DATA}/{dataset}",
        "--config", _config(config),
        "--out", f"{RESULTS}/_warmup/{dataset}",
        "--limit", "1",
    )
    results_vol.commit()
    return _trt_cache()


@app.function(gpu="A10G", volumes=VOLUMES, timeout=GPU_TIMEOUT)
def run_benchmark(dataset: str, config: str, tracker_name: str) -> str:
    """Track every sequence of a dataset and write per-sequence boxes to the results volume."""
    data_vol.reload()
    results_vol.reload()
    # <dataset>/<tracker> so bench/eval.py can take {RESULTS}/<dataset> as its results root.
    out = f"{RESULTS}/{dataset}/{tracker_name}"
    _run(
        "bench/run_tracker.py",
        "--dataset", dataset,
        "--data-root", f"{DATA}/{dataset}",
        "--config", _config(config),
        "--out", out,
    )
    results_vol.commit()
    return out


@app.function(volumes=VOLUMES, timeout=60 * 30)
def evaluate(dataset: str, trackers: list[str]) -> str:
    """Score tracked results for a dataset and return the metrics CSV."""
    results_vol.reload()
    out = f"{RESULTS}/metrics/{dataset}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # TrackingNet withholds test ground truth: pack a server submission instead of scoring.
    if dataset == "trackingnet":
        zips = []
        for tracker in trackers:
            zip_path = f"{RESULTS}/metrics/trackingnet_{tracker}.zip"
            _run("bench/pack_trackingnet.py",
                 "--results", f"{RESULTS}/{dataset}/{tracker}", "--out", zip_path)
            zips.append(zip_path)
        results_vol.commit()
        return "note\n" + "\n".join(
            f"upload to the TrackingNet eval server: {z}" for z in zips)
    _run(
        "bench/eval.py",
        "--results", f"{RESULTS}/{dataset}",
        "--dataset", dataset,
        "--data-root", f"{DATA}/{dataset}",
        "--trackers", ",".join(trackers),
        "--out", out,
        "--protocol-check",
    )
    results_vol.commit()
    return Path(out).read_text()


def _markdown(csv_text: str) -> str:
    """Render a CSV as a markdown table."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return "_(no rows)_"
    head, *body = rows
    sep = ["---"] * len(head)
    return "\n".join("| " + " | ".join(r) + " |" for r in (head, sep, *body))


@app.local_entrypoint()
def main(
    datasets: str,
    configs: str = "inference_config.yaml",
    gpu: str = "A10G",
    out_dir: str = "bench_results",
):
    """Fan out tracking over datasets x configs, evaluate, print and save the tables."""
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    cfgs = [c.strip() for c in configs.split(",") if c.strip()]
    trackers = [Path(c).stem for c in cfgs]

    warm = warm_trt_cache.with_options(gpu=gpu)
    bench = run_benchmark.with_options(gpu=gpu)

    # Pay the TensorRT build cost once per config, not once per sequence or dataset.
    for cfg in cfgs:
        print("warmed TRT cache:", warm.remote(ds[0], cfg))

    # return_exceptions: one bad sequence must not discard a multi-hour sweep.
    jobs = [(d, c, t) for d in ds for c, t in zip(cfgs, trackers)]
    failed = []
    for job, out in zip(jobs, bench.starmap(jobs, return_exceptions=True)):
        print("failed:" if isinstance(out, Exception) else "tracked:", job, out)
        if isinstance(out, Exception):
            failed.append(job)

    local = Path(out_dir)
    local.mkdir(parents=True, exist_ok=True)
    evals = evaluate.starmap([(d, trackers) for d in ds], return_exceptions=True)
    for dataset, csv_text in zip(ds, evals):
        if isinstance(csv_text, Exception):
            print(f"\n### {dataset}\n\neval failed: {csv_text}")
            continue
        (local / f"{dataset}.csv").write_text(csv_text)
        print(f"\n### {dataset}\n\n{_markdown(csv_text)}")
    print(f"\nCSVs written to {local.resolve()}")
    if failed:
        print(f"\n{len(failed)} job(s) failed, tables above are incomplete: {failed}")


def _test_markdown():
    assert _markdown("a,b\n1,2\n") == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert _markdown("") == "_(no rows)_"


if __name__ == "__main__":
    _test_markdown()
    print("ok")
