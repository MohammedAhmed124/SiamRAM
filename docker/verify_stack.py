"""Sanity-check the JetPack 7.2 container stack (Dockerfile.jetson).

Usage (on the Jetson, with the NVIDIA runtime):

    docker compose -f docker-compose.jetson.yml run --rm jetson \
        python docker/verify_stack.py --gpu

Without --gpu (e.g. during an x86/qemu build check) only the CPU-side checks
run: exact package versions, allowlisted `pip check`, cv2 GStreamer support,
and GStreamer element registration. With --gpu it additionally certifies the
two things only the device can prove: an ONNX -> TensorRT engine build +
execution (the same API path raw_trt.py uses), and an NVDEC decode through
the same pipeline shape predictor.py builds.

Exits non-zero if any fatal check fails. NVDEC element registration is
non-fatal without --gpu (predictor.py falls back to CPU decode) and fatal
with --gpu.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

FAILURES: list[str] = []

EXPECTED_VERSIONS = {
    "torch": "2.9.1+cu130",
    "torchvision": "0.24.1",
    "tensorrt": "10.16.1.11",
    "onnx": "1.18.0",
    "ultralytics": "8.4.45",
    "cv2": "4.6.0",
}

# `pip check` complaints that are deliberate (see requirements.jetson7.txt):
# opencv-python must never be pip-installed (system cv2 carries GStreamer),
# torchreid's linters/tb-nightly are dev-only, and pip flags cusparselt's
# platform markers on aarch64 even though torch treats it as optional.
PIP_CHECK_ALLOWED = (
    "requires opencv-python",
    "requires flake8",
    "requires isort",
    "requires tb-nightly",
    "requires yapf",
    "nvidia-cusparselt-cu13",
)


def check(name: str, fn, fatal: bool = True) -> None:
    try:
        detail = fn()
        print(f"[ok]   {name}: {detail}")
    except Exception as exc:  # noqa: BLE001 - report everything, decide via exit code
        print(f"[FAIL] {name}: {exc}")
        if fatal:
            FAILURES.append(name)


def expected_version(module_name: str) -> str:
    import importlib

    module = importlib.import_module(module_name)
    actual = module.__version__
    expected = EXPECTED_VERSIONS[module_name]
    # Some wheels carry a +cu130 local tag only on certain arches (e.g.
    # torchvision 0.24.1+cu130 on x86_64 vs 0.24.1 on aarch64) — accept both.
    if actual != expected and actual.split("+")[0] != expected:
        raise RuntimeError(f"expected {expected}, found {actual}")
    return actual


def pip_check_allowlisted() -> str:
    res = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    unexpected = [
        line for line in res.stdout.splitlines()
        if line.strip() and not any(allowed in line for allowed in PIP_CHECK_ALLOWED)
    ]
    if unexpected:
        raise RuntimeError("; ".join(unexpected))
    return "only the documented, deliberate exceptions"


def torch_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"torch {torch.__version__} present but CUDA unavailable — "
            "container must run with `runtime: nvidia` on the Jetson"
        )
    return f"device={torch.cuda.get_device_name(0)}"


def opencv_gstreamer() -> str:
    import cv2

    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line:
            if "YES" not in line:
                raise RuntimeError(f"cv2 built without GStreamer: {line.strip()}")
            return line.strip()
    raise RuntimeError("GStreamer not mentioned in cv2 build info")


def numpy_abi() -> str:
    import numpy

    if int(numpy.__version__.split(".")[0]) >= 2:
        raise RuntimeError(
            f"numpy {numpy.__version__} — Ubuntu's cv2 is built against the "
            "numpy 1.x ABI; a pip install must have replaced it"
        )
    return numpy.__version__


def app_imports() -> str:
    import polars  # noqa: F401  (ultralytics dependency, lazily imported there)
    import torchreid
    import ultralytics  # noqa: F401
    import mobile_cv  # noqa: F401

    return f"torchreid {torchreid.__version__}, polars {polars.__version__}"


def gst_element(element: str) -> str:
    exe = shutil.which("gst-inspect-1.0")
    if exe is None:
        raise RuntimeError("gst-inspect-1.0 not installed")
    res = subprocess.run(
        [exe, element], capture_output=True, text=True, timeout=60, check=False
    )
    if res.returncode != 0:
        raise RuntimeError("element not found (CPU decode fallback will be used)")
    return "available"


def trt_engine_smoke() -> str:
    """Build and execute a tiny ONNX->TensorRT engine — the raw_trt.py path."""
    import io

    import tensorrt as trt
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU()
    ).eval().cuda()
    x = torch.randn(1, 3, 64, 64, device="cuda")
    buf = io.BytesIO()
    torch.onnx.export(model, (x,), buf, opset_version=17, dynamo=False)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(buf.getvalue()):
        raise RuntimeError(parser.get_error(0))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")

    engine = trt.Runtime(logger).deserialize_cuda_engine(bytes(serialized))
    context = engine.create_execution_context()
    out = torch.empty(1, 8, 64, 64, device="cuda")
    context.set_tensor_address(engine.get_tensor_name(0), x.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), out.data_ptr())
    stream = torch.cuda.Stream()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("execute_async_v3 failed")
    stream.synchronize()
    expected = torch.nn.functional.relu(model[0](x))
    if not torch.allclose(out, expected, atol=1e-2):
        raise RuntimeError("engine output mismatch vs torch reference")
    return f"built + executed, TRT {trt.__version__}"


def nvdec_smoke() -> str:
    """Decode a generated clip through the same pipeline predictor.py uses."""
    import cv2
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="nvdec_smoke_")
    path = os.path.join(tmp, "clip.mp4")
    written = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*fourcc), 25, (320, 240)
        )
        if writer.isOpened():
            rng = np.random.default_rng(0)
            for _ in range(25):
                writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
            writer.release()
            written = fourcc
            break
        writer.release()
    if written is None:
        raise RuntimeError("cv2.VideoWriter could not encode a test clip")

    pipeline = (
        f"filesrc location={path} ! qtdemux ! queue ! parsebin ! "
        "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
        "appsink sync=false drop=false max-buffers=2"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"NVDEC pipeline failed to open ({written} clip)")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("NVDEC pipeline opened but produced no frame")
        return f"decoded {written} -> frame {frame.shape}"
    finally:
        cap.release()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    gpu = "--gpu" in sys.argv[1:]
    print(f"[ok]   python: {sys.version.split()[0]}")

    for module_name in EXPECTED_VERSIONS:
        check(f"{module_name}=={EXPECTED_VERSIONS[module_name]}",
              lambda m=module_name: expected_version(m))
    check("numpy<2 (cv2 ABI)", numpy_abi)
    check("cv2 GStreamer", opencv_gstreamer)
    check("app imports (torchreid/ultralytics/polars/mobile_cv)", app_imports)
    check("pip check (allowlisted)", pip_check_allowlisted)
    for element in ("qtdemux", "parsebin", "nvv4l2decoder", "nvvidconv"):
        check(f"gst:{element}", lambda e=element: gst_element(e),
              fatal=gpu or element in ("qtdemux", "parsebin"))

    if gpu:
        check("torch CUDA", torch_cuda)
        check("TensorRT engine build+run", trt_engine_smoke)
        check("NVDEC decode", nvdec_smoke)
    else:
        print("[note] run with --gpu on the Jetson to certify TensorRT engine "
              "build/run and NVDEC decode (not possible off-device)")

    if FAILURES:
        print(f"\n{len(FAILURES)} fatal check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("\nAll fatal checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
