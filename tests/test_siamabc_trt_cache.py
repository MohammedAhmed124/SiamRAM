from __future__ import annotations

from pathlib import Path

from models.SiamABC.tracker.trt_engine.cache_utils import siamabc_cache_prefix


def _prefix(checkpoint: Path, **overrides) -> str:
    values = {
        "model_config": {
            "_target_": "models.SiamABC.model.SiamABC.SiamABCNet",
            "model_size": "M",
            "towernum": 2,
        },
        "weights_path": str(checkpoint),
        "template_size": 128,
        "instance_size": 256,
        "lambda_tta": 0.1,
        "backbone_mode": "dynamic_fp16",
        "cuda_id": 0,
        "disable_tf32": False,
        "min_batch": 1,
        "opt_batch": 1,
        "max_batch": 1,
        "software_versions": {
            "torch": "2.11.0",
            "torch_tensorrt": "2.11.0",
            "tensorrt": "10.0",
        },
        "gpu_identity": {
            "name": "test-gpu",
            "compute_capability": "8.9",
            "total_memory": 123,
        },
    }
    values.update(overrides)
    return siamabc_cache_prefix(**values)


def test_cache_prefix_changes_for_behavior_affecting_settings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-a")
    base = _prefix(checkpoint)

    variants = [
        _prefix(checkpoint, backbone_mode="static_fp16"),
        _prefix(checkpoint, disable_tf32=True),
        _prefix(checkpoint, template_size=96),
        _prefix(checkpoint, instance_size=224),
        _prefix(checkpoint, lambda_tta=0.2),
        _prefix(checkpoint, cuda_id=1),
        _prefix(checkpoint, max_batch=2),
        _prefix(
            checkpoint,
            model_config={"model_size": "M", "towernum": 3},
        ),
        _prefix(
            checkpoint,
            software_versions={
                "torch": "2.12.0",
                "torch_tensorrt": "2.11.0",
                "tensorrt": "10.0",
            },
        ),
        _prefix(
            checkpoint,
            gpu_identity={
                "name": "other-gpu",
                "compute_capability": "9.0",
                "total_memory": 456,
            },
        ),
    ]
    assert all(prefix != base for prefix in variants)

    checkpoint.write_bytes(b"checkpoint-b")
    assert _prefix(checkpoint) != base
