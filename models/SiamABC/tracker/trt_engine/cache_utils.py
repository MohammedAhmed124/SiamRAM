"""TensorRT cache fingerprinting helpers that do not import TensorRT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


def checkpoint_fingerprint(path_value: str) -> dict[str, Any]:
    """Return a content-based checkpoint fingerprint with metadata fallback."""
    path = Path(path_value)
    result: dict[str, Any] = {"path": str(path.expanduser().resolve())}
    try:
        stat = path.stat()
        result.update(
            {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
        digest = hashlib.sha256()
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    except OSError as exc:
        result["fingerprint_error"] = type(exc).__name__
    return result


def plain_config(value: Any) -> Any:
    """Convert OmegaConf and mapping values into deterministic JSON data."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(key): plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_config(item) for item in value]
    return value


def siamabc_cache_prefix(
    *,
    model_config: Any,
    weights_path: str,
    template_size: int,
    instance_size: int,
    lambda_tta: float,
    backbone_mode: str,
    cuda_id: int,
    disable_tf32: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    software_versions: Mapping[str, str],
    gpu_identity: Mapping[str, Any],
) -> str:
    """Fingerprint every setting that can affect SiamABC TensorRT engines."""
    payload = {
        "kind": "siamabc",
        "cache_schema": 2,
        "model": plain_config(model_config),
        "backbone": {
            "mode": str(backbone_mode),
            "template_size": int(template_size),
            "instance_size": int(instance_size),
            "min_batch": int(min_batch),
            "opt_batch": int(opt_batch),
            "max_batch": int(max_batch),
            "disable_tf32": bool(disable_tf32),
            "torchscript_truncate_long_and_double": True,
        },
        "connector": {
            "lambda_tta": float(lambda_tta),
            "precision": "fp32",
            "optimization_level": 3,
            "use_fast_partitioner": True,
        },
        "cuda_id": int(cuda_id),
        "software": dict(software_versions),
        "gpu": dict(gpu_identity),
        "weights": checkpoint_fingerprint(weights_path),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    model_size = str(payload["model"].get("model_size", "unknown"))
    return f"siamabc_{model_size}_{backbone_mode}_{digest}"
