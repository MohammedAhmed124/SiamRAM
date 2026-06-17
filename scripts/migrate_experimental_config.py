#!/usr/bin/env python3
"""
Migrate flat SiamRAM config keys into `ram_tracker_subsystems`.

This script is intentionally non-destructive: by default it writes a sibling
`*.nested.yaml` file and preserves all original flat keys for backward
compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _maybe_get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if key in obj:
        return obj[key]
    return default


def migrate_config(path: Path) -> Any:
    cfg = OmegaConf.load(str(path))
    ram = getattr(cfg, "ram_tracker", OmegaConf.create({}))
    exp = getattr(cfg, "ram_tracker_experiment", OmegaConf.create({}))
    subsystems = OmegaConf.create({})

    subsystems.descriptor = {
        "backend": _maybe_get(ram, "descriptor_backend", "osnet"),
        "osnet_model_name": _maybe_get(ram, "osnet_model_name", "osnet_x1_0"),
        "osnet_pretrained_checkpoint": _maybe_get(
            ram, "osnet_pretrained_checkpoint", "imagenet"
        ),
        "osnet_model_path": _maybe_get(ram, "osnet_model_path", ""),
        "osnet_device": _maybe_get(ram, "osnet_device", "auto"),
        "osnet_max_candidate_batch": _maybe_get(ram, "osnet_max_candidate_batch", 0),
    }
    subsystems.reacquisition = {
        "threshold": _maybe_get(ram, "reacq_threshold", 0.55),
        "confirm_frames": _maybe_get(ram, "reacq_confirm_frames", 1),
        "occ_siam_threshold": _maybe_get(ram, "occ_siam_reacq_threshold", 0.80),
    }
    subsystems.gmc_prior = {
        "enabled": _maybe_get(ram, "gmc_prior_enabled", True),
        "require_reliable_h": _maybe_get(ram, "gmc_prior_require_reliable_h", True),
        "max_translation_frac": _maybe_get(ram, "gmc_prior_max_translation_frac", 0.25),
        "min_scale": _maybe_get(ram, "gmc_prior_min_scale", 0.70),
        "max_scale": _maybe_get(ram, "gmc_prior_max_scale", 1.40),
        "max_rotation_deg": _maybe_get(ram, "gmc_prior_max_rotation_deg", 25.0),
        "max_corner_displacement_frac": _maybe_get(
            ram, "gmc_prior_max_corner_displacement_frac", 0.25
        ),
    }

    cfg.ram_tracker_subsystems = subsystems
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="config/inference_config_experimental.yaml",
        help="Input YAML path.",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Output YAML path. Defaults to <input>.nested.yaml",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file instead of writing a sibling output file.",
    )
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_path = (
        input_path
        if args.in_place
        else Path(args.output).resolve()
        if args.output
        else input_path.with_suffix(".nested.yaml")
    )

    migrated = migrate_config(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(migrated, str(out_path), resolve=False)
    print(f"Wrote migrated config: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
